"""Shared pytest fixtures for the Linux Whisper test suite.

Provides helpers for temporary config files, mock audio data, and
import-guarded fixtures that work even when optional dependencies
(evdev, sounddevice, pystray, moonshine, llama_cpp, onnxruntime)
are not installed.
"""

from __future__ import annotations

import itertools
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Ensure optional heavy dependencies can be faked during test collection.
# We insert lightweight stubs into sys.modules BEFORE any linux_whisper code
# tries to import them, so that import-time guards (`try: import X`) see a
# usable (mock) module rather than raising ImportError.
# ---------------------------------------------------------------------------

_OPTIONAL_DEPS = [
    "evdev",
    "sounddevice",
    "pystray",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "faster-whisper",
    "llama_cpp",
    "whispercpp",
    # onnxruntime is used in disfluency/punctuation but we test regex fallback
    # so we do NOT stub it here — the code's own try/except handles ImportError.
]


def _ensure_stub(name: str) -> None:
    """Insert a MagicMock into sys.modules if the real package is missing."""
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = MagicMock()


for _dep in _OPTIONAL_DEPS:
    _ensure_stub(_dep)

# After stubbing, make sure evdev has the constants the hotkey module needs
_evdev_stub = sys.modules.get("evdev")
if isinstance(_evdev_stub, MagicMock):
    # Create a proper ecodes sub-attribute with real integer constants
    ecodes = types.SimpleNamespace(
        EV_KEY=1,
        KEY_LEFTCTRL=29,
        KEY_RIGHTCTRL=97,
        KEY_LEFTSHIFT=42,
        KEY_RIGHTSHIFT=54,
        KEY_LEFTALT=56,
        KEY_RIGHTALT=100,
        KEY_LEFTMETA=125,
        KEY_RIGHTMETA=126,
        KEY_A=30,
        KEY_B=48,
        KEY_E=18,
        KEY_F1=59,
        KEY_ENTER=28,
        KEY_SPACE=57,
    )
    _evdev_stub.ecodes = ecodes
    _evdev_stub.InputDevice = MagicMock
    _evdev_stub.InputEvent = MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Return a temporary directory suitable for config files."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    return cfg_dir


@pytest.fixture()
def tmp_config_file(tmp_config_dir: Path) -> Path:
    """Write a minimal YAML config and return its path."""
    cfg = {
        "hotkey": "fn",
        "mode": "auto",
        "stt": {"backend": "whisper-cpp", "model": "whisper-large-v3-turbo"},
        "polish": {"enabled": True},
        "audio": {"sample_rate": 16000},
        "inject": {"method": "auto"},
        "tray": {"enabled": False},
    }
    path = tmp_config_dir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


@pytest.fixture()
def empty_config_file(tmp_config_dir: Path) -> Path:
    """Write an empty YAML config and return its path."""
    path = tmp_config_dir / "config.yaml"
    path.write_text("")
    return path


@pytest.fixture()
def mock_audio_f32():
    """Return a factory that creates numpy float32 audio arrays."""
    import numpy as np

    def _make(n_samples: int = 1600, frequency: float = 440.0, sr: int = 16000):
        t = np.linspace(0, n_samples / sr, n_samples, endpoint=False, dtype=np.float32)
        return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    return _make


class _FakeGLibSource:
    """Stand-in for a `GLib.Source` (from `idle_source_new`/`timeout_source_new`).

    Real GLib only invokes a source's callback once the main loop it is
    attached to actually runs. Tests have no such loop pumping, so `attach()`
    here invokes the callback immediately and once — but critically, only
    `attach()` does that, not `set_callback()`. That means a test can tell
    the difference between code that dispatches through the overlay's own
    context (`Overlay._dispatch`, which calls `set_callback` then `attach`)
    and code that (incorrectly) mutates the window directly: only the former
    shows up as a recorded `attach()` call.

    Nothing in `overlay.py` uses `idle_source_new`/`timeout_source_new`
    anymore (see `_next_glib_source_id` below) — this is kept only in case a
    future addition needs the lower-level Source API again.
    """

    def __init__(self) -> None:
        self.callback = None
        self.args: tuple = ()
        self.attached_to: object | None = None
        self.destroyed = False

    def set_callback(self, callback, *args):
        self.callback = callback
        self.args = args

    def attach(self, context):
        self.attached_to = context
        if self.callback is not None:
            self.callback(*self.args)
        return 1  # fake source id

    def destroy(self):
        self.destroyed = True


class _FakeMainLoop:
    """Stand-in for `GLib.MainLoop`, real enough for threading tests to mean
    something: `run()` blocks the calling (overlay) thread until `quit()` is
    called, rather than returning instantly the way a bare `MagicMock()`
    would.

    That blocking matters: without it, `Overlay._run_gtk()`'s teardown
    (destroy the window, clear `self._window` — see the MAJOR finding on the
    dangling window reference) would race every test that starts the
    overlay and inspects it before calling `stop()` — a plain MagicMock
    `run()` returns immediately, so the background thread can race straight
    through teardown before the test's assertion runs at all. `quit()` is
    safe to call from another thread, same as the real API, and is a no-op
    if `run()` was never entered — matching real GLib, a stray `quit()`
    doesn't raise.
    """

    def __init__(self, context: object = None) -> None:
        self.context = context
        self._quit_event = threading.Event()
        self.run = MagicMock(side_effect=self._run)
        self.quit = MagicMock(side_effect=self._quit)

    def _run(self) -> None:
        self._quit_event.wait(timeout=5.0)

    def _quit(self) -> None:
        self._quit_event.set()


@pytest.fixture()
def mock_gtk(monkeypatch):
    """Replace linux_whisper.overlay's GTK bindings with MagicMocks.

    overlay.py talks to GTK3/Gdk directly (module-level `Gtk`/`Gdk`/`GLib`
    names) rather than through an injectable interface, so tests that need
    to exercise window-construction code paths patch those names instead of
    relying on whatever GTK happens to be installed on the host — this
    machine has a real GTK3 + a real display, but CI does not, and no test
    may depend on either being present.
    """
    from unittest.mock import MagicMock

    from linux_whisper import overlay as overlay_module

    fake_gtk = MagicMock()
    fake_gdk = MagicMock()
    fake_glib = MagicMock()

    # GLib.idle_source_new()/timeout_source_new() return a fresh fake source
    # each call; attach() runs the callback synchronously (see _FakeGLibSource)
    # so tests don't need to pump a real main loop. Nothing in overlay.py
    # calls these anymore (it uses plain idle_add/timeout_add — see below,
    # attached to the *default* main context, which is the whole point of
    # the BLOCKER fix), but they're kept working in case something needs
    # the lower-level Source API again.
    fake_glib.idle_source_new.side_effect = lambda: _FakeGLibSource()
    fake_glib.timeout_source_new.side_effect = lambda *a, **kw: _FakeGLibSource()
    fake_glib.SOURCE_REMOVE = False
    fake_glib.SOURCE_CONTINUE = True

    # GLib.idle_add()/timeout_add() attach to the process-wide default main
    # context (unlike idle_source_new()/timeout_source_new(), which need an
    # explicit .attach(context)) — real GLib only invokes the callback once
    # something actually pumps that context. Tests have no real loop
    # pumping, so invoke synchronously here, same rationale as
    # _FakeGLibSource.attach() above. Each call gets a distinct fake GLib
    # source id (a plain incrementing int, like the real API) so
    # GLib.source_remove() calls can be matched back to the timeout_add()
    # call that produced them.
    _next_glib_source_id = itertools.count(1)

    def _fake_idle_add(fn, *args, **kwargs):
        fn(*args)
        return next(_next_glib_source_id)

    def _fake_timeout_add(interval, fn, *args, **kwargs):
        fn(*args)
        return next(_next_glib_source_id)

    fake_glib.idle_add.side_effect = _fake_idle_add
    fake_glib.timeout_add.side_effect = _fake_timeout_add
    fake_glib.source_remove.return_value = True

    # GLib.MainLoop(context) — a real blocking stand-in (see _FakeMainLoop)
    # so `_run_gtk`'s call to `loop.run()` behaves enough like the real
    # thing for tests to exercise the start/stop lifecycle meaningfully,
    # instead of the whole GTK thread completing (and tearing itself down)
    # before the test even gets to assert anything about it.
    fake_glib.MainLoop.side_effect = lambda context=None: _FakeMainLoop(context)

    # Gdk.Display.get_default() reports a class named "X11Display" — the
    # overlay's backend check (Overlay._run_gtk) matches on the substring
    # "X11" in the type name, the same way the real GdkX11Display/
    # GdkWaylandDisplay classes are named. Tests that need to exercise the
    # non-X11 path override this return value.
    x11_display_cls = type("X11Display", (MagicMock,), {})
    fake_display = x11_display_cls()

    # A concrete (not auto-mocked) pointer/monitor geometry, so
    # `_monitor_geometry_at_pointer()` — exercised for real now that
    # `set_recording(True)` calls `_reposition()` directly — has real ints
    # to do arithmetic on instead of chained MagicMocks.
    fake_pointer = MagicMock()
    fake_pointer.get_position.return_value = (None, 960, 540)
    fake_seat = MagicMock()
    fake_seat.get_pointer.return_value = fake_pointer
    fake_display.get_default_seat.return_value = fake_seat
    fake_geometry = types.SimpleNamespace(x=0, y=0, width=1920, height=1080)
    fake_monitor = MagicMock()
    fake_monitor.get_geometry.return_value = fake_geometry
    fake_display.get_monitor_at_point.return_value = fake_monitor
    fake_gdk.Display.get_default.return_value = fake_display

    monkeypatch.setattr(overlay_module, "Gtk", fake_gtk)
    monkeypatch.setattr(overlay_module, "Gdk", fake_gdk)
    monkeypatch.setattr(overlay_module, "GLib", fake_glib)
    monkeypatch.setattr(overlay_module, "_HAS_GTK", True)
    monkeypatch.setattr(overlay_module, "_UNAVAILABLE_REASON", "")

    return types.SimpleNamespace(Gtk=fake_gtk, Gdk=fake_gdk, GLib=fake_glib)


@pytest.fixture()
def mock_audio_pcm_bytes(mock_audio_f32):
    """Return a factory that creates 16-bit PCM audio bytes."""
    import numpy as np

    def _make(n_samples: int = 1600, **kwargs):
        audio_f32 = mock_audio_f32(n_samples=n_samples, **kwargs)
        audio_i16 = (audio_f32 * 32767).astype(np.int16)
        return audio_i16.tobytes()

    return _make
