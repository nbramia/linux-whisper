"""Floating pill overlay showing recording state and live audio levels.

Displays a small semi-transparent pill on screen:
- Hidden when idle (not recording)
- Visible with muted indicators when recording but no speech detected
- Visible with animated audio level bars when speech is detected

GTK4 has no window positioning API on Wayland (``gtk_window_move()`` was
removed) and Mutter (GNOME's compositor) does not implement
``wlr-layer-shell``, so a native-Wayland or layer-shell overlay cannot be
positioned on this desktop. The one path that works, verified on GNOME 46 /
Mutter: GTK3 running through XWayland, using an override-redirect
``Gtk.WindowType.POPUP`` window. That window type never takes input focus —
load-bearing, since text injection targets whatever window *does* have
focus, and a focus-stealing overlay would swallow dictated text instead of
delivering it.

Getting to XWayland requires ``GDK_BACKEND=x11`` set **before this process
imports anything GTK-related at all** — not merely before a window is
constructed. PyGObject resolves and locks in the GDK backend as a side
effect of ``from gi.repository import Gdk`` (or ``Gtk``) itself, at *import*
time — verified against a real GNOME/Wayland session, this happens even
with no ``Gtk.init()`` call and no window ever constructed. Since this
module's own top-level import block below does exactly that, the env var
must be set by whatever imports this module for the first time, *before*
that import — see the comment in ``app.py``'s ``setup()``, which sets it
before importing either this module or ``tray.py`` (whose GTK3-backed
pystray backend imports Gdk/Gtk too, and starts its own thread first).
Setting it inside this module, even at the top of a function that runs
before any window is built, is too late: importing this module to reach
that function already ran the lines below first.

Runs entirely in its own daemon thread, holding its own ``GLib.MainLoop``.
That loop is bound to the process-wide *default* ``GLib.MainContext`` —
**not** a private one. A private context was tried first (to keep this
overlay fully isolated from the tray's GTK loop) and broke rendering
outright: GTK3's draw/expose dispatch is wired to the default context only,
so a private context never delivers it. Verified empirically against a real
window on a real display with the tray disabled: the animation timer fired
at a correct 30fps and called ``queue_draw()`` 60 times, and the ``draw``
handler was invoked zero times. Switching the same window back to the
default context, with everything else unchanged, produced 45+ draws.

The trade-off this leaves: the overlay and the system tray's GTK loop now
share one ``GLib.MainContext``, so a GTK source attached here (the 30fps
animation timer, the marshalled state-mutation calls below) may end up
dispatched by whichever thread's ``MainLoop.run()`` is currently iterating
that shared context — this overlay's own thread, or the tray's, if the tray
is enabled. That is inherent to running two GLib loops in one process, and
it is exposure the tray already introduces on its own; it is not something
a second main loop object avoids. What *is* avoided by holding our own
``GLib.MainLoop`` instance (instead of calling ``Gtk.main()``/
``Gtk.main_quit()``) is the sharper bug that motivated this: ``stop()``
quitting the tray's loop instead of (or as well as) this one.
``GLib.MainLoop.quit()`` only quits that specific loop object, so this
overlay's ``stop()`` cannot take the tray down with it. Every GTK-state
mutation is still routed through ``GLib.idle_add()``/``GLib.timeout_add()``
(see ``Overlay._dispatch()`` and ``_OverlayWindow._start_tick()``) rather
than called directly across threads, so dispatch is at least consistently
serialised through the context regardless of which thread services it.

Isolated from the asyncio event loop and the real-time audio callback
thread — both of those never touch GTK at all, only this dedicated thread
does.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque

from linux_whisper.config import OverlayConfig

logger = logging.getLogger(__name__)

_HAS_GTK = False
_UNAVAILABLE_REASON = "GTK 3.0 / PyGObject (python3-gi) not installed"

Gtk = None
Gdk = None
GLib = None

try:
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    import cairo
    from gi.repository import Gdk, GLib, Gtk  # noqa: F811 (populated on success)

    _HAS_GTK = True
except (ImportError, ValueError) as exc:
    _UNAVAILABLE_REASON = f"GTK 3.0 unavailable: {exc}"
    logger.debug("GTK3 not available — overlay disabled: %s", exc)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PILL_WIDTH = 200
_PILL_HEIGHT = 40
_PILL_RADIUS = 20
_BAR_COUNT = 16  # number of audio level bars
_BAR_WIDTH = 6
_BAR_GAP = 3
_BAR_MIN_HEIGHT = 4
_BAR_MAX_HEIGHT = 28
_MARGIN = 48  # pixels from the edge of the monitor for top/bottom placement
_FPS = 30
_LEVEL_HISTORY = 32  # frames of audio level history for smoothing
_IDLE_LEVEL = 0.06   # resting bar height when the mic is quiet
_BAR_ATTACK = 0.55   # rise quickly toward a louder sample
_BAR_RELEASE = 0.18  # fall back gently, so speech reads as continuous

# cairo operator constants, spelled out numerically to avoid a hard `import
# cairo` dependency purely for two constants (pycairo ships with PyGObject's
# cairo integration, but is not otherwise used by this module).
_CAIRO_OPERATOR_CLEAR = 0
_CAIRO_OPERATOR_OVER = 2


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------


class _Colors:
    # Pill background
    BG = (0.1, 0.1, 0.12, 0.85)
    # Border
    BORDER = (0.3, 0.3, 0.35, 0.6)
    # Bars when speech detected
    BAR_ACTIVE = (0.35, 0.75, 0.55, 0.9)  # green
    BAR_ACTIVE_PEAK = (0.45, 0.9, 0.65, 1.0)
    # Bars when listening but no speech
    BAR_IDLE = (0.4, 0.4, 0.45, 0.4)  # dim gray


# ---------------------------------------------------------------------------
# Positioning
# ---------------------------------------------------------------------------


def compute_pill_position(
    monitor_x: int,
    monitor_y: int,
    monitor_width: int,
    monitor_height: int,
    position: str,
    pill_width: int = _PILL_WIDTH,
    pill_height: int = _PILL_HEIGHT,
    margin: int = _MARGIN,
) -> tuple[int, int]:
    """Compute the top-left (x, y) window coordinates for the pill.

    Pure function of monitor geometry and config — no GTK/Gdk dependency, so
    it can be unit-tested against a fake monitor without a display.
    Horizontally always centred; ``position`` controls the vertical anchor.
    Unrecognised positions fall back to "center" (config validation is
    responsible for rejecting bad values before they reach here). The result
    is clamped to the monitor's bounds, so a narrow or short monitor can't
    push the pill partly off-screen.
    """
    x = monitor_x + (monitor_width - pill_width) // 2
    if position == "top-center":
        y = monitor_y + margin
    elif position == "bottom-center":
        y = monitor_y + monitor_height - pill_height - margin
    else:
        y = monitor_y + (monitor_height - pill_height) // 2

    x = _clamp(x, monitor_x, monitor_x + monitor_width - pill_width)
    y = _clamp(y, monitor_y, monitor_y + monitor_height - pill_height)
    return x, y


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp ``value`` to ``[lo, hi]``. If the range is inverted (the pill is
    larger than the monitor on this axis), pin to ``lo`` rather than raise."""
    if hi < lo:
        return lo
    return max(lo, min(value, hi))


def _monitor_geometry_at_pointer() -> tuple[int, int, int, int] | None:
    """Return (x, y, width, height) of the monitor under the pointer.

    Uses the pointer rather than window focus: on this machine
    `xdotool getactivewindow` fails because the focused window is a native
    Wayland surface, so X11 focus queries cannot resolve the active monitor.
    The pointer position is reliable under XWayland regardless.
    """
    display = Gdk.Display.get_default()
    if display is None:
        return None
    seat = display.get_default_seat()
    if seat is None:
        return None
    pointer = seat.get_pointer()
    if pointer is None:
        return None
    _, px, py = pointer.get_position()
    monitor = display.get_monitor_at_point(px, py)
    if monitor is None:
        return None
    geom = monitor.get_geometry()
    return geom.x, geom.y, geom.width, geom.height


# ---------------------------------------------------------------------------
# Overlay window
# ---------------------------------------------------------------------------


class _OverlayWindow:
    """Wraps a GTK3 override-redirect popup window that draws the pill.

    Composition rather than subclassing `Gtk.Window`: this class must be
    importable and constructible in tests even when GTK is mocked out, and a
    mock object cannot be used as a base class.
    """

    def __init__(self, position: str) -> None:
        self._position = position
        # GLib source id from GLib.timeout_add() (see _start_tick) — attached
        # to the default main context only while the pill is visible, so it
        # doesn't tick at 30fps forever while hidden. None means "not ticking".
        self._tick_source: int | None = None

        self._window = Gtk.Window(type=Gtk.WindowType.POPUP)
        self._window.set_title("linux-whisper-overlay")
        self._window.set_decorated(False)
        self._window.set_resizable(False)
        self._window.set_default_size(_PILL_WIDTH, _PILL_HEIGHT)
        # set_default_size alone is not enough: a POPUP with no child
        # widget has no natural size to shrink to, and GTK3 allocates a
        # 200x200 square instead of the 200x40 pill. The size request
        # pins it, and also keeps the 30fps redraw to a fifth of the
        # pixels -- this loop holds the GIL, so its cost is not free.
        self._window.set_size_request(_PILL_WIDTH, _PILL_HEIGHT)
        self._window.set_app_paintable(True)
        self._window.set_keep_above(True)
        self._window.set_skip_taskbar_hint(True)
        self._window.set_skip_pager_hint(True)

        # Never take input focus. Text injection targets the focused
        # window; a focus-stealing overlay would swallow dictated text.
        self._window.set_accept_focus(False)
        self._window.set_focus_on_map(False)
        self._window.set_can_focus(False)

        screen = self._window.get_screen()
        if screen is not None:
            visual = screen.get_rgba_visual()
            if visual is not None:
                self._window.set_visual(visual)

        self._window.connect("draw", self._draw)

        # State
        self._visible_state = False
        self._speech_active = False
        self._audio_levels: deque[float] = deque(
            [0.0] * _LEVEL_HISTORY, maxlen=_LEVEL_HISTORY
        )
        self._bar_heights: list[float] = [0.0] * _BAR_COUNT
        self._phase: float = 0.0  # animation phase
        self._lock = threading.Lock()

    # -- entry points, always run on the overlay's own GTK thread (marshalled
    #    there by Overlay._dispatch — see the facade below) --

    def set_recording(self, active: bool) -> None:
        """Show or hide the pill. Runs on the overlay thread only — GTK
        calls (`show_all`/`hide`/`move`) are not thread-safe, so this must
        never be invoked directly from another thread."""
        with self._lock:
            was_active = self._visible_state
            self._visible_state = active
            if not active:
                self._speech_active = False
                self._audio_levels.clear()
                self._audio_levels.extend([0.0] * _LEVEL_HISTORY)

        if active and not was_active:
            # Position first, THEN reveal. The window is already mapped (see
            # prime()); showing is only an opacity change, so the compositor
            # never has to create a surface on the hot path.
            self._reposition()
            self._window.set_opacity(1.0)
            self._start_tick()
        elif not active and was_active:
            self._stop_tick()
            self._window.set_opacity(0.0)

    def prime(self) -> None:
        """Map the window once, fully transparent, and keep it mapped.

        Measured cost of the old map-on-demand path, from the keypress:
        audio 0.9ms, show() dispatch 0.1ms, show_all() 6.0ms, reposition
        2.5ms -- under 10ms in-process, against roughly a second before the
        pill was actually visible. The gap is Mutter creating and presenting
        a brand-new XWayland surface each time the window was mapped, which
        nothing inside this process can measure or speed up.

        So the surface is created once at startup and never torn down;
        show/hide became an opacity change. The window is given an empty
        input region so that, although permanently mapped, it can never
        receive a pointer event -- without that, a 200x40 dead zone would sit
        over the bottom of the screen swallowing clicks.
        """
        self._window.set_opacity(0.0)
        self._reposition()
        self._window.show_all()

        # Empty input region == click-through. Must happen after realize.
        gdk_window = self._window.get_window()
        if gdk_window is not None:
            gdk_window.input_shape_combine_region(cairo.Region(), 0, 0)

    def set_speech_active(self, active: bool) -> None:
        with self._lock:
            self._speech_active = active

    def push_audio_level(self, level: float) -> None:
        """Record one audio sample. Called directly from the async audio
        monitor thread, not marshalled — it must never block that thread, so
        it drops the sample rather than waiting for a tick in progress."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._audio_levels.append(min(1.0, max(0.0, level)))
        finally:
            self._lock.release()

    def destroy(self) -> None:
        self._stop_tick()
        self._window.destroy()

    def _reposition(self) -> None:
        """Move the window to its configured position. Must be re-asserted
        after show_all() — GTK/the window manager can reposition on map."""
        geom = _monitor_geometry_at_pointer()
        if geom is None:
            return
        x, y = compute_pill_position(*geom, self._position)
        self._window.move(x, y)

    # -- animation timer, attached to the default main context (see module
    #    docstring) only while the pill is visible --

    def _start_tick(self) -> None:
        if self._tick_source is not None:
            return
        self._tick_source = GLib.timeout_add(1000 // _FPS, self._tick)

    def _stop_tick(self) -> None:
        if self._tick_source is not None:
            GLib.source_remove(self._tick_source)
            self._tick_source = None

    def _tick(self) -> bool:
        """GLib timeout callback while the pill is visible. Returns True to
        keep running — the source is explicitly destroyed in `_stop_tick()`
        rather than by returning False, so it can be re-attached on the next
        `set_recording(True)`."""
        self._update_bars()
        self._window.queue_draw()
        return True

    def _update_bars(self) -> None:
        """Update bar heights from audio level history."""
        with self._lock:
            levels = list(self._audio_levels)
            self._phase += 0.1

        # Bars always follow the measured level. They used to be driven by the
        # `speech` flag instead -- audio-reactive while it was true, and a
        # decorative sine "breathing" animation whenever it was false. Since
        # that flag went permanently false a few seconds into any continuous
        # utterance, the pill spent most of a dictation animating something
        # unrelated to the microphone. `speech` now only picks the colour.
        n = len(levels)
        for i in range(_BAR_COUNT):
            # Oldest sample at the left edge, newest at the right, so the pill
            # reads as a waveform scrolling past.
            idx = min(int((i / _BAR_COUNT) * n), n - 1)
            target = levels[idx]

            if target < _IDLE_LEVEL:
                # Near-silence: a low shimmer, so the pill still looks live
                # while you are thinking rather than sitting completely flat.
                target = _IDLE_LEVEL + 0.03 * math.sin(self._phase * 0.5 + i * 0.3)

            # Fast attack, slower release -- a meter that rises instantly and
            # falls gently tracks speech far better than symmetric smoothing.
            current = self._bar_heights[i]
            rate = _BAR_ATTACK if target > current else _BAR_RELEASE
            self._bar_heights[i] += (target - current) * rate

    def _draw(self, widget: object, cr: object) -> bool:
        """Draw the pill with audio level bars."""
        width = self._window.get_allocated_width()
        height = self._window.get_allocated_height()

        cr.set_operator(_CAIRO_OPERATOR_CLEAR)
        cr.paint()
        cr.set_operator(_CAIRO_OPERATOR_OVER)

        self._draw_rounded_rect(cr, 0, 0, width, height, _PILL_RADIUS)
        cr.set_source_rgba(*_Colors.BG)
        cr.fill_preserve()
        cr.set_source_rgba(*_Colors.BORDER)
        cr.set_line_width(1.0)
        cr.stroke()

        with self._lock:
            speech = self._speech_active
            bars = list(self._bar_heights)

        total_bar_width = _BAR_COUNT * _BAR_WIDTH + (_BAR_COUNT - 1) * _BAR_GAP
        start_x = (width - total_bar_width) / 2
        center_y = height / 2

        for i in range(_BAR_COUNT):
            h = _BAR_MIN_HEIGHT + bars[i] * (_BAR_MAX_HEIGHT - _BAR_MIN_HEIGHT)
            x = start_x + i * (_BAR_WIDTH + _BAR_GAP)
            y = center_y - h / 2

            if speech:
                t = bars[i]
                r = _Colors.BAR_ACTIVE[0] + t * (_Colors.BAR_ACTIVE_PEAK[0] - _Colors.BAR_ACTIVE[0])
                g = _Colors.BAR_ACTIVE[1] + t * (_Colors.BAR_ACTIVE_PEAK[1] - _Colors.BAR_ACTIVE[1])
                b = _Colors.BAR_ACTIVE[2] + t * (_Colors.BAR_ACTIVE_PEAK[2] - _Colors.BAR_ACTIVE[2])
                a = _Colors.BAR_ACTIVE[3]
                cr.set_source_rgba(r, g, b, a)
            else:
                cr.set_source_rgba(*_Colors.BAR_IDLE)

            self._draw_rounded_rect(cr, x, y, _BAR_WIDTH, h, 2)
            cr.fill()

        return True

    @staticmethod
    def _draw_rounded_rect(
        cr: object, x: float, y: float, w: float, h: float, r: float
    ) -> None:
        """Draw a rounded rectangle path."""
        pi = math.pi
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, pi / 2)
        cr.arc(x + r, y + h - r, r, pi / 2, pi)
        cr.arc(x + r, y + r, r, pi, 3 * pi / 2)
        cr.close_path()


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class Overlay:
    """Public API for the recording overlay.

    Thread-safe: all methods can be called from any thread. The GTK event
    loop runs in a dedicated daemon thread, holding its own `GLib.MainLoop`
    bound to the process-wide default `GLib.MainContext` (see the module
    docstring for why it must be the default context, and the trade-off that
    comes with sharing it with the system tray's GTK loop) — isolated from
    the asyncio loop and the real-time audio callback thread.
    """

    def __init__(self, config: OverlayConfig | None = None) -> None:
        self._config = config or OverlayConfig()
        self._window: _OverlayWindow | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._main_loop: object | None = None

    @property
    def available(self) -> bool:
        return _HAS_GTK

    @property
    def unavailable_reason(self) -> str:
        return _UNAVAILABLE_REASON

    def start(self) -> None:
        """Start the overlay in a background thread.

        Does not silently claim success: if the GTK thread never signals
        ready, or signals ready but failed to produce a window (backend
        mismatch, GTK init failure — see `_run_gtk`), this logs a WARNING
        naming what happened instead of an unconditional "Overlay started".
        """
        if not _HAS_GTK:
            logger.info("GTK not available — overlay disabled: %s", _UNAVAILABLE_REASON)
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._ready.clear()
        self._thread = threading.Thread(target=self._run_gtk, name="overlay", daemon=True)
        self._thread.start()
        became_ready = self._ready.wait(timeout=5.0)
        if not became_ready:
            logger.warning("Overlay startup timed out after 5s — GTK thread unresponsive")
            return
        if self._window is None:
            logger.warning("Overlay failed to start — running without recording pill")
            return
        logger.info("Overlay started")

    def stop(self) -> None:
        """Stop the overlay: quit its own main loop and join its thread."""
        if self._thread is None:
            logger.info("Overlay stopped")
            return

        if self._main_loop is not None:
            # Schedule the quit as an idle source on the default context
            # rather than calling `MainLoop.quit()` directly. `_ready` is
            # set (in `_run_gtk`) before the thread reaches `loop.run()`, so
            # a caller that starts and immediately stops can reach this line
            # before `run()` has actually started. GLib does not carry a
            # pre-quit forward — `quit()` on a not-yet-running loop is a
            # no-op, and the thread would then be parked in `run()` forever,
            # with every later `start()` becoming a no-op because that
            # thread is still alive. Queuing the quit as a pending source on
            # the context `run()` is about to iterate has no such gap: the
            # source exists the moment `idle_add()` returns, so it fires as
            # soon as `run()` starts pumping, whatever the exact timing.
            loop = self._main_loop
            GLib.idle_add(loop.quit)

        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            logger.warning(
                "Overlay thread did not exit within 3s — abandoning join "
                "(it is a daemon thread, so it will not block process exit)"
            )
        else:
            self._thread = None
            self._main_loop = None
        logger.info("Overlay stopped")

    def show(self) -> None:
        """Show the pill (recording started)."""
        if self._window is not None:
            self._dispatch(self._window.set_recording, True)

    def hide(self) -> None:
        """Hide the pill (recording stopped)."""
        if self._window is not None:
            self._dispatch(self._window.set_recording, False)

    def set_speech_active(self, active: bool) -> None:
        """Update whether speech is currently detected."""
        if self._window is not None:
            self._dispatch(self._window.set_speech_active, active)

    def push_audio_level(self, level: float) -> None:
        """Push a new audio level (0.0-1.0) for visualization.

        Called directly, not marshalled through `_dispatch()`: this runs on
        the async audio monitor loop, which must never block on the overlay
        thread. `_OverlayWindow.push_audio_level()` is itself non-blocking —
        it drops the sample instead of waiting for a lock held by a tick in
        progress.
        """
        if self._window is not None:
            self._window.push_audio_level(level)

    def _dispatch(self, fn: object, *args: object) -> None:
        """Marshal a call onto the overlay's own GTK thread.

        `GLib.idle_add()` attaches to the process-wide default main
        context — see the module docstring for why that is deliberate here
        (GTK3 draw dispatch only fires on the default context) and for the
        trade-off it carries (the callback may run on whichever thread is
        currently iterating that context, this overlay's own or the tray's).
        Either way it is never a direct cross-thread call into GTK.
        """
        GLib.idle_add(fn, *args)

    def _run_gtk(self) -> None:
        """GTK setup and main loop — runs entirely on the overlay thread.

        `GDK_BACKEND=x11` must already be set in the environment (`App.setup()`
        does this before importing this module or `tray.py` at all — see the
        module docstring) — by the time this thread runs, the backend is
        whatever it is going to be. Verify it rather than assume: if
        something beat us to opening the display natively on Wayland, the
        window would report requested-but-refused positions (see the module
        docstring's positioning caveat), which is worse than no pill at all.
        """
        window: _OverlayWindow | None = None
        loop: object | None = None
        try:
            window = _OverlayWindow(self._config.position)
            backend = type(Gdk.Display.get_default()).__name__
            if "X11" not in backend:
                raise RuntimeError(
                    f"expected the X11 GDK backend, got {backend!r} — "
                    "GDK_BACKEND=x11 did not take effect before the display "
                    "opened (see App.setup() and this module's docstring)"
                )
            # Backend is confirmed good — create and keep the compositor
            # surface up front, transparent and click-through, so revealing
            # the pill later is only an opacity change rather than a fresh
            # XWayland surface (which took ~1s to present).
            window.prime()

            # Bind to the default main context (pass None), not a private
            # one — see the module docstring: GTK3's draw/expose dispatch is
            # only ever delivered through the default context, and a private
            # context silently never renders anything.
            loop = GLib.MainLoop(None)
        except Exception as exc:
            logger.warning("Overlay disabled: %s", exc)
            if window is not None:
                window.destroy()
            window = None
        finally:
            self._window = window
            self._main_loop = loop
            # Always signal readiness — success or failure — so start()
            # never blocks the full timeout waiting on a thread that has
            # already finished failing.
            self._ready.set()

        if loop is None:
            return

        logger.debug("Overlay window created")
        try:
            loop.run()
        except Exception:
            logger.exception("Overlay GTK loop crashed")
        finally:
            if self._window is not None:
                self._window.destroy()
                self._window = None
