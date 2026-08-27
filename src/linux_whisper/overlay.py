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

Runs entirely in its own daemon thread with its own GTK main loop, isolated
from the asyncio event loop and the real-time audio callback thread.
"""

from __future__ import annotations

import logging
import math
import os
import threading
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
    responsible for rejecting bad values before they reach here).
    """
    x = monitor_x + (monitor_width - pill_width) // 2
    if position == "top-center":
        y = monitor_y + margin
    elif position == "bottom-center":
        y = monitor_y + monitor_height - pill_height - margin
    else:
        y = monitor_y + (monitor_height - pill_height) // 2
    return x, y


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

        self._window = Gtk.Window(type=Gtk.WindowType.POPUP)
        self._window.set_title("linux-whisper-overlay")
        self._window.set_decorated(False)
        self._window.set_resizable(False)
        self._window.set_default_size(_PILL_WIDTH, _PILL_HEIGHT)
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

    # -- thread-safe state mutation (called via GLib.idle_add or directly) --

    def set_recording(self, active: bool) -> None:
        with self._lock:
            self._visible_state = active
            if not active:
                self._speech_active = False
                self._audio_levels.clear()
                self._audio_levels.extend([0.0] * _LEVEL_HISTORY)

    def set_speech_active(self, active: bool) -> None:
        with self._lock:
            self._speech_active = active

    def push_audio_level(self, level: float) -> None:
        with self._lock:
            self._audio_levels.append(min(1.0, max(0.0, level)))

    # -- GTK main-loop callbacks (always run on the overlay thread) --

    def tick(self) -> bool:
        """Called by a GLib timeout for animation. Returns True to continue."""
        with self._lock:
            visible = self._visible_state

        if visible and not self._window.get_visible():
            self._window.show_all()
            self._reposition()
        elif not visible and self._window.get_visible():
            self._window.hide()

        if visible:
            self._update_bars()
            self._window.queue_draw()

        return True

    def destroy(self) -> None:
        self._window.destroy()

    def _reposition(self) -> None:
        """Move the window to its configured position. Must be re-asserted
        after show_all() — GTK/the window manager can reposition on map."""
        geom = _monitor_geometry_at_pointer()
        if geom is None:
            return
        x, y = compute_pill_position(*geom, self._position)
        self._window.move(x, y)

    def _update_bars(self) -> None:
        """Update bar heights from audio level history."""
        with self._lock:
            levels = list(self._audio_levels)
            speech = self._speech_active
            self._phase += 0.1

        if speech:
            n = len(levels)
            for i in range(_BAR_COUNT):
                idx = min(int((i / _BAR_COUNT) * n), n - 1)
                target = levels[idx]
                wave = 0.15 * math.sin(self._phase + i * 0.4)
                target = max(0.05, min(1.0, target + wave))
                self._bar_heights[i] += (target - self._bar_heights[i]) * 0.3
        else:
            for i in range(_BAR_COUNT):
                breath = 0.08 + 0.04 * math.sin(self._phase * 0.5 + i * 0.3)
                self._bar_heights[i] += (breath - self._bar_heights[i]) * 0.1

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
    loop runs in a dedicated daemon thread, isolated from the asyncio loop
    and the real-time audio callback thread.
    """

    def __init__(self, config: OverlayConfig | None = None) -> None:
        self._config = config or OverlayConfig()
        self._window: _OverlayWindow | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def available(self) -> bool:
        return _HAS_GTK

    @property
    def unavailable_reason(self) -> str:
        return _UNAVAILABLE_REASON

    def start(self) -> None:
        """Start the overlay in a background thread."""
        if not _HAS_GTK:
            logger.info("GTK not available — overlay disabled: %s", _UNAVAILABLE_REASON)
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._ready.clear()
        self._thread = threading.Thread(target=self._run_gtk, name="overlay", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        logger.info("Overlay started")

    def stop(self) -> None:
        """Stop the overlay."""
        if self._thread is not None:
            GLib.idle_add(Gtk.main_quit)
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Overlay stopped")

    def show(self) -> None:
        """Show the pill (recording started)."""
        if self._window is not None:
            GLib.idle_add(self._window.set_recording, True)

    def hide(self) -> None:
        """Hide the pill (recording stopped)."""
        if self._window is not None:
            GLib.idle_add(self._window.set_recording, False)

    def set_speech_active(self, active: bool) -> None:
        """Update whether speech is currently detected."""
        if self._window is not None:
            GLib.idle_add(self._window.set_speech_active, active)

    def push_audio_level(self, level: float) -> None:
        """Push a new audio level (0.0-1.0) for visualization."""
        if self._window is not None:
            # Direct call is fine — guarded by a lock, no GTK API touched.
            self._window.push_audio_level(level)

    def _run_gtk(self) -> None:
        """GTK main loop — runs in the overlay thread.

        GDK_BACKEND is forced to "x11" only for the brief window in which
        the window is constructed and its display connection is opened
        (GTK4 native-Wayland has no positioning API, and Mutter does not
        implement wlr-layer-shell — XWayland is the only path that
        positions correctly on this desktop). The environment variable is
        restored immediately afterward: the backend is chosen once per
        process, on first display access, so nothing later in the process
        needs it set, and this avoids mutating os.environ for the life of
        the app.
        """
        prior_backend = os.environ.get("GDK_BACKEND")
        os.environ["GDK_BACKEND"] = "x11"
        try:
            self._window = _OverlayWindow(self._config.position)
        finally:
            if prior_backend is None:
                os.environ.pop("GDK_BACKEND", None)
            else:
                os.environ["GDK_BACKEND"] = prior_backend

        GLib.timeout_add(1000 // _FPS, self._window.tick)
        self._ready.set()
        logger.debug("Overlay window created")

        try:
            Gtk.main()
        except Exception:
            logger.exception("Overlay GTK loop crashed")
        finally:
            if self._window is not None:
                self._window.destroy()
