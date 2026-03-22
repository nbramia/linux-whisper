"""Floating pill overlay showing recording state and live audio levels.

Displays a small semi-transparent pill at the bottom-center of the screen:
- Hidden when idle (not recording)
- Visible with muted indicators when recording but no speech detected
- Visible with animated audio level bars when speech is detected

Uses GTK4 with Cairo for custom drawing. Runs in its own thread.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    from gi.repository import Gdk, GLib, Gtk

    _HAS_GTK = True
except (ImportError, ValueError):
    _HAS_GTK = False
    logger.debug("GTK4 not available — overlay disabled")

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
_MARGIN_BOTTOM = 48  # pixels from bottom of screen
_FPS = 30
_LEVEL_HISTORY = 32  # frames of audio level history for smoothing


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
    # Small dot indicator
    DOT_LISTENING = (0.35, 0.75, 0.55, 0.8)  # green
    DOT_SILENT = (0.5, 0.5, 0.55, 0.5)  # gray


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

if _HAS_GTK:

    class _OverlayWindow(Gtk.Window):
        """Custom GTK4 window for the pill overlay."""

        def __init__(self) -> None:
            super().__init__(title="linux-whisper-overlay")
            self.set_decorated(False)
            self.set_resizable(False)
            self.set_default_size(_PILL_WIDTH, _PILL_HEIGHT)

            # Transparent background
            self.add_css_class("overlay-transparent")

            # State
            self._visible_state = False
            self._speech_active = False
            self._audio_levels: deque[float] = deque(
                [0.0] * _LEVEL_HISTORY, maxlen=_LEVEL_HISTORY
            )
            self._bar_heights: list[float] = [0.0] * _BAR_COUNT
            self._phase: float = 0.0  # animation phase
            self._lock = threading.Lock()

            # Drawing area
            self._canvas = Gtk.DrawingArea()
            self._canvas.set_content_width(_PILL_WIDTH)
            self._canvas.set_content_height(_PILL_HEIGHT)
            self._canvas.set_draw_func(self._draw)
            self.set_child(self._canvas)

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

        def tick(self) -> bool:
            """Called by GLib timeout for animation. Returns True to continue."""
            with self._lock:
                visible = self._visible_state

            if visible and not self.get_visible():
                self.set_visible(True)
                self._position_at_bottom()
            elif not visible and self.get_visible():
                self.set_visible(False)

            if visible:
                self._update_bars()
                self._canvas.queue_draw()

            return True  # keep timer running

        def _position_at_bottom(self) -> None:
            """Position the window at bottom-center of the primary monitor."""
            display = Gdk.Display.get_default()
            if display is None:
                return
            monitors = display.get_monitors()
            if monitors.get_n_items() == 0:
                return
            monitor = monitors.get_item(0)
            geom = monitor.get_geometry()
            x = geom.x + (geom.width - _PILL_WIDTH) // 2
            y = geom.y + geom.height - _PILL_HEIGHT - _MARGIN_BOTTOM
            # GTK4 on Wayland doesn't support move() — we use a CSS margin hack
            # or just let the WM place it. For X11, we could use present().
            # The window will appear at the default position on Wayland;
            # for proper positioning, layer-shell is needed.
            # We set a hint via the title for compositor rules.
            self.set_title("linux-whisper-overlay")

        def _update_bars(self) -> None:
            """Update bar heights from audio level history."""
            with self._lock:
                levels = list(self._audio_levels)
                speech = self._speech_active
                self._phase += 0.1

            if speech:
                # Distribute recent levels across bars with slight offset
                n = len(levels)
                for i in range(_BAR_COUNT):
                    # Map bar index to a position in the level history
                    idx = int((i / _BAR_COUNT) * n)
                    idx = min(idx, n - 1)
                    target = levels[idx]
                    # Add some wave motion
                    wave = 0.15 * math.sin(self._phase + i * 0.4)
                    target = max(0.05, min(1.0, target + wave))
                    # Smooth
                    self._bar_heights[i] += (target - self._bar_heights[i]) * 0.3
            else:
                # Gentle breathing animation when listening but no speech
                for i in range(_BAR_COUNT):
                    breath = 0.08 + 0.04 * math.sin(self._phase * 0.5 + i * 0.3)
                    self._bar_heights[i] += (breath - self._bar_heights[i]) * 0.1

        def _draw(
            self,
            area: Gtk.DrawingArea,
            cr: object,  # cairo.Context — not typed to avoid import
            width: int,
            height: int,
        ) -> None:
            """Draw the pill with audio level bars."""
            # Clear (transparent)
            cr.set_operator(0)  # CAIRO_OPERATOR_CLEAR
            cr.paint()
            cr.set_operator(2)  # CAIRO_OPERATOR_OVER

            # Pill background
            self._draw_rounded_rect(cr, 0, 0, width, height, _PILL_RADIUS)
            cr.set_source_rgba(*_Colors.BG)
            cr.fill_preserve()
            cr.set_source_rgba(*_Colors.BORDER)
            cr.set_line_width(1.0)
            cr.stroke()

            with self._lock:
                speech = self._speech_active
                bars = list(self._bar_heights)

            # Draw bars
            total_bar_width = _BAR_COUNT * _BAR_WIDTH + (_BAR_COUNT - 1) * _BAR_GAP
            start_x = (width - total_bar_width) / 2
            center_y = height / 2

            for i in range(_BAR_COUNT):
                h = _BAR_MIN_HEIGHT + bars[i] * (_BAR_MAX_HEIGHT - _BAR_MIN_HEIGHT)
                x = start_x + i * (_BAR_WIDTH + _BAR_GAP)
                y = center_y - h / 2

                # Color based on state
                if speech:
                    # Gradient from base to peak based on height
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


class Overlay:
    """Public API for the recording overlay.

    Thread-safe: all methods can be called from any thread.
    The GTK event loop runs in a dedicated thread.
    """

    def __init__(self) -> None:
        self._window: _OverlayWindow | None = None
        self._app: Gtk.Application | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def available(self) -> bool:
        return _HAS_GTK

    def start(self) -> None:
        """Start the overlay in a background thread."""
        if not _HAS_GTK:
            logger.info("GTK4 not available — overlay disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run_gtk, name="overlay", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)
        logger.info("Overlay started")

    def stop(self) -> None:
        """Stop the overlay."""
        if self._app is not None:
            GLib.idle_add(self._app.quit)
        if self._thread is not None:
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
            # Direct call is fine — deque is thread-safe for append
            self._window.push_audio_level(level)

    def _run_gtk(self) -> None:
        """GTK main loop — runs in the overlay thread."""
        self._app = Gtk.Application(application_id="dev.linuxwhisper.overlay")
        self._app.connect("activate", self._on_activate)
        try:
            self._app.run(None)
        except Exception:
            logger.exception("Overlay GTK loop crashed")

    def _on_activate(self, app: Gtk.Application) -> None:
        """Called when the GTK application is ready."""
        self._window = _OverlayWindow()
        self._window.set_application(app)
        self._window.set_visible(False)

        # Apply transparent CSS
        css = Gtk.CssProvider()
        css.load_from_string(
            """
            .overlay-transparent {
                background-color: transparent;
            }
            """
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # Animation timer
        GLib.timeout_add(1000 // _FPS, self._window.tick)

        self._ready.set()
        logger.debug("Overlay window created")
