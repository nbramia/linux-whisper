"""System tray integration using pystray + Pillow.

Provides a system-tray icon that reflects application state (idle, recording,
processing, error) and exposes a context menu for quick actions.  All public
methods are thread-safe so the main asyncio loop can push updates freely.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from typing import Any

from linux_whisper.config import Config
from linux_whisper.state import AppState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — pystray / PIL may not be installed (headless mode)
# ---------------------------------------------------------------------------

try:
    import pystray
    from PIL import Image, ImageDraw

    _HAS_TRAY = True
except ImportError:  # pragma: no cover
    pystray = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment,misc]
    ImageDraw = None  # type: ignore[assignment,misc]
    _HAS_TRAY = False

# ---------------------------------------------------------------------------
# Icon generation helpers
# ---------------------------------------------------------------------------

_ICON_SIZE = 64
_BG = (0, 0, 0, 0)  # transparent background

# Palette
_GRAY = (140, 140, 140, 255)
_RED = (220, 50, 50, 255)
_AMBER = (230, 180, 30, 255)
_WHITE = (255, 255, 255, 255)
_DARK_RED = (180, 30, 30, 255)


def _new_image() -> tuple[Any, Any]:
    """Return a blank RGBA image and its ``ImageDraw`` handle."""
    img = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), _BG)
    draw = ImageDraw.Draw(img)
    return img, draw


def _draw_circle(draw: Any, color: tuple[int, ...], *, inset: int = 4) -> None:
    """Draw a filled circle with an optional inset from the edges."""
    draw.ellipse(
        [inset, inset, _ICON_SIZE - inset - 1, _ICON_SIZE - inset - 1],
        fill=color,
    )


def _draw_mic(draw: Any, color: tuple[int, ...] = _WHITE) -> None:
    """Draw a simplified microphone silhouette inside the icon.

    The mic is composed of:
    * a rounded-rectangle capsule (the mic head),
    * a thin vertical stem, and
    * a short horizontal base line.
    """
    cx = _ICON_SIZE // 2
    # Mic head (capsule) — a tall, narrow rounded rect
    head_w, head_h = 10, 20
    head_left = cx - head_w // 2
    head_top = 14
    draw.rounded_rectangle(
        [head_left, head_top, head_left + head_w, head_top + head_h],
        radius=head_w // 2,
        fill=color,
    )
    # Arc around the mic head (the U-shaped holder)
    arc_w = 18
    arc_left = cx - arc_w // 2
    arc_top = head_top + 4
    arc_bottom = head_top + head_h + 6
    draw.arc(
        [arc_left, arc_top, arc_left + arc_w, arc_bottom],
        start=0,
        end=180,
        fill=color,
        width=2,
    )
    # Stem
    stem_top = arc_bottom - (arc_bottom - arc_top) // 2 + 3
    stem_bottom = stem_top + 8
    draw.line([(cx, stem_top), (cx, stem_bottom)], fill=color, width=2)
    # Base
    base_half = 6
    draw.line(
        [(cx - base_half, stem_bottom), (cx + base_half, stem_bottom)],
        fill=color,
        width=2,
    )


def _draw_x(draw: Any, color: tuple[int, ...] = _WHITE) -> None:
    """Draw an X mark centered in the icon."""
    margin = 20
    draw.line(
        [(margin, margin), (_ICON_SIZE - margin, _ICON_SIZE - margin)],
        fill=color,
        width=4,
    )
    draw.line(
        [(margin, _ICON_SIZE - margin), (_ICON_SIZE - margin, margin)],
        fill=color,
        width=4,
    )


def _make_idle_icon() -> Any:
    """Gray circle with a microphone — resting state."""
    img, draw = _new_image()
    _draw_circle(draw, _GRAY)
    _draw_mic(draw)
    return img


def _make_recording_icon() -> Any:
    """Bright red circle with a mic — actively recording."""
    img, draw = _new_image()
    _draw_circle(draw, _RED)
    _draw_mic(draw)
    # Add a subtle outer glow ring to convey "active"
    draw.ellipse(
        [1, 1, _ICON_SIZE - 2, _ICON_SIZE - 2],
        outline=_RED,
        width=2,
    )
    return img


def _make_processing_icon() -> Any:
    """Amber/yellow circle with a mic — transcribing/polishing."""
    img, draw = _new_image()
    _draw_circle(draw, _AMBER)
    _draw_mic(draw)
    return img


def _make_error_icon() -> Any:
    """Red circle with an X — something went wrong."""
    img, draw = _new_image()
    _draw_circle(draw, _DARK_RED)
    _draw_x(draw)
    return img


# Map each application state to an icon factory.
_ICON_FACTORIES: dict[AppState, Callable[[], Any]] = {
    AppState.IDLE: _make_idle_icon,
    AppState.RECORDING: _make_recording_icon,
    AppState.PROCESSING: _make_processing_icon,
    AppState.ERROR: _make_error_icon,
}

_TOOLTIPS: dict[AppState, str] = {
    AppState.IDLE: "Linux Whisper — idle",
    AppState.RECORDING: "Linux Whisper — recording…",
    AppState.PROCESSING: "Linux Whisper — processing…",
    AppState.ERROR: "Linux Whisper — error",
}

# ---------------------------------------------------------------------------
# SystemTray
# ---------------------------------------------------------------------------


class SystemTray:
    """Manages the pystray icon in a dedicated background thread.

    Parameters
    ----------
    config:
        Application configuration (used to display current mode/model info).
    on_quit:
        Callback invoked when the user chooses *Quit* from the context menu.
    on_mode_change:
        Callback invoked with the newly selected mode string
        (``"hold"`` | ``"toggle"`` | ``"vad-auto"``).
    on_open_settings:
        Callback invoked when the user chooses *Settings*.
    """

    def __init__(
        self,
        config: Config,
        *,
        on_quit: Callable[[], None] | None = None,
        on_mode_change: Callable[[str], None] | None = None,
        on_open_settings: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._on_quit = on_quit
        self._on_mode_change = on_mode_change
        self._on_open_settings = on_open_settings

        # Mutable state guarded by ``_lock``
        self._lock = threading.Lock()
        self._current_state: AppState = AppState.IDLE
        self._current_mode: str = config.mode
        self._last_latency: float | None = None
        self._avg_latency: float | None = None

        # pystray internals
        self._icon: pystray.Icon | None = None  # type: ignore[union-attr]
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Launch the tray icon on a background daemon thread.

        Safe to call even when ``pystray`` is unavailable — the method will
        simply log a warning and return.
        """
        if not _HAS_TRAY:
            logger.warning("pystray or Pillow not installed — tray disabled")
            return

        if self._thread is not None and self._thread.is_alive():
            logger.debug("Tray thread already running")
            return

        self._thread = threading.Thread(
            target=self._run,
            name="linux-whisper-tray",
            daemon=True,
        )
        self._thread.start()
        logger.info("System tray thread started")

    def stop(self) -> None:
        """Tear down the tray icon and join the background thread."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                logger.debug("Error stopping tray icon", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._running.clear()
        logger.info("System tray stopped")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # -- public, thread-safe update methods ---------------------------------

    def update_state(self, state: AppState) -> None:
        """Change the displayed icon and tooltip to reflect *state*.

        May be called from any thread (including the asyncio event-loop thread).
        """
        with self._lock:
            if state == self._current_state:
                return
            self._current_state = state

        icon = self._icon
        if icon is None:
            return

        try:
            icon.icon = _ICON_FACTORIES[state]()
            icon.title = _TOOLTIPS.get(state, "Linux Whisper")
        except Exception:
            logger.debug("Failed to update tray icon", exc_info=True)

    def update_mode(self, mode: str) -> None:
        """Update the internally tracked mode so the menu reflects it."""
        with self._lock:
            self._current_mode = mode
        self._refresh_menu()

    def update_stats(self, last_latency: float, avg_latency: float) -> None:
        """Push new latency numbers into the tray menu.

        Parameters
        ----------
        last_latency:
            Most recent end-to-end latency in **seconds**.
        avg_latency:
            Rolling-average latency in **seconds**.
        """
        with self._lock:
            self._last_latency = last_latency
            self._avg_latency = avg_latency
        self._refresh_menu()

    # -- internal -----------------------------------------------------------

    def _run(self) -> None:
        """Entry point executed inside the background thread."""
        try:
            self._icon = pystray.Icon(
                name="linux-whisper",
                icon=_ICON_FACTORIES[AppState.IDLE](),
                title=_TOOLTIPS[AppState.IDLE],
                menu=self._build_menu(),
            )
            self._running.set()
            # pystray.Icon.run() blocks until Icon.stop() is called.
            self._icon.run()
        except Exception:
            logger.exception("Tray thread crashed")
        finally:
            self._running.clear()

    def _refresh_menu(self) -> None:
        """Rebuild and assign the context menu (must be called after state changes)."""
        icon = self._icon
        if icon is None:
            return
        try:
            icon.menu = self._build_menu()
            icon.update_menu()
        except Exception:
            logger.debug("Failed to refresh tray menu", exc_info=True)

    def _build_menu(self) -> pystray.Menu:  # type: ignore[name-defined]
        """Construct the right-click context menu."""
        with self._lock:
            current_mode = self._current_mode
            last_lat = self._last_latency
            avg_lat = self._avg_latency

        mode_items = [
            pystray.MenuItem(
                f"{'● ' if current_mode == m else '  '}{m.replace('-', ' ').title()}",
                self._make_mode_handler(m),
            )
            for m in Config.VALID_MODES
        ]

        model_label = f"Model: {self._config.stt.model}"
        backend_label = f"Backend: {self._config.stt.backend}"

        if last_lat is not None and avg_lat is not None:
            latency_label = f"Latency: {_fmt_ms(last_lat)} (avg {_fmt_ms(avg_lat)})"
        else:
            latency_label = "Latency: —"

        return pystray.Menu(
            pystray.MenuItem("Mode", pystray.Menu(*mode_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(model_label, None, enabled=False),
            pystray.MenuItem(backend_label, None, enabled=False),
            pystray.MenuItem(latency_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings…", self._handle_settings),
            pystray.MenuItem("Quit", self._handle_quit),
        )

    # -- menu action handlers -----------------------------------------------

    def _make_mode_handler(self, mode: str) -> Callable[..., None]:
        """Return a click handler that switches to *mode*."""

        def handler(icon: Any, item: Any) -> None:
            with self._lock:
                self._current_mode = mode
            logger.info("Tray: mode changed to %s", mode)
            if self._on_mode_change is not None:
                try:
                    self._on_mode_change(mode)
                except Exception:
                    logger.exception("on_mode_change callback failed")
            self._refresh_menu()

        return handler

    def _handle_settings(self, icon: Any, item: Any) -> None:
        logger.info("Tray: open settings requested")
        if self._on_open_settings is not None:
            try:
                self._on_open_settings()
            except Exception:
                logger.exception("on_open_settings callback failed")

    def _handle_quit(self, icon: Any, item: Any) -> None:
        logger.info("Tray: quit requested")
        if self._on_quit is not None:
            try:
                self._on_quit()
            except Exception:
                logger.exception("on_quit callback failed")
        self.stop()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fmt_ms(seconds: float) -> str:
    """Format a duration in seconds as a human-friendly millisecond string."""
    ms = seconds * 1000
    if ms < 10:
        return f"{ms:.1f} ms"
    return f"{math.floor(ms)} ms"
