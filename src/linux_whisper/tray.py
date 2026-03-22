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
    """Bright red circle with a mic — actively recording (no speech)."""
    img, draw = _new_image()
    _draw_circle(draw, _RED)
    _draw_mic(draw)
    draw.ellipse(
        [1, 1, _ICON_SIZE - 2, _ICON_SIZE - 2],
        outline=_RED,
        width=2,
    )
    return img


_GREEN = (80, 200, 120, 255)


def _make_recording_speech_icon() -> Any:
    """Green circle with mic and sound arcs - speech detected."""
    img, draw = _new_image()
    _draw_circle(draw, _GREEN)
    _draw_mic(draw)
    # Sound wave arcs
    for i in range(3):
        inset = 6 + i * 5
        draw.arc(
            [inset, inset, _ICON_SIZE - inset, _ICON_SIZE - inset],
            start=150, end=210, fill=_WHITE, width=2,
        )
        draw.arc(
            [inset, inset, _ICON_SIZE - inset, _ICON_SIZE - inset],
            start=-30, end=30, fill=_WHITE, width=2,
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

# Pre-built icons for audio level states (avoid regenerating on every update)
_CACHED_ICONS: dict[str, Any] = {}


def _get_cached_icon(key: str, factory: Callable[[], Any]) -> Any:
    if key not in _CACHED_ICONS:
        _CACHED_ICONS[key] = factory()
    return _CACHED_ICONS[key]

_TOOLTIPS: dict[AppState, str] = {
    AppState.IDLE: "Linux Whisper - idle",
    AppState.RECORDING: "Linux Whisper - recording",
    AppState.PROCESSING: "Linux Whisper - processing",
    AppState.ERROR: "Linux Whisper - error",
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

    # Available models shown in the tray menu, grouped by backend
    _MODEL_OPTIONS: list[tuple[str, str, str]] = [
        # (display_name, backend, model)
        ("Large V3 Turbo (best quality)", "faster-whisper", "large-v3-turbo"),
        ("Distil Large V3 (fast, English)", "faster-whisper", "distil-large-v3"),
        ("Medium English (faster)", "faster-whisper", "medium.en"),
        ("Small English (fastest)", "faster-whisper", "small.en"),
        ("Moonshine Medium (low latency)", "moonshine", "moonshine-medium"),
        ("Moonshine Tiny (instant)", "moonshine", "moonshine-tiny"),
    ]

    def __init__(
        self,
        config: Config,
        *,
        on_quit: Callable[[], None] | None = None,
        on_mode_change: Callable[[str], None] | None = None,
        on_model_change: Callable[[str, str], None] | None = None,
        on_open_settings: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._on_quit = on_quit
        self._on_mode_change = on_mode_change
        self._on_model_change = on_model_change
        self._on_open_settings = on_open_settings

        # Mutable state guarded by ``_lock``
        self._lock = threading.Lock()
        self._current_state: AppState = AppState.IDLE
        self._current_mode: str = config.mode
        self._current_backend: str = config.stt.backend
        self._current_model: str = config.stt.model
        self._last_latency: float | None = None
        self._avg_latency: float | None = None
        self._last_transcription: str | None = None

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

    def set_speech_active(self, speech: bool) -> None:
        """Switch the tray icon between recording-silent and recording-speech.

        Call when VAD speech state changes (not every frame).
        """
        with self._lock:
            if self._current_state != AppState.RECORDING:
                return

        icon = self._icon
        if icon is None:
            return

        try:
            if speech:
                icon.icon = _get_cached_icon("rec_speech", _make_recording_speech_icon)
                icon.title = "Linux Whisper - listening (speech)"
            else:
                icon.icon = _get_cached_icon("rec_silent", _make_recording_icon)
                icon.title = "Linux Whisper - listening"
        except Exception:
            logger.debug("Failed to update speech icon", exc_info=True)

    def set_last_transcription(self, text: str) -> None:
        """Store the most recent transcription for the Copy Last menu item."""
        with self._lock:
            self._last_transcription = text
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
            current_backend = self._current_backend
            current_model = self._current_model
            last_lat = self._last_latency
            avg_lat = self._avg_latency
            last_text = self._last_transcription

        mode_items = [
            pystray.MenuItem(
                f"{'● ' if current_mode == m else '  '}{m.replace('-', ' ').title()}",
                self._make_mode_handler(m),
            )
            for m in Config.VALID_MODES
        ]

        model_items = [
            pystray.MenuItem(
                f"{'● ' if (current_backend == backend and current_model == model) else '  '}{name}",
                self._make_model_handler(backend, model),
            )
            for name, backend, model in self._MODEL_OPTIONS
        ]

        if last_lat is not None and avg_lat is not None:
            latency_label = f"Latency: {_fmt_ms(last_lat)} (avg {_fmt_ms(avg_lat)})"
        else:
            latency_label = "Latency: -"

        # Truncate last transcription for display
        if last_text:
            display_text = last_text[:50] + ("..." if len(last_text) > 50 else "")
            copy_label = f"Copy Last: \"{display_text}\""
        else:
            copy_label = "Copy Last (none yet)"

        return pystray.Menu(
            pystray.MenuItem(copy_label, self._handle_copy_last, enabled=bool(last_text)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Model", pystray.Menu(*model_items)),
            pystray.MenuItem("Mode", pystray.Menu(*mode_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(latency_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
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

    def _make_model_handler(self, backend: str, model: str) -> Callable[..., None]:
        """Return a click handler that switches to *backend*/*model*."""

        def handler(icon: Any, item: Any) -> None:
            with self._lock:
                if self._current_backend == backend and self._current_model == model:
                    return  # already selected
                self._current_backend = backend
                self._current_model = model
            logger.info("Tray: model changed to %s/%s", backend, model)
            if self._on_model_change is not None:
                try:
                    self._on_model_change(backend, model)
                except Exception:
                    logger.exception("on_model_change callback failed")
            self._refresh_menu()

        return handler

    def _handle_copy_last(self, icon: Any, item: Any) -> None:
        """Copy the most recent transcription to the clipboard."""
        with self._lock:
            text = self._last_transcription
        if not text:
            return
        try:
            import subprocess
            subprocess.run(
                ["wl-copy", "--", text],
                check=True,
                timeout=5,
            )
            logger.info("Copied last transcription to clipboard (%d chars)", len(text))
        except FileNotFoundError:
            # Try xclip fallback
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text.encode(),
                    check=True,
                    timeout=5,
                )
                logger.info("Copied last transcription to clipboard via xclip")
            except Exception:
                logger.warning("No clipboard tool found (wl-copy/xclip)")
        except Exception:
            logger.warning("Failed to copy to clipboard", exc_info=True)

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
