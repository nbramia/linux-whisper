"""Text injection backends for X11 and Wayland display servers.

Auto-detects the display server and compositor, then selects the best
available tool to inject transcribed text at the current cursor position.

Injection priority:
  X11      → xdotool  → clipboard fallback (xclip)
  wlroots  → wtype    → ydotool → clipboard fallback (wl-copy/wl-paste)
  Wayland  → ydotool  → clipboard fallback (wl-copy/wl-paste)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from abc import ABC, abstractmethod
from enum import Enum

from linux_whisper.config import InjectConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display server detection
# ---------------------------------------------------------------------------


class DisplayServer(Enum):
    X11 = "x11"
    WAYLAND = "wayland"
    UNKNOWN = "unknown"


def _detect_display_server() -> DisplayServer:
    """Detect whether the session is running on X11 or Wayland."""
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type == "x11":
        return DisplayServer.X11
    if session_type == "wayland":
        return DisplayServer.WAYLAND
    # Fallback heuristics
    if os.environ.get("WAYLAND_DISPLAY"):
        return DisplayServer.WAYLAND
    if os.environ.get("DISPLAY"):
        return DisplayServer.X11
    logger.warning("Cannot determine display server from environment")
    return DisplayServer.UNKNOWN


def _is_wlroots_compositor() -> bool:
    """Check whether the running Wayland compositor is wlroots-based.

    Sway, Hyprland, river, and other wlroots compositors support the
    virtual-keyboard-unstable-v1 protocol that wtype relies on.
    """
    if os.environ.get("SWAYSOCK"):
        return True
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return True
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    wlroots_desktops = {"sway", "hyprland", "river", "wayfire", "labwc", "dwl"}
    return desktop in wlroots_desktops


# ---------------------------------------------------------------------------
# Async subprocess helpers
# ---------------------------------------------------------------------------


async def _run(
    *cmd: str,
    timeout: float = 10.0,
    stdin_data: bytes | None = None,
) -> tuple[int, bytes, bytes]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    logger.debug("Running: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_data),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.error("Subprocess timed out: %s", " ".join(cmd))
        return -1, b"", b"timeout"
    return proc.returncode or 0, stdout, stderr


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TextInjector(ABC):
    """Base class for all text injection backends."""

    name: str = "base"

    def __init__(self, config: InjectConfig) -> None:
        self.config = config

    @abstractmethod
    async def inject(self, text: str) -> bool:
        """Inject *text* at the current cursor position.

        Returns True on success, False on failure.
        """
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


# ---------------------------------------------------------------------------
# xdotool — X11
# ---------------------------------------------------------------------------


class XdotoolInjector(TextInjector):
    """Inject text via ``xdotool type`` on X11."""

    name = "xdotool"

    async def inject(self, text: str) -> bool:
        cmd: list[str] = ["xdotool", "type", "--clearmodifiers"]
        if self.config.typing_delay > 0:
            cmd.extend(["--delay", str(self.config.typing_delay)])
        cmd.append("--")
        cmd.append(text)
        rc, _out, err = await _run(*cmd)
        if rc != 0:
            logger.error("xdotool failed (rc=%d): %s", rc, err.decode(errors="replace"))
            return False
        return True


# ---------------------------------------------------------------------------
# wtype — wlroots Wayland
# ---------------------------------------------------------------------------


class WtypeInjector(TextInjector):
    """Inject text via ``wtype`` on wlroots-based Wayland compositors."""

    name = "wtype"

    async def inject(self, text: str) -> bool:
        cmd: list[str] = ["wtype"]
        if self.config.typing_delay > 0:
            cmd.extend(["-d", str(self.config.typing_delay)])
        cmd.append("--")
        cmd.append(text)
        rc, _out, err = await _run(*cmd)
        if rc != 0:
            logger.error("wtype failed (rc=%d): %s", rc, err.decode(errors="replace"))
            return False
        return True


# ---------------------------------------------------------------------------
# ydotool — any Wayland (or X11)
# ---------------------------------------------------------------------------


class YdotoolInjector(TextInjector):
    """Inject text via ``ydotool type`` on any Wayland session.

    Requires the ``ydotoold`` daemon to be running.
    """

    name = "ydotool"

    async def inject(self, text: str) -> bool:
        cmd: list[str] = ["ydotool", "type"]
        if self.config.typing_delay > 0:
            cmd.extend(["--key-delay", str(self.config.typing_delay)])
        cmd.append("--")
        cmd.append(text)
        rc, _out, err = await _run(*cmd)
        if rc != 0:
            logger.error("ydotool failed (rc=%d): %s", rc, err.decode(errors="replace"))
            return False
        return True


# ---------------------------------------------------------------------------
# Clipboard fallback
# ---------------------------------------------------------------------------

_CLIPBOARD_RESTORE_DELAY: float = 5.0  # seconds before restoring clipboard


class ClipboardInjector(TextInjector):
    """Inject text by temporarily placing it on the clipboard and pasting.

    The original clipboard contents are saved before injection and restored
    after a short delay so the user does not lose their clipboard.

    Uses ``xclip`` on X11 and ``wl-copy``/``wl-paste`` on Wayland.
    """

    name = "clipboard"

    def __init__(self, config: InjectConfig, display_server: DisplayServer) -> None:
        super().__init__(config)
        self._display_server = display_server

    # -- clipboard primitives -----------------------------------------------

    async def _get_clipboard(self) -> bytes | None:
        """Read current clipboard contents (best-effort)."""
        if self._display_server == DisplayServer.X11:
            rc, data, _ = await _run("xclip", "-selection", "clipboard", "-o")
        else:
            rc, data, _ = await _run("wl-paste", "--no-newline")
        if rc != 0:
            return None
        return data

    async def _set_clipboard(self, data: bytes) -> bool:
        """Write *data* to the system clipboard."""
        if self._display_server == DisplayServer.X11:
            rc, _, err = await _run(
                "xclip", "-selection", "clipboard", stdin_data=data,
            )
        else:
            rc, _, err = await _run("wl-copy", stdin_data=data)
        if rc != 0:
            logger.error("Clipboard write failed: %s", err.decode(errors="replace"))
            return False
        return True

    async def _paste_keystroke(self) -> bool:
        """Simulate Ctrl+V to paste from clipboard."""
        if self._display_server == DisplayServer.X11:
            rc, _, err = await _run(
                "xdotool", "key", "--clearmodifiers", "ctrl+v",
            )
        elif _is_wlroots_compositor() and shutil.which("wtype"):
            rc, _, err = await _run("wtype", "-M", "ctrl", "-P", "v", "-p", "v", "-m", "ctrl")
        elif shutil.which("ydotool"):
            rc, _, err = await _run("ydotool", "key", "29:1", "47:1", "47:0", "29:0")
        else:
            logger.error("No tool available to simulate Ctrl+V paste keystroke")
            return False
        if rc != 0:
            logger.error("Paste keystroke failed: %s", err.decode(errors="replace"))
            return False
        return True

    # -- delayed restore ----------------------------------------------------

    async def _schedule_restore(self, original: bytes | None) -> None:
        """Restore clipboard contents after a delay."""
        if original is None:
            return
        await asyncio.sleep(_CLIPBOARD_RESTORE_DELAY)
        ok = await self._set_clipboard(original)
        if ok:
            logger.debug("Clipboard contents restored after %.0fs", _CLIPBOARD_RESTORE_DELAY)
        else:
            logger.warning("Failed to restore original clipboard contents")

    # -- main inject --------------------------------------------------------

    async def inject(self, text: str) -> bool:
        # 1. Save current clipboard
        original = await self._get_clipboard()

        # 2. Set clipboard to the transcribed text
        if not await self._set_clipboard(text.encode("utf-8")):
            return False

        # Small delay so the clipboard manager registers the change
        await asyncio.sleep(0.05)

        # 3. Simulate Ctrl+V
        if not await self._paste_keystroke():
            # Attempt to restore immediately on failure
            if original is not None:
                await self._set_clipboard(original)
            return False

        # 4. Schedule clipboard restoration in the background
        asyncio.get_running_loop().create_task(self._schedule_restore(original))

        return True


# ---------------------------------------------------------------------------
# Factory — auto-detection
# ---------------------------------------------------------------------------


def detect_injector(config: InjectConfig) -> TextInjector:
    """Select the best available text injector based on config and environment.

    When ``config.method`` is ``"auto"``, the display server and available
    tools are probed automatically.  Otherwise the explicitly requested
    backend is used (falling back to clipboard if the tool is missing).
    """
    display = _detect_display_server()
    logger.info("Display server: %s", display.value)

    # Explicit method override -----------------------------------------
    if config.method != "auto":
        return _build_explicit(config, display)

    # Auto-detect ------------------------------------------------------
    if display == DisplayServer.X11:
        return _auto_x11(config, display)
    if display == DisplayServer.WAYLAND:
        return _auto_wayland(config, display)

    # Unknown display server — try everything
    logger.warning("Unknown display server, trying all injection methods")
    return _auto_unknown(config, display)


def _build_explicit(config: InjectConfig, display: DisplayServer) -> TextInjector:
    """Build the injector explicitly requested by the user."""
    match config.method:
        case "xdotool":
            if shutil.which("xdotool"):
                return XdotoolInjector(config)
            logger.warning("xdotool not found, falling back to clipboard")
        case "wtype":
            if shutil.which("wtype"):
                return WtypeInjector(config)
            logger.warning("wtype not found, falling back to clipboard")
        case "ydotool":
            if shutil.which("ydotool"):
                return YdotoolInjector(config)
            logger.warning("ydotool not found, falling back to clipboard")
        case "clipboard":
            return ClipboardInjector(config, display)
        case _:
            logger.error("Unknown injection method '%s'", config.method)

    return ClipboardInjector(config, display)


def _auto_x11(config: InjectConfig, display: DisplayServer) -> TextInjector:
    """Auto-detect for an X11 session."""
    if shutil.which("xdotool"):
        logger.info("Selected injector: xdotool (X11)")
        return XdotoolInjector(config)
    logger.info("xdotool not found, falling back to clipboard (X11)")
    return ClipboardInjector(config, display)


def _auto_wayland(config: InjectConfig, display: DisplayServer) -> TextInjector:
    """Auto-detect for a Wayland session."""
    if _is_wlroots_compositor() and shutil.which("wtype"):
        logger.info("Selected injector: wtype (wlroots Wayland)")
        return WtypeInjector(config)
    if shutil.which("ydotool"):
        logger.info("Selected injector: ydotool (Wayland)")
        return YdotoolInjector(config)
    logger.info("No typing tool found, falling back to clipboard (Wayland)")
    return ClipboardInjector(config, display)


def _auto_unknown(config: InjectConfig, display: DisplayServer) -> TextInjector:
    """Best-effort detection when the display server is unknown."""
    if shutil.which("xdotool"):
        logger.info("Selected injector: xdotool (unknown display)")
        return XdotoolInjector(config)
    if shutil.which("wtype"):
        logger.info("Selected injector: wtype (unknown display)")
        return WtypeInjector(config)
    if shutil.which("ydotool"):
        logger.info("Selected injector: ydotool (unknown display)")
        return YdotoolInjector(config)
    logger.info("Falling back to clipboard (unknown display)")
    return ClipboardInjector(config, display)
