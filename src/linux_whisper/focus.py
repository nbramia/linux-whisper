"""Focused application detection for context-aware LLM prompts.

Detects the currently focused window on X11 and Wayland (Sway, Hyprland),
maps the WM_CLASS / app_id to a category, and produces a context string
for the LLM system prompt.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App categories and tone guidance
# ---------------------------------------------------------------------------


class AppCategory(Enum):
    """Application categories with associated tone guidance."""

    MESSAGING = "messaging"
    EMAIL = "email"
    CODE = "code"
    TERMINAL = "terminal"
    DOCUMENT = "document"
    BROWSER = "browser"
    UNKNOWN = "unknown"

    @property
    def guidance(self) -> str:
        """Return a short tone instruction for the LLM."""
        return _CATEGORY_GUIDANCE.get(self, "")


_CATEGORY_GUIDANCE: dict[AppCategory, str] = {
    AppCategory.MESSAGING: (
        "Conversational and casual. No trailing period on short messages. Lowercase OK."
    ),
    AppCategory.EMAIL: "Professional tone. Proper punctuation and capitalization.",
    AppCategory.CODE: "Terse and technical. No pleasantries.",
    AppCategory.TERMINAL: "Terse and technical. No pleasantries.",
    AppCategory.DOCUMENT: "Clear, well-structured prose.",
    AppCategory.BROWSER: "Match the context — could be casual or professional.",
    AppCategory.UNKNOWN: "",
}


# ---------------------------------------------------------------------------
# WM_CLASS to category mapping
# ---------------------------------------------------------------------------

_WM_CLASS_CATEGORIES: dict[str, AppCategory] = {
    # Messaging
    "slack": AppCategory.MESSAGING,
    "discord": AppCategory.MESSAGING,
    "telegram": AppCategory.MESSAGING,
    "telegram-desktop": AppCategory.MESSAGING,
    "signal": AppCategory.MESSAGING,
    "element": AppCategory.MESSAGING,
    "whatsapp": AppCategory.MESSAGING,
    "teams": AppCategory.MESSAGING,
    "microsoft teams": AppCategory.MESSAGING,
    "zoom": AppCategory.MESSAGING,
    # Email
    "thunderbird": AppCategory.EMAIL,
    "geary": AppCategory.EMAIL,
    "mailspring": AppCategory.EMAIL,
    "evolution": AppCategory.EMAIL,
    # Code editors
    "code": AppCategory.CODE,
    "vscodium": AppCategory.CODE,
    "vim": AppCategory.CODE,
    "nvim": AppCategory.CODE,
    "neovim": AppCategory.CODE,
    "emacs": AppCategory.CODE,
    "idea": AppCategory.CODE,
    "pycharm": AppCategory.CODE,
    "webstorm": AppCategory.CODE,
    "goland": AppCategory.CODE,
    "clion": AppCategory.CODE,
    "rustrover": AppCategory.CODE,
    "rider": AppCategory.CODE,
    "zed": AppCategory.CODE,
    "sublime_text": AppCategory.CODE,
    "kate": AppCategory.CODE,
    # Terminals
    "kitty": AppCategory.TERMINAL,
    "alacritty": AppCategory.TERMINAL,
    "foot": AppCategory.TERMINAL,
    "gnome-terminal": AppCategory.TERMINAL,
    "konsole": AppCategory.TERMINAL,
    "wezterm": AppCategory.TERMINAL,
    "ghostty": AppCategory.TERMINAL,
    "terminator": AppCategory.TERMINAL,
    "tilix": AppCategory.TERMINAL,
    "xterm": AppCategory.TERMINAL,
    "st": AppCategory.TERMINAL,
    # Browsers
    "firefox": AppCategory.BROWSER,
    "chromium": AppCategory.BROWSER,
    "chromium-browser": AppCategory.BROWSER,
    "google-chrome": AppCategory.BROWSER,
    "brave-browser": AppCategory.BROWSER,
    "zen-browser": AppCategory.BROWSER,
    "vivaldi": AppCategory.BROWSER,
    # Documents
    "libreoffice": AppCategory.DOCUMENT,
    "soffice": AppCategory.DOCUMENT,
    "obsidian": AppCategory.DOCUMENT,
    "logseq": AppCategory.DOCUMENT,
    "notion": AppCategory.DOCUMENT,
    "typora": AppCategory.DOCUMENT,
}


def _classify_wm_class(wm_class: str) -> AppCategory:
    """Map a WM_CLASS string to an AppCategory.

    Tries exact match first (case-insensitive), then substring match.
    """
    lower = wm_class.lower().strip()
    if not lower:
        return AppCategory.UNKNOWN

    # Exact match
    if lower in _WM_CLASS_CATEGORIES:
        return _WM_CLASS_CATEGORIES[lower]

    # Substring match — check if any known key is contained in the wm_class
    for key, category in _WM_CLASS_CATEGORIES.items():
        if key in lower:
            return category

    return AppCategory.UNKNOWN


# ---------------------------------------------------------------------------
# FocusedApp result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FocusedApp:
    """Information about the currently focused application."""

    wm_class: str
    app_name: str
    category: AppCategory


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_SUBPROCESS_TIMEOUT = 0.1  # 100ms — xdotool/swaymsg are very fast


def detect_focused_app() -> FocusedApp | None:
    """Detect the currently focused application window.

    Returns a :class:`FocusedApp` with the WM_CLASS, display name, and
    category, or ``None`` if detection fails or the display server is
    unsupported.
    """
    try:
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "x11" or (
            not session_type and os.environ.get("DISPLAY")
        ):
            return _detect_x11()
        if session_type == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
            return _detect_wayland()
    except Exception:
        logger.debug("Focus detection failed", exc_info=True)
    return None


def _detect_x11() -> FocusedApp | None:
    """Detect focused app on X11 using xdotool."""
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowclassname"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            return None
        wm_class = result.stdout.strip()
        if not wm_class:
            return None
        category = _classify_wm_class(wm_class)
        return FocusedApp(wm_class=wm_class, app_name=wm_class, category=category)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("xdotool not available or timed out")
        return None


def _detect_wayland() -> FocusedApp | None:
    """Detect focused app on Wayland (Sway or Hyprland)."""
    if os.environ.get("SWAYSOCK"):
        return _detect_sway()
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return _detect_hyprland()
    # GNOME/KDE Wayland — no reliable detection without extensions
    logger.debug("Unsupported Wayland compositor for focus detection")
    return None


def _detect_sway() -> FocusedApp | None:
    """Detect focused app on Sway via swaymsg."""
    try:
        result = subprocess.run(
            ["swaymsg", "-t", "get_tree"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            return None
        tree = json.loads(result.stdout)
        focused = _find_focused_node(tree)
        if focused is None:
            return None
        # Sway uses app_id for native Wayland apps, window_properties.class for XWayland
        wm_class = focused.get("app_id") or ""
        if not wm_class:
            props = focused.get("window_properties", {})
            wm_class = props.get("class", "")
        app_name = focused.get("name", wm_class) or wm_class
        if not wm_class:
            return None
        category = _classify_wm_class(wm_class)
        return FocusedApp(wm_class=wm_class, app_name=app_name, category=category)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("swaymsg failed or returned invalid JSON")
        return None


def _find_focused_node(node: dict) -> dict | None:
    """Recursively find the focused leaf node in a Sway tree."""
    if node.get("focused") and not node.get("nodes") and not node.get("floating_nodes"):
        return node
    for child in node.get("nodes", []):
        result = _find_focused_node(child)
        if result is not None:
            return result
    for child in node.get("floating_nodes", []):
        result = _find_focused_node(child)
        if result is not None:
            return result
    return None


def _detect_hyprland() -> FocusedApp | None:
    """Detect focused app on Hyprland via hyprctl."""
    try:
        result = subprocess.run(
            ["hyprctl", "activewindow", "-j"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        wm_class = data.get("class", "")
        app_name = data.get("title", wm_class) or wm_class
        if not wm_class:
            return None
        category = _classify_wm_class(wm_class)
        return FocusedApp(wm_class=wm_class, app_name=app_name, category=category)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("hyprctl failed or returned invalid JSON")
        return None


# ---------------------------------------------------------------------------
# Context string builder
# ---------------------------------------------------------------------------


def build_context_string(app: FocusedApp) -> str:
    """Build a context string for the LLM system prompt."""
    parts = [f"The user is typing in {app.app_name} ({app.category.value})."]
    if app.category.guidance:
        parts.append(f"Adjust tone: {app.category.guidance}")
    return " ".join(parts)
