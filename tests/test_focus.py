"""Tests for linux_whisper.focus — focused app detection and classification."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from linux_whisper.focus import (
    AppCategory,
    FocusedApp,
    _classify_wm_class,
    _detect_hyprland,
    _detect_sway,
    _detect_x11,
    _find_focused_node,
    build_context_string,
    detect_focused_app,
)

# ── AppCategory ────────────────────────────────────────────────────────────


class TestAppCategory:
    """Test AppCategory enum values and guidance."""

    def test_all_values(self):
        assert AppCategory.MESSAGING.value == "messaging"
        assert AppCategory.EMAIL.value == "email"
        assert AppCategory.CODE.value == "code"
        assert AppCategory.TERMINAL.value == "terminal"
        assert AppCategory.DOCUMENT.value == "document"
        assert AppCategory.BROWSER.value == "browser"
        assert AppCategory.UNKNOWN.value == "unknown"

    def test_guidance_messaging(self):
        assert "casual" in AppCategory.MESSAGING.guidance.lower()

    def test_guidance_email(self):
        assert "professional" in AppCategory.EMAIL.guidance.lower()

    def test_guidance_code(self):
        assert "terse" in AppCategory.CODE.guidance.lower()

    def test_guidance_unknown_empty(self):
        assert AppCategory.UNKNOWN.guidance == ""


# ── FocusedApp ─────────────────────────────────────────────────────────────


class TestFocusedApp:
    """Test FocusedApp dataclass."""

    def test_creation(self):
        app = FocusedApp(wm_class="slack", app_name="Slack", category=AppCategory.MESSAGING)
        assert app.wm_class == "slack"
        assert app.app_name == "Slack"
        assert app.category == AppCategory.MESSAGING

    def test_frozen(self):
        app = FocusedApp(wm_class="slack", app_name="Slack", category=AppCategory.MESSAGING)
        with pytest.raises(AttributeError):
            app.wm_class = "other"  # type: ignore[misc]


# ── _classify_wm_class ────────────────────────────────────────────────────


class TestClassifyWmClass:
    """Test WM_CLASS to category mapping."""

    def test_exact_match_slack(self):
        assert _classify_wm_class("slack") == AppCategory.MESSAGING

    def test_exact_match_discord(self):
        assert _classify_wm_class("discord") == AppCategory.MESSAGING

    def test_exact_match_thunderbird(self):
        assert _classify_wm_class("thunderbird") == AppCategory.EMAIL

    def test_exact_match_code(self):
        assert _classify_wm_class("code") == AppCategory.CODE

    def test_exact_match_kitty(self):
        assert _classify_wm_class("kitty") == AppCategory.TERMINAL

    def test_exact_match_firefox(self):
        assert _classify_wm_class("firefox") == AppCategory.BROWSER

    def test_exact_match_obsidian(self):
        assert _classify_wm_class("obsidian") == AppCategory.DOCUMENT

    def test_case_insensitive(self):
        assert _classify_wm_class("Slack") == AppCategory.MESSAGING
        assert _classify_wm_class("FIREFOX") == AppCategory.BROWSER

    def test_substring_match(self):
        # "google-chrome" contains "chrome"
        assert _classify_wm_class("google-chrome") == AppCategory.BROWSER

    def test_unknown_app(self):
        assert _classify_wm_class("gimp") == AppCategory.UNKNOWN

    def test_empty_string(self):
        assert _classify_wm_class("") == AppCategory.UNKNOWN

    def test_whitespace(self):
        assert _classify_wm_class("  slack  ") == AppCategory.MESSAGING

    def test_jetbrains_idea(self):
        assert _classify_wm_class("idea") == AppCategory.CODE

    def test_telegram_desktop(self):
        assert _classify_wm_class("telegram-desktop") == AppCategory.MESSAGING


# ── _find_focused_node ─────────────────────────────────────────────────────


class TestFindFocusedNode:
    """Test Sway tree traversal."""

    def test_finds_focused_leaf(self):
        tree = {
            "focused": False,
            "nodes": [
                {"focused": False, "nodes": []},
                {"focused": True, "nodes": [], "floating_nodes": [], "app_id": "kitty"},
            ],
        }
        result = _find_focused_node(tree)
        assert result is not None
        assert result["app_id"] == "kitty"

    def test_finds_nested_focused(self):
        tree = {
            "focused": False,
            "nodes": [
                {
                    "focused": False,
                    "nodes": [
                        {"focused": True, "nodes": [], "floating_nodes": [], "app_id": "firefox"},
                    ],
                },
            ],
        }
        result = _find_focused_node(tree)
        assert result is not None
        assert result["app_id"] == "firefox"

    def test_returns_none_when_no_focused(self):
        tree = {"focused": False, "nodes": [{"focused": False, "nodes": []}]}
        assert _find_focused_node(tree) is None

    def test_finds_in_floating_nodes(self):
        tree = {
            "focused": False,
            "nodes": [],
            "floating_nodes": [
                {"focused": True, "nodes": [], "floating_nodes": [], "app_id": "slack"},
            ],
        }
        result = _find_focused_node(tree)
        assert result is not None
        assert result["app_id"] == "slack"


# ── _detect_x11 ───────────────────────────────────────────────────────────


class TestDetectX11:
    """Test X11 detection via xdotool."""

    @patch("linux_whisper.focus.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="firefox\n")
        result = _detect_x11()
        assert result is not None
        assert result.wm_class == "firefox"
        assert result.category == AppCategory.BROWSER

    @patch("linux_whisper.focus.subprocess.run")
    def test_xdotool_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert _detect_x11() is None

    @patch("linux_whisper.focus.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="xdotool", timeout=0.1)
        assert _detect_x11() is None

    @patch("linux_whisper.focus.subprocess.run")
    def test_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert _detect_x11() is None

    @patch("linux_whisper.focus.subprocess.run")
    def test_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        assert _detect_x11() is None


# ── _detect_sway ───────────────────────────────────────────────────────────


class TestDetectSway:
    """Test Sway detection via swaymsg."""

    @patch("linux_whisper.focus.subprocess.run")
    def test_native_wayland_app(self, mock_run):
        tree = {
            "focused": False,
            "nodes": [
                {
                    "focused": True,
                    "nodes": [],
                    "floating_nodes": [],
                    "app_id": "kitty",
                    "name": "kitty",
                },
            ],
        }
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(tree)
        )
        result = _detect_sway()
        assert result is not None
        assert result.wm_class == "kitty"
        assert result.category == AppCategory.TERMINAL

    @patch("linux_whisper.focus.subprocess.run")
    def test_xwayland_app(self, mock_run):
        tree = {
            "focused": False,
            "nodes": [
                {
                    "focused": True,
                    "nodes": [],
                    "floating_nodes": [],
                    "app_id": None,
                    "window_properties": {"class": "Slack"},
                    "name": "Slack",
                },
            ],
        }
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(tree)
        )
        result = _detect_sway()
        assert result is not None
        assert result.wm_class == "Slack"
        assert result.category == AppCategory.MESSAGING

    @patch("linux_whisper.focus.subprocess.run")
    def test_swaymsg_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert _detect_sway() is None

    @patch("linux_whisper.focus.subprocess.run")
    def test_bad_json(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json")
        assert _detect_sway() is None


# ── _detect_hyprland ───────────────────────────────────────────────────────


class TestDetectHyprland:
    """Test Hyprland detection via hyprctl."""

    @patch("linux_whisper.focus.subprocess.run")
    def test_success(self, mock_run):
        data = {"class": "discord", "title": "Discord"}
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(data)
        )
        result = _detect_hyprland()
        assert result is not None
        assert result.wm_class == "discord"
        assert result.category == AppCategory.MESSAGING

    @patch("linux_whisper.focus.subprocess.run")
    def test_hyprctl_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert _detect_hyprland() is None

    @patch("linux_whisper.focus.subprocess.run")
    def test_empty_class(self, mock_run):
        data = {"class": "", "title": ""}
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps(data)
        )
        assert _detect_hyprland() is None


# ── detect_focused_app (integration) ───────────────────────────────────────


class TestDetectFocusedApp:
    """Test the main detect_focused_app dispatcher."""

    @patch("linux_whisper.focus.subprocess.run")
    def test_x11_dispatch(self, mock_run, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        mock_run.return_value = MagicMock(returncode=0, stdout="slack\n")
        result = detect_focused_app()
        assert result is not None
        assert result.category == AppCategory.MESSAGING

    @patch("linux_whisper.focus._detect_wayland")
    def test_wayland_dispatch(self, mock_detect, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        mock_detect.return_value = FocusedApp(
            wm_class="kitty", app_name="kitty", category=AppCategory.TERMINAL
        )
        result = detect_focused_app()
        assert result is not None
        assert result.category == AppCategory.TERMINAL

    def test_unknown_session_no_env(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert detect_focused_app() is None


# ── build_context_string ───────────────────────────────────────────────────


class TestBuildContextString:
    """Test context string formatting."""

    def test_messaging_context(self):
        app = FocusedApp(wm_class="slack", app_name="Slack", category=AppCategory.MESSAGING)
        ctx = build_context_string(app)
        assert "Slack" in ctx
        assert "messaging" in ctx
        assert "casual" in ctx.lower()

    def test_email_context(self):
        app = FocusedApp(
            wm_class="thunderbird", app_name="Thunderbird", category=AppCategory.EMAIL
        )
        ctx = build_context_string(app)
        assert "Thunderbird" in ctx
        assert "professional" in ctx.lower()

    def test_unknown_no_guidance(self):
        app = FocusedApp(wm_class="gimp", app_name="GIMP", category=AppCategory.UNKNOWN)
        ctx = build_context_string(app)
        assert "GIMP" in ctx
        assert "Adjust tone" not in ctx
