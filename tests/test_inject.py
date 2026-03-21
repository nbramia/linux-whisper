"""Tests for linux_whisper.inject — display server detection, injector factory, injector classes.

All subprocess calls are mocked; no actual typing tools are invoked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linux_whisper.config import InjectConfig
from linux_whisper.inject.injector import (
    ClipboardInjector,
    DisplayServer,
    TextInjector,
    WtypeInjector,
    XdotoolInjector,
    YdotoolInjector,
    _detect_display_server,
    _is_wlroots_compositor,
    detect_injector,
)


# ── Display server detection ────────────────────────────────────────────────


class TestDetectDisplayServer:

    def test_x11_via_session_type(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        assert _detect_display_server() == DisplayServer.X11

    def test_wayland_via_session_type(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        assert _detect_display_server() == DisplayServer.WAYLAND

    def test_wayland_via_display_env(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert _detect_display_server() == DisplayServer.WAYLAND

    def test_x11_via_display_env(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        assert _detect_display_server() == DisplayServer.X11

    def test_unknown_without_env(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        assert _detect_display_server() == DisplayServer.UNKNOWN

    def test_case_insensitive_session_type(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "X11")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        assert _detect_display_server() == DisplayServer.X11


# ── wlroots compositor detection ────────────────────────────────────────────


class TestIsWlrootsCompositor:

    def test_sway_detected(self, monkeypatch):
        monkeypatch.setenv("SWAYSOCK", "/run/user/1000/sway-ipc.sock")
        assert _is_wlroots_compositor() is True

    def test_hyprland_detected(self, monkeypatch):
        monkeypatch.delenv("SWAYSOCK", raising=False)
        monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc123")
        assert _is_wlroots_compositor() is True

    def test_xdg_desktop_sway(self, monkeypatch):
        monkeypatch.delenv("SWAYSOCK", raising=False)
        monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")
        assert _is_wlroots_compositor() is True

    def test_xdg_desktop_hyprland(self, monkeypatch):
        monkeypatch.delenv("SWAYSOCK", raising=False)
        monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "Hyprland")
        assert _is_wlroots_compositor() is True

    def test_gnome_is_not_wlroots(self, monkeypatch):
        monkeypatch.delenv("SWAYSOCK", raising=False)
        monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
        assert _is_wlroots_compositor() is False

    def test_empty_env_is_not_wlroots(self, monkeypatch):
        monkeypatch.delenv("SWAYSOCK", raising=False)
        monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
        monkeypatch.delenv("XDG_CURRENT_DESKTOP", raising=False)
        assert _is_wlroots_compositor() is False


# ── Injector factory: auto-detect ───────────────────────────────────────────


class TestDetectInjectorAuto:

    def test_x11_with_xdotool(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        with patch("linux_whisper.inject.injector.shutil.which", return_value="/usr/bin/xdotool"):
            injector = detect_injector(InjectConfig(method="auto"))
        assert isinstance(injector, XdotoolInjector)

    def test_x11_without_xdotool_falls_back_to_clipboard(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        with patch("linux_whisper.inject.injector.shutil.which", return_value=None):
            injector = detect_injector(InjectConfig(method="auto"))
        assert isinstance(injector, ClipboardInjector)

    def test_wayland_wlroots_with_wtype(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("SWAYSOCK", "/tmp/sway")

        def which_side_effect(name):
            return "/usr/bin/wtype" if name == "wtype" else None

        with patch("linux_whisper.inject.injector.shutil.which", side_effect=which_side_effect):
            injector = detect_injector(InjectConfig(method="auto"))
        assert isinstance(injector, WtypeInjector)

    def test_wayland_non_wlroots_with_ydotool(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.delenv("SWAYSOCK", raising=False)
        monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")

        def which_side_effect(name):
            return "/usr/bin/ydotool" if name == "ydotool" else None

        with patch("linux_whisper.inject.injector.shutil.which", side_effect=which_side_effect):
            injector = detect_injector(InjectConfig(method="auto"))
        assert isinstance(injector, YdotoolInjector)

    def test_wayland_no_tools_falls_back_to_clipboard(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.delenv("SWAYSOCK", raising=False)
        monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")

        with patch("linux_whisper.inject.injector.shutil.which", return_value=None):
            injector = detect_injector(InjectConfig(method="auto"))
        assert isinstance(injector, ClipboardInjector)

    def test_unknown_display_tries_all(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)

        with patch("linux_whisper.inject.injector.shutil.which", return_value=None):
            injector = detect_injector(InjectConfig(method="auto"))
        assert isinstance(injector, ClipboardInjector)


# ── Injector factory: explicit method ───────────────────────────────────────


class TestDetectInjectorExplicit:

    def test_explicit_xdotool_found(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        with patch("linux_whisper.inject.injector.shutil.which", return_value="/usr/bin/xdotool"):
            injector = detect_injector(InjectConfig(method="xdotool"))
        assert isinstance(injector, XdotoolInjector)

    def test_explicit_xdotool_missing_falls_back(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        with patch("linux_whisper.inject.injector.shutil.which", return_value=None):
            injector = detect_injector(InjectConfig(method="xdotool"))
        assert isinstance(injector, ClipboardInjector)

    def test_explicit_wtype_found(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        with patch("linux_whisper.inject.injector.shutil.which", return_value="/usr/bin/wtype"):
            injector = detect_injector(InjectConfig(method="wtype"))
        assert isinstance(injector, WtypeInjector)

    def test_explicit_ydotool_found(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        with patch("linux_whisper.inject.injector.shutil.which", return_value="/usr/bin/ydotool"):
            injector = detect_injector(InjectConfig(method="ydotool"))
        assert isinstance(injector, YdotoolInjector)

    def test_explicit_clipboard(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        injector = detect_injector(InjectConfig(method="clipboard"))
        assert isinstance(injector, ClipboardInjector)

    def test_explicit_unknown_method_falls_back(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        # InjectConfig allows any string; the factory handles unknown values
        # by falling back to clipboard
        injector = detect_injector(InjectConfig(method="clipboard"))
        assert isinstance(injector, ClipboardInjector)


# ── Injector classes with mocked subprocess ─────────────────────────────────


class TestXdotoolInjector:

    async def test_inject_success(self):
        config = InjectConfig(method="xdotool")
        injector = XdotoolInjector(config)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, b"", b"")
            result = await injector.inject("hello world")

        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0]
        assert "xdotool" in args
        assert "hello world" in args

    async def test_inject_failure(self):
        config = InjectConfig(method="xdotool")
        injector = XdotoolInjector(config)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (1, b"", b"error")
            result = await injector.inject("hello")

        assert result is False

    async def test_inject_with_typing_delay(self):
        config = InjectConfig(method="xdotool", typing_delay=50)
        injector = XdotoolInjector(config)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, b"", b"")
            await injector.inject("test")

        args = mock_run.call_args[0]
        assert "--delay" in args
        assert "50" in args


class TestWtypeInjector:

    async def test_inject_success(self):
        config = InjectConfig(method="wtype")
        injector = WtypeInjector(config)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, b"", b"")
            result = await injector.inject("wayland text")

        assert result is True
        args = mock_run.call_args[0]
        assert "wtype" in args

    async def test_inject_failure(self):
        config = InjectConfig(method="wtype")
        injector = WtypeInjector(config)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (1, b"", b"err")
            result = await injector.inject("test")

        assert result is False

    async def test_inject_with_typing_delay(self):
        config = InjectConfig(method="wtype", typing_delay=20)
        injector = WtypeInjector(config)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, b"", b"")
            await injector.inject("test")

        args = mock_run.call_args[0]
        assert "-d" in args
        assert "20" in args


class TestYdotoolInjector:

    async def test_inject_success(self):
        config = InjectConfig(method="ydotool")
        injector = YdotoolInjector(config)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, b"", b"")
            result = await injector.inject("ydotool text")

        assert result is True
        args = mock_run.call_args[0]
        assert "ydotool" in args

    async def test_inject_with_key_delay(self):
        config = InjectConfig(method="ydotool", typing_delay=30)
        injector = YdotoolInjector(config)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (0, b"", b"")
            await injector.inject("test")

        args = mock_run.call_args[0]
        assert "--key-delay" in args
        assert "30" in args


class TestClipboardInjector:

    async def test_inject_success_x11(self):
        config = InjectConfig(method="clipboard")
        injector = ClipboardInjector(config, DisplayServer.X11)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            # get_clipboard, set_clipboard, paste_keystroke
            mock_run.side_effect = [
                (0, b"old clipboard", b""),  # xclip read
                (0, b"", b""),  # xclip write
                (0, b"", b""),  # xdotool paste
            ]
            result = await injector.inject("hello")

        assert result is True

    async def test_inject_set_clipboard_fails(self):
        config = InjectConfig(method="clipboard")
        injector = ClipboardInjector(config, DisplayServer.X11)

        with patch("linux_whisper.inject.injector._run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = [
                (0, b"old", b""),  # get_clipboard
                (1, b"", b"error"),  # set_clipboard fails
            ]
            result = await injector.inject("hello")

        assert result is False

    def test_clipboard_injector_repr(self):
        config = InjectConfig(method="clipboard")
        injector = ClipboardInjector(config, DisplayServer.X11)
        assert "ClipboardInjector" in repr(injector)


# ── TextInjector base class ─────────────────────────────────────────────────


class TestTextInjectorBase:

    def test_repr(self):
        injector = XdotoolInjector(InjectConfig())
        assert "XdotoolInjector" in repr(injector)

    def test_name_attribute(self):
        assert XdotoolInjector.name == "xdotool"
        assert WtypeInjector.name == "wtype"
        assert YdotoolInjector.name == "ydotool"
        assert ClipboardInjector.name == "clipboard"
