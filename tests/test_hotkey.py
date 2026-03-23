"""Tests for linux_whisper.hotkey — key parsing, daemon modes, callbacks."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# The evdev stub is set up in conftest.py. Grab the ecodes namespace
# from the stub so we can reference the integer constants in assertions.
_evdev_stub = sys.modules["evdev"]
ecodes = _evdev_stub.ecodes

# Extend the stub with additional key codes that hotkey.py will resolve at
# import time via getattr(ecodes, "KEY_...").  We attach them before importing
# the module under test.
_EXTRA_KEYS = {
    "KEY_FN": 0x1D0,
    "KEY_GRAVE": 41,
    "KEY_MINUS": 12,
    "KEY_EQUAL": 13,
    "KEY_LEFTBRACE": 26,
    "KEY_RIGHTBRACE": 27,
    "KEY_BACKSLASH": 43,
    "KEY_SEMICOLON": 39,
    "KEY_APOSTROPHE": 40,
    "KEY_COMMA": 51,
    "KEY_DOT": 52,
    "KEY_SLASH": 53,
    "KEY_C": 46,
    "KEY_Z": 44,
    "KEY_X": 45,
    "KEY_V": 47,
    "KEY_D": 32,
    "KEY_F": 33,
    "KEY_G": 34,
    "KEY_H": 35,
}

for _name, _val in _EXTRA_KEYS.items():
    if not hasattr(ecodes, _name):
        setattr(ecodes, _name, _val)

from linux_whisper.hotkey import (  # noqa: E402
    HotkeyCombination,
    HotkeyDaemon,
    _find_keyboard_devices,
    _key_name_to_code,
    _normalize_modifier,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_event(type_: int, code: int, value: int):
    """Create a lightweight event-like object for _handle_event."""
    ev = types.SimpleNamespace(type=type_, code=code, value=value)
    return ev


def _make_daemon(
    hotkey: str = "fn",
    mode: str = "hold",
) -> tuple[HotkeyDaemon, MagicMock, MagicMock]:
    """Return (daemon, on_start_mock, on_stop_mock)."""
    on_start = MagicMock()
    on_stop = MagicMock()
    daemon = HotkeyDaemon(hotkey, mode, on_start, on_stop)
    return daemon, on_start, on_stop


# ── _key_name_to_code ────────────────────────────────────────────────────────


class TestKeyNameToCode:
    """Verify _key_name_to_code resolves names to evdev codes."""

    def test_fn_key(self):
        assert _key_name_to_code("fn") == ecodes.KEY_FN

    def test_ctrl_modifier(self):
        assert _key_name_to_code("ctrl") == ecodes.KEY_LEFTCTRL

    def test_shift_modifier(self):
        assert _key_name_to_code("shift") == ecodes.KEY_LEFTSHIFT

    def test_alt_modifier(self):
        assert _key_name_to_code("alt") == ecodes.KEY_LEFTALT

    def test_super_modifier(self):
        assert _key_name_to_code("super") == ecodes.KEY_LEFTMETA

    def test_meta_modifier(self):
        assert _key_name_to_code("meta") == ecodes.KEY_LEFTMETA

    def test_space_key(self):
        assert _key_name_to_code("space") == ecodes.KEY_SPACE

    def test_enter_key(self):
        assert _key_name_to_code("enter") == ecodes.KEY_ENTER

    def test_letter_e(self):
        assert _key_name_to_code("e") == ecodes.KEY_E

    def test_letter_a(self):
        assert _key_name_to_code("a") == ecodes.KEY_A

    def test_grave_alias(self):
        assert _key_name_to_code("grave") == ecodes.KEY_GRAVE

    def test_backtick_alias(self):
        assert _key_name_to_code("`") == ecodes.KEY_GRAVE

    def test_case_insensitive(self):
        assert _key_name_to_code("CTRL") == ecodes.KEY_LEFTCTRL
        assert _key_name_to_code("Shift") == ecodes.KEY_LEFTSHIFT

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown key name"):
            _key_name_to_code("nonexistentkey123")


# ── _normalize_modifier ──────────────────────────────────────────────────────


class TestNormalizeModifier:
    """Verify right-hand modifiers are mapped to left-hand equivalents."""

    def test_right_ctrl_to_left(self):
        assert _normalize_modifier(ecodes.KEY_RIGHTCTRL) == ecodes.KEY_LEFTCTRL

    def test_right_shift_to_left(self):
        assert _normalize_modifier(ecodes.KEY_RIGHTSHIFT) == ecodes.KEY_LEFTSHIFT

    def test_right_alt_to_left(self):
        assert _normalize_modifier(ecodes.KEY_RIGHTALT) == ecodes.KEY_LEFTALT

    def test_right_meta_to_left(self):
        assert _normalize_modifier(ecodes.KEY_RIGHTMETA) == ecodes.KEY_LEFTMETA

    def test_non_modifier_passes_through(self):
        assert _normalize_modifier(ecodes.KEY_A) == ecodes.KEY_A

    def test_left_ctrl_passes_through(self):
        assert _normalize_modifier(ecodes.KEY_LEFTCTRL) == ecodes.KEY_LEFTCTRL


# ── HotkeyCombination.parse ──────────────────────────────────────────────────


class TestHotkeyCombinationParse:
    """Verify hotkey string parsing."""

    def test_single_key_fn(self):
        combo = HotkeyCombination.parse("fn")
        assert combo.key == ecodes.KEY_FN
        assert combo.modifiers == frozenset()

    def test_ctrl_shift_e(self):
        combo = HotkeyCombination.parse("ctrl+shift+e")
        assert combo.key == ecodes.KEY_E
        assert combo.modifiers == frozenset({ecodes.KEY_LEFTCTRL, ecodes.KEY_LEFTSHIFT})

    def test_super_grave(self):
        combo = HotkeyCombination.parse("super+grave")
        assert combo.key == ecodes.KEY_GRAVE
        assert combo.modifiers == frozenset({ecodes.KEY_LEFTMETA})

    def test_case_insensitive_parse(self):
        combo = HotkeyCombination.parse("CTRL+SHIFT+E")
        assert combo.key == ecodes.KEY_E
        assert combo.modifiers == frozenset({ecodes.KEY_LEFTCTRL, ecodes.KEY_LEFTSHIFT})

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty hotkey string"):
            HotkeyCombination.parse("")

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unknown key name"):
            HotkeyCombination.parse("ctrl+nosuchkey")

    def test_repr(self):
        combo = HotkeyCombination.parse("ctrl+e")
        assert repr(combo) == "HotkeyCombination('ctrl+e')"

    def test_whitespace_tolerant(self):
        combo = HotkeyCombination.parse("ctrl + e")
        assert combo.key == ecodes.KEY_E
        assert combo.modifiers == frozenset({ecodes.KEY_LEFTCTRL})

    def test_right_modifier_normalized_in_combo(self):
        """rightctrl should be normalized to leftctrl in the modifiers set."""
        combo = HotkeyCombination.parse("rightctrl+a")
        assert ecodes.KEY_LEFTCTRL in combo.modifiers


# ── HotkeyDaemon construction ────────────────────────────────────────────────


class TestHotkeyDaemonConstruction:
    """Verify daemon construction and mode validation."""

    @pytest.mark.parametrize("mode", ["hold", "toggle", "auto", "vad-auto"])
    def test_valid_modes(self, mode):
        daemon, _, _ = _make_daemon(mode=mode)
        assert daemon._mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            HotkeyDaemon("fn", "invalid", MagicMock(), MagicMock())


# ── HotkeyDaemon start/stop lifecycle ────────────────────────────────────────


class TestHotkeyDaemonLifecycle:
    """Verify start/stop behaviour without running the main loop."""

    def test_start_creates_thread(self):
        daemon, _, _ = _make_daemon()
        with patch.object(daemon, "_run"):
            daemon.start()
            assert daemon._thread is not None
            daemon.stop(timeout=1.0)

    def test_stop_clears_thread(self):
        daemon, _, _ = _make_daemon()
        with patch.object(daemon, "_run"):
            daemon.start()
            daemon.stop(timeout=1.0)
            assert daemon._thread is None

    def test_alive_property(self):
        daemon, _, _ = _make_daemon()
        assert daemon.alive is False

    def test_double_start_is_safe(self):
        daemon, _, _ = _make_daemon()
        with patch.object(daemon, "_run"):
            daemon.start()
            daemon.start()  # should log warning but not crash
            daemon.stop(timeout=1.0)


# ── Hold mode ────────────────────────────────────────────────────────────────


class TestHoldMode:
    """Hold-to-talk: key down starts, key up stops."""

    def test_key_down_fires_start(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="hold")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        on_start.assert_called_once()
        on_stop.assert_not_called()

    def test_key_up_fires_stop(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="hold")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
        on_start.assert_called_once()
        on_stop.assert_called_once()

    def test_key_up_without_recording_no_stop(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="hold")
        # Key up without prior key down that started recording
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
        on_start.assert_not_called()
        on_stop.assert_not_called()

    def test_repeat_event_does_not_re_fire_start(self):
        daemon, on_start, _ = _make_daemon(hotkey="fn", mode="hold")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        # value=2 is autorepeat
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 2))
        # _fire_start guards against double-start
        on_start.assert_called_once()

    def test_non_ev_key_event_ignored(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="hold")
        # type=3 is EV_ABS, should be ignored
        daemon._handle_event(_make_event(3, ecodes.KEY_FN, 1))
        on_start.assert_not_called()

    def test_wrong_key_ignored(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="hold")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_A, 1))
        on_start.assert_not_called()


# ── Toggle mode ──────────────────────────────────────────────────────────────


class TestToggleMode:
    """Toggle: first press starts, second press stops."""

    def test_first_down_starts(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="toggle")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        on_start.assert_called_once()
        on_stop.assert_not_called()

    def test_second_down_stops(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="toggle")
        # First press: start
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
        # Second press: stop
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        on_start.assert_called_once()
        on_stop.assert_called_once()

    def test_key_up_does_not_stop_in_toggle(self):
        """In toggle mode, key-up should NOT fire stop."""
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="toggle")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
        on_stop.assert_not_called()


# ── Modifier tracking ────────────────────────────────────────────────────────


class TestModifierTracking:
    """Verify that combos with modifiers require all modifiers to be held."""

    def test_combo_requires_modifier(self):
        daemon, on_start, _ = _make_daemon(hotkey="ctrl+e", mode="hold")
        # Press E without ctrl — should NOT start
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_E, 1))
        on_start.assert_not_called()

    def test_combo_with_modifier_held(self):
        daemon, on_start, _ = _make_daemon(hotkey="ctrl+e", mode="hold")
        # Hold ctrl, then press e
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_E, 1))
        on_start.assert_called_once()

    def test_right_ctrl_satisfies_ctrl_modifier(self):
        daemon, on_start, _ = _make_daemon(hotkey="ctrl+e", mode="hold")
        # Right ctrl should be normalized and satisfy the "ctrl" modifier
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_RIGHTCTRL, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_E, 1))
        on_start.assert_called_once()

    def test_modifier_release_stops_recording_in_hold(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="ctrl+e", mode="hold")
        # Hold ctrl + e
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_E, 1))
        on_start.assert_called_once()
        # Release ctrl while e is still held — should fire stop
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 0))
        on_stop.assert_called_once()

    def test_two_modifiers_required(self):
        daemon, on_start, _ = _make_daemon(hotkey="ctrl+shift+e", mode="hold")
        # Only shift held, press e — should NOT start
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_E, 1))
        on_start.assert_not_called()

    def test_two_modifiers_both_held(self):
        daemon, on_start, _ = _make_daemon(hotkey="ctrl+shift+e", mode="hold")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_E, 1))
        on_start.assert_called_once()


# ── Auto mode ────────────────────────────────────────────────────────────────


class TestAutoMode:
    """Auto: hold > threshold = hold-to-talk; double-tap = toggle."""

    def test_hold_longer_than_threshold_stops_on_release(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="auto")

        with patch("time.monotonic") as mock_mono:
            # Key down at t=0
            mock_mono.return_value = 0.0
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
            on_start.assert_called_once()

            # Key up at t=0.5 (> 0.3 threshold)
            mock_mono.return_value = 0.5
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
            on_stop.assert_called_once()

    def test_single_short_tap_stops_recording(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="auto")

        with patch("time.monotonic") as mock_mono:
            # Quick tap: down at t=0, up at t=0.1
            mock_mono.return_value = 0.0
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
            on_start.assert_called_once()

            mock_mono.return_value = 0.1
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
            # Single tap with no previous tap — stops recording
            on_stop.assert_called_once()

    def test_double_tap_enters_toggle_mode(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="auto")

        with patch("time.monotonic") as mock_mono:
            # First tap: down at t=0, up at t=0.1
            mock_mono.return_value = 0.0
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
            assert on_start.call_count == 1

            mock_mono.return_value = 0.1
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
            assert on_stop.call_count == 1  # first tap stops

            # Second tap: down at t=0.2 (within 0.4 window), up at t=0.25
            mock_mono.return_value = 0.2
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
            assert on_start.call_count == 2  # starts again

            mock_mono.return_value = 0.25
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
            # Should NOT stop — we're in toggle mode now
            assert on_stop.call_count == 1
            assert daemon._in_toggle_mode is True

    def test_third_tap_stops_toggle_mode(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="auto")

        with patch("time.monotonic") as mock_mono:
            # First tap
            mock_mono.return_value = 0.0
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
            mock_mono.return_value = 0.1
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))

            # Second tap (double-tap)
            mock_mono.return_value = 0.2
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
            mock_mono.return_value = 0.25
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
            assert daemon._in_toggle_mode is True

            # Third tap: should stop toggle recording
            mock_mono.return_value = 2.0
            daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
            assert on_stop.call_count == 2
            assert daemon._in_toggle_mode is False


# ── VAD-auto mode ────────────────────────────────────────────────────────────


class TestVadAutoMode:
    """vad-auto: press starts recording, stop is external."""

    def test_key_down_starts(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="vad-auto")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        on_start.assert_called_once()

    def test_key_up_does_not_stop(self):
        daemon, on_start, on_stop = _make_daemon(hotkey="fn", mode="vad-auto")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
        on_stop.assert_not_called()

    def test_second_press_while_recording_no_double_start(self):
        daemon, on_start, _ = _make_daemon(hotkey="fn", mode="vad-auto")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 0))
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        on_start.assert_called_once()

    def test_notify_external_stop_resets_state(self):
        daemon, on_start, _ = _make_daemon(hotkey="fn", mode="vad-auto")
        daemon._handle_event(_make_event(ecodes.EV_KEY, ecodes.KEY_FN, 1))
        on_start.assert_called_once()
        daemon.notify_external_stop()
        assert daemon._recording is False


# ── _fire_start / _fire_stop ─────────────────────────────────────────────────


class TestFireCallbacks:
    """Verify _fire_start and _fire_stop invoke callbacks correctly."""

    def test_fire_start_invokes_callback(self):
        daemon, on_start, _ = _make_daemon()
        daemon._fire_start()
        on_start.assert_called_once()
        assert daemon._recording is True

    def test_fire_start_idempotent(self):
        daemon, on_start, _ = _make_daemon()
        daemon._fire_start()
        daemon._fire_start()
        on_start.assert_called_once()

    def test_fire_stop_invokes_callback(self):
        daemon, _, on_stop = _make_daemon()
        daemon._fire_start()
        daemon._fire_stop()
        on_stop.assert_called_once()
        assert daemon._recording is False

    def test_fire_stop_idempotent(self):
        daemon, _, on_stop = _make_daemon()
        daemon._fire_stop()
        on_stop.assert_not_called()

    def test_fire_start_handles_callback_exception(self):
        on_start = MagicMock(side_effect=RuntimeError("boom"))
        on_stop = MagicMock()
        daemon = HotkeyDaemon("fn", "hold", on_start, on_stop)
        daemon._fire_start()
        # Recording should be reset to False after exception
        assert daemon._recording is False

    def test_fire_stop_handles_callback_exception(self):
        on_start = MagicMock()
        on_stop = MagicMock(side_effect=RuntimeError("boom"))
        daemon = HotkeyDaemon("fn", "hold", on_start, on_stop)
        daemon._fire_start()
        daemon._fire_stop()
        # Should not raise, recording should be False
        assert daemon._recording is False


# ── _find_keyboard_devices ───────────────────────────────────────────────────


class TestFindKeyboardDevices:
    """Verify device discovery with mocked /dev/input."""

    def test_finds_keyboard_device(self):
        mock_dev = MagicMock()
        mock_dev.path = "/dev/input/event0"
        mock_dev.capabilities.return_value = {
            ecodes.EV_KEY: [ecodes.KEY_A, ecodes.KEY_ENTER],
        }

        with patch("linux_whisper.hotkey.Path") as MockPath:
            mock_input_dir = MagicMock()
            mock_input_dir.glob.return_value = ["/dev/input/event0"]
            MockPath.return_value = mock_input_dir

            with patch("linux_whisper.hotkey.InputDevice", return_value=mock_dev):
                devices = _find_keyboard_devices()

        assert len(devices) == 1
        assert devices[0] is mock_dev

    def test_skips_non_keyboard_device(self):
        mock_dev = MagicMock()
        mock_dev.path = "/dev/input/event0"
        # Device has EV_KEY but no common keyboard keys
        mock_dev.capabilities.return_value = {
            ecodes.EV_KEY: [116],  # KEY_POWER only
        }

        with patch("linux_whisper.hotkey.Path") as MockPath:
            mock_input_dir = MagicMock()
            mock_input_dir.glob.return_value = ["/dev/input/event0"]
            MockPath.return_value = mock_input_dir

            with patch("linux_whisper.hotkey.InputDevice", return_value=mock_dev):
                devices = _find_keyboard_devices()

        assert len(devices) == 0
        mock_dev.close.assert_called_once()

    def test_skips_permission_error(self):
        with patch("linux_whisper.hotkey.Path") as MockPath:
            mock_input_dir = MagicMock()
            mock_input_dir.glob.return_value = ["/dev/input/event0"]
            MockPath.return_value = mock_input_dir

            with patch(
                "linux_whisper.hotkey.InputDevice",
                side_effect=PermissionError("no access"),
            ):
                devices = _find_keyboard_devices()

        assert len(devices) == 0
