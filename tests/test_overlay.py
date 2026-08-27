"""Tests for linux_whisper.overlay — GTK3 pill overlay, positioning, facade.

GTK is always mocked here (see the `mock_gtk` fixture in conftest.py) — no
test in this module may require a real GTK installation or a display,
regardless of what happens to be available on the machine running the tests.
"""

from __future__ import annotations

import pytest

from linux_whisper.config import OverlayConfig
from linux_whisper.overlay import Overlay, compute_pill_position

pytestmark = pytest.mark.usefixtures("mock_gtk")


# ---------------------------------------------------------------------------
# compute_pill_position — pure arithmetic, no GTK involved
# ---------------------------------------------------------------------------


class TestComputePillPosition:
    """Verified against the two real monitors from the issue's probe:
    DP-1 (472, 0, 2304x1296) and HDMI-1 primary (0, 1296, 3072x1728)."""

    @pytest.mark.parametrize(
        ("monitor", "position", "expected"),
        [
            # HDMI-1: (0, 1296, 3072, 1728)
            ((0, 1296, 3072, 1728), "center", (1436, 2140)),
            ((0, 1296, 3072, 1728), "bottom-center", (1436, 2936)),
            ((0, 1296, 3072, 1728), "top-center", (1436, 1344)),
            # DP-1: (472, 0, 2304, 1296)
            ((472, 0, 2304, 1296), "center", (1524, 628)),
            ((472, 0, 2304, 1296), "bottom-center", (1524, 1208)),
            ((472, 0, 2304, 1296), "top-center", (1524, 48)),
        ],
    )
    def test_position_for_each_monitor(self, monitor, position, expected):
        assert compute_pill_position(*monitor, position) == expected

    def test_unrecognised_position_falls_back_to_center(self):
        monitor = (0, 1296, 3072, 1728)
        assert compute_pill_position(*monitor, "bogus") == compute_pill_position(
            *monitor, "center"
        )

    def test_horizontally_always_centered(self):
        monitor = (0, 1296, 3072, 1728)
        xs = {
            compute_pill_position(*monitor, p)[0]
            for p in ("center", "bottom-center", "top-center")
        }
        assert len(xs) == 1

    def test_custom_pill_and_margin_sizes(self):
        x, y = compute_pill_position(
            0, 0, 1000, 1000, "top-center", pill_width=100, pill_height=50, margin=10
        )
        assert x == 450
        assert y == 10


# ---------------------------------------------------------------------------
# Overlay facade — availability
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_reflects_has_gtk(self):
        overlay = Overlay()
        assert overlay.available is True

    def test_unavailable_when_gtk_missing(self, monkeypatch):
        import linux_whisper.overlay as overlay_module

        monkeypatch.setattr(overlay_module, "_HAS_GTK", False)
        monkeypatch.setattr(
            overlay_module, "_UNAVAILABLE_REASON", "GTK 3.0 unavailable: no gi"
        )
        overlay = Overlay()
        assert overlay.available is False
        assert overlay.unavailable_reason == "GTK 3.0 unavailable: no gi"

    def test_start_noop_when_unavailable(self, monkeypatch):
        import linux_whisper.overlay as overlay_module

        monkeypatch.setattr(overlay_module, "_HAS_GTK", False)
        overlay = Overlay()
        overlay.start()
        assert overlay._thread is None


# ---------------------------------------------------------------------------
# _OverlayWindow — construction never allows focus stealing
# ---------------------------------------------------------------------------


class TestFocusSafety:
    """The highest-risk requirement: text injection targets the focused
    window, so the overlay must never be able to steal focus."""

    def test_window_uses_override_redirect_popup_type(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        _OverlayWindow("center")
        mock_gtk.Gtk.Window.assert_called_once_with(type=mock_gtk.Gtk.WindowType.POPUP)

    def test_window_never_accepts_focus(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center")
        window._window.set_accept_focus.assert_called_once_with(False)
        window._window.set_can_focus.assert_called_once_with(False)
        window._window.set_focus_on_map.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# _OverlayWindow — state mutation
# ---------------------------------------------------------------------------


class TestOverlayWindowState:
    def test_set_recording_true_sets_visible_state(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center")
        window.set_recording(True)
        assert window._visible_state is True

    def test_set_recording_false_clears_speech_and_levels(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center")
        window.set_recording(True)
        window.set_speech_active(True)
        window.push_audio_level(0.9)

        window.set_recording(False)

        assert window._visible_state is False
        assert window._speech_active is False
        assert all(level == 0.0 for level in window._audio_levels)

    def test_push_audio_level_clamps_to_unit_range(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center")
        window.push_audio_level(5.0)
        window.push_audio_level(-5.0)
        assert window._audio_levels[-2] == 1.0
        assert window._audio_levels[-1] == 0.0


# ---------------------------------------------------------------------------
# Overlay facade — lifecycle and dispatch to the window
# ---------------------------------------------------------------------------


class TestOverlayLifecycle:
    def test_start_creates_window_and_thread(self, mock_gtk):
        overlay = Overlay(OverlayConfig(position="bottom-center"))
        overlay.start()
        try:
            assert overlay._window is not None
            mock_gtk.Gtk.Window.assert_called_once()
        finally:
            overlay.stop()

    def test_show_marks_window_recording(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            overlay.show()
            assert overlay._window._visible_state is True
        finally:
            overlay.stop()

    def test_hide_clears_window_recording(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            overlay.show()
            overlay.hide()
            assert overlay._window._visible_state is False
        finally:
            overlay.stop()

    def test_set_speech_active_dispatches_to_window(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            overlay.set_speech_active(True)
            assert overlay._window._speech_active is True
        finally:
            overlay.stop()

    def test_push_audio_level_dispatches_to_window(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            overlay.push_audio_level(0.42)
            assert overlay._window._audio_levels[-1] == pytest.approx(0.42)
        finally:
            overlay.stop()

    def test_show_hide_noop_before_start(self, mock_gtk):
        overlay = Overlay()
        # Should not raise even though no window/thread exists yet.
        overlay.show()
        overlay.hide()
        overlay.set_speech_active(True)
        overlay.push_audio_level(0.5)

    def test_stop_noop_before_start(self, mock_gtk):
        overlay = Overlay()
        # Should not raise.
        overlay.stop()
