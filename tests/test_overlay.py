"""Tests for linux_whisper.overlay — GTK3 pill overlay, positioning, facade.

GTK is always mocked here (see the `mock_gtk` fixture in conftest.py) — no
test in this module may require a real GTK installation or a display,
regardless of what happens to be available on the machine running the tests.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

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

    def test_narrow_monitor_clamps_x_instead_of_going_negative(self):
        # A 150px-wide monitor with the default 200px pill: unclamped math
        # would put x at monitor_x - 25 — off-screen to the left. It must
        # clamp to the monitor's own left edge instead.
        x, y = compute_pill_position(500, 0, 150, 1080, "center")
        assert x == 500  # monitor's left edge, not 475
        assert y == 520  # (1080 - 40) // 2 — unaffected by the x clamp

    def test_short_monitor_clamps_y_to_top_edge(self):
        # A 30px-tall monitor is shorter than the 40px pill itself — there is
        # no valid on-screen range, so both anchors must pin to the
        # monitor's top edge rather than compute a nonsensical position.
        x, y = compute_pill_position(0, 100, 1920, 30, "top-center", margin=48)
        assert y == 100  # clamped to the monitor's top edge, not 148

        x, y = compute_pill_position(0, 100, 1920, 30, "bottom-center", margin=48)
        assert y == 100  # clamped to the monitor's top edge, not 42

    def test_oversized_margin_clamps_to_bottom_edge_not_off_screen(self):
        # The monitor is tall enough to fit the pill, but the configured
        # margin alone would push top-center's y past the bottom edge.
        x, y = compute_pill_position(0, 0, 1920, 200, "top-center", margin=300)
        assert y == 160  # 200 - 40, the monitor's bottom edge, not 300


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

        _OverlayWindow("center", MagicMock())
        mock_gtk.Gtk.Window.assert_called_once_with(type=mock_gtk.Gtk.WindowType.POPUP)

    def test_window_never_accepts_focus(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center", MagicMock())
        window._window.set_accept_focus.assert_called_once_with(False)
        window._window.set_can_focus.assert_called_once_with(False)
        window._window.set_focus_on_map.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# _OverlayWindow — state mutation
# ---------------------------------------------------------------------------


class TestOverlayWindowState:
    def test_set_recording_true_sets_visible_state(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center", MagicMock())
        window.set_recording(True)
        assert window._visible_state is True

    def test_set_recording_false_clears_speech_and_levels(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center", MagicMock())
        window.set_recording(True)
        window.set_speech_active(True)
        window.push_audio_level(0.9)

        window.set_recording(False)

        assert window._visible_state is False
        assert window._speech_active is False
        assert all(level == 0.0 for level in window._audio_levels)

    def test_push_audio_level_clamps_to_unit_range(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center", MagicMock())
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
            assert overlay._ready.is_set()
            assert overlay._window is not None
            mock_gtk.Gtk.Window.assert_called_once()
        finally:
            overlay.stop()

    def test_show_marks_window_recording(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            calls_before = mock_gtk.GLib.idle_source_new.call_count
            overlay.show()
            # Not just "the state ended up right" — prove it got there via a
            # real dispatch onto the overlay's own context, not a direct
            # cross-thread call that would happen to pass this assertion too.
            assert mock_gtk.GLib.idle_source_new.call_count > calls_before
            assert overlay._window._visible_state is True
        finally:
            overlay.stop()

    def test_hide_clears_window_recording(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            overlay.show()
            calls_before = mock_gtk.GLib.idle_source_new.call_count
            overlay.hide()
            assert mock_gtk.GLib.idle_source_new.call_count > calls_before
            assert overlay._window._visible_state is False
        finally:
            overlay.stop()

    def test_set_speech_active_dispatches_to_window(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            calls_before = mock_gtk.GLib.idle_source_new.call_count
            overlay.set_speech_active(True)
            assert mock_gtk.GLib.idle_source_new.call_count > calls_before
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


# ---------------------------------------------------------------------------
# BLOCKER — the overlay must never silently run on the wrong GDK backend.
# Wayland refuses to position an override-redirect window at all, so a
# backend mismatch must disable the overlay loudly rather than show a
# mispositioned pill.
# ---------------------------------------------------------------------------


class TestBackendVerification:
    def test_x11_backend_starts_normally(self, mock_gtk, caplog):
        """Sanity check: the happy path (what mock_gtk sets up by default)
        must not trip the new backend guard."""
        overlay = Overlay()
        with caplog.at_level(logging.INFO):
            overlay.start()
        try:
            assert overlay._window is not None
            assert any("Overlay started" in r.message for r in caplog.records)
        finally:
            overlay.stop()

    def test_non_x11_backend_disables_overlay_and_names_it(self, mock_gtk, caplog):
        """Reproduces the exact failure mode from the review: GDK_BACKEND=x11
        set too late (after the tray already opened the default Wayland
        display) leaves GDK on GdkWaylandDisplay. That must be caught and
        the overlay disabled, not silently shown mispositioned."""
        wayland_display_cls = type("WaylandDisplay", (MagicMock,), {})
        mock_gtk.Gdk.Display.get_default.return_value = wayland_display_cls()

        overlay = Overlay()
        with caplog.at_level(logging.WARNING):
            overlay.start()
        try:
            assert overlay._window is None
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert warnings, "expected a WARNING when the backend isn't X11"
            assert any("WaylandDisplay" in r.message for r in warnings), (
                "the WARNING must name the actual backend, not just say 'failed'"
            )
        finally:
            overlay.stop()

    def test_non_x11_backend_still_destroys_the_constructed_window(self, mock_gtk):
        """The window is fully constructed before the backend is checked
        (the check needs a live display) — it must be torn down again, not
        leaked, when the check fails."""
        wayland_display_cls = type("WaylandDisplay", (MagicMock,), {})
        mock_gtk.Gdk.Display.get_default.return_value = wayland_display_cls()

        overlay = Overlay()
        overlay.start()
        try:
            mock_gtk.Gtk.Window.return_value.destroy.assert_called_once()
        finally:
            overlay.stop()


# ---------------------------------------------------------------------------
# MAJOR — a failed GTK init must not hang startup and then lie about it.
# ---------------------------------------------------------------------------


class TestStartupFailureIsReportedNotHidden:
    def test_construction_failure_sets_ready_with_no_window(self, mock_gtk, caplog):
        """Window construction must run inside the crash handler, and
        `_ready` must always be set in a `finally` — otherwise `start()`
        blocks for the full 5s timeout on a thread that already died."""
        mock_gtk.Gtk.Window.side_effect = RuntimeError("Gtk couldn't be initialized")
        overlay = Overlay()

        # Call the thread body directly and synchronously — this isolates
        # "does _ready get set on the failure path" from thread-timing.
        with caplog.at_level(logging.WARNING):
            overlay._run_gtk()

        assert overlay._ready.is_set()
        assert overlay._window is None
        assert any("Overlay disabled" in r.message for r in caplog.records)

    def test_start_fails_fast_instead_of_stalling_the_full_timeout(self, mock_gtk, caplog):
        """The original bug: with no display, construction raised, `_ready`
        was never set, and `start()` blocked for the full 5s timeout before
        logging "Overlay started" anyway. This must return almost
        immediately and must not claim success."""
        import time

        mock_gtk.Gtk.Window.side_effect = RuntimeError("Gtk couldn't be initialized")
        overlay = Overlay()

        began = time.monotonic()
        with caplog.at_level(logging.WARNING):
            overlay.start()
        elapsed = time.monotonic() - began
        try:
            assert elapsed < 2.0, "start() stalled — _ready was not set promptly on failure"
            assert not any(r.message == "Overlay started" for r in caplog.records), (
                "start() must not claim success when the window failed to build"
            )
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert warnings
        finally:
            overlay.stop()

    def test_start_noop_thread_does_not_stay_alive_after_construction_failure(
        self, mock_gtk
    ):
        mock_gtk.Gtk.Window.side_effect = RuntimeError("boom")
        overlay = Overlay()
        overlay.start()
        try:
            overlay._thread.join(timeout=1.0)
            assert not overlay._thread.is_alive()
        finally:
            overlay.stop()


# ---------------------------------------------------------------------------
# MINOR — the 30fps animation timer must not run forever while hidden.
# ---------------------------------------------------------------------------


class TestAnimationTimerLifecycle:
    def test_showing_attaches_a_tick_timer(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            calls_before = mock_gtk.GLib.timeout_source_new.call_count
            overlay.show()
            assert mock_gtk.GLib.timeout_source_new.call_count > calls_before
            assert overlay._window._tick_source is not None
        finally:
            overlay.stop()

    def test_hiding_destroys_the_tick_timer_instead_of_leaving_it_running(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        try:
            overlay.show()
            source = overlay._window._tick_source
            overlay.hide()
            assert source.destroyed is True
            assert overlay._window._tick_source is None
        finally:
            overlay.stop()

    def test_repeated_show_does_not_leak_a_second_timer(self, mock_gtk):
        """Calling show() while already visible (e.g. a duplicate hotkey
        event) must not attach a second competing timer."""
        overlay = Overlay()
        overlay.start()
        try:
            overlay.show()
            first_source = overlay._window._tick_source
            overlay.show()
            assert overlay._window._tick_source is first_source
        finally:
            overlay.stop()


# ---------------------------------------------------------------------------
# MINOR — push_audio_level() must never block the real-time audio monitor.
# ---------------------------------------------------------------------------


class TestPushAudioLevelNonBlocking:
    def test_drops_sample_instead_of_blocking_when_lock_is_held(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center", MagicMock())
        window._lock.acquire()  # simulate a tick in progress on the GTK thread
        try:
            import time

            started = time.monotonic()
            window.push_audio_level(0.9)  # must return immediately, not block
            elapsed = time.monotonic() - started
        finally:
            window._lock.release()

        assert elapsed < 0.1
        # The sample was dropped, not queued — nothing after the pre-filled
        # history should read back as 0.9.
        assert 0.9 not in window._audio_levels

    def test_still_records_the_sample_when_the_lock_is_free(self, mock_gtk):
        from linux_whisper.overlay import _OverlayWindow

        window = _OverlayWindow("center", MagicMock())
        window.push_audio_level(0.9)
        assert window._audio_levels[-1] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# MAJOR — stop() must quit the overlay's OWN loop, never the shared default
# main loop (which the tray's GTK backend also runs).
# ---------------------------------------------------------------------------


class TestStopUsesItsOwnMainLoop:
    def test_stop_quits_the_overlays_own_loop_object(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        loop = overlay._main_loop
        assert loop is not None

        overlay.stop()

        loop.quit.assert_called_once()
        # Never the old Gtk.main_quit() posted to the shared default context.
        mock_gtk.Gtk.main_quit.assert_not_called()

    def test_stop_never_touches_gtk_main_quit(self, mock_gtk):
        overlay = Overlay()
        overlay.start()
        overlay.stop()
        mock_gtk.Gtk.main_quit.assert_not_called()

    def test_stop_logs_a_warning_when_the_thread_does_not_join(self, mock_gtk, caplog):
        """If the overlay thread hangs, stop() must say so instead of
        silently discarding the (still-alive) thread handle."""
        overlay = Overlay()
        overlay.start()

        # Simulate a hung thread: join() returns without the thread having
        # finished (a real timeout would do the same — is_alive() stays True).
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(overlay._thread, "join", lambda timeout=None: None)
            mp.setattr(overlay._thread, "is_alive", lambda: True)
            with caplog.at_level(logging.WARNING):
                overlay.stop()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("did not exit" in r.message for r in warnings)
        # The handle must be kept, not discarded, so a stuck thread stays
        # observable instead of vanishing untracked.
        assert overlay._thread is not None
