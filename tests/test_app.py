"""Tests for linux_whisper.app — App orchestration, pipeline, state transitions."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from linux_whisper.config import (
    AudioConfig,
    Config,
    InjectConfig,
    OverlayConfig,
    PolishConfig,
    STTConfig,
    TrayConfig,
)
from linux_whisper.state import AppState

if TYPE_CHECKING:
    from linux_whisper.app import App

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> Config:
    """Create a valid Config with optional overrides."""
    defaults = dict(
        hotkey="fn",
        mode="hold",
        stt=STTConfig(backend="whisper-cpp", model="whisper-large-v3-turbo"),
        polish=PolishConfig(enabled=True),
        audio=AudioConfig(auto_gain=True),
        inject=InjectConfig(method="auto"),
        tray=TrayConfig(enabled=False),
        overlay=OverlayConfig(enabled=False),
        snippets={},
    )
    defaults.update(overrides)
    return Config(**defaults)


@dataclass(frozen=True, slots=True)
class FakeAudioChunk:
    """Lightweight stand-in for AudioChunk to avoid importing audio module."""

    samples: np.ndarray
    timestamp: float = 0.0
    is_speech: bool = True
    is_final: bool = False


@dataclass(slots=True)
class FakeTranscriptResult:
    """Stand-in for TranscriptResult."""

    full_text: str = ""
    segments: list = None  # type: ignore[assignment]
    language: str | None = None
    duration: float = 0.0

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


def _make_app(config: Config | None = None) -> App:
    """Create an App with all heavy imports mocked."""
    from linux_whisper.app import App

    return App(config or _make_config())


# ---------------------------------------------------------------------------
# 1. App.__init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_components_none_before_setup(self):
        app = _make_app()
        assert app._audio is None
        assert app._hotkey is None
        assert app._stt is None
        assert app._polish is None
        assert app._snippets is None
        assert app._injector is None
        assert app._tray is None
        assert app._overlay is None

    def test_state_starts_idle(self):
        app = _make_app()
        assert app.state.is_idle

    def test_shutdown_event_not_set(self):
        app = _make_app()
        assert not app._shutdown_event.is_set()

    def test_latencies_empty(self):
        app = _make_app()
        assert app._latencies == []


# ---------------------------------------------------------------------------
# 2. App.setup()
# ---------------------------------------------------------------------------


class TestSetup:
    async def test_setup_calls_all_setup_methods(self):
        app = _make_app()
        methods = [
            "_setup_audio",
            "_setup_stt",
            "_setup_polish",
            "_setup_snippets",
            "_setup_injector",
            "_setup_hotkey",
            "_setup_tray",
            "_setup_overlay",
        ]
        for m in methods:
            setattr(app, m, AsyncMock())

        await app.setup()

        for m in methods:
            getattr(app, m).assert_awaited_once()

    async def test_setup_raises_on_invalid_config(self):
        bad_config = _make_config(mode="invalid_mode")
        app = _make_app(bad_config)

        with pytest.raises(ValueError, match="Invalid configuration"):
            await app.setup()

    async def test_setup_wires_tray_state_listener(self):
        app = _make_app(_make_config(tray=TrayConfig(enabled=True)))
        for m in [
            "_setup_audio",
            "_setup_stt",
            "_setup_polish",
            "_setup_snippets",
            "_setup_injector",
            "_setup_hotkey",
            "_setup_overlay",
        ]:
            setattr(app, m, AsyncMock())

        # Simulate tray being created
        app._tray = MagicMock()
        app._setup_tray = AsyncMock()

        await app.setup()

        # State machine should have a listener now
        assert len(app.state._listeners) == 1

    async def test_setup_forces_x11_backend_before_tray_starts_when_overlay_enabled(
        self, monkeypatch
    ):
        """BLOCKER regression guard: the tray starts its GTK3 thread before
        the overlay does (setup() calls _setup_tray() before
        _setup_overlay(); run() starts tray before overlay), and PyGObject
        locks in the GDK backend as a side effect of importing `Gdk`/`Gtk` —
        verified against a real display, this happens even before any
        window is built. Setting GDK_BACKEND=x11 must happen as a bare
        os.environ write before _setup_tray() runs (and before importing
        `linux_whisper.overlay` or `linux_whisper.tray` at all) — calling a
        helper that lives inside overlay.py would already be too late,
        because importing overlay.py to reach it would run its top-level
        `from gi.repository import Gdk` first. See app.py's setup()."""
        monkeypatch.delenv("GDK_BACKEND", raising=False)
        app = _make_app(_make_config(overlay=OverlayConfig(enabled=True)))
        order: list[tuple[str, str | None]] = []

        async def _record_tray():
            order.append(("_setup_tray", os.environ.get("GDK_BACKEND")))

        for m in [
            "_setup_audio",
            "_setup_stt",
            "_setup_polish",
            "_setup_snippets",
            "_setup_injector",
            "_setup_hotkey",
            "_setup_overlay",
        ]:
            setattr(app, m, AsyncMock())
        app._setup_tray = _record_tray

        await app.setup()

        # By the time _setup_tray() ran, the env var was already "x11".
        assert order == [("_setup_tray", "x11")]

    async def test_setup_does_not_force_x11_backend_when_overlay_disabled(self, monkeypatch):
        monkeypatch.delenv("GDK_BACKEND", raising=False)
        app = _make_app(_make_config(overlay=OverlayConfig(enabled=False)))
        for m in [
            "_setup_audio",
            "_setup_stt",
            "_setup_polish",
            "_setup_snippets",
            "_setup_injector",
            "_setup_hotkey",
            "_setup_tray",
            "_setup_overlay",
        ]:
            setattr(app, m, AsyncMock())

        await app.setup()

        assert "GDK_BACKEND" not in os.environ


# ---------------------------------------------------------------------------
# 2b. App._setup_overlay()
# ---------------------------------------------------------------------------


class TestSetupOverlay:
    """The overlay used to fail silently (self._overlay = None, no log) when
    unavailable — that silence is what let it rot undetected for five
    months. These tests guard the three states an operator needs visible:
    disabled by config, enabled-but-unavailable (must WARN with a reason),
    and enabled-and-available."""

    async def test_disabled_by_config_sets_no_overlay(self, caplog):
        app = _make_app(_make_config(overlay=OverlayConfig(enabled=False)))
        with caplog.at_level(logging.INFO):
            await app._setup_overlay()

        assert app._overlay is None
        assert any("disabled" in r.message.lower() for r in caplog.records)

    async def test_enabled_but_unavailable_logs_warning_with_reason(self, monkeypatch, caplog):
        import linux_whisper.overlay as overlay_module

        monkeypatch.setattr(overlay_module, "_HAS_GTK", False)
        monkeypatch.setattr(
            overlay_module, "_UNAVAILABLE_REASON", "GTK 3.0 unavailable: no gi (test)"
        )

        app = _make_app(_make_config(overlay=OverlayConfig(enabled=True)))
        with caplog.at_level(logging.WARNING):
            await app._setup_overlay()

        assert app._overlay is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING when the overlay is unavailable"
        assert any("GTK 3.0 unavailable: no gi (test)" in r.message for r in warnings)

    async def test_enabled_and_available_sets_overlay(self, mock_gtk, caplog):
        app = _make_app(_make_config(overlay=OverlayConfig(enabled=True, position="top-center")))
        with caplog.at_level(logging.INFO):
            await app._setup_overlay()

        assert app._overlay is not None
        assert app._overlay.available is True
        assert any("Overlay ready" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 3. _process_pipeline()
# ---------------------------------------------------------------------------


class TestProcessPipeline:
    async def test_returns_none_when_no_audio_or_stt(self):
        app = _make_app()
        assert app._audio is None
        result = await app._process_pipeline()
        assert result is None

    async def test_returns_none_on_empty_audio_segments(self):
        app = _make_app()
        app._audio = MagicMock()
        app._stt = MagicMock()

        # stop_recording does nothing special
        app._audio.stop_recording = MagicMock()

        # audio_chunks yields only a final marker with empty samples
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True, is_speech=False
        )

        async def _fake_chunks():
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks

        result = await app._process_pipeline()
        assert result is None

    async def test_returns_none_on_empty_stt_result(self):
        app = _make_app(_make_config(audio=AudioConfig(auto_gain=False)))
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._audio.stop_recording = MagicMock()

        speech = np.random.randn(16000).astype(np.float32) * 0.5
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = FakeTranscriptResult(full_text="")

        result = await app._process_pipeline()
        assert result is None
        app._stt.reset.assert_called_once()

    async def test_returns_none_when_stt_returns_none(self):
        app = _make_app(_make_config(audio=AudioConfig(auto_gain=False)))
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._audio.stop_recording = MagicMock()

        speech = np.ones(1600, dtype=np.float32) * 0.1
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = None

        result = await app._process_pipeline()
        assert result is None

    async def test_agc_applied_when_auto_gain_true(self):
        config = _make_config(audio=AudioConfig(auto_gain=True), polish=PolishConfig(enabled=False))
        app = _make_app(config)
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._audio.stop_recording = MagicMock()

        speech = np.random.randn(16000).astype(np.float32) * 0.1
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = FakeTranscriptResult(full_text="hello world")

        mock_agc = MagicMock(return_value=speech)
        # Patch the module that gets imported inside _process_pipeline
        import linux_whisper.audio as audio_mod
        with patch.object(audio_mod, "apply_agc", mock_agc):
            result = await app._process_pipeline()

        mock_agc.assert_called_once()
        assert result == "hello world"

    async def test_agc_not_applied_when_auto_gain_false(self):
        config = _make_config(
            audio=AudioConfig(auto_gain=False), polish=PolishConfig(enabled=False)
        )
        app = _make_app(config)
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._audio.stop_recording = MagicMock()

        speech = np.random.randn(16000).astype(np.float32) * 0.3
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = FakeTranscriptResult(full_text="test text")

        result = await app._process_pipeline()
        assert result == "test text"

    async def test_snippet_match_bypasses_polish(self):
        config = _make_config(
            snippets={"my email": "user@example.com"},
            polish=PolishConfig(enabled=True),
        )
        app = _make_app(config)
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._audio.stop_recording = MagicMock()

        speech = np.ones(16000, dtype=np.float32) * 0.3
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = FakeTranscriptResult(full_text="my email")

        # Set up snippet matcher mock
        app._snippets = MagicMock()
        app._snippets.match.return_value = "user@example.com"

        # Polish should NOT be called
        app._polish = MagicMock()

        with patch("linux_whisper.audio.apply_agc", side_effect=lambda x: x):
            result = await app._process_pipeline()

        assert result == "user@example.com"
        app._polish.process.assert_not_called()

    async def test_polish_called_when_no_snippet_match(self):
        config = _make_config(
            polish=PolishConfig(enabled=True, context_awareness=False),
            audio=AudioConfig(auto_gain=False),
        )
        app = _make_app(config)
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._polish = MagicMock()
        app._audio.stop_recording = MagicMock()

        speech = np.ones(16000, dtype=np.float32) * 0.3
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = FakeTranscriptResult(full_text="hello there")

        # No snippet matcher
        app._snippets = None

        # Polish returns polished text via to_thread
        app._polish.process.return_value = "Hello there."

        result = await app._process_pipeline()
        assert result == "Hello there."
        app._polish.process.assert_called_once_with("hello there", None)

    async def test_context_awareness_calls_detect_focused_app(self):
        config = _make_config(
            polish=PolishConfig(enabled=True, context_awareness=True),
            audio=AudioConfig(auto_gain=False),
        )
        app = _make_app(config)
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._polish = MagicMock()
        app._snippets = None
        app._audio.stop_recording = MagicMock()

        speech = np.ones(16000, dtype=np.float32) * 0.3
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = FakeTranscriptResult(full_text="send message")

        mock_focused = MagicMock()
        mock_focused.app_name = "Slack"
        mock_focused.category.value = "messaging"

        app._polish.process.return_value = "Send message."

        with patch(
            "linux_whisper.app.detect_focused_app", return_value=mock_focused, create=True
        ) as mock_detect, patch(
            "linux_whisper.app.build_context_string",
            return_value="The user is typing in Slack (messaging).",
            create=True,
        ) as mock_ctx:
            # Patch at the import site inside _process_pipeline
            focus_mod = MagicMock()
            focus_mod.detect_focused_app = mock_detect
            focus_mod.build_context_string = mock_ctx
            with patch.dict("sys.modules", {"linux_whisper.focus": focus_mod}):
                result = await app._process_pipeline()

        assert result == "Send message."
        mock_detect.assert_called_once()
        mock_ctx.assert_called_once_with(mock_focused)
        app._polish.process.assert_called_once_with(
            "send message", "The user is typing in Slack (messaging)."
        )

    async def test_context_awareness_skipped_when_disabled(self):
        config = _make_config(
            polish=PolishConfig(enabled=True, context_awareness=False),
            audio=AudioConfig(auto_gain=False),
        )
        app = _make_app(config)
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._polish = MagicMock()
        app._snippets = None
        app._audio.stop_recording = MagicMock()

        speech = np.ones(16000, dtype=np.float32) * 0.3
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = FakeTranscriptResult(full_text="test")
        app._polish.process.return_value = "Test."

        result = await app._process_pipeline()
        assert result == "Test."
        # Polish called with None context
        app._polish.process.assert_called_once_with("test", None)

    async def test_pipeline_no_polish_when_disabled(self):
        config = _make_config(
            polish=PolishConfig(enabled=False),
            audio=AudioConfig(auto_gain=False),
        )
        app = _make_app(config)
        app._audio = MagicMock()
        app._stt = MagicMock()
        app._polish = None  # disabled
        app._audio.stop_recording = MagicMock()

        speech = np.ones(16000, dtype=np.float32) * 0.3
        data_chunk = FakeAudioChunk(samples=speech, is_final=False)
        final_chunk = FakeAudioChunk(
            samples=np.empty(0, dtype=np.float32), is_final=True
        )

        async def _fake_chunks():
            yield data_chunk
            yield final_chunk

        app._audio.audio_chunks = _fake_chunks
        app._stt.finalize.return_value = FakeTranscriptResult(full_text="raw text")

        result = await app._process_pipeline()
        assert result == "raw text"


# ---------------------------------------------------------------------------
# 4. _handle_recording_stop()
# ---------------------------------------------------------------------------


class TestHandleRecordingStop:
    async def test_noop_when_not_recording(self):
        app = _make_app()
        # state is IDLE, so should return early
        await app._handle_recording_stop()
        assert app.state.is_idle

    async def test_transitions_and_processes_pipeline(self):
        app = _make_app()
        app._injector = AsyncMock()
        app._injector.inject = AsyncMock(return_value=True)
        app._overlay = MagicMock()

        # Start in RECORDING state
        await app.state.transition(AppState.RECORDING)

        with patch.object(app, "_process_pipeline", new_callable=AsyncMock) as mock_pipe:
            mock_pipe.return_value = "Hello world"
            await app._handle_recording_stop()

        mock_pipe.assert_awaited_once()
        app._injector.inject.assert_awaited_once_with("Hello world")
        assert app.state.is_idle

    async def test_empty_result_no_injection(self):
        app = _make_app()
        app._injector = AsyncMock()
        app._overlay = MagicMock()

        await app.state.transition(AppState.RECORDING)

        with patch.object(app, "_process_pipeline", new_callable=AsyncMock) as mock_pipe:
            mock_pipe.return_value = None
            await app._handle_recording_stop()

        app._injector.inject.assert_not_awaited()
        assert app.state.is_idle

    async def test_pipeline_error_transitions_to_error_then_idle(self):
        app = _make_app()
        app._overlay = MagicMock()

        await app.state.transition(AppState.RECORDING)

        with patch.object(app, "_process_pipeline", new_callable=AsyncMock) as mock_pipe:
            mock_pipe.side_effect = RuntimeError("STT crashed")
            await app._handle_recording_stop()

        # Should end up in IDLE after error recovery
        assert app.state.is_idle

    async def test_tray_gets_last_transcription(self):
        app = _make_app()
        app._tray = MagicMock()
        app._injector = AsyncMock()
        app._injector.inject = AsyncMock(return_value=True)
        app._overlay = MagicMock()

        await app.state.transition(AppState.RECORDING)

        with patch.object(app, "_process_pipeline", new_callable=AsyncMock) as mock_pipe:
            mock_pipe.return_value = "some text"
            await app._handle_recording_stop()

        app._tray.set_last_transcription.assert_called_once_with("some text")

    async def test_overlay_hidden_after_processing(self):
        app = _make_app()
        app._overlay = MagicMock()
        app._injector = AsyncMock()
        app._injector.inject = AsyncMock(return_value=True)

        await app.state.transition(AppState.RECORDING)

        with patch.object(app, "_process_pipeline", new_callable=AsyncMock) as mock_pipe:
            mock_pipe.return_value = "text"
            await app._handle_recording_stop()

        app._overlay.hide.assert_called_once()


# ---------------------------------------------------------------------------
# 4b. _feed_audio_levels() — the overlay's bars were driven by math.sin()
# because push_audio_level() was never wired to anything real. This is the
# regression guard: the audio monitor loop must push real levels.
# ---------------------------------------------------------------------------


class TestStartOrder:
    """GTK initialisation order. Both the overlay and the tray (via pystray's
    appindicator backend) build GTK3 objects on their own threads, and GTK is
    not thread-safe -- constructing them concurrently segfaults the process.
    Measured with tray-first: 8 SIGSEGVs across 4 consecutive restarts, unit
    crash-looping. Overlay-first: 0 across 6. Overlay.start() blocks until its
    window exists, which serialises the two.
    """

    async def test_overlay_starts_before_tray(self):
        app = _make_app()
        order: list[str] = []
        app._overlay = MagicMock()
        app._overlay.start.side_effect = lambda *a, **k: order.append("overlay")
        app._tray = MagicMock()
        app._tray.start.side_effect = lambda *a, **k: order.append("tray")
        app._hotkey = None
        app._audio = None
        app._shutdown_event.set()  # return immediately after startup

        await app.run()

        assert order == ["overlay", "tray"], order


class TestFeedAudioLevels:
    async def test_pushes_audio_level_to_overlay_while_recording(self):
        app = _make_app()
        app._overlay = MagicMock()
        app._audio = MagicMock()
        app._audio.get_pre_roll.return_value = np.array([0.5, -0.5, 0.3], dtype=np.float32)

        await app.state.transition(AppState.RECORDING)

        async def _stop_after_one_iteration(*_args, **_kwargs):
            # End the loop after a single pass by leaving the RECORDING state.
            await app.state.transition(AppState.IDLE)

        with patch("asyncio.sleep", new=AsyncMock(side_effect=_stop_after_one_iteration)):
            await app._feed_audio_levels()

        # A loud sample well above the ambient floor must arrive as a near-full
        # bar. The overlay is fed a level normalised in dB above the noise
        # floor, not raw amplitude: ordinary speech only reaches ~0.02-0.06 rms
        # on a real mic, so raw amplitude left the bars flat.
        app._overlay.push_audio_level.assert_called_once()
        (level,) = app._overlay.push_audio_level.call_args.args
        assert 0.0 <= level <= 1.0
        assert level > 0.9, f"loud audio should nearly fill the bar, got {level}"

    async def test_recording_start_wires_up_the_feed_loop_end_to_end(self):
        """Regression guard for the actual bug (push_audio_level() wired to
        nothing): goes through the real recording-start path — the one the
        hotkey daemon triggers — rather than calling _feed_audio_levels()
        directly, so a break in the `_handle_recording_start` wiring itself
        (e.g. someone removing the `ensure_future` call) would fail this."""
        app = _make_app()
        app._overlay = MagicMock()
        app._audio = MagicMock()
        app._audio.get_pre_roll.return_value = np.array([0.2, -0.9, 0.4], dtype=np.float32)

        async def _stop_after_one_iteration(*_args, **_kwargs):
            await app.state.transition(AppState.IDLE)

        with patch("asyncio.sleep", new=AsyncMock(side_effect=_stop_after_one_iteration)):
            await app._handle_recording_start()
            # _handle_recording_start schedules _feed_audio_levels via
            # ensure_future rather than awaiting it directly — let it run.
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            await asyncio.gather(*pending)

        # show() is NOT called here any more — it moved to the synchronous
        # _on_recording_start so visible feedback does not wait on the event
        # loop (see test_show_happens_on_the_hotkey_thread_before_stt).
        app._overlay.show.assert_not_called()
        app._overlay.push_audio_level.assert_called_once()
        (level,) = app._overlay.push_audio_level.call_args.args
        assert 0.0 <= level <= 1.0 and level > 0.9

    async def test_speech_stays_detected_through_a_long_utterance(self):
        """The noise floor must not learn your voice.

        It used to adapt on every sample regardless of whether speech was
        present, so during continuous talking it climbed toward the speech
        level and the `rms > floor * 3` test went false and stayed false.
        Simulated at a steady rms of 0.05 the old detector flipped to "no
        speech" at t=6.9s and never came back. This drives ~20s of continuous
        speech and asserts it never gets switched off.
        """
        app = _make_app()
        app._overlay = MagicMock()
        app._audio = MagicMock()
        # Steady tone at a normal speaking level, well above ambient.
        app._audio.get_pre_roll.return_value = np.full(1600, 0.05, dtype=np.float32)

        await app.state.transition(AppState.RECORDING)

        iterations = 600  # 20s at 30Hz

        async def _advance(*_args, **_kwargs):
            nonlocal iterations
            iterations -= 1
            if iterations <= 0:
                await app.state.transition(AppState.IDLE)

        with patch("asyncio.sleep", new=AsyncMock(side_effect=_advance)):
            await app._feed_audio_levels()

        # set_speech_active is only called on a change. Speech should latch on
        # once and never be turned back off while the level is held steady.
        calls = [c.args[0] for c in app._overlay.set_speech_active.call_args_list]
        assert calls == [True], f"speech toggled during a steady utterance: {calls}"

    async def test_silence_is_not_reported_as_speech(self):
        app = _make_app()
        app._overlay = MagicMock()
        app._audio = MagicMock()
        app._audio.get_pre_roll.return_value = np.full(1600, 0.0015, dtype=np.float32)

        await app.state.transition(AppState.RECORDING)
        remaining = 100

        async def _advance(*_args, **_kwargs):
            nonlocal remaining
            remaining -= 1
            if remaining <= 0:
                await app.state.transition(AppState.IDLE)

        with patch("asyncio.sleep", new=AsyncMock(side_effect=_advance)):
            await app._feed_audio_levels()

        calls = [c.args[0] for c in app._overlay.set_speech_active.call_args_list]
        assert True not in calls, f"ambient silence reported as speech: {calls}"

    async def test_show_happens_on_the_hotkey_thread_before_stt(self):
        """The pill must be shown from the synchronous hotkey path, BEFORE
        _stt.start_stream().

        Two regressions this guards. (1) show() used to live in the async
        _handle_recording_start, so feedback waited on call_soon_threadsafe ->
        ensure_future -> a state transition. (2) start_stream() calls
        _ensure_worker(), which blocks for ~4.3s when it has to spawn the GPU
        worker — showing the pill after it means no feedback at all for the
        first dictation after startup.

        Audio capture must still be started before both, so the recording
        itself never waits on the UI.
        """
        app = _make_app()
        order: list[str] = []

        app._overlay = MagicMock()
        app._overlay.show.side_effect = lambda *a, **k: order.append("overlay.show")
        app._audio = MagicMock()
        app._audio.get_pre_roll.return_value = np.array([0.1], dtype=np.float32)
        app._audio.start_recording.side_effect = lambda *a, **k: order.append("audio.start")
        app._stt = MagicMock()
        app._stt.start_stream.side_effect = lambda *a, **k: order.append("stt.start_stream")
        app._loop = None  # returns before the async hand-off; irrelevant here

        app._on_recording_start()

        assert order == ["audio.start", "overlay.show", "stt.start_stream"], order

    async def test_does_not_push_when_no_overlay(self):
        app = _make_app()
        app._overlay = None
        app._audio = MagicMock()
        app._audio.get_pre_roll.return_value = np.array([0.5, -0.5, 0.3], dtype=np.float32)

        await app.state.transition(AppState.RECORDING)

        async def _stop_after_one_iteration(*_args, **_kwargs):
            await app.state.transition(AppState.IDLE)

        with patch("asyncio.sleep", new=AsyncMock(side_effect=_stop_after_one_iteration)):
            # Should not raise even though there's no overlay to push to.
            await app._feed_audio_levels()

    async def test_noop_without_audio(self):
        app = _make_app()
        app._audio = None
        app._overlay = MagicMock()
        # Should return immediately without touching the overlay.
        await app._feed_audio_levels()
        app._overlay.push_audio_level.assert_not_called()


# ---------------------------------------------------------------------------
# 5. _record_latency()
# ---------------------------------------------------------------------------


class TestRecordLatency:
    def test_tracks_latency(self):
        app = _make_app()
        app._record_latency(0.5)
        assert app._latencies == [0.5]

    def test_updates_tray_stats(self):
        app = _make_app()
        app._tray = MagicMock()
        app._record_latency(0.4)
        app._tray.update_stats.assert_called_once_with(last_latency=0.4, avg_latency=0.4)

    def test_respects_max_history(self):
        app = _make_app()
        app._max_latency_history = 5
        for i in range(10):
            app._record_latency(float(i))
        assert len(app._latencies) == 5
        assert app._latencies == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_no_tray_no_error(self):
        app = _make_app()
        app._tray = None
        # Should not raise
        app._record_latency(0.3)
        assert app._latencies == [0.3]

    def test_avg_latency_computed_correctly(self):
        app = _make_app()
        app._tray = MagicMock()
        app._record_latency(0.2)
        app._record_latency(0.4)
        # Second call: avg should be 0.3
        call_args = app._tray.update_stats.call_args_list[-1]
        assert call_args.kwargs["avg_latency"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 6. latency_stats property
# ---------------------------------------------------------------------------


class TestLatencyStats:
    def test_empty_stats(self):
        app = _make_app()
        stats = app.latency_stats
        assert stats == {"last": 0, "avg": 0, "p95": 0}

    def test_single_entry(self):
        app = _make_app()
        app._latencies = [0.5]
        stats = app.latency_stats
        assert stats["last"] == 0.5
        assert stats["avg"] == 0.5
        assert stats["p95"] == 0.5

    def test_multiple_entries(self):
        app = _make_app()
        app._latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        stats = app.latency_stats
        assert stats["last"] == 1.0
        assert stats["avg"] == pytest.approx(0.55)
        # p95 index = int(10 * 0.95) = 9, sorted[9] = 1.0
        assert stats["p95"] == 1.0

    def test_p95_with_20_entries(self):
        app = _make_app()
        app._latencies = [float(i) for i in range(1, 21)]
        stats = app.latency_stats
        # p95 index = int(20 * 0.95) = 19, sorted[19] = 20.0
        assert stats["p95"] == 20.0
        assert stats["last"] == 20.0
        assert stats["avg"] == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# 7. _trim_silence()
# ---------------------------------------------------------------------------


class TestTrimSilence:
    def test_short_audio_returned_unchanged(self):
        from linux_whisper.app import App

        # Less than one frame (30ms * 16000 = 480 samples)
        short = np.zeros(100, dtype=np.float32)
        result = App._trim_silence(short)
        np.testing.assert_array_equal(result, short)

    def test_silent_audio_returned_unchanged(self):
        from linux_whisper.app import App

        # All silence — no speech frames detected, returns original
        silence = np.zeros(16000, dtype=np.float32)
        result = App._trim_silence(silence)
        np.testing.assert_array_equal(result, silence)

    def test_speech_audio_trimmed(self):
        from linux_whisper.app import App

        sample_rate = 16000
        # 1 second of silence + 0.5 second of loud sine + 1 second of silence
        silence_before = np.zeros(sample_rate, dtype=np.float32)
        t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False, dtype=np.float32)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        silence_after = np.zeros(sample_rate, dtype=np.float32)

        audio = np.concatenate([silence_before, speech, silence_after])
        result = App._trim_silence(audio)

        # Result should be shorter than input (trimmed silence)
        assert len(result) < len(audio)
        # But should contain the speech portion
        assert len(result) > 0

    def test_all_speech_mostly_preserved(self):
        from linux_whisper.app import App

        # Audio that is entirely speech — should be mostly preserved
        t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = App._trim_silence(speech)
        # Should preserve most of the audio (within padding tolerance)
        assert len(result) >= len(speech) * 0.8


# ---------------------------------------------------------------------------
# 8. _request_shutdown()
# ---------------------------------------------------------------------------


class TestRequestShutdown:
    def test_sets_shutdown_event(self):
        app = _make_app()
        assert not app._shutdown_event.is_set()
        app._request_shutdown()
        assert app._shutdown_event.is_set()

    def test_idempotent(self):
        app = _make_app()
        app._request_shutdown()
        app._request_shutdown()
        assert app._shutdown_event.is_set()


# ---------------------------------------------------------------------------
# 9. Config reconstruction in _handle_mode_change / _handle_model_change
# ---------------------------------------------------------------------------


class TestConfigReconstruction:
    async def test_mode_change_preserves_snippets(self):
        snippets = {"hello": "world", "email": "user@example.com"}
        config = _make_config(mode="hold", snippets=snippets)
        app = _make_app(config)
        app._hotkey = MagicMock()
        app._loop = asyncio.get_running_loop()

        with patch("linux_whisper.app.HotkeyDaemon", create=True), \
             patch("linux_whisper.app.CONFIG_PATH", create=True), \
             patch("linux_whisper.app._dataclass_to_dict", create=True, return_value={}), \
             patch("linux_whisper.app.yaml", create=True):
            # Patch the imports inside _handle_mode_change
            import linux_whisper.config as cfg_mod
            mock_path_obj = MagicMock()
            mock_path_obj.parent.mkdir = MagicMock()
            with (
                patch.object(cfg_mod, "CONFIG_PATH", mock_path_obj),
                patch("builtins.open", MagicMock()),
                patch("yaml.dump"),
                # Need to patch the local import too
                patch.dict("sys.modules", {}),
            ):
                await app._handle_mode_change("toggle")
        assert app.config.mode == "toggle"
        assert app.config.snippets == snippets
        assert app.config.hotkey == "fn"

    async def test_model_change_preserves_snippets(self):
        snippets = {"greet": "Hi there!"}
        config = _make_config(snippets=snippets)
        app = _make_app(config)
        app._loop = asyncio.get_running_loop()

        mock_engine = MagicMock()
        with patch("linux_whisper.stt.engine.create_engine", return_value=mock_engine), \
             patch("builtins.open", MagicMock()), \
             patch("yaml.dump"):
            import linux_whisper.config as cfg_mod
            mock_path_obj = MagicMock()
            mock_path_obj.parent.mkdir = MagicMock()
            with patch.object(cfg_mod, "CONFIG_PATH", mock_path_obj):
                await app._handle_model_change("faster-whisper", "distil-large-v3.5")

        assert app.config.snippets == snippets
        assert app.config.stt.backend == "faster-whisper"
        assert app.config.stt.model == "distil-large-v3.5"

    async def test_mode_change_preserves_all_config_fields(self):
        config = _make_config(
            hotkey="ctrl+space",
            mode="hold",
            stt=STTConfig(backend="faster-whisper", model="distil-large-v3.5", threads=4),
            polish=PolishConfig(enabled=True, llm=True),
            audio=AudioConfig(auto_gain=True, sample_rate=16000),
            inject=InjectConfig(method="clipboard"),
            tray=TrayConfig(enabled=True),
            overlay=OverlayConfig(enabled=True, position="bottom-center"),
            snippets={"test": "value"},
        )
        app = _make_app(config)
        app._hotkey = MagicMock()
        app._loop = asyncio.get_running_loop()

        with patch("builtins.open", MagicMock()), \
             patch("yaml.dump"):
            import linux_whisper.config as cfg_mod
            mock_path_obj = MagicMock()
            mock_path_obj.parent.mkdir = MagicMock()
            with (
                patch.object(cfg_mod, "CONFIG_PATH", mock_path_obj),
                patch("linux_whisper.hotkey.HotkeyDaemon") as mock_hk,
            ):
                mock_hk_instance = MagicMock()
                mock_hk.return_value = mock_hk_instance
                await app._handle_mode_change("auto")

        assert app.config.hotkey == "ctrl+space"
        assert app.config.mode == "auto"
        assert app.config.stt.backend == "faster-whisper"
        assert app.config.stt.threads == 4
        assert app.config.polish.enabled is True
        assert app.config.audio.auto_gain is True
        assert app.config.inject.method == "clipboard"
        assert app.config.tray.enabled is True
        assert app.config.overlay.position == "bottom-center"
        assert app.config.snippets == {"test": "value"}

    async def test_model_change_preserves_stt_threads(self):
        config = _make_config(
            stt=STTConfig(backend="whisper-cpp", model="whisper-large-v3-turbo", threads=8),
        )
        app = _make_app(config)
        app._loop = asyncio.get_running_loop()

        mock_engine = MagicMock()
        with patch("linux_whisper.stt.engine.create_engine", return_value=mock_engine), \
             patch("builtins.open", MagicMock()), \
             patch("yaml.dump"):
            import linux_whisper.config as cfg_mod
            mock_path_obj = MagicMock()
            mock_path_obj.parent.mkdir = MagicMock()
            with patch.object(cfg_mod, "CONFIG_PATH", mock_path_obj):
                await app._handle_model_change("moonshine", "moonshine-tiny")

        assert app.config.stt.threads == 8
        assert app.config.stt.backend == "moonshine"
        assert app.config.stt.model == "moonshine-tiny"


# ---------------------------------------------------------------------------
# 10. _on_recording_start / _on_recording_stop thread-safety guards
# ---------------------------------------------------------------------------


class TestRecordingCallbacks:
    def test_on_recording_stop_noop_when_no_loop(self):
        app = _make_app()
        app._loop = None
        # Should not raise
        app._on_recording_stop()

    def test_on_recording_start_starts_audio_capture(self):
        app = _make_app()
        app._audio = MagicMock()
        app._audio.get_pre_roll.return_value = np.zeros(12000, dtype=np.float32)
        app._stt = MagicMock()
        app._loop = None  # prevents scheduling

        app._on_recording_start()

        app._audio.start_recording.assert_called_once()
        app._stt.start_stream.assert_called_once()

    def test_on_recording_stop_noop_when_loop_closed(self):
        app = _make_app()
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = True
        app._loop = mock_loop
        # Should not raise or schedule anything
        app._on_recording_stop()
        mock_loop.call_soon_threadsafe.assert_not_called()

    def test_on_mode_change_noop_when_no_loop(self):
        app = _make_app()
        app._loop = None
        # Should not raise
        app._on_mode_change("toggle")

    def test_on_model_change_noop_when_no_loop(self):
        app = _make_app()
        app._loop = None
        # Should not raise
        app._on_model_change("faster-whisper", "distil-large-v3.5")


# ---------------------------------------------------------------------------
# 11. _inject_text
# ---------------------------------------------------------------------------


class TestInjectText:
    async def test_inject_text_calls_injector(self):
        app = _make_app()
        app._injector = AsyncMock()
        app._injector.inject = AsyncMock(return_value=True)

        await app._inject_text("hello")
        app._injector.inject.assert_awaited_once_with("hello")

    async def test_inject_text_no_injector(self):
        app = _make_app()
        app._injector = None
        # Should not raise
        await app._inject_text("hello")


# ---------------------------------------------------------------------------
# 12. create_app helper
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_create_app_with_config(self):
        from linux_whisper.app import create_app

        config = _make_config()
        app = create_app(config)
        assert app.config is config

    def test_create_app_default_config(self):
        from linux_whisper.app import create_app

        with patch("linux_whisper.app.Config.load", return_value=_make_config()):
            app = create_app()
        assert app.config.hotkey == "fn"
