"""Tests for linux_whisper.app — App orchestration, pipeline, state transitions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from linux_whisper.config import (
    AudioConfig,
    Config,
    InjectConfig,
    PolishConfig,
    STTConfig,
    TrayConfig,
)
from linux_whisper.state import AppState


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


def _make_app(config: Config | None = None) -> "App":
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
        config = _make_config(audio=AudioConfig(auto_gain=False), polish=PolishConfig(enabled=False))
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

        with patch("linux_whisper.app.HotkeyDaemon", create=True) as MockHK, \
             patch("linux_whisper.app.CONFIG_PATH", create=True) as mock_path, \
             patch("linux_whisper.app._dataclass_to_dict", create=True, return_value={}), \
             patch("linux_whisper.app.yaml", create=True):
            # Patch the imports inside _handle_mode_change
            import linux_whisper.config as cfg_mod
            mock_path_obj = MagicMock()
            mock_path_obj.parent.mkdir = MagicMock()
            with patch.object(cfg_mod, "CONFIG_PATH", mock_path_obj):
                with patch("builtins.open", MagicMock()):
                    with patch("yaml.dump"):
                        # Need to patch the local import too
                        with patch.dict("sys.modules", {}):
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
            with patch.object(cfg_mod, "CONFIG_PATH", mock_path_obj):
                with patch("linux_whisper.hotkey.HotkeyDaemon") as MockHK:
                    mock_hk_instance = MagicMock()
                    MockHK.return_value = mock_hk_instance
                    await app._handle_mode_change("auto")

        assert app.config.hotkey == "ctrl+space"
        assert app.config.mode == "auto"
        assert app.config.stt.backend == "faster-whisper"
        assert app.config.stt.threads == 4
        assert app.config.polish.enabled is True
        assert app.config.audio.auto_gain is True
        assert app.config.inject.method == "clipboard"
        assert app.config.tray.enabled is True
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
