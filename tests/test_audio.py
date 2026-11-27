"""Tests for linux_whisper.audio — RingBuffer, AudioChunk, tone generation.

Does NOT test actual audio device capture (requires hardware and sounddevice).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from linux_whisper.audio import (
    FEEDBACK_AMPLITUDE,
    FEEDBACK_DURATION_S,
    SAMPLE_RATE,
    AudioChunk,
    PipelineMode,
    RingBuffer,
    _generate_sweep,
    apply_agc,
    generate_start_tone,
    generate_stop_tone,
)

# ── RingBuffer ──────────────────────────────────────────────────────────────


class TestRingBufferBasic:
    """Basic read/write operations on the ring buffer."""

    def test_empty_buffer(self):
        rb = RingBuffer(1024)
        assert rb.available() == 0
        assert rb.capacity == 1024

    def test_write_and_read(self):
        rb = RingBuffer(1024)
        data = np.arange(100, dtype=np.float32)
        written = rb.write(data)
        assert written == 100
        assert rb.available() == 100

        out = rb.read(100)
        assert len(out) == 100
        np.testing.assert_array_equal(out, data)
        assert rb.available() == 0

    def test_read_partial(self):
        rb = RingBuffer(1024)
        data = np.arange(100, dtype=np.float32)
        rb.write(data)

        out = rb.read(50)
        assert len(out) == 50
        np.testing.assert_array_equal(out, data[:50])
        assert rb.available() == 50

    def test_read_more_than_available(self):
        rb = RingBuffer(1024)
        data = np.arange(10, dtype=np.float32)
        rb.write(data)

        out = rb.read(1000)
        assert len(out) == 10
        np.testing.assert_array_equal(out, data)

    def test_read_empty_buffer(self):
        rb = RingBuffer(1024)
        out = rb.read(100)
        assert len(out) == 0
        assert out.dtype == np.float32

    def test_write_empty_array(self):
        rb = RingBuffer(1024)
        written = rb.write(np.empty(0, dtype=np.float32))
        assert written == 0
        assert rb.available() == 0

    def test_read_all(self):
        rb = RingBuffer(1024)
        data = np.arange(50, dtype=np.float32)
        rb.write(data)

        out = rb.read_all()
        assert len(out) == 50
        np.testing.assert_array_equal(out, data)
        assert rb.available() == 0

    def test_read_all_empty(self):
        rb = RingBuffer(1024)
        out = rb.read_all()
        assert len(out) == 0


class TestRingBufferWrapAround:
    """Test ring buffer behavior when writes wrap around the buffer."""

    def test_wrap_around_write(self):
        rb = RingBuffer(100)
        # Fill most of the buffer
        data1 = np.arange(80, dtype=np.float32)
        rb.write(data1)
        rb.read(80)  # consume all

        # Now write data that wraps around
        data2 = np.arange(50, dtype=np.float32) + 100
        rb.write(data2)
        out = rb.read_all()
        assert len(out) == 50
        np.testing.assert_array_equal(out, data2)

    def test_multiple_wrap_arounds(self):
        rb = RingBuffer(64)
        for i in range(10):
            data = np.full(30, fill_value=float(i), dtype=np.float32)
            rb.write(data)
            out = rb.read_all()
            assert len(out) == 30
            np.testing.assert_array_equal(out, data)


class TestRingBufferOverwrite:
    """Test behavior when writes exceed buffer capacity."""

    def test_overwrite_advances_tail(self):
        rb = RingBuffer(100)
        data = np.arange(200, dtype=np.float32)
        rb.write(data)
        # Buffer can hold at most capacity-1 = 99 samples
        avail = rb.available()
        assert avail <= 99
        # We should still be able to read the most recent data
        out = rb.read_all()
        assert len(out) > 0

    def test_overwrite_large_data(self):
        rb = RingBuffer(50)
        # Write much more than capacity
        data = np.arange(1000, dtype=np.float32)
        rb.write(data)
        # Should still be consistent
        avail = rb.available()
        assert avail <= 49
        out = rb.read_all()
        assert len(out) == avail

    def test_free_space(self):
        rb = RingBuffer(100)
        assert rb.free_space() == 99  # capacity - 1

        rb.write(np.arange(50, dtype=np.float32))
        assert rb.free_space() == 49


class TestRingBufferPeek:
    """Test peek (non-consuming read)."""

    def test_peek_does_not_advance_tail(self):
        rb = RingBuffer(1024)
        data = np.arange(100, dtype=np.float32)
        rb.write(data)

        peeked = rb.peek(50)
        assert len(peeked) == 50
        np.testing.assert_array_equal(peeked, data[:50])
        assert rb.available() == 100  # unchanged

    def test_peek_more_than_available(self):
        rb = RingBuffer(1024)
        data = np.arange(10, dtype=np.float32)
        rb.write(data)

        peeked = rb.peek(1000)
        assert len(peeked) == 10

    def test_peek_empty(self):
        rb = RingBuffer(1024)
        peeked = rb.peek(10)
        assert len(peeked) == 0

    def test_peek_with_wrap_around(self):
        rb = RingBuffer(100)
        # Fill and consume to advance tail near the end
        data1 = np.arange(90, dtype=np.float32)
        rb.write(data1)
        rb.read(90)

        # Write data that wraps around
        data2 = np.arange(30, dtype=np.float32) + 200
        rb.write(data2)

        peeked = rb.peek(30)
        np.testing.assert_array_equal(peeked, data2)
        assert rb.available() == 30  # peek did not consume


class TestRingBufferClear:
    """Test the clear method."""

    def test_clear_discards_data(self):
        rb = RingBuffer(1024)
        rb.write(np.arange(500, dtype=np.float32))
        assert rb.available() > 0

        rb.clear()
        assert rb.available() == 0

    def test_clear_then_write(self):
        rb = RingBuffer(1024)
        rb.write(np.arange(500, dtype=np.float32))
        rb.clear()

        new_data = np.full(10, 42.0, dtype=np.float32)
        rb.write(new_data)
        out = rb.read_all()
        np.testing.assert_array_equal(out, new_data)


# ── AudioChunk ──────────────────────────────────────────────────────────────


class TestAudioChunk:
    """Test the AudioChunk dataclass."""

    def test_audio_chunk_fields(self):
        samples = np.zeros(512, dtype=np.float32)
        t = time.monotonic()
        chunk = AudioChunk(samples=samples, timestamp=t, is_speech=True)
        assert chunk.is_speech is True
        assert chunk.is_final is False
        assert len(chunk.samples) == 512

    def test_audio_chunk_final(self):
        chunk = AudioChunk(
            samples=np.empty(0, dtype=np.float32),
            timestamp=0.0,
            is_speech=False,
            is_final=True,
        )
        assert chunk.is_final is True

    def test_audio_chunk_is_frozen(self):
        chunk = AudioChunk(
            samples=np.zeros(10, dtype=np.float32),
            timestamp=0.0,
            is_speech=False,
        )
        with pytest.raises(AttributeError):
            chunk.is_speech = True  # type: ignore[misc]


# ── PipelineMode ────────────────────────────────────────────────────────────


class TestPipelineMode:

    def test_streaming_value(self):
        assert PipelineMode.STREAMING.value == "streaming"

    def test_batch_value(self):
        assert PipelineMode.BATCH.value == "batch"


# ── Tone generation ─────────────────────────────────────────────────────────


class TestToneGeneration:
    """Test the feedback tone generators produce valid audio."""

    def test_start_tone_shape(self):
        tone = generate_start_tone()
        expected_samples = int(SAMPLE_RATE * FEEDBACK_DURATION_S)
        assert len(tone) == expected_samples
        assert tone.dtype == np.float32

    def test_stop_tone_shape(self):
        tone = generate_stop_tone()
        expected_samples = int(SAMPLE_RATE * FEEDBACK_DURATION_S)
        assert len(tone) == expected_samples
        assert tone.dtype == np.float32

    def test_tones_are_bounded(self):
        """Tone amplitude should not exceed the configured amplitude (plus small tolerance)."""
        for tone in (generate_start_tone(), generate_stop_tone()):
            assert np.max(np.abs(tone)) <= FEEDBACK_AMPLITUDE + 0.01

    def test_tones_are_different(self):
        """Start and stop tones should be different (different sweep direction)."""
        start = generate_start_tone()
        stop = generate_stop_tone()
        assert not np.allclose(start, stop)

    def test_generate_sweep_basic(self):
        sweep = _generate_sweep(440.0, 880.0, 0.1)
        expected = int(SAMPLE_RATE * 0.1)
        assert len(sweep) == expected
        assert sweep.dtype == np.float32

    def test_generate_sweep_fade_envelope(self):
        """The first and last samples should be close to zero due to fade."""
        sweep = _generate_sweep(440.0, 880.0, 0.1)
        # First sample should be near zero (fade-in)
        assert abs(sweep[0]) < 0.05
        # Last sample should be near zero (fade-out)
        assert abs(sweep[-1]) < 0.05

    def test_generate_sweep_custom_params(self):
        sweep = _generate_sweep(
            freq_start=200.0,
            freq_end=400.0,
            duration=0.05,
            sample_rate=8000,
            amplitude=0.5,
        )
        expected = int(8000 * 0.05)
        assert len(sweep) == expected

    def test_sweep_not_silent(self):
        """The sweep should contain non-zero values (not all silence)."""
        sweep = _generate_sweep(440.0, 880.0, 0.05)
        assert np.any(np.abs(sweep) > 0.01)


# ── SileroVAD fallback (no model file) ─────────────────────────────────────


class TestSileroVADFallback:
    """Test that AudioPipeline works without a VAD model."""

    def test_pipeline_init_without_vad_model(self):
        """AudioPipeline should initialize without VAD when model is missing."""
        from linux_whisper.audio import AudioPipeline
        from linux_whisper.config import AudioConfig

        # Pass a nonexistent model path — should gracefully disable VAD
        pipeline = AudioPipeline(
            AudioConfig(),
            vad_model_path=Path("/nonexistent/silero_vad.onnx"),
        )
        assert pipeline.vad_enabled is False

    def test_pipeline_properties_before_start(self):
        from linux_whisper.audio import AudioPipeline
        from linux_whisper.config import AudioConfig

        pipeline = AudioPipeline(
            AudioConfig(),
            vad_model_path=Path("/nonexistent/silero_vad.onnx"),
        )
        assert pipeline.is_running is False
        assert pipeline.is_recording is False
        assert pipeline.speech_active is False


# Need to import Path for the tests above
from pathlib import Path

# ── Automatic Gain Control ─────────────────────────────────────────────────


class TestApplyAGC:
    """Test the apply_agc function."""

    def test_boosts_quiet_audio(self):
        # Peak of 0.1 should be boosted to ~0.7
        audio = np.array([0.1, -0.05, 0.08, -0.1, 0.03], dtype=np.float32)
        result = apply_agc(audio, target_peak=0.7)
        assert np.isclose(np.max(np.abs(result)), 0.7, atol=1e-6)

    def test_no_change_loud_audio(self):
        # Peak of 0.8 is above 0.7 target — should return unchanged
        audio = np.array([0.8, -0.5, 0.3, -0.8, 0.1], dtype=np.float32)
        result = apply_agc(audio, target_peak=0.7)
        np.testing.assert_array_equal(result, audio)

    def test_exactly_at_target(self):
        # Peak of exactly 0.7 — should return unchanged
        audio = np.array([0.7, -0.3, 0.5, -0.7, 0.2], dtype=np.float32)
        result = apply_agc(audio, target_peak=0.7)
        np.testing.assert_array_equal(result, audio)

    def test_silent_audio(self):
        # All zeros — should return unchanged (no division by zero)
        audio = np.zeros(100, dtype=np.float32)
        result = apply_agc(audio)
        np.testing.assert_array_equal(result, audio)

    def test_clips_to_bounds(self):
        # After gain, all values should be in [-1.0, 1.0]
        audio = np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32)
        result = apply_agc(audio, target_peak=0.7)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_preserves_dtype(self):
        audio = np.array([0.1, -0.05], dtype=np.float32)
        result = apply_agc(audio)
        assert result.dtype == np.float32

    def test_custom_target(self):
        audio = np.array([0.2, -0.1, 0.15], dtype=np.float32)
        result = apply_agc(audio, target_peak=0.5)
        assert np.isclose(np.max(np.abs(result)), 0.5, atol=1e-6)

    def test_single_sample(self):
        audio = np.array([0.1], dtype=np.float32)
        result = apply_agc(audio, target_peak=0.7)
        assert np.isclose(result[0], 0.7, atol=1e-6)

    def test_realistic_whispered_audio(self):
        # Simulate whispered speech: low peak ~0.05
        np.random.seed(42)
        audio = (np.random.randn(16000).astype(np.float32) * 0.02)  # very quiet
        peak_before = np.max(np.abs(audio))
        result = apply_agc(audio, target_peak=0.7)
        peak_after = np.max(np.abs(result))
        assert peak_before < 0.1
        assert np.isclose(peak_after, 0.7, atol=1e-6)

    def test_default_target_is_0_7(self):
        audio = np.array([0.1, -0.1], dtype=np.float32)
        result = apply_agc(audio)
        assert np.isclose(np.max(np.abs(result)), 0.7, atol=1e-6)


# ── Silero v6 window size ──────────────────────────────────────────────────


class TestVadWindowSize:
    """Guard the VAD window constant against silent breakage.

    Silero v6 requires 576-sample windows.  Feeding it v5's 512 raises no
    error — the graph accepts the shorter window and returns ~0.001 for every
    frame, including unambiguous speech.  Voice activity detection is then
    entirely dead while every test and log line still looks healthy, so the
    constant needs an explicit assertion.
    """

    def test_window_is_576_samples(self):
        from linux_whisper.audio import VAD_WINDOW_SAMPLES

        assert VAD_WINDOW_SAMPLES == 576

    def test_window_is_36ms_at_16khz(self):
        from linux_whisper.audio import SAMPLE_RATE, VAD_WINDOW_SAMPLES

        assert pytest.approx(0.036, abs=0.001) == VAD_WINDOW_SAMPLES / SAMPLE_RATE

    def test_vad_rejects_wrong_window_length(self):
        """__call__ must reject a mis-sized window rather than pass it through."""
        from unittest.mock import MagicMock

        from linux_whisper.audio import VAD_WINDOW_SAMPLES, SileroVAD

        vad = SileroVAD.__new__(SileroVAD)
        vad._session = MagicMock()
        vad._state_dim = (2, 1, 128)
        vad._state = np.zeros((2, 1, 128), dtype=np.float32)
        vad._sr = np.array(16000, dtype=np.int64)

        with pytest.raises(ValueError, match="576"):
            vad(np.zeros(512, dtype=np.float32))

        vad._session.run.assert_not_called()
        assert VAD_WINDOW_SAMPLES == 576

    def test_capture_blocksize_need_not_match_vad_window(self):
        """The accumulator decouples capture chunks from the VAD window."""
        from linux_whisper.audio import VAD_WINDOW_SAMPLES
        from linux_whisper.config import AudioConfig

        # 512-sample capture blocks feeding 576-sample VAD windows is fine and
        # intentional; they are separate concerns.
        assert AudioConfig().buffer_size != VAD_WINDOW_SAMPLES


class TestSileroStateIntrospection:
    """State shape comes from the model graph, not a hard-coded constant."""

    def test_state_dim_read_from_graph(self):
        from unittest.mock import MagicMock

        from linux_whisper.audio import SileroVAD

        state_input = MagicMock()
        state_input.name = "state"
        state_input.shape = [2, "batch", 128]
        session = MagicMock()
        session.get_inputs.return_value = [state_input]

        assert SileroVAD._infer_state_dim(session) == (2, 1, 128)

    def test_non_default_state_dim_is_honoured(self):
        from unittest.mock import MagicMock

        from linux_whisper.audio import SileroVAD

        state_input = MagicMock()
        state_input.name = "state"
        state_input.shape = [4, 1, 256]
        session = MagicMock()
        session.get_inputs.return_value = [state_input]

        # A future model with different dimensions must not be silently fed a
        # zeroed tensor of the old shape.
        assert SileroVAD._infer_state_dim(session) == (4, 1, 256)

    def test_missing_state_input_falls_back(self):
        from unittest.mock import MagicMock

        from linux_whisper.audio import SileroVAD

        other = MagicMock()
        other.name = "input"
        other.shape = [1, 576]
        session = MagicMock()
        session.get_inputs.return_value = [other]

        assert SileroVAD._infer_state_dim(session) == (2, 1, 128)


# ── Input loss: watchdog reconnect and bounded chunk queue ─────────────────


class _FakeStream:
    """Stands in for ``sd.InputStream``.  A healthy one calls back every 5 ms
    from its own thread, like PortAudio; a dead one never calls back."""

    def __init__(self, *, callback, healthy: bool, **_kwargs) -> None:
        import threading

        self._callback = callback
        self._healthy = healthy
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.active = False
        self.aborted = False
        self.closed = False

    def start(self) -> None:
        self.active = True
        self._thread.start()

    def _run(self) -> None:
        block = np.zeros((512, 1), dtype=np.float32)
        while not self._halt.wait(0.005):
            if self._healthy:
                self._callback(block, 512, None, 0)

    def kill(self) -> None:
        """Simulate the audio server going away: callbacks just stop."""
        self._healthy = False

    def abort(self) -> None:
        self.aborted = True
        self.active = False
        self._halt.set()
        if self._thread.is_alive():
            self._thread.join()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fast_watchdog(monkeypatch):
    import linux_whisper.audio as audio_mod

    monkeypatch.setattr(audio_mod, "STREAM_STALL_S", 0.1)
    monkeypatch.setattr(audio_mod, "STREAM_CHECK_INTERVAL_S", 0.02)
    monkeypatch.setattr(audio_mod, "REOPEN_BACKOFF_INITIAL_S", 0.02)
    monkeypatch.setattr(audio_mod, "REOPEN_BACKOFF_MAX_S", 0.08)
    return audio_mod


def _install_streams(monkeypatch, audio_mod, plan):
    """Make each ``sd.InputStream(...)`` follow *plan*: "ok", "dead" or "fail"."""
    opened: list[_FakeStream] = []
    plan = list(plan)

    def factory(**kwargs):
        step = plan.pop(0) if plan else "ok"
        if step == "fail":
            raise RuntimeError("PortAudio: device unavailable")
        stream = _FakeStream(healthy=(step == "ok"), **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(audio_mod.sd, "InputStream", factory)
    return opened


def _pipeline(audio_mod):
    return audio_mod.AudioPipeline(vad_model_path=Path("/nonexistent/silero_vad.onnx"))


async def _until(predicate, timeout: float = 2.0) -> None:
    import asyncio

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


class TestInputLossRecovery:
    async def test_reopens_when_callbacks_stop(self, monkeypatch, fast_watchdog):
        opened = _install_streams(monkeypatch, fast_watchdog, ["ok", "ok"])
        pipeline = _pipeline(fast_watchdog)
        await pipeline.start()
        try:
            opened[0].kill()
            await _until(lambda: pipeline.reconnects == 1)
            assert opened[0].aborted and opened[0].closed
            assert pipeline._stream is opened[1]
            assert opened[1].active
        finally:
            await pipeline.stop()

    async def test_retries_with_backoff_until_audio_flows(self, monkeypatch, fast_watchdog):
        # The audio server is still restarting: opens fail, then one opens but
        # stays silent, then a stream finally delivers audio.
        opened = _install_streams(
            monkeypatch, fast_watchdog, ["ok", "fail", "fail", "dead", "ok"]
        )
        pipeline = _pipeline(fast_watchdog)
        await pipeline.start()
        try:
            opened[0].kill()
            await _until(lambda: pipeline.reconnects == 1)
            assert len(opened) == 3
            assert opened[1].aborted  # the silent one was discarded
            assert pipeline._stream is opened[2]
        finally:
            await pipeline.stop()

    async def test_reopens_when_portaudio_ends_the_stream(self, monkeypatch, fast_watchdog):
        opened = _install_streams(monkeypatch, fast_watchdog, ["ok", "ok"])
        pipeline = _pipeline(fast_watchdog)
        await pipeline.start()
        try:
            opened[0].active = False
            await _until(lambda: pipeline.reconnects == 1)
        finally:
            await pipeline.stop()

    async def test_healthy_stream_is_left_alone(self, monkeypatch, fast_watchdog):
        import asyncio

        opened = _install_streams(monkeypatch, fast_watchdog, ["ok"])
        pipeline = _pipeline(fast_watchdog)
        await pipeline.start()
        try:
            await asyncio.sleep(0.4)
            assert pipeline.reconnects == 0
            assert len(opened) == 1
        finally:
            await pipeline.stop()

    async def test_stop_during_reconnect_closes_everything(self, monkeypatch, fast_watchdog):
        import asyncio

        opened = _install_streams(monkeypatch, fast_watchdog, ["ok"] + ["fail"] * 50)
        pipeline = _pipeline(fast_watchdog)
        await pipeline.start()
        opened[0].kill()
        await asyncio.sleep(0.3)  # watchdog is now in its retry loop
        await pipeline.stop()
        assert pipeline._watchdog is None
        assert pipeline._stream is None
        assert all(s.closed for s in opened)

    async def test_recording_survives_reconnect(self, monkeypatch, fast_watchdog):
        opened = _install_streams(monkeypatch, fast_watchdog, ["ok", "ok"])
        pipeline = _pipeline(fast_watchdog)
        await pipeline.start()
        try:
            pipeline.start_recording()
            opened[0].kill()
            await _until(lambda: pipeline.reconnects == 1)
            await _until(lambda: pipeline._chunk_queue.qsize() > 0)
            pipeline._chunk_queue = type(pipeline._chunk_queue)(
                maxsize=pipeline._chunk_queue.maxsize
            )
            await _until(lambda: pipeline._chunk_queue.qsize() > 0)  # chunks flow again
        finally:
            await pipeline.stop()


class TestChunkQueueBound:
    def test_drops_oldest_when_full(self, monkeypatch, fast_watchdog):
        import asyncio

        pipeline = _pipeline(fast_watchdog)
        pipeline._chunk_queue = asyncio.Queue(maxsize=3)

        def chunk(i: int, final: bool = False) -> AudioChunk:
            return AudioChunk(np.full(4, i, dtype=np.float32), float(i), True, final)

        for i in range(5):
            pipeline._put_chunk(chunk(i))
        pipeline._put_chunk(chunk(5, final=True))

        kept = [pipeline._chunk_queue.get_nowait() for _ in range(3)]
        assert [c.timestamp for c in kept] == [3.0, 4.0, 5.0]
        assert kept[-1].is_final  # the end-of-recording marker survives
        assert pipeline.dropped_chunks == 3

    def test_default_bound_matches_ring_buffer(self, fast_watchdog):
        pipeline = _pipeline(fast_watchdog)
        seconds = (
            fast_watchdog.CHUNK_QUEUE_MAX * fast_watchdog.VAD_WINDOW_SAMPLES / SAMPLE_RATE
        )
        assert pipeline._chunk_queue.maxsize == fast_watchdog.CHUNK_QUEUE_MAX
        assert seconds == pytest.approx(fast_watchdog.DEFAULT_RING_BUFFER_SECONDS, abs=0.1)
