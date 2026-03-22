"""Tests for linux_whisper.audio — RingBuffer, AudioChunk, tone generation.

Does NOT test actual audio device capture (requires hardware and sounddevice).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from linux_whisper.audio import (
    FEEDBACK_AMPLITUDE,
    FEEDBACK_DURATION_S,
    FEEDBACK_FREQ_HIGH,
    FEEDBACK_FREQ_LOW,
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
