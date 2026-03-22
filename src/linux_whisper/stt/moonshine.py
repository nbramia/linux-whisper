"""Moonshine v2 STT backend — natively streaming inference via ONNX."""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from linux_whisper.config import Config
from linux_whisper.stt.engine import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import moonshine_onnx  # type: ignore[import-untyped]

    _HAS_MOONSHINE = True
except ImportError:
    _HAS_MOONSHINE = False

_MOONSHINE_MODELS: dict[str, str] = {
    "moonshine-tiny": "moonshine/tiny",
    "moonshine-medium": "moonshine/base",
}

_SAMPLE_RATE = 16_000


def _require_moonshine() -> None:
    if not _HAS_MOONSHINE:
        raise ImportError(
            "The 'moonshine_onnx' package is required for the Moonshine backend but "
            "is not installed.  Install it with:\n"
            "    pip install useful-moonshine-onnx\n"
        )


class MoonshineEngine:
    """Moonshine v2 speech-to-text engine.

    Uses moonshine_onnx.transcribe() for inference. Audio is buffered
    during feed_audio() and transcribed in finalize().
    """

    def __init__(self, config: Config) -> None:
        _require_moonshine()

        self._model_name = config.stt.model
        if self._model_name not in _MOONSHINE_MODELS:
            raise ValueError(
                f"Unknown Moonshine model '{self._model_name}'. "
                f"Valid models: {list(_MOONSHINE_MODELS)}"
            )

        self._threads = config.stt.threads or os.cpu_count() or 4
        self._model_tag = _MOONSHINE_MODELS[self._model_name]

        self._stream_started = False
        self._audio_buffer = bytearray()
        self._segments: list[TranscriptSegment] = []

        # Set thread count for ONNX Runtime
        os.environ["OMP_NUM_THREADS"] = str(self._threads)

        logger.info(
            "MoonshineEngine created: model=%s (%s), threads=%d",
            self._model_name,
            self._model_tag,
            self._threads,
        )

    def _pcm_to_float32(self, pcm: bytes | bytearray) -> np.ndarray:
        """Convert 16-bit signed PCM bytes to float32 numpy array in [-1, 1]."""
        int16 = np.frombuffer(pcm, dtype=np.int16)
        return int16.astype(np.float32) / 32768.0

    def _audio_duration(self) -> float:
        """Duration of buffered audio in seconds."""
        return len(self._audio_buffer) / (2 * _SAMPLE_RATE)  # 2 bytes per int16 sample

    # ------------------------------------------------------------------
    # STTEngine protocol
    # ------------------------------------------------------------------

    def start_stream(self) -> None:
        self._audio_buffer = bytearray()
        self._segments = []
        self._stream_started = True
        logger.debug("Moonshine stream started")

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        if not self._stream_started:
            raise RuntimeError("start_stream() must be called before feed_audio()")
        self._audio_buffer.extend(chunk)
        return []  # Moonshine ONNX doesn't support incremental; we transcribe in finalize()

    def finalize(self) -> TranscriptResult:
        if not self._stream_started:
            return TranscriptResult()

        duration = self._audio_duration()
        self._stream_started = False

        if not self._audio_buffer:
            return TranscriptResult(duration=duration)

        audio_float = self._pcm_to_float32(self._audio_buffer)

        t0 = time.perf_counter()
        try:
            result = moonshine_onnx.transcribe(audio_float, self._model_tag)
            text = result[0].strip() if result and result[0] else ""
        except Exception:
            logger.exception("Moonshine transcription failed")
            text = ""

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Moonshine: %.1fs audio → %d chars in %.0fms (%.1fx realtime)",
            duration,
            len(text),
            elapsed_ms,
            (duration * 1000) / elapsed_ms if elapsed_ms > 0 else 0,
        )

        segments = []
        if text:
            segments.append(TranscriptSegment(
                text=text, start_time=0.0, end_time=duration, is_partial=False,
            ))

        return TranscriptResult(
            segments=segments,
            full_text=text,
            language="en",
            duration=duration,
        )

    def reset(self) -> None:
        self._audio_buffer = bytearray()
        self._segments = []
        self._stream_started = False
