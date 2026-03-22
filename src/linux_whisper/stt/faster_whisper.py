"""faster-whisper STT backend — high-quality batch transcription on CPU.

Uses CTranslate2 under the hood with INT8 quantization for fast CPU
inference. Supports all Whisper model variants including large-v3-turbo.

The built-in Silero VAD filter handles silence trimming automatically,
so upstream silence trimming is not strictly necessary (but doesn't hurt).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from linux_whisper.config import Config
from linux_whisper.stt.engine import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel

    _HAS_FASTER_WHISPER = True
except ImportError:
    _HAS_FASTER_WHISPER = False
    WhisperModel = None  # type: ignore[assignment,misc]

_FASTER_WHISPER_MODELS: dict[str, str] = {
    "large-v3-turbo": "large-v3-turbo",
    "large-v3": "large-v3",
    "distil-large-v3": "distil-large-v3",
    "distil-large-v3.5": "distil-large-v3.5",
    "medium.en": "medium.en",
    "small.en": "small.en",
}

_SAMPLE_RATE = 16_000


def _require_faster_whisper() -> None:
    if not _HAS_FASTER_WHISPER:
        raise ImportError(
            "The 'faster-whisper' package is required for this backend but "
            "is not installed.  Install it with:\n"
            "    pip install faster-whisper\n"
        )


class FasterWhisperEngine:
    """High-quality speech-to-text using faster-whisper (CTranslate2).

    Not streaming — buffers all audio during feed_audio() and transcribes
    in finalize(). Uses INT8 quantization on CPU for best speed/quality
    tradeoff on x86_64 with AVX-512.
    """

    def __init__(self, config: Config) -> None:
        _require_faster_whisper()

        self._model_name = config.stt.model
        if self._model_name not in _FASTER_WHISPER_MODELS:
            raise ValueError(
                f"Unknown faster-whisper model '{self._model_name}'. "
                f"Valid models: {list(_FASTER_WHISPER_MODELS)}"
            )

        self._threads = config.stt.threads or os.cpu_count() or 4
        self._model_tag = _FASTER_WHISPER_MODELS[self._model_name]
        self._model: WhisperModel | None = None
        self._stream_started = False
        self._audio_buffer = bytearray()

        # Eagerly load the model at init so first recording has zero delay
        self._ensure_model()

        logger.info(
            "FasterWhisperEngine ready: model=%s, threads=%d",
            self._model_name,
            self._threads,
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading faster-whisper model '%s' (INT8, CPU)...", self._model_tag)
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self._model_tag,
            device="cpu",
            compute_type="int8",
            cpu_threads=self._threads,
        )
        logger.info(
            "faster-whisper model loaded in %.1fs",
            time.perf_counter() - t0,
        )

    def start_stream(self) -> None:
        self._ensure_model()
        self._audio_buffer = bytearray()
        self._stream_started = True

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        if not self._stream_started:
            raise RuntimeError("start_stream() must be called before feed_audio()")
        self._audio_buffer.extend(chunk)
        return []

    def finalize(self) -> TranscriptResult:
        if not self._stream_started:
            return TranscriptResult()

        self._stream_started = False

        if not self._audio_buffer:
            return TranscriptResult()

        # Convert int16 PCM bytes to float32
        audio_int16 = np.frombuffer(self._audio_buffer, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        duration = len(audio_float) / _SAMPLE_RATE

        t0 = time.perf_counter()
        try:
            raw_segments, info = self._model.transcribe(
                audio_float,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=300,
                    threshold=0.35,
                    min_speech_duration_ms=100,
                ),
                condition_on_previous_text=True,
                no_speech_threshold=0.5,
            )

            segments = []
            texts = []
            for seg in raw_segments:
                text = seg.text.strip()
                if text:
                    segments.append(TranscriptSegment(
                        text=text,
                        start_time=seg.start,
                        end_time=seg.end,
                        is_partial=False,
                    ))
                    texts.append(text)

            full_text = " ".join(texts)

        except Exception:
            logger.exception("faster-whisper transcription failed")
            return TranscriptResult(duration=duration)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "faster-whisper: %.1fs audio → %d chars in %.0fms (%.1fx realtime)",
            duration,
            len(full_text),
            elapsed_ms,
            (duration * 1000) / elapsed_ms if elapsed_ms > 0 else 0,
        )

        return TranscriptResult(
            segments=segments,
            full_text=full_text,
            language=info.language if info else "en",
            duration=duration,
        )

    def reset(self) -> None:
        self._audio_buffer = bytearray()
        self._stream_started = False
