"""whisper.cpp STT backend — non-streaming, buffer-and-process inference."""

from __future__ import annotations

import array
import logging
import os
import time
from pathlib import Path

from linux_whisper.config import MODELS_DIR, Config
from linux_whisper.stt.engine import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from whispercpp import Whisper  # type: ignore[import-untyped]

    _HAS_WHISPERCPP = True
except ImportError:
    _HAS_WHISPERCPP = False

# Model name → expected GGML filename in the cache directory
_WHISPER_CPP_MODELS: dict[str, str] = {
    "whisper-large-v3-turbo": "ggml-large-v3-turbo.bin",
    "distil-large-v3.5": "ggml-distil-large-v3.5.bin",
}

# Audio format constants
_SAMPLE_RATE = 16_000  # Hz
_SAMPLE_WIDTH = 2  # 16-bit PCM → 2 bytes per sample


def _require_whispercpp() -> None:
    if not _HAS_WHISPERCPP:
        raise ImportError(
            "The 'whispercpp' package is required for the whisper-cpp backend "
            "but is not installed.  Install it with:\n"
            "    pip install whispercpp\n"
            "or install linux-whisper with the whisper-cpp extra:\n"
            "    pip install linux-whisper[whisper-cpp]"
        )


def _resolve_model_path(model_name: str) -> Path:
    """Return the path to the GGML model file, raising if it does not exist."""
    if model_name not in _WHISPER_CPP_MODELS:
        raise ValueError(
            f"Unknown whisper.cpp model '{model_name}'. "
            f"Valid models: {list(_WHISPER_CPP_MODELS)}"
        )

    model_file = MODELS_DIR / _WHISPER_CPP_MODELS[model_name]
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}\n"
            f"Download the GGML-quantised '{model_name}' model and place it at:\n"
            f"    {model_file}\n"
            f"You can obtain models from https://huggingface.co/ggerganov/whisper.cpp"
        )
    return model_file


class WhisperCppEngine:
    """whisper.cpp speech-to-text engine (non-streaming).

    This backend buffers all audio fed via ``feed_audio`` and performs a
    single inference pass in ``finalize()``.  It is better suited for
    shorter utterances or when streaming latency is not critical.
    """

    def __init__(self, config: Config) -> None:
        _require_whispercpp()

        self._model_name = config.stt.model
        self._threads = config.stt.threads or os.cpu_count() or 4
        self._model_path = _resolve_model_path(self._model_name)

        self._whisper: Whisper | None = None
        self._stream_started = False
        self._audio_buffer = bytearray()
        self._stream_start_time: float = 0.0

        logger.info(
            "WhisperCppEngine created: model=%s, threads=%d, path=%s",
            self._model_name,
            self._threads,
            self._model_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load the whisper.cpp model on first use."""
        if self._whisper is not None:
            return

        logger.info(
            "Loading whisper.cpp model from '%s' ...", self._model_path
        )

        self._whisper = Whisper.from_pretrained(
            str(self._model_path),
            n_threads=self._threads,
        )

        logger.info("whisper.cpp model loaded successfully")

    def _pcm_bytes_to_float_list(self, pcm: bytes | bytearray) -> list[float]:
        """Convert raw 16-bit signed PCM bytes to a list of floats in [-1, 1]."""
        samples = array.array("h", pcm)
        return [s / 32768.0 for s in samples]

    def _audio_duration(self) -> float:
        """Duration of buffered audio in seconds."""
        return len(self._audio_buffer) / (_SAMPLE_RATE * _SAMPLE_WIDTH)

    # ------------------------------------------------------------------
    # STTEngine protocol
    # ------------------------------------------------------------------

    def start_stream(self) -> None:
        """Prepare for a new audio stream."""
        self._ensure_model()
        self._audio_buffer = bytearray()
        self._stream_started = True
        self._stream_start_time = time.monotonic()
        logger.debug("whisper.cpp stream started")

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        """Buffer audio for later processing.

        whisper.cpp is not streaming, so this simply accumulates audio
        and always returns an empty list.
        """
        if not self._stream_started:
            raise RuntimeError(
                "start_stream() must be called before feed_audio()"
            )

        self._audio_buffer.extend(chunk)
        return []

    def finalize(self) -> TranscriptResult:
        """Run whisper.cpp inference on the full buffered audio."""
        if not self._stream_started:
            return TranscriptResult()

        duration = self._audio_duration()
        self._stream_started = False

        if not self._audio_buffer:
            logger.debug("whisper.cpp finalize called with empty buffer")
            return TranscriptResult(duration=0.0)

        audio_float = self._pcm_bytes_to_float_list(self._audio_buffer)

        logger.debug(
            "Running whisper.cpp inference on %.1fs of audio ...", duration
        )

        try:
            result = self._whisper.transcribe(audio_float)  # type: ignore[union-attr]
        except Exception:
            logger.exception("whisper.cpp inference failed")
            return TranscriptResult(duration=duration)

        # Parse whisper.cpp result into segments
        segments: list[TranscriptSegment] = []
        full_parts: list[str] = []

        for seg in result:
            text = seg.text.strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_time=seg.t0 / 100.0,  # whisper.cpp times are in centiseconds
                    end_time=seg.t1 / 100.0,
                    is_partial=False,
                )
            )
            full_parts.append(text)

        # If whisper.cpp returned a flat string instead of segments, handle that
        if not segments and hasattr(result, "strip"):
            text = result.strip()  # type: ignore[union-attr]
            if text:
                segments.append(
                    TranscriptSegment(
                        text=text,
                        start_time=0.0,
                        end_time=duration,
                        is_partial=False,
                    )
                )
                full_parts.append(text)

        full_text = " ".join(full_parts)
        detected_lang = getattr(result, "lang", None)

        logger.debug(
            "whisper.cpp finalized: %.1fs audio, %d segments, %d chars",
            duration,
            len(segments),
            len(full_text),
        )

        return TranscriptResult(
            segments=segments,
            full_text=full_text,
            language=detected_lang,
            duration=duration,
        )

    def reset(self) -> None:
        """Clear all internal buffers and state."""
        self._audio_buffer = bytearray()
        self._stream_started = False
        logger.debug("whisper.cpp engine reset")
