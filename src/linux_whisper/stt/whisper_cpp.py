"""whisper.cpp STT backend via pywhispercpp — non-streaming, buffer-and-process.

Uses pywhispercpp which bundles whisper.cpp with ggml backends.  When compiled
with HIP support (the default pre-built wheel includes ``libggml-hip.so``),
GPU acceleration is used automatically on AMD ROCm devices.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from linux_whisper.config import MODELS_DIR, Config
from linux_whisper.stt.engine import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard (lazy — avoids loading the C extension at import time)
# ---------------------------------------------------------------------------
_HAS_WHISPERCPP: bool | None = None  # resolved on first check


def _check_whispercpp() -> bool:
    """Check if pywhispercpp is available (cached)."""
    global _HAS_WHISPERCPP  # noqa: PLW0603
    if _HAS_WHISPERCPP is None:
        try:
            from pywhispercpp.model import Model as _  # noqa: F401

            _HAS_WHISPERCPP = True
        except ImportError:
            _HAS_WHISPERCPP = False
    return _HAS_WHISPERCPP

# Model name → expected GGML filename in the cache directory
_WHISPER_CPP_MODELS: dict[str, str] = {
    "whisper-large-v3-turbo": "ggml-large-v3-turbo.bin",
    "distil-large-v3.5": "ggml-distil-large-v3.5.bin",
}

# Audio format constants
_SAMPLE_RATE = 16_000  # Hz
_SAMPLE_WIDTH = 2  # 16-bit PCM → 2 bytes per sample


def _require_whispercpp() -> None:
    if not _check_whispercpp():
        raise ImportError(
            "The 'pywhispercpp' package is required for the whisper-cpp backend "
            "but is not installed.  Install it with:\n"
            "    pip install pywhispercpp\n"
            "or install linux-whisper with the whisper extra:\n"
            "    pip install linux-whisper[whisper]"
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


def _detect_gpu_available() -> bool:
    """Check if the ggml HIP backend can see a ROCm GPU."""
    if not _check_whispercpp():
        return False
    try:
        import _pywhispercpp as pw

        info = pw.whisper_print_system_info()
        return isinstance(info, str) and "ROCm" in info
    except Exception:
        return False


class WhisperCppEngine:
    """whisper.cpp speech-to-text engine (non-streaming).

    This backend buffers all audio fed via ``feed_audio`` and performs a
    single inference pass in ``finalize()``.  When pywhispercpp is built
    with HIP support and ``config.stt.device`` is ``"rocm"``, inference
    runs on the GPU automatically via ggml's HIP backend.
    """

    def __init__(self, config: Config) -> None:
        _require_whispercpp()

        self._model_name = config.stt.model
        self._threads = config.stt.threads or os.cpu_count() or 4
        self._device = config.stt.device
        self._model_path = _resolve_model_path(self._model_name)

        self._whisper: Any | None = None  # WhisperModel instance
        self._stream_started = False
        self._audio_buffer = bytearray()
        self._stream_start_time: float = 0.0

        # Check GPU availability upfront
        self._use_gpu = False
        if self._device == "rocm":
            if _detect_gpu_available():
                self._use_gpu = True
                logger.info("ROCm GPU detected — whisper.cpp will use GPU acceleration")
            else:
                logger.warning(
                    "stt.device='rocm' but no ROCm GPU available — falling back to CPU"
                )

        device_label = "ROCm GPU" if self._use_gpu else "CPU"
        logger.info(
            "WhisperCppEngine created: model=%s, device=%s, threads=%d, path=%s",
            self._model_name,
            device_label,
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

        from pywhispercpp.model import Model as WhisperModel

        # pywhispercpp auto-detects GPU via ggml when HIP is compiled in.
        # When device is "cpu", we disable GPU by setting the env var before init.
        if not self._use_gpu:
            os.environ["GGML_CUDA_NO_PINNED"] = "1"

        self._whisper = WhisperModel(
            str(self._model_path),
            n_threads=self._threads,
            redirect_whispercpp_logs_to=False,
        )

        device_label = "ROCm GPU" if self._use_gpu else "CPU"
        logger.info("whisper.cpp model loaded successfully (device=%s)", device_label)

    def _pcm_bytes_to_float_array(self, pcm: bytes | bytearray) -> np.ndarray:
        """Convert raw 16-bit signed PCM bytes to a float32 numpy array in [-1, 1]."""
        samples = np.frombuffer(pcm, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

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

        audio_float = self._pcm_bytes_to_float_array(self._audio_buffer)

        logger.debug(
            "Running whisper.cpp inference on %.1fs of audio ...", duration
        )

        try:
            result = self._whisper.transcribe(audio_float)  # type: ignore[union-attr]
        except Exception:
            logger.exception("whisper.cpp inference failed")
            return TranscriptResult(duration=duration)

        # Parse pywhispercpp Segment objects into our TranscriptSegments
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

        full_text = " ".join(full_parts)

        logger.debug(
            "whisper.cpp finalized: %.1fs audio, %d segments, %d chars",
            duration,
            len(segments),
            len(full_text),
        )

        return TranscriptResult(
            segments=segments,
            full_text=full_text,
            duration=duration,
        )

    def reset(self) -> None:
        """Clear all internal buffers and state."""
        self._audio_buffer = bytearray()
        self._stream_started = False
        logger.debug("whisper.cpp engine reset")
