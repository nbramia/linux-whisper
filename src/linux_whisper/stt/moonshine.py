"""Moonshine v2 STT backend — natively streaming inference."""

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
    import moonshine  # type: ignore[import-untyped]

    _HAS_MOONSHINE = True
except ImportError:
    _HAS_MOONSHINE = False

_MOONSHINE_MODELS: dict[str, str] = {
    "moonshine-tiny": "moonshine/tiny",
    "moonshine-medium": "moonshine/medium",
}

# Audio format constants
_SAMPLE_RATE = 16_000  # Hz
_SAMPLE_WIDTH = 2  # 16-bit PCM → 2 bytes per sample


def _require_moonshine() -> None:
    if not _HAS_MOONSHINE:
        raise ImportError(
            "The 'moonshine' package is required for the Moonshine backend but "
            "is not installed.  Install it with:\n"
            "    pip install useful-moonshine-onnx\n"
            "or install linux-whisper with the moonshine extra:\n"
            "    pip install linux-whisper[moonshine]"
        )


class MoonshineEngine:
    """Moonshine v2 streaming speech-to-text engine.

    Moonshine v2 supports native streaming: each call to ``feed_audio``
    runs incremental inference and can yield partial transcript segments
    that may be revised as more audio arrives.
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
        self._model_dir = MODELS_DIR / self._model_name
        self._model_tag = _MOONSHINE_MODELS[self._model_name]

        self._model: object | None = None
        self._stream_started = False
        self._audio_buffer = bytearray()
        self._segments: list[TranscriptSegment] = []
        self._stream_start_time: float = 0.0

        logger.info(
            "MoonshineEngine created: model=%s, threads=%d",
            self._model_name,
            self._threads,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load the Moonshine model on first use."""
        if self._model is not None:
            return

        logger.info("Loading Moonshine model '%s' ...", self._model_tag)

        # Ensure cache directory exists
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Set thread count via environment before model load
        os.environ["OMP_NUM_THREADS"] = str(self._threads)
        os.environ["ONNX_NUM_THREADS"] = str(self._threads)

        self._model = moonshine.load_model(self._model_tag)

        logger.info("Moonshine model loaded successfully")

    def _pcm_bytes_to_float_list(self, pcm: bytes | bytearray) -> list[float]:
        """Convert raw 16-bit signed PCM bytes to a list of floats in [-1, 1]."""
        samples = array.array("h", pcm)
        return [s / 32768.0 for s in samples]

    def _audio_offset(self) -> float:
        """Current audio position in seconds based on buffered bytes."""
        return len(self._audio_buffer) / (_SAMPLE_RATE * _SAMPLE_WIDTH)

    # ------------------------------------------------------------------
    # STTEngine protocol
    # ------------------------------------------------------------------

    def start_stream(self) -> None:
        """Prepare for a new audio stream."""
        self._ensure_model()
        self._audio_buffer = bytearray()
        self._segments = []
        self._stream_started = True
        self._stream_start_time = time.monotonic()
        logger.debug("Moonshine stream started")

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        """Feed a chunk of 16-bit 16 kHz mono PCM and run incremental inference.

        Returns partial segments that may be revised as more audio arrives.
        """
        if not self._stream_started:
            raise RuntimeError(
                "start_stream() must be called before feed_audio()"
            )

        start_offset = self._audio_offset()
        self._audio_buffer.extend(chunk)
        end_offset = self._audio_offset()

        # Run streaming inference on the full buffer so far
        audio_float = self._pcm_bytes_to_float_list(self._audio_buffer)

        try:
            tokens = moonshine.transcribe(audio_float, self._model)
        except Exception:
            logger.exception("Moonshine inference failed on audio chunk")
            return []

        if not tokens or not tokens[0]:
            return []

        text = tokens[0].strip()
        if not text:
            return []

        # Build a single partial segment covering the entire buffer so far
        segment = TranscriptSegment(
            text=text,
            start_time=0.0,
            end_time=end_offset,
            is_partial=True,
        )

        # Replace prior partials with the updated text
        self._segments = [segment]

        return [segment]

    def finalize(self) -> TranscriptResult:
        """Finalize the stream and return the completed transcript."""
        if not self._stream_started:
            return TranscriptResult()

        duration = self._audio_offset()

        # Run final inference on the complete audio
        if self._audio_buffer:
            audio_float = self._pcm_bytes_to_float_list(self._audio_buffer)
            try:
                tokens = moonshine.transcribe(audio_float, self._model)
                text = tokens[0].strip() if tokens and tokens[0] else ""
            except Exception:
                logger.exception("Moonshine final inference failed")
                text = self._segments[-1].text if self._segments else ""
        else:
            text = ""

        # Build final segments (mark as non-partial)
        final_segments: list[TranscriptSegment] = []
        if text:
            final_segments.append(
                TranscriptSegment(
                    text=text,
                    start_time=0.0,
                    end_time=duration,
                    is_partial=False,
                )
            )

        self._stream_started = False

        logger.debug(
            "Moonshine stream finalized: %.1fs audio, %d chars",
            duration,
            len(text),
        )

        return TranscriptResult(
            segments=final_segments,
            full_text=text,
            language="en",
            duration=duration,
        )

    def reset(self) -> None:
        """Clear all internal buffers and state."""
        self._audio_buffer = bytearray()
        self._segments = []
        self._stream_started = False
        logger.debug("Moonshine engine reset")
