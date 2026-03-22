"""STT engine protocol, data types, and factory function."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from linux_whisper.config import Config, STTConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptSegment:
    """A single segment of transcribed speech."""

    text: str
    start_time: float  # seconds from stream start
    end_time: float  # seconds from stream start
    is_partial: bool = False  # True while the segment may still be revised


@dataclass(slots=True)
class TranscriptResult:
    """Complete result returned when a stream is finalized."""

    segments: list[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    language: str | None = None
    duration: float = 0.0  # total audio duration in seconds


@runtime_checkable
class STTEngine(Protocol):
    """Protocol that all speech-to-text backends must implement.

    Lifecycle::

        engine = create_engine(config)
        engine.start_stream()
        for chunk in audio_chunks:
            partials = engine.feed_audio(chunk)
        result = engine.finalize()
        engine.reset()   # optional — ready for next utterance
    """

    def start_stream(self) -> None:
        """Prepare the engine for a new audio stream.

        Must be called before the first ``feed_audio`` call.  Resets any
        internal buffers from a previous stream.
        """
        ...

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        """Feed a chunk of raw 16-bit 16 kHz mono PCM audio.

        Returns a (possibly empty) list of transcript segments.  Streaming
        backends may return partial segments that get revised on subsequent
        calls.  Non-streaming backends typically return an empty list here
        and defer all processing to :meth:`finalize`.
        """
        ...

    def finalize(self) -> TranscriptResult:
        """Signal end-of-audio and return the final transcript.

        For non-streaming backends this is where inference happens.  For
        streaming backends this flushes any remaining partial segments.
        """
        ...

    def reset(self) -> None:
        """Release buffers and prepare for the next utterance.

        Safe to call even if ``start_stream`` was never called.
        """
        ...


def create_engine(config: Config) -> STTEngine:
    """Instantiate the STT backend specified in *config*.

    Raises
    ------
    ValueError
        If the backend name is not recognised.
    ImportError
        If the backend's optional dependency is not installed.
    """
    stt: STTConfig = config.stt
    backend = stt.backend

    logger.info(
        "Creating STT engine: backend=%s, model=%s, threads=%s",
        backend,
        stt.model,
        stt.threads,
    )

    if backend == "moonshine":
        from linux_whisper.stt.moonshine import MoonshineEngine

        return MoonshineEngine(config)

    if backend == "faster-whisper":
        from linux_whisper.stt.faster_whisper import FasterWhisperEngine

        return FasterWhisperEngine(config)

    if backend == "whisper-cpp":
        from linux_whisper.stt.whisper_cpp import WhisperCppEngine

        return WhisperCppEngine(config)

    raise ValueError(
        f"Unknown STT backend '{backend}'. "
        f"Valid backends: {STTConfig.VALID_BACKENDS}"
    )
