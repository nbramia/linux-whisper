"""Moonshine v2 STT backend — ONNX inference via the ``moonshine-voice`` package.

Moonshine v2 (February 2026) replaced the v1 ``useful-moonshine-onnx`` package
with ``moonshine-voice``: a portable C++ core with bundled ONNX Runtime and a
``Transcriber`` API.  The models are memory-mappable ``.ort`` flatbuffers and
emit punctuation and capitalisation natively, unlike v1.

The bundled ONNX Runtime lives inside the wheel, so this backend does **not**
share a runtime with the Silero VAD or the polish encoders — no repeat of the
pywhispercpp/onnxruntime ROCm library conflict.

Unlike every other backend here, this one does its real work in
:meth:`feed_audio` rather than :meth:`finalize`.  Each chunk is pushed into a
per-utterance ``Stream``, so the encode cost overlaps the user still speaking
and only a short flush lands after they stop.  Measured on LibriSpeech
test-clean (medium-streaming, CPU): ``add_audio`` runs at ~0.38x realtime while
recording, and ``finalize`` costs **11-49ms** versus ~290ms for whisper.cpp on
the GPU.

The per-utterance stream is also required for correctness, not just latency.
``Transcriber.transcribe_without_streaming`` carries state across calls with no
working reset — ``stop()``/``start()``, ``create_stream()``, and
``remove_all_listeners()`` all leave it in place — so the second and later
utterances inherit the previous one's context and pick up hallucinated leading
tokens.  A fresh ``Stream`` per utterance is the only reliable isolation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np

from linux_whisper.stt.engine import TranscriptResult, TranscriptSegment

if TYPE_CHECKING:
    from linux_whisper.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import moonshine_voice  # type: ignore[import-untyped]
    from moonshine_voice.download import get_model_for_language  # type: ignore[import-untyped]
    from moonshine_voice.transcriber import Transcriber  # type: ignore[import-untyped]

    _HAS_MOONSHINE = True
except ImportError:
    _HAS_MOONSHINE = False

# Config model name → Moonshine v2 architecture attribute name.
#
# The config names are unchanged from v1 so existing config files keep working,
# but they now resolve to v2 streaming checkpoints.  "moonshine-medium" mapped
# to v1's *base* model, which was already misleading; under v2 it maps to the
# genuine medium checkpoint (245M params, 6.65% WER — better than Whisper
# large-v3 at a sixth of the size).
_MOONSHINE_MODELS: dict[str, str] = {
    "moonshine-tiny": "TINY_STREAMING",
    "moonshine-small": "SMALL_STREAMING",
    "moonshine-medium": "MEDIUM_STREAMING",
}

_SAMPLE_RATE = 16_000
_LANGUAGE = "en"


def _require_moonshine() -> None:
    if not _HAS_MOONSHINE:
        raise ImportError(
            "The 'moonshine-voice' package is required for the Moonshine backend "
            "but is not installed.  Install it with:\n"
            "    pip install moonshine-voice\n"
            "\n"
            "Note: this replaces the v1 'useful-moonshine-onnx' package, which "
            "provided v1 models only."
        )


class MoonshineEngine:
    """Moonshine v2 speech-to-text engine.

    Audio is encoded incrementally in :meth:`feed_audio`; :meth:`finalize`
    only flushes the decoder.  See the module docstring for why this backend
    inverts the usual work split.
    """

    def __init__(self, config: Config) -> None:
        _require_moonshine()

        self._model_name = config.stt.model
        if self._model_name not in _MOONSHINE_MODELS:
            raise ValueError(
                f"Unknown Moonshine model '{self._model_name}'. "
                f"Valid models: {list(_MOONSHINE_MODELS)}"
            )

        arch_name = _MOONSHINE_MODELS[self._model_name]
        arch = getattr(moonshine_voice.ModelArch, arch_name, None)
        if arch is None:
            raise ImportError(
                f"The installed moonshine-voice package has no ModelArch."
                f"{arch_name}.  This backend needs Moonshine v2 (moonshine-voice "
                f">= 0.1); the v1 'useful-moonshine-onnx' package will not work."
            )

        self._threads = config.stt.threads or os.cpu_count() or 4
        self._model_tag = arch_name

        self._stream_started = False
        self._stream: object | None = None
        self._samples_fed = 0
        self._segments: list[TranscriptSegment] = []

        # Set thread count for ONNX Runtime
        os.environ["OMP_NUM_THREADS"] = str(self._threads)

        # Downloads on first use, then resolves from cache.
        model_path, resolved_arch = get_model_for_language(_LANGUAGE, arch)
        self._transcriber = Transcriber(model_path=model_path, model_arch=resolved_arch)

        logger.info(
            "MoonshineEngine created: model=%s (%s), threads=%d, path=%s",
            self._model_name,
            self._model_tag,
            self._threads,
            model_path,
        )

    def _pcm_to_float32(self, pcm: bytes | bytearray) -> np.ndarray:
        """Convert 16-bit signed PCM bytes to float32 numpy array in [-1, 1]."""
        int16 = np.frombuffer(pcm, dtype=np.int16)
        return int16.astype(np.float32) / 32768.0

    @staticmethod
    def _extract_segments(transcript: object, duration: float) -> list[TranscriptSegment]:
        """Flatten a Moonshine ``Transcript`` into engine segments.

        v2 returns line-level results with timestamps.  Empty lines are dropped
        — the streaming decoder emits a trailing blank line at end-of-audio.
        """
        segments: list[TranscriptSegment] = []
        for line in getattr(transcript, "lines", []) or []:
            text = (getattr(line, "text", "") or "").strip()
            if not text:
                continue
            start = float(getattr(line, "start_time", 0.0) or 0.0)
            end = float(getattr(line, "end_time", 0.0) or 0.0)
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_time=start,
                    end_time=end if end > start else duration,
                    is_partial=False,
                )
            )
        return segments

    # ------------------------------------------------------------------
    # STTEngine protocol
    # ------------------------------------------------------------------

    def start_stream(self) -> None:
        # A fresh Stream per utterance is mandatory — see the module docstring
        # on cross-utterance state leakage.
        self._close_stream()
        self._samples_fed = 0
        self._segments = []
        self._stream = self._transcriber.create_stream()
        self._stream.start()
        self._stream_started = True
        logger.debug("Moonshine stream started")

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        if not self._stream_started or self._stream is None:
            raise RuntimeError("start_stream() must be called before feed_audio()")

        samples = self._pcm_to_float32(chunk)
        self._samples_fed += len(samples)
        try:
            self._stream.add_audio(samples)
        except Exception:
            # A mid-stream failure must not kill the recording; finalize() will
            # return whatever the decoder managed to produce.
            logger.exception("Moonshine add_audio failed")

        # Partial results are available here but deliberately not surfaced:
        # the pipeline has no revision protocol, so emitting text that later
        # changes would inject and then contradict itself.  See issue #5.
        return []

    def finalize(self) -> TranscriptResult:
        if not self._stream_started or self._stream is None:
            return TranscriptResult()

        duration = self._samples_fed / _SAMPLE_RATE
        self._stream_started = False

        if not self._samples_fed:
            self._close_stream()
            return TranscriptResult(duration=duration)

        t0 = time.perf_counter()
        try:
            self._stream.stop()
            transcript = self._stream.update_transcription()
            segments = self._extract_segments(transcript, duration)
        except Exception:
            logger.exception("Moonshine transcription failed")
            segments = []
        finally:
            self._close_stream()

        text = " ".join(segment.text for segment in segments).strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Moonshine: %.1fs audio → %d chars, flush %.0fms",
            duration,
            len(text),
            elapsed_ms,
        )

        return TranscriptResult(
            segments=segments,
            full_text=text,
            language=_LANGUAGE,
            duration=duration,
        )

    def reset(self) -> None:
        self._close_stream()
        self._samples_fed = 0
        self._segments = []
        self._stream_started = False

    def _close_stream(self) -> None:
        """Release the current stream, tolerating an already-closed one."""
        if self._stream is None:
            return
        try:
            self._stream.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            logger.debug("Moonshine stream close failed", exc_info=True)
        finally:
            self._stream = None
