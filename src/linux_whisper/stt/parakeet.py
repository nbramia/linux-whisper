"""Parakeet TDT STT backend — NVIDIA FastConformer-TDT via ONNX Runtime.

Parakeet TDT 0.6B v3 is a 600M-parameter token-and-duration transducer covering
25 European languages.  Two properties matter here:

* **Throughput.** RTFx ~3300 on the Open ASR Leaderboard versus ~216 for
  whisper large-v3-turbo.  On this machine the INT8 ONNX build transcribes
  faster on CPU than whisper.cpp manages on the ROCm GPU, which takes STT off
  the critical path without touching the GPU at all.
* **Native punctuation and capitalisation.**  Unlike a raw CTC model, the
  transcript arrives already cased and punctuated.

Inference runs through ``onnx-asr`` on the CPU execution provider, reusing the
``onnxruntime`` already present for the Silero VAD and the polish encoders.
That is deliberate: the GPU path would need the ROCm execution provider, which
is exactly the shared-library conflict that forced ``whisper_gpu`` into a
subprocess.  Parakeet is fast enough on CPU that the conflict is not worth
re-litigating.

**Licence note:** Parakeet weights are CC-BY-4.0, not MIT like Whisper's.
Attribution is required for redistribution.
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
    import onnx_asr  # type: ignore[import-untyped]

    _HAS_ONNX_ASR = True
except ImportError:
    _HAS_ONNX_ASR = False

# Config model name → onnx-asr model identifier.
_PARAKEET_MODELS: dict[str, str] = {
    "parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
}

# INT8 is the shipping quantisation: it is the variant benchmarked at RTFx
# 3300, and the accuracy cost against fp32 is within noise on dictation-length
# audio.  Override with stt.quantization if that ever stops being true.
_DEFAULT_QUANTIZATION = "int8"

# Thread cap.  The project convention elsewhere is ``threads or cpu_count()``,
# which is actively harmful here — a 600M model on dictation-length audio does
# not have enough work per op to amortise the synchronisation, so oversubscribing
# costs more than it buys.  Measured over 12 LibriSpeech utterances on a 32-thread
# Ryzen AI MAX+ 395:
#
#     threads    p50      p95
#           4    188ms    246ms
#           8    189ms    235ms   <- cap
#          16    224ms    305ms
#          32    344ms    456ms   <- what cpu_count() would pick
#
# An explicit ``stt.threads`` in config still wins; this only bounds the default.
_MAX_DEFAULT_THREADS = 8

_SAMPLE_RATE = 16_000
_LANGUAGE = "en"


def _require_onnx_asr() -> None:
    if not _HAS_ONNX_ASR:
        raise ImportError(
            "The 'onnx-asr' package is required for the Parakeet backend but is "
            "not installed.  Install it with:\n"
            "    pip install onnx-asr\n"
        )


class ParakeetEngine:
    """Parakeet TDT speech-to-text engine.

    Audio is buffered during :meth:`feed_audio` and transcribed in
    :meth:`finalize` — Parakeet TDT is a batch model, so there is no partial
    output to surface mid-stream.
    """

    def __init__(self, config: Config) -> None:
        _require_onnx_asr()

        self._model_name = config.stt.model
        if self._model_name not in _PARAKEET_MODELS:
            raise ValueError(
                f"Unknown Parakeet model '{self._model_name}'. "
                f"Valid models: {list(_PARAKEET_MODELS)}"
            )

        self._threads = config.stt.threads or min(
            os.cpu_count() or 4, _MAX_DEFAULT_THREADS
        )
        self._model_tag = _PARAKEET_MODELS[self._model_name]

        self._stream_started = False
        self._audio_buffer = bytearray()

        os.environ["OMP_NUM_THREADS"] = str(self._threads)

        if config.stt.device == "rocm":
            # Not a silent downgrade — the caller asked for the GPU and is not
            # getting it, so say why.  onnxruntime-rocm and pywhispercpp both
            # link libamdhip64; loading both in one process is the conflict
            # that put whisper_gpu in a subprocess.  Parakeet on CPU already
            # beats whisper on the GPU here, so CPU is the right answer.
            logger.info(
                "Parakeet runs on CPU regardless of stt.device=rocm — INT8 CPU "
                "inference is faster than the GPU path here and avoids the "
                "onnxruntime/pywhispercpp ROCm library conflict."
            )

        # Downloads from Hugging Face on first use, then resolves from cache.
        # Thread count goes through SessionOptions rather than only the
        # OMP_NUM_THREADS env var, which onnxruntime may have already read.
        self._model = onnx_asr.load_model(
            self._model_tag,
            quantization=_DEFAULT_QUANTIZATION,
            providers=["CPUExecutionProvider"],
            sess_options=self._make_session_options(self._threads),
        )

        logger.info(
            "ParakeetEngine created: model=%s (%s, %s), threads=%d",
            self._model_name,
            self._model_tag,
            _DEFAULT_QUANTIZATION,
            self._threads,
        )

    @staticmethod
    def _make_session_options(threads: int):  # noqa: ANN205
        """ONNX Runtime session options with an explicit intra-op thread count."""
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

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
        self._stream_started = True
        logger.debug("Parakeet stream started")

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        if not self._stream_started:
            raise RuntimeError("start_stream() must be called before feed_audio()")
        self._audio_buffer.extend(chunk)
        # TDT decoding is batch-only; everything happens in finalize().
        return []

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
            text = (self._model.recognize(audio_float) or "").strip()
        except Exception:
            logger.exception("Parakeet transcription failed")
            text = ""

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Parakeet: %.1fs audio → %d chars in %.0fms (%.1fx realtime)",
            duration,
            len(text),
            elapsed_ms,
            (duration * 1000) / elapsed_ms if elapsed_ms > 0 else 0,
        )

        segments: list[TranscriptSegment] = []
        if text:
            segments.append(
                TranscriptSegment(
                    text=text, start_time=0.0, end_time=duration, is_partial=False
                )
            )

        return TranscriptResult(
            segments=segments,
            full_text=text,
            language=_LANGUAGE,
            duration=duration,
        )

    def reset(self) -> None:
        self._audio_buffer = bytearray()
        self._stream_started = False
