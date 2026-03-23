"""GPU-accelerated whisper.cpp backend via subprocess isolation.

Runs pywhispercpp in a separate process to avoid the ROCm shared-library
conflict with onnxruntime.  The worker process loads the model once and
stays warm between transcriptions.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from multiprocessing.connection import Connection
from pathlib import Path

from linux_whisper.config import MODELS_DIR, Config
from linux_whisper.stt.engine import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)

# Model name → GGML filename
_WHISPER_CPP_MODELS: dict[str, str] = {
    "whisper-large-v3-turbo": "ggml-large-v3-turbo.bin",
    "distil-large-v3.5": "ggml-distil-large-v3.5.bin",
}

_SAMPLE_RATE = 16_000
_SAMPLE_WIDTH = 2  # 16-bit PCM
_WORKER_STARTUP_TIMEOUT = 30.0  # seconds to wait for model load


class WhisperGPUEngine:
    """whisper.cpp STT engine with GPU acceleration via process isolation.

    Spawns a worker process that loads pywhispercpp (with ROCm/HIP),
    keeping it isolated from onnxruntime in the main process.  Audio
    is sent via pipe, transcription results returned via pipe.
    """

    def __init__(self, config: Config) -> None:
        self._model_name = config.stt.model
        self._threads = config.stt.threads or os.cpu_count() or 4
        self._model_path = self._resolve_model_path(self._model_name)

        self._process: multiprocessing.Process | None = None
        self._conn: Connection | None = None

        self._stream_started = False
        self._audio_buffer = bytearray()
        self._stream_start_time: float = 0.0

        logger.info(
            "WhisperGPUEngine created: model=%s, threads=%d, path=%s",
            self._model_name,
            self._threads,
            self._model_path,
        )

    # ------------------------------------------------------------------
    # Model path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path(model_name: str) -> Path:
        if model_name not in _WHISPER_CPP_MODELS:
            raise ValueError(
                f"Unknown whisper.cpp model '{model_name}'. "
                f"Valid models: {list(_WHISPER_CPP_MODELS)}"
            )
        model_file = MODELS_DIR / _WHISPER_CPP_MODELS[model_name]
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}\n"
                f"Download the GGML model and place it at:\n"
                f"    {model_file}\n"
                f"Models: https://huggingface.co/ggerganov/whisper.cpp"
            )
        return model_file

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        """Start the GPU worker process if not already running."""
        if self._process is not None and self._process.is_alive():
            return

        logger.info("Starting whisper.cpp GPU worker process...")

        parent_conn, child_conn = multiprocessing.Pipe()
        self._conn = parent_conn

        self._process = multiprocessing.Process(
            target=self._worker_entry,
            args=(child_conn, str(self._model_path), self._threads),
            daemon=True,
            name="whisper-gpu-worker",
        )
        self._process.start()

        # Wait for the worker to signal ready (model loaded)
        if parent_conn.poll(timeout=_WORKER_STARTUP_TIMEOUT):
            msg = parent_conn.recv()
            if msg.get("status") == "ready":
                logger.info("Whisper GPU worker ready (pid=%d)", self._process.pid)
                return

        raise RuntimeError("Whisper GPU worker failed to start within timeout")

    @staticmethod
    def _worker_entry(conn: Connection, model_path: str, n_threads: int) -> None:
        """Entry point for the worker — delegates to the worker module."""
        from linux_whisper.stt.whisper_gpu_worker import worker_main

        worker_main(conn, model_path, n_threads)

    def _shutdown_worker(self) -> None:
        """Gracefully shut down the worker process."""
        if self._conn is not None:
            try:
                self._conn.send({"cmd": "shutdown"})
            except Exception:
                pass
            self._conn.close()
            self._conn = None

        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _audio_duration(self) -> float:
        return len(self._audio_buffer) / (_SAMPLE_RATE * _SAMPLE_WIDTH)

    # ------------------------------------------------------------------
    # STTEngine protocol
    # ------------------------------------------------------------------

    def start_stream(self) -> None:
        self._ensure_worker()
        self._audio_buffer = bytearray()
        self._stream_started = True
        self._stream_start_time = time.monotonic()
        logger.debug("whisper-gpu stream started")

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        if not self._stream_started:
            raise RuntimeError("start_stream() must be called before feed_audio()")
        self._audio_buffer.extend(chunk)
        return []

    def finalize(self) -> TranscriptResult:
        if not self._stream_started:
            return TranscriptResult()

        duration = self._audio_duration()
        self._stream_started = False

        if not self._audio_buffer:
            return TranscriptResult(duration=0.0)

        if self._conn is None:
            logger.error("Worker connection lost")
            return TranscriptResult(duration=duration)

        logger.debug("Sending %.1fs audio to GPU worker...", duration)

        try:
            self._conn.send({
                "cmd": "transcribe",
                "audio": bytes(self._audio_buffer),
            })

            # Wait for result (generous timeout for first inference + model warmup)
            if not self._conn.poll(timeout=30.0):
                logger.warning("GPU worker timed out")
                return TranscriptResult(duration=duration)

            msg = self._conn.recv()
        except Exception:
            logger.exception("Failed to communicate with GPU worker")
            return TranscriptResult(duration=duration)

        if msg.get("status") != "ok":
            logger.warning("GPU worker error: %s", msg.get("error", "unknown"))
            return TranscriptResult(duration=duration)

        segments = []
        for seg in msg.get("segments", []):
            segments.append(TranscriptSegment(
                text=seg["text"],
                start_time=seg["t0"],
                end_time=seg["t1"],
                is_partial=False,
            ))

        full_text = msg.get("full_text", "")
        logger.debug(
            "GPU worker: %.1fs audio → %d segments, %d chars",
            duration, len(segments), len(full_text),
        )

        return TranscriptResult(
            segments=segments,
            full_text=full_text,
            duration=duration,
        )

    def reset(self) -> None:
        self._audio_buffer = bytearray()
        self._stream_started = False

    def __del__(self) -> None:
        self._shutdown_worker()
