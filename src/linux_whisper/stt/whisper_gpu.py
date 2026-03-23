"""GPU-accelerated whisper.cpp backend via subprocess isolation.

Runs pywhispercpp in a completely separate process (subprocess.Popen)
to avoid the ROCm shared-library conflict with onnxruntime.  The worker
loads the model once and stays warm between transcriptions.

Communication uses length-prefixed JSON over stdin/stdout pipes.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import subprocess
import sys
import time
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
_SAMPLE_WIDTH = 2
_WORKER_STARTUP_TIMEOUT = 60.0  # model load can take ~10-15s on first run


def _send_msg(pipe, msg: dict) -> None:
    """Write a length-prefixed JSON message."""
    data = json.dumps(msg).encode()
    pipe.write(struct.pack(">I", len(data)))
    pipe.write(data)
    pipe.flush()


def _recv_msg(pipe) -> dict | None:
    """Read a length-prefixed JSON message."""
    header = pipe.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack(">I", header)[0]
    data = pipe.read(length)
    if len(data) < length:
        return None
    return json.loads(data)


class WhisperGPUEngine:
    """whisper.cpp STT engine with GPU acceleration via process isolation.

    Spawns a worker subprocess that loads pywhispercpp (with ROCm/HIP),
    keeping it isolated from onnxruntime in the main process.
    """

    def __init__(self, config: Config) -> None:
        self._model_name = config.stt.model
        self._threads = config.stt.threads or os.cpu_count() or 4
        self._model_path = self._resolve_model_path(self._model_name)

        self._process: subprocess.Popen | None = None

        self._stream_started = False
        self._audio_buffer = bytearray()
        self._stream_start_time: float = 0.0

        logger.info(
            "WhisperGPUEngine created: model=%s, threads=%d, path=%s",
            self._model_name,
            self._threads,
            self._model_path,
        )

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
        """Start the GPU worker subprocess if not already running."""
        if self._process is not None and self._process.poll() is None:
            return

        logger.info("Starting whisper.cpp GPU worker subprocess...")

        # Run the worker as a script file (not -m module) to avoid the
        # linux_whisper package __init__ chain importing numpy before
        # pywhispercpp — which causes a ROCm segfault.
        worker_script = Path(__file__).parent / "whisper_gpu_worker.py"
        self._process = subprocess.Popen(
            [sys.executable, str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent stderr for logging
        )

        # Send init message with model path
        _send_msg(self._process.stdin, {
            "cmd": "init",
            "model_path": str(self._model_path),
            "n_threads": self._threads,
        })

        # Wait for ready signal
        t0 = time.monotonic()
        msg = _recv_msg(self._process.stdout)
        dt = time.monotonic() - t0

        if msg and msg.get("status") == "ready":
            logger.info(
                "Whisper GPU worker ready (pid=%d, %.1fs)",
                self._process.pid,
                dt,
            )
            return

        # Worker failed
        if self._process.poll() is not None:
            logger.error(
                "GPU worker exited with code %d", self._process.returncode
            )
        else:
            logger.error("GPU worker did not signal ready after %.1fs", dt)
            self._shutdown_worker()

        raise RuntimeError("Whisper GPU worker failed to start")

    def _shutdown_worker(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                _send_msg(self._process.stdin, {"cmd": "shutdown"})
                self._process.wait(timeout=5)
        except Exception:
            pass
        finally:
            if self._process.poll() is None:
                self._process.terminate()
            self._process = None

    # ------------------------------------------------------------------
    # STTEngine protocol
    # ------------------------------------------------------------------

    def _audio_duration(self) -> float:
        return len(self._audio_buffer) / (_SAMPLE_RATE * _SAMPLE_WIDTH)

    def start_stream(self) -> None:
        self._ensure_worker()
        self._audio_buffer = bytearray()
        self._stream_started = True
        self._stream_start_time = time.monotonic()

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

        if not self._audio_buffer or self._process is None:
            return TranscriptResult(duration=duration)

        audio_bytes = bytes(self._audio_buffer)

        logger.debug("Sending %.1fs audio to GPU worker...", duration)

        try:
            # Send transcribe command + raw audio
            _send_msg(self._process.stdin, {
                "cmd": "transcribe",
                "audio_length": len(audio_bytes),
            })
            self._process.stdin.write(audio_bytes)
            self._process.stdin.flush()

            msg = _recv_msg(self._process.stdout)
        except Exception:
            logger.exception("Failed to communicate with GPU worker")
            return TranscriptResult(duration=duration)

        if msg is None or msg.get("status") != "ok":
            error = msg.get("error", "unknown") if msg else "no response"
            logger.warning("GPU worker error: %s", error)
            return TranscriptResult(duration=duration)

        segments = [
            TranscriptSegment(
                text=seg["text"],
                start_time=seg["t0"],
                end_time=seg["t1"],
                is_partial=False,
            )
            for seg in msg.get("segments", [])
        ]

        full_text = msg.get("full_text", "")
        logger.debug(
            "GPU STT: %.1fs → %d segments, %d chars", duration, len(segments), len(full_text)
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
