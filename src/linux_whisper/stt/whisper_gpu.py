"""GPU-accelerated whisper.cpp backend via subprocess isolation.

Runs pywhispercpp in a completely separate process (subprocess.Popen)
to avoid the ROCm shared-library conflict with onnxruntime.  The worker
loads the model once and stays warm between transcriptions, until it has
been idle for ``stt.gpu_idle_unload_s`` seconds; it is then stopped to release
its GPU memory and SDMA queue, and the next stream starts a fresh worker.

Communication uses length-prefixed JSON over stdin/stdout pipes.
"""

from __future__ import annotations

import json
import logging
import os
import select
import struct
import subprocess
import sys
import threading
import time
import weakref
from math import isfinite
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
_WORKER_INFERENCE_TIMEOUT = 120.0
_MAX_RESPONSE_BYTES = 1_048_576
# whisper.cpp timestamps are quantized and may include a final padded decoding
# frame. One second permits that boundary effect without accepting timings that
# are unrelated to the submitted clip.
_SEGMENT_END_TOLERANCE_S = 1.0


class GPUWorkerError(RuntimeError):
    """The isolated whisper.cpp worker could not safely complete a request."""


class GPUWorkerTimeoutError(GPUWorkerError):
    """The worker did not complete its IPC operation before its deadline."""


class GPUWorkerProtocolError(GPUWorkerError):
    """The worker returned a truncated, malformed, or unexpected response."""


def _parse_result(msg: dict, *, duration: float) -> TranscriptResult:
    """Validate a successful worker payload before it can resemble silence."""
    if msg.get("status") != "ok":
        raise GPUWorkerError("GPU worker reported an inference failure")
    if not isinstance(msg.get("full_text"), str):
        raise GPUWorkerProtocolError("GPU worker response has no text field")
    raw_segments = msg.get("segments")
    if not isinstance(raw_segments, list):
        raise GPUWorkerProtocolError("GPU worker response has no segment list")

    segments: list[TranscriptSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise GPUWorkerProtocolError("GPU worker response has an invalid segment")
        start = raw.get("t0")
        end = raw.get("t1")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int | float)
            or not isinstance(end, int | float)
            or not isfinite(start)
            or not isfinite(end)
            or start < 0
            or end < start
            or end > duration + _SEGMENT_END_TOLERANCE_S
        ):
            raise GPUWorkerProtocolError("GPU worker response has invalid segment timing")
        text = raw["text"]
        if not text or text != text.strip():
            raise GPUWorkerProtocolError("GPU worker response has invalid segment text")
        segments.append(
            TranscriptSegment(
                text=text,
                start_time=float(start),
                end_time=float(end),
                is_partial=False,
            )
        )
    if msg["full_text"] != " ".join(segment.text for segment in segments):
        raise GPUWorkerProtocolError("GPU worker response has inconsistent text")
    return TranscriptResult(segments=segments, full_text=msg["full_text"], duration=duration)


def _wait_for_fd(fd: int, *, write: bool, deadline: float, operation: str) -> None:
    """Wait for a pipe endpoint without letting an unresponsive worker block us."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GPUWorkerTimeoutError(f"GPU worker {operation} timed out")
    readable, writable, _ = select.select(
        [] if write else [fd], [fd] if write else [], [], remaining
    )
    if not (writable if write else readable):
        raise GPUWorkerTimeoutError(f"GPU worker {operation} timed out")


def _write_all(pipe, data: bytes, *, deadline: float, operation: str) -> None:
    """Write *data* using OS-level bounded writes rather than buffered I/O."""
    fd = pipe.fileno()
    offset = 0
    while offset < len(data):
        _wait_for_fd(fd, write=True, deadline=deadline, operation=operation)
        try:
            written = os.write(fd, data[offset:])
        except BlockingIOError:
            continue
        except BrokenPipeError as exc:
            raise GPUWorkerError(f"GPU worker closed stdin during {operation}") from exc
        if written <= 0:
            raise GPUWorkerError(f"GPU worker accepted no input during {operation}")
        offset += written


def _read_exact(pipe, length: int, *, deadline: float, operation: str) -> bytes:
    """Read exactly *length* bytes with a single deadline for the whole frame."""
    fd = pipe.fileno()
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        _wait_for_fd(fd, write=False, deadline=deadline, operation=operation)
        try:
            chunk = os.read(fd, remaining)
        except BlockingIOError:
            continue
        if not chunk:
            raise GPUWorkerProtocolError(f"GPU worker closed stdout during {operation}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_msg(pipe, msg: dict, *, deadline: float, operation: str) -> None:
    """Write a length-prefixed JSON message."""
    data = json.dumps(msg).encode()
    _write_all(pipe, struct.pack(">I", len(data)) + data, deadline=deadline, operation=operation)


def _recv_msg(pipe, *, deadline: float, operation: str) -> dict:
    """Read a length-prefixed JSON message."""
    header = _read_exact(pipe, 4, deadline=deadline, operation=operation)
    length = struct.unpack(">I", header)[0]
    if length > _MAX_RESPONSE_BYTES:
        raise GPUWorkerProtocolError(f"GPU worker response exceeds {_MAX_RESPONSE_BYTES} bytes")
    try:
        msg = json.loads(_read_exact(pipe, length, deadline=deadline, operation=operation))
    except json.JSONDecodeError as exc:
        raise GPUWorkerProtocolError("GPU worker sent invalid JSON") from exc
    if not isinstance(msg, dict):
        raise GPUWorkerProtocolError("GPU worker response is not an object")
    return msg


def _idle_unload_callback(engine_ref: weakref.ref[WhisperGPUEngine], generation: int) -> None:
    """Timer entry point that does not keep a discarded engine alive."""
    engine = engine_ref()
    if engine is not None:
        engine._unload_if_idle(generation)


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
        self._operation_timeout_s = _WORKER_INFERENCE_TIMEOUT

        self._stream_started = False
        self._audio_buffer = bytearray()
        self._stream_start_time: float = 0.0
        self._init_idle_unload(config.stt.gpu_idle_unload_s)

        logger.info(
            "WhisperGPUEngine created: model=%s, threads=%d, idle_unload=%ss, path=%s",
            self._model_name,
            self._threads,
            self._idle_unload_s,
            self._model_path,
        )

    def _init_idle_unload(self, idle_unload_s: float) -> None:
        # ``_lock`` guards ``_process``, the stream flag and the in-flight
        # count against the idle timer thread. It is never held across
        # inference IPC, so ``start_stream`` on the hotkey thread cannot wait
        # behind a transcription. Re-entrant because ``_ensure_worker`` reaps
        # a dead worker through ``_shutdown_worker`` while holding it.
        self._lock = threading.RLock()
        # Serialises request/response pairs on the worker's pipes. Only
        # ``finalize`` takes it, and never while holding ``_lock``.
        self._ipc_lock = threading.Lock()
        # Transcriptions sent to the worker and not yet answered. Like an
        # open stream, an in-flight transcription blocks idle unload.
        self._inflight = 0
        self._idle_unload_s: float = idle_unload_s
        self._idle_timer: threading.Timer | None = None
        # Bumped on every arm/cancel so a timer that fired just before being
        # cancelled recognises itself as stale once it gets the lock.
        self._idle_generation = 0

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
        if self._process is not None:
            # ``poll`` observed an exit. Close the dead process's pipe ends
            # before replacing it so repeated recovery cannot leak descriptors.
            self._shutdown_worker()

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
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)

        deadline = time.monotonic() + _WORKER_STARTUP_TIMEOUT
        try:
            _send_msg(
                self._process.stdin,
                {"cmd": "init", "model_path": str(self._model_path), "n_threads": self._threads},
                deadline=deadline,
                operation="startup",
            )
            msg = _recv_msg(self._process.stdout, deadline=deadline, operation="startup")
            if msg.get("status") != "ready":
                raise GPUWorkerProtocolError("GPU worker did not acknowledge startup")
        except Exception as exc:
            self._shutdown_worker()
            if isinstance(exc, GPUWorkerError):
                raise
            raise GPUWorkerError("GPU worker failed to start") from exc

        logger.info("Whisper GPU worker ready (pid=%d)", self._process.pid)

    def _arm_idle_timer(self) -> None:
        """(Re)start the idle countdown; caller holds ``_lock``.

        A no-op while a stream is open or a transcription is in flight: the
        countdown restarts when the last of them finishes.
        """
        self._cancel_idle_timer()
        if (
            self._idle_unload_s <= 0
            or self._process is None
            or self._stream_started
            or self._inflight
        ):
            return
        timer = threading.Timer(
            self._idle_unload_s,
            _idle_unload_callback,
            args=(weakref.ref(self), self._idle_generation),
        )
        timer.daemon = True
        timer.name = "whisper-gpu-idle-unload"
        self._idle_timer = timer
        timer.start()

    def _cancel_idle_timer(self) -> None:
        """Stop any pending idle countdown; caller holds ``_lock``."""
        lock = getattr(self, "_lock", None)
        if lock is None:  # partially constructed engine
            return
        self._idle_generation += 1
        timer, self._idle_timer = self._idle_timer, None
        if timer is not None:
            timer.cancel()

    def _unload_if_idle(self, generation: int) -> None:
        with self._lock:
            if (
                generation != self._idle_generation
                or self._stream_started
                or self._inflight
                or self._process is None
            ):
                return
            logger.info(
                "Unloading idle whisper.cpp GPU worker (pid=%d) after %ss idle",
                self._process.pid,
                self._idle_unload_s,
            )
            self._shutdown_worker()

    def _shutdown_worker(self) -> None:
        self._cancel_idle_timer()
        process = getattr(self, "_process", None)
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    _send_msg(
                        process.stdin,
                        {"cmd": "shutdown"},
                        deadline=time.monotonic() + 1.0,
                        operation="shutdown",
                    )
                    process.wait(timeout=1.0)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
        finally:
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    pipe.close()

    # ------------------------------------------------------------------
    # STTEngine protocol
    # ------------------------------------------------------------------

    def _audio_duration(self) -> float:
        return len(self._audio_buffer) / (_SAMPLE_RATE * _SAMPLE_WIDTH)

    def set_operation_timeout(self, timeout_s: float) -> None:
        """Set the bounded IPC deadline for a caller that owns this engine."""
        if timeout_s <= 0:
            raise ValueError("operation timeout must be positive")
        self._operation_timeout_s = timeout_s

    def start_stream(self) -> None:
        with self._lock:
            self._cancel_idle_timer()
            # Mark the stream before (re)loading so an idle unload can never
            # act on this worker, even if startup raises part-way through.
            self._stream_started = True
            try:
                self._ensure_worker()
            except BaseException:
                self._stream_started = False
                raise
            self._audio_buffer = bytearray()
            self._stream_start_time = time.monotonic()

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        if not self._stream_started:
            raise RuntimeError("start_stream() must be called before feed_audio()")
        self._audio_buffer.extend(chunk)
        return []

    def finalize(self) -> TranscriptResult:
        # Snapshot under the lock, run the IPC without it, then settle under
        # it again. Holding ``_lock`` across inference would block a
        # concurrent ``start_stream`` (the hotkey thread) for the whole
        # transcription — up to the operation timeout on a hung worker.
        with self._lock:
            if not self._stream_started:
                return TranscriptResult()

            duration = self._audio_duration()
            self._stream_started = False

            if not self._audio_buffer:
                self._arm_idle_timer()
                return TranscriptResult(duration=duration)
            process = self._process
            if process is None or process.poll() is not None:
                self._shutdown_worker()
                raise GPUWorkerError("GPU worker is unavailable")

            audio_bytes = bytes(self._audio_buffer)
            timeout_s = self._operation_timeout_s
            self._inflight += 1

        try:
            return self._transcribe(process, audio_bytes, duration, timeout_s)
        except Exception as exc:
            with self._lock:
                # Reap only the worker this request used; if it has already
                # been replaced (or closed), the replacement is not at fault.
                if self._process is process:
                    self._shutdown_worker()
            if isinstance(exc, GPUWorkerError):
                raise
            raise GPUWorkerError("failed to communicate with GPU worker") from exc
        finally:
            with self._lock:
                self._inflight -= 1
                # Restart the countdown unless a new stream has begun (or
                # another request is still in flight); a no-op after a
                # failure, which has already reaped the worker.
                self._arm_idle_timer()

    def _transcribe(
        self,
        process: subprocess.Popen,
        audio_bytes: bytes,
        duration: float,
        timeout_s: float,
    ) -> TranscriptResult:
        """Send one transcription to *process* and parse its reply."""
        logger.debug("Sending %.1fs audio to GPU worker...", duration)

        with self._ipc_lock:
            deadline = time.monotonic() + timeout_s
            _send_msg(
                process.stdin,
                {"cmd": "transcribe", "audio_length": len(audio_bytes)},
                deadline=deadline,
                operation="inference request",
            )
            _write_all(
                process.stdin,
                audio_bytes,
                deadline=deadline,
                operation="inference audio",
            )
            msg = _recv_msg(process.stdout, deadline=deadline, operation="inference response")
        result = _parse_result(msg, duration=duration)

        logger.debug(
            "GPU STT: %.1fs → %d segments, %d chars",
            duration,
            len(result.segments),
            len(result.full_text),
        )
        return result

    def reset(self) -> None:
        with self._lock:
            self._audio_buffer = bytearray()
            was_streaming = self._stream_started
            self._stream_started = False
            if was_streaming:
                self._arm_idle_timer()

    def close(self) -> None:
        """Stop the idle timer and the worker. Safe to call more than once."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            self._shutdown_worker()
            return
        with lock:
            self._shutdown_worker()

    def __del__(self) -> None:
        self.close()
