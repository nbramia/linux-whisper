"""GPU-accelerated whisper.cpp backend via subprocess isolation.

Runs pywhispercpp in a completely separate process (subprocess.Popen)
to avoid the ROCm shared-library conflict with onnxruntime.  The worker
loads the model once and stays warm between transcriptions, until it has
been idle for ``stt.gpu_idle_unload_s`` seconds; it is then stopped to release
its GPU memory and SDMA queue, and the next stream starts a fresh worker.

Communication uses length-prefixed JSON over stdin/stdout pipes.

Long dictations are transcribed while they are being recorded. Whenever the
pending audio of an open stream reaches one whisper window (30 s), the engine
cuts it at the quietest 200 ms in the window's last 5 s and sends the part
before the cut to the worker on a background thread, carrying the rest into
the next window. ``finalize`` then only has the tail (at most one window) left
to transcribe. Recordings shorter than one window are sent as a single request
at ``finalize``.
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
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

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

# Windowed transcription. A window never exceeds whisper's own 30 s input, so
# each one is a single encoder pass; it is cut at the quietest 200 ms frame in
# its last 5 s so a word is unlikely to straddle the boundary.
_WINDOW_SAMPLES = 30 * _SAMPLE_RATE
_CUT_SEARCH_SAMPLES = 5 * _SAMPLE_RATE
_CUT_FRAME_SAMPLES = _SAMPLE_RATE // 5
_CUT_HOP_SAMPLES = _SAMPLE_RATE // 100
# Whisper invents text ("Thank you.", "you") for audio with no speech in it.
# A tail left after the last window can be nothing but the pause before the
# key was released, so it is skipped when no 100 ms of it is louder than this
# multiple of the stream's quietest 200 ms (its noise floor). On the recorded
# dictation fixtures, trailing silence peaks at 1.6x and speech at 2.3x+.
_SILENT_TAIL_RATIO = 2.0
# ...and only when it is also below this absolute level (int16 RMS; 0.01 of
# full scale, -40 dBFS), so a quiet final phrase is never dropped just because
# the "floor" came from quiet continuous speech. On the 28 recorded dictation
# fixtures (Silero VAD labelling speech), the room floor is 0.0032-0.0042,
# trailing silence peaks at 0.0049-0.0077 and the final word at 0.016 or more.
# Deliberately above app.py's _ABSOLUTE_SILENCE_RMS (0.004), which sits inside
# this microphone's trailing-silence range and would never skip a real pause.
_SILENT_TAIL_CEILING_RMS = 0.01 * 32768
_FLOOR_FRAME_SAMPLES = _SAMPLE_RATE // 5
_TAIL_FRAME_SAMPLES = _SAMPLE_RATE // 10


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
        ):
            raise GPUWorkerProtocolError("GPU worker response has invalid segment timing")
        if end > duration:
            # whisper.cpp predicts end times on silence-padded audio, so they can
            # land past the clip. The overrun says nothing about the text.
            logger.debug(
                "Clamping segment timing %.2f-%.2fs to clip duration %.2fs", start, end, duration
            )
            start = min(start, duration)
            end = duration
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


def _choose_cut(window: np.ndarray) -> int:
    """Return where to end a full window of int16 *window* samples.

    The cut is the centre of the lowest-energy 200 ms frame (on a 10 ms hop)
    within the last 5 s of the window; ties go to the earliest frame.
    """
    total = len(window)
    search_start = max(0, total - _CUT_SEARCH_SAMPLES)
    region = window[search_start:].astype(np.float64)
    if len(region) <= _CUT_FRAME_SAMPLES:
        return total
    energy = np.concatenate(([0.0], np.cumsum(region * region)))
    starts = np.arange(0, len(region) - _CUT_FRAME_SAMPLES + 1, _CUT_HOP_SAMPLES)
    frame_energy = energy[starts + _CUT_FRAME_SAMPLES] - energy[starts]
    best = int(starts[int(np.argmin(frame_energy))])
    return search_start + best + _CUT_FRAME_SAMPLES // 2


def _frame_rms(samples: np.ndarray, frame: int) -> np.ndarray:
    """RMS of each *frame*-sample frame of int16 *samples*.

    A partial final frame is measured as a frame of its own, so a short burst
    at the very end is never averaged away or dropped.
    """
    values = samples.astype(np.float64)
    if len(values) == 0:
        return np.zeros(1)
    whole = len(values) // frame * frame
    rms = np.sqrt(np.mean(values[:whole].reshape(-1, frame) ** 2, axis=1))
    if whole < len(values):
        rms = np.append(rms, np.sqrt(np.mean(values[whole:] ** 2)))
    return np.asarray(rms, dtype=np.float64)


def _is_silent_tail(tail: bytes, floor_rms: float) -> bool:
    """Whether *tail* holds nothing louder than the stream's noise floor.

    Both tests must hold: no 100 ms of it above ``_SILENT_TAIL_RATIO`` times
    the floor, and none above the absolute ``_SILENT_TAIL_CEILING_RMS``.
    """
    loudest = float(_frame_rms(np.frombuffer(tail, dtype=np.int16), _TAIL_FRAME_SAMPLES).max())
    return loudest <= _SILENT_TAIL_RATIO * floor_rms and loudest < _SILENT_TAIL_CEILING_RMS


def _apply_gain(audio_bytes: bytes) -> bytes:
    """Apply automatic gain control to one window of int16 PCM."""
    from linux_whisper.audio import apply_agc

    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    gained = apply_agc(samples)
    if gained is samples:  # already loud enough (or silent): unchanged
        return audio_bytes
    return (gained * 32767).astype(np.int16).tobytes()


@dataclass(slots=True, eq=False)
class _Window:
    """One background window transcription of an open stream.

    The thread fills in ``result`` or ``error`` and then sets ``done`` under
    the engine's ``_lock``; ``finalize`` joins the thread before reading them.
    """

    start_sample: int
    audio: bytes
    previous: _Window | None
    thread: threading.Thread | None = None
    result: TranscriptResult | None = None
    error: BaseException | None = None
    done: bool = False
    # RMS of the window's quietest 200 ms (or of the frame it was cut
    # through, if quieter), before gain control.
    floor_rms: float = 0.0
    # Not sent because an earlier window of the same stream failed.
    skipped: bool = False
    # Already returned from ``feed_audio`` as partial segments.
    reported: bool = False


def _offset_segments(
    result: TranscriptResult, start_sample: int, *, partial: bool
) -> list[TranscriptSegment]:
    offset = start_sample / _SAMPLE_RATE
    return [
        TranscriptSegment(
            text=segment.text,
            start_time=segment.start_time + offset,
            end_time=segment.end_time + offset,
            is_partial=partial,
        )
        for segment in result.segments
    ]


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


def _stop_process(process: subprocess.Popen) -> None:
    """Ask *process* to exit, escalating to terminate and kill; close its pipes."""
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

    # Gain control (``audio.auto_gain``) is applied here, per window, so the
    # app feeds raw audio as it is recorded instead of gaining it as a whole.
    handles_gain_control = True

    def __init__(self, config: Config) -> None:
        self._model_name = config.stt.model
        self._threads = config.stt.threads or os.cpu_count() or 4
        self._model_path = self._resolve_model_path(self._model_name)

        self._process: subprocess.Popen | None = None
        self._operation_timeout_s = _WORKER_INFERENCE_TIMEOUT

        self._init_stream_state(auto_gain=config.audio.auto_gain)
        self._init_idle_unload(config.stt.gpu_idle_unload_s)

        logger.info(
            "WhisperGPUEngine created: model=%s, threads=%d, idle_unload=%ss, path=%s",
            self._model_name,
            self._threads,
            self._idle_unload_s,
            self._model_path,
        )

    def _init_stream_state(self, *, auto_gain: bool) -> None:
        # Gain control runs per window here, not on the whole recording in
        # the app, because windows leave before the recording is complete.
        self._auto_gain = auto_gain
        self._stream_started = False
        # Audio of the open stream not yet sent in a window.
        self._audio_buffer = bytearray()
        # Samples of the open stream already sent in background windows.
        self._dispatched_samples = 0
        # The open stream's background windows, in audio order. Replaced
        # rather than cleared when a stream starts, ends or is reset, so a
        # window thread that outlives its stream only touches its own objects.
        self._windows: list[_Window] = []
        self._stream_start_time: float = 0.0

    def _clear_stream_locked(self) -> None:
        """Drop the open stream's audio and windows; caller holds ``_lock``."""
        self._audio_buffer = bytearray()
        self._dispatched_samples = 0
        self._windows = []

    def _init_idle_unload(self, idle_unload_s: float) -> None:
        # ``_lock`` guards ``_process``, the stream flag and the in-flight
        # count against the idle timer thread. It is never held across
        # inference IPC, so ``start_stream`` on the hotkey thread cannot wait
        # behind a transcription. Re-entrant because ``_ensure_worker`` reaps
        # a dead worker through ``_shutdown_worker`` while holding it.
        self._lock = threading.RLock()
        # Serialises request/response pairs on the worker's pipes. Taken by
        # window threads and ``finalize``, never while holding ``_lock``
        # (``_lock`` may be taken briefly inside it).
        self._ipc_lock = threading.Lock()
        # Transcriptions (background windows and ``finalize``) not yet
        # settled. Like an open stream, they block idle unload.
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
        if process is not None:
            _stop_process(process)

    def _reap(self, process: subprocess.Popen | None) -> None:
        """Stop *process* after a failed request, unless it was already replaced.

        The process is detached under ``_lock`` and stopped outside it, so a
        hung worker's escalation to kill never holds up ``feed_audio`` or
        ``start_stream``. The next ``start_stream`` starts a replacement.
        """
        with self._lock:
            if process is None or self._process is not process:
                return
            self._cancel_idle_timer()
            self._process = None
        _stop_process(process)

    # ------------------------------------------------------------------
    # STTEngine protocol
    # ------------------------------------------------------------------

    def _audio_duration(self) -> float:
        pending = len(self._audio_buffer) // _SAMPLE_WIDTH
        return (self._dispatched_samples + pending) / _SAMPLE_RATE

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
            self._clear_stream_locked()
            self._stream_start_time = time.monotonic()

    def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
        """Buffer *chunk*; send each completed window to the worker in the background.

        Returns the segments of background windows that have finished since
        the last call, in order and offset to the stream's timeline, marked
        partial because ``finalize`` returns them again in the full result.
        Never waits on the worker: a window is handed to its own thread.
        """
        with self._lock:
            if not self._stream_started:
                raise RuntimeError("start_stream() must be called before feed_audio()")
            self._audio_buffer.extend(chunk)
            while len(self._audio_buffer) >= _WINDOW_SAMPLES * _SAMPLE_WIDTH:
                self._dispatch_window_locked()
            return self._completed_partials_locked()

    def _dispatch_window_locked(self) -> None:
        """Cut one window off the pending audio and start its thread; caller holds ``_lock``."""
        window_samples = np.frombuffer(
            bytes(self._audio_buffer[: _WINDOW_SAMPLES * _SAMPLE_WIDTH]), dtype=np.int16
        )
        cut = _choose_cut(window_samples)
        half = _CUT_FRAME_SAMPLES // 2
        previous = self._windows[-1] if self._windows else None
        window = _Window(
            start_sample=self._dispatched_samples,
            audio=window_samples[:cut].tobytes(),
            previous=previous,
            # The frame cut through straddles the carried-over audio, so it
            # counts towards the noise floor along with the window's own.
            floor_rms=float(_frame_rms(window_samples[cut - half : cut + half], 2 * half)[0]),
        )
        del self._audio_buffer[: cut * _SAMPLE_WIDTH]
        self._dispatched_samples += cut
        self._windows.append(window)
        logger.debug(
            "Dispatching %.1fs window at %.1fs to GPU worker",
            cut / _SAMPLE_RATE,
            window.start_sample / _SAMPLE_RATE,
        )

        thread = threading.Thread(
            target=self._run_window,
            args=(window, self._process, self._operation_timeout_s),
            name="whisper-gpu-window",
            daemon=True,
        )
        window.thread = thread
        self._inflight += 1
        try:
            thread.start()
        except BaseException as exc:
            self._inflight -= 1
            window.thread = None
            window.error = exc
            window.done = True

    def _run_window(
        self,
        window: _Window,
        process: subprocess.Popen | None,
        timeout_s: float,
    ) -> None:
        """Window thread: transcribe *window* once every earlier window has settled.

        Windows of one stream go to the worker strictly in order, and none is
        sent after one has failed, so ``finalize`` reports the first failure.
        """
        try:
            window.floor_rms = min(
                window.floor_rms,
                float(
                    _frame_rms(
                        np.frombuffer(window.audio, dtype=np.int16), _FLOOR_FRAME_SAMPLES
                    ).min()
                ),
            )
            previous = window.previous
            if previous is not None:
                if previous.thread is not None:
                    previous.thread.join()
                if previous.error is not None or previous.skipped:
                    window.skipped = True
                    return
            window.result = self._request(process, window.audio, timeout_s)
        except BaseException as exc:
            window.error = exc
            logger.warning("Background window transcription failed: %s", exc)
        finally:
            window.audio = b""
            window.previous = None
            with self._lock:
                window.done = True
                self._inflight -= 1
                self._arm_idle_timer()

    def _completed_partials_locked(self) -> list[TranscriptSegment]:
        """Segments of windows finished since the last call; caller holds ``_lock``."""
        partials: list[TranscriptSegment] = []
        for window in self._windows:
            if window.reported:
                continue
            if not window.done:
                break
            window.reported = True
            if window.result is not None:
                partials.extend(
                    _offset_segments(window.result, window.start_sample, partial=True)
                )
        return partials

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
            windows = self._windows
            tail = bytes(self._audio_buffer)
            tail_start = self._dispatched_samples
            self._clear_stream_locked()

            if not windows and not tail:
                self._arm_idle_timer()
                return TranscriptResult(duration=duration)
            process = self._process
            timeout_s = self._operation_timeout_s
            self._inflight += 1

        try:
            parts: list[tuple[int, TranscriptResult]] = []
            for window in windows:
                if window.thread is not None:
                    window.thread.join()
            for window in windows:
                if window.error is not None:
                    if isinstance(window.error, GPUWorkerError):
                        raise window.error
                    raise GPUWorkerError("GPU window transcription failed") from window.error
                if window.result is None:
                    raise GPUWorkerError("GPU window transcription did not complete")
                parts.append((window.start_sample, window.result))
            if tail and windows:
                floor_rms = min(window.floor_rms for window in windows)
                if _is_silent_tail(tail, floor_rms):
                    logger.debug(
                        "Skipping %.1fs tail with no audio above the noise floor",
                        len(tail) / (_SAMPLE_RATE * _SAMPLE_WIDTH),
                    )
                    tail = b""
            if tail:
                parts.append((tail_start, self._request(process, tail, timeout_s)))
        finally:
            with self._lock:
                self._inflight -= 1
                # Restart the countdown unless a new stream has begun (or
                # another request is still in flight); a no-op after a
                # failure, which has already reaped the worker.
                self._arm_idle_timer()

        segments: list[TranscriptSegment] = []
        for start_sample, result in parts:
            segments.extend(_offset_segments(result, start_sample, partial=False))
        full_text = " ".join(result.full_text for _, result in parts if result.full_text)
        return TranscriptResult(segments=segments, full_text=full_text, duration=duration)

    def _request(
        self,
        process: subprocess.Popen | None,
        audio_bytes: bytes,
        timeout_s: float,
    ) -> TranscriptResult:
        """Transcribe one window on *process*, reaping it if the request fails.

        Holds ``_ipc_lock`` for the request/response pair and the reap, so a
        following request can never interleave with a worker being stopped.
        """
        if self._auto_gain:
            audio_bytes = _apply_gain(audio_bytes)
        with self._ipc_lock:
            try:
                with self._lock:
                    current = process is not None and self._process is process
                if not current or process.poll() is not None:
                    raise GPUWorkerError("GPU worker is unavailable")
                duration = len(audio_bytes) / (_SAMPLE_RATE * _SAMPLE_WIDTH)
                return self._transcribe(process, audio_bytes, duration, timeout_s)
            except Exception as exc:
                # Reap only the worker this request used; if it has already
                # been replaced (or closed), the replacement is not at fault.
                self._reap(process)
                if isinstance(exc, GPUWorkerError):
                    raise
                raise GPUWorkerError("failed to communicate with GPU worker") from exc

    def _transcribe(
        self,
        process: subprocess.Popen,
        audio_bytes: bytes,
        duration: float,
        timeout_s: float,
    ) -> TranscriptResult:
        """Send one transcription to *process* and parse its reply; caller holds ``_ipc_lock``."""
        logger.debug("Sending %.1fs audio to GPU worker...", duration)

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
        # Background windows of an abandoned stream run to completion on
        # their own threads; their results are dropped with the list.
        with self._lock:
            self._clear_stream_locked()
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
