"""Audio capture pipeline with ring buffer, Silero VAD, and feedback tones.

Captures audio from the default input device via sounddevice (PipeWire/PulseAudio/ALSA),
runs Silero VAD v5 (ONNX) for voice activity detection, and exposes an async generator
that yields speech audio chunks in streaming or batch mode.

Sample format: 16kHz, mono, float32.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    import numpy.typing as npt

from linux_whisper.config import AudioConfig, CACHE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"

# VAD operates on 512-sample windows (32ms at 16kHz) — matches Silero's expected input
VAD_WINDOW_SAMPLES = 512

# Ring buffer holds 30 seconds of audio by default
DEFAULT_RING_BUFFER_SECONDS = 120  # 2 minutes max recording

# Feedback tone parameters
FEEDBACK_DURATION_S = 0.05  # 50ms
FEEDBACK_FREQ_LOW = 880.0  # Hz
FEEDBACK_FREQ_HIGH = 1760.0  # Hz
FEEDBACK_AMPLITUDE = 0.3

# Silero VAD ONNX model path
SILERO_MODEL_DIR = CACHE_DIR / "models" / "silero-vad"
SILERO_MODEL_PATH = SILERO_MODEL_DIR / "silero_vad.onnx"


# ---------------------------------------------------------------------------
# Ring Buffer — lock-free circular buffer for audio samples
# ---------------------------------------------------------------------------


class RingBuffer:
    """Lock-free single-producer single-consumer ring buffer backed by numpy.

    The audio callback thread is the sole producer (writes via `write`).
    The consumer thread reads via `read` or `read_all`.  Thread safety is
    guaranteed by atomic head/tail index updates — no mutexes needed because
    there is exactly one writer and one reader, and Python integer assignment
    is atomic on CPython.
    """

    __slots__ = ("_buf", "_capacity", "_head", "_tail")

    def __init__(self, capacity: int) -> None:
        """Create a ring buffer that can hold *capacity* float32 samples."""
        self._buf: npt.NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._head = 0  # next write position (producer)
        self._tail = 0  # next read position (consumer)

    @property
    def capacity(self) -> int:
        return self._capacity

    def available(self) -> int:
        """Number of samples available for reading."""
        head = self._head
        tail = self._tail
        if head >= tail:
            return head - tail
        return self._capacity - tail + head

    def free_space(self) -> int:
        """Number of samples that can be written without overwriting unread data."""
        return self._capacity - 1 - self.available()

    def write(self, data: npt.NDArray[np.float32]) -> int:
        """Write samples into the buffer.  Returns the number of samples written.

        If *data* is larger than the free space, the oldest unread samples are
        silently overwritten (the tail pointer advances).  This keeps the
        producer (real-time audio callback) wait-free.
        """
        n = len(data)
        if n == 0:
            return 0

        head = self._head

        # If data exceeds capacity, only keep the last capacity-1 samples
        if n >= self._capacity:
            data = data[-(self._capacity - 1) :]
            n = len(data)
            self._buf[:n] = data
            self._head = n
            self._tail = 0
            return n

        end = head + n
        if end <= self._capacity:
            self._buf[head:end] = data
        else:
            first = self._capacity - head
            self._buf[head:] = data[:first]
            self._buf[: n - first] = data[first:]

        new_head = end % self._capacity

        # Advance tail if we overwrote unread data
        avail_after = (new_head - self._tail) % self._capacity
        if avail_after > self._capacity - 1:
            self._tail = (new_head + 1) % self._capacity

        self._head = new_head
        return n

    def read(self, count: int) -> npt.NDArray[np.float32]:
        """Read up to *count* samples from the buffer, advancing the tail."""
        avail = self.available()
        n = min(count, avail)
        if n == 0:
            return np.empty(0, dtype=np.float32)

        tail = self._tail
        end = tail + n
        if end <= self._capacity:
            out = self._buf[tail:end].copy()
        else:
            first = self._capacity - tail
            out = np.concatenate([self._buf[tail:], self._buf[: n - first]])

        self._tail = end % self._capacity
        return out

    def read_all(self) -> npt.NDArray[np.float32]:
        """Read and consume all available samples."""
        return self.read(self.available())

    def peek(self, count: int) -> npt.NDArray[np.float32]:
        """Read up to *count* samples without advancing the tail."""
        avail = self.available()
        n = min(count, avail)
        if n == 0:
            return np.empty(0, dtype=np.float32)

        tail = self._tail
        end = tail + n
        if end <= self._capacity:
            return self._buf[tail:end].copy()
        first = self._capacity - tail
        return np.concatenate([self._buf[tail:], self._buf[: n - first]])

    def peek_recent(self, count: int) -> npt.NDArray[np.float32]:
        """Read the most recent *count* samples without advancing the tail.

        Unlike ``peek`` which reads from the oldest data, this reads the
        *count* samples immediately before the write head (newest data).
        """
        avail = self.available()
        n = min(count, avail)
        if n == 0:
            return np.empty(0, dtype=np.float32)

        head = self._head
        start = head - n
        if start >= 0:
            return self._buf[start:head].copy()
        # Wraps around
        return np.concatenate([self._buf[start:], self._buf[:head]])

    def clear(self) -> None:
        """Discard all unread data."""
        self._tail = self._head


# ---------------------------------------------------------------------------
# Silero VAD v5 — ONNX Runtime wrapper
# ---------------------------------------------------------------------------


class SileroVAD:
    """Silero VAD v5 voice activity detector using ONNX Runtime.

    The model expects 512 samples (32ms) of float32 mono audio at 16kHz and
    returns a speech probability in [0, 1].  Internal RNN state (h/c) is
    maintained between calls for temporal coherence.

    If the ONNX model file does not exist, construction raises FileNotFoundError.
    """

    # Silero v5 state tensor dimensions
    _STATE_DIM = (2, 1, 128)

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD ONNX model not found at {model_path}.  "
                "Download it to enable voice activity detection."
            )

        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=self._make_session_options(),
        )

        # RNN hidden state — shape [2, batch, 128] for this Silero version
        self._state = np.zeros(self._STATE_DIM, dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

        logger.info("Silero VAD loaded from %s", model_path)

    @staticmethod
    def _make_session_options():  # noqa: ANN205
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def __call__(self, audio: npt.NDArray[np.float32]) -> float:
        """Return speech probability for a single 512-sample window."""
        if len(audio) != VAD_WINDOW_SAMPLES:
            raise ValueError(
                f"Expected {VAD_WINDOW_SAMPLES} samples, got {len(audio)}"
            )

        input_data = audio.reshape(1, -1)

        ort_inputs = {
            "input": input_data,
            "sr": self._sr,
            "state": self._state,
        }

        output, state_out = self._session.run(None, ort_inputs)
        self._state = state_out

        return float(output.squeeze())

    def reset_state(self) -> None:
        """Reset the RNN hidden state.  Call between utterances."""
        self._state = np.zeros(self._STATE_DIM, dtype=np.float32)


# ---------------------------------------------------------------------------
# Audio Feedback Tone Generator
# ---------------------------------------------------------------------------


def _generate_sweep(
    freq_start: float,
    freq_end: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = FEEDBACK_AMPLITUDE,
) -> npt.NDArray[np.float32]:
    """Generate a linear frequency sweep (chirp) as float32 samples.

    Used for start/stop recording feedback tones.
    """
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float64)

    # Linear frequency sweep: f(t) = f0 + (f1 - f0) * t / duration
    # Phase integral: phi(t) = 2*pi * (f0*t + (f1-f0)*t^2 / (2*duration))
    phase = 2.0 * math.pi * (freq_start * t + (freq_end - freq_start) * t**2 / (2.0 * duration))
    sweep = amplitude * np.sin(phase)

    # Apply a short fade-in/fade-out envelope to avoid clicks
    fade_samples = min(n_samples // 5, int(sample_rate * 0.005))
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float64)
        sweep[:fade_samples] *= fade_in
        sweep[-fade_samples:] *= fade_out

    return sweep.astype(np.float32)


def generate_start_tone() -> npt.NDArray[np.float32]:
    """Rising tone (880 Hz -> 1760 Hz) played when recording starts."""
    return _generate_sweep(FEEDBACK_FREQ_LOW, FEEDBACK_FREQ_HIGH, FEEDBACK_DURATION_S)


def generate_stop_tone() -> npt.NDArray[np.float32]:
    """Falling tone (1760 Hz -> 880 Hz) played when recording stops."""
    return _generate_sweep(FEEDBACK_FREQ_HIGH, FEEDBACK_FREQ_LOW, FEEDBACK_DURATION_S)


def play_tone(tone: npt.NDArray[np.float32], sample_rate: int = SAMPLE_RATE) -> None:
    """Play a tone through the default output device (non-blocking).

    Errors are logged but never propagated — feedback is best-effort.
    """
    try:
        sd.play(tone, samplerate=sample_rate, blocking=False)
    except Exception:
        logger.debug("Failed to play feedback tone", exc_info=True)


# ---------------------------------------------------------------------------
# Pipeline Mode
# ---------------------------------------------------------------------------


class PipelineMode(enum.Enum):
    """How the audio pipeline delivers audio to consumers."""

    STREAMING = "streaming"
    """Forward chunks to consumers as soon as they pass VAD."""

    BATCH = "batch"
    """Collect all speech audio, emit as a single block on stop."""


# ---------------------------------------------------------------------------
# Audio Chunk — yielded by the async generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A chunk of audio data yielded by the pipeline."""

    samples: npt.NDArray[np.float32]
    """Float32 mono audio at 16kHz."""

    timestamp: float
    """Monotonic time when this chunk was captured."""

    is_speech: bool
    """Whether VAD classified this chunk as containing speech."""

    is_final: bool = False
    """True for the last chunk in a recording session (only in batch mode, this
    is the single emitted block; in streaming mode, this marks the end)."""


# ---------------------------------------------------------------------------
# Audio Pipeline
# ---------------------------------------------------------------------------


class AudioPipeline:
    """Captures audio, runs VAD, and yields speech chunks.

    Lifecycle::

        pipeline = AudioPipeline(config)
        await pipeline.start()
        async for chunk in pipeline.audio_chunks():
            process(chunk.samples)
        await pipeline.stop()

    The pipeline can be started and stopped multiple times (e.g., per
    recording session triggered by the hotkey).
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        *,
        mode: PipelineMode = PipelineMode.STREAMING,
        vad_model_path: Path | None = None,
        ring_buffer_seconds: float = DEFAULT_RING_BUFFER_SECONDS,
    ) -> None:
        self._config = config or AudioConfig()
        self._mode = mode
        self._sample_rate = self._config.sample_rate
        self._buffer_size = self._config.buffer_size
        self._vad_threshold = self._config.vad_threshold
        self._silence_timeout = self._config.silence_timeout
        self._feedback_enabled = self._config.feedback_sounds

        # Ring buffer
        capacity = int(self._sample_rate * ring_buffer_seconds)
        self._ring = RingBuffer(capacity)

        # VAD — optional; if the model isn't present, all audio is treated as speech
        self._vad: SileroVAD | None = None
        vad_path = vad_model_path or SILERO_MODEL_PATH
        try:
            self._vad = SileroVAD(vad_path)
        except FileNotFoundError:
            logger.warning(
                "Silero VAD model not found at %s — VAD disabled, all audio treated as speech.",
                vad_path,
            )
        except Exception:
            logger.warning("Failed to load Silero VAD — VAD disabled.", exc_info=True)

        # Sounddevice input stream (created on start)
        self._stream: sd.InputStream | None = None

        # Async plumbing
        self._loop: asyncio.AbstractEventLoop | None = None
        self._chunk_queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()

        # State tracking
        self._running = False
        self._recording = False
        self._speech_active = False
        self._last_speech_time: float = 0.0
        self._batch_accumulator: list[npt.NDArray[np.float32]] = []

        # Pre-generate feedback tones
        self._start_tone = generate_start_tone()
        self._stop_tone = generate_stop_tone()

        # VAD accumulation buffer for collecting a full window from potentially
        # smaller sounddevice frames
        self._vad_accum: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32)

        # Lock to protect _recording flag from concurrent start/stop_recording calls
        self._recording_lock = threading.Lock()

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Open the audio input stream and begin capture.

        Does not start *recording* — call :meth:`start_recording` to begin
        forwarding audio chunks to consumers.
        """
        if self._running:
            logger.warning("AudioPipeline.start() called while already running")
            return

        self._loop = asyncio.get_running_loop()
        self._chunk_queue = asyncio.Queue()

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=self._buffer_size,
                callback=self._audio_callback,
                latency="low",
            )
            self._stream.start()
        except Exception:
            logger.exception("Failed to open audio input stream")
            raise

        self._running = True
        logger.info(
            "Audio pipeline started: %d Hz, %d-sample blocks, mode=%s, VAD=%s",
            self._sample_rate,
            self._buffer_size,
            self._mode.value,
            "enabled" if self._vad else "disabled",
        )

    async def stop(self) -> None:
        """Stop capture and release audio resources."""
        if not self._running:
            return

        # If still recording, stop gracefully
        if self._recording:
            self.stop_recording()

        self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.debug("Error closing audio stream", exc_info=True)
            self._stream = None

        # Signal end-of-stream to any waiting consumers
        try:
            self._chunk_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        logger.info("Audio pipeline stopped")

    # -- Recording session ---------------------------------------------------

    def start_recording(self) -> None:
        """Begin a recording session — audio chunks will be forwarded to consumers."""
        with self._recording_lock:
            if self._recording:
                logger.debug("start_recording() called while already recording")
                return
            if not self._running:
                logger.warning("Cannot start recording: pipeline not running")
                return

            self._recording = True
            self._speech_active = False
            self._last_speech_time = time.monotonic()
            self._batch_accumulator.clear()
            self._ring.clear()
            self._vad_accum = np.empty(0, dtype=np.float32)

            # Drain any stale chunks from the queue (leftover from previous recording)
            while not self._chunk_queue.empty():
                try:
                    self._chunk_queue.get_nowait()
                except Exception:
                    break

            if self._vad is not None:
                self._vad.reset_state()

        if self._feedback_enabled:
            play_tone(self._start_tone, self._sample_rate)

        logger.debug("Recording started")

    def stop_recording(self) -> None:
        """End the recording session.

        In batch mode, emits all accumulated speech audio as a single final chunk.
        In streaming mode, emits a zero-length final sentinel chunk.
        """
        with self._recording_lock:
            if not self._recording:
                return
            self._recording = False

        if self._feedback_enabled:
            play_tone(self._stop_tone, self._sample_rate)

        # Emit final chunk
        if self._mode == PipelineMode.BATCH and self._batch_accumulator:
            combined = np.concatenate(self._batch_accumulator)
            self._batch_accumulator.clear()
            self._enqueue_chunk(AudioChunk(
                samples=combined,
                timestamp=time.monotonic(),
                is_speech=True,
                is_final=True,
            ))
        else:
            # Streaming mode (or empty batch): emit a sentinel
            self._enqueue_chunk(AudioChunk(
                samples=np.empty(0, dtype=np.float32),
                timestamp=time.monotonic(),
                is_speech=False,
                is_final=True,
            ))

        logger.debug("Recording stopped")

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def speech_active(self) -> bool:
        """Whether VAD currently detects speech."""
        return self._speech_active

    @property
    def silence_duration(self) -> float:
        """Seconds since the last detected speech frame."""
        if self._speech_active:
            return 0.0
        return time.monotonic() - self._last_speech_time

    @property
    def vad_enabled(self) -> bool:
        return self._vad is not None

    # -- Async generator for consumers --------------------------------------

    async def audio_chunks(self) -> AsyncGenerator[AudioChunk, None]:
        """Async generator yielding :class:`AudioChunk` objects.

        Yields chunks as they become available.  A chunk with ``is_final=True``
        signals the end of a recording session.  The generator exits when
        ``is_final`` is received or the pipeline is stopped.

        Usage::

            async for chunk in pipeline.audio_chunks():
                if chunk.is_final:
                    break
                process(chunk.samples)
        """
        while True:
            chunk = await self._chunk_queue.get()
            if chunk is None:
                # Pipeline shut down
                return
            yield chunk
            if chunk.is_final:
                return

    # -- Private: sounddevice callback (real-time thread) --------------------

    def _audio_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice from the PortAudio callback thread.

        Must be fast (<1ms).  No blocking, no allocation beyond numpy views.
        """
        if status:
            logger.debug("Audio callback status: %s", status)

        # Flatten to 1-D mono
        audio = indata[:, 0] if indata.ndim > 1 else indata.ravel()

        # Always write to ring buffer (allows pre-roll capture)
        self._ring.write(audio)

        if not self._recording:
            return

        # --- VAD processing ---
        # Accumulate samples until we have a full VAD window
        self._vad_accum = np.concatenate([self._vad_accum, audio])

        while len(self._vad_accum) >= VAD_WINDOW_SAMPLES:
            window = self._vad_accum[:VAD_WINDOW_SAMPLES]
            self._vad_accum = self._vad_accum[VAD_WINDOW_SAMPLES:]

            is_speech = self._run_vad(window)
            now = time.monotonic()

            if is_speech:
                self._speech_active = True
                self._last_speech_time = now
            else:
                if self._speech_active:
                    elapsed = now - self._last_speech_time
                    if elapsed >= self._silence_timeout:
                        self._speech_active = False

            self._dispatch_window(window, is_speech, now)

    def _run_vad(self, window: npt.NDArray[np.float32]) -> bool:
        """Run VAD on a single window.  Returns True if speech detected."""
        if self._vad is None:
            return True  # No VAD — treat everything as speech

        try:
            probability = self._vad(window)
            return probability >= self._vad_threshold
        except Exception:
            # VAD failure is non-fatal — assume speech
            logger.debug("VAD inference failed", exc_info=True)
            return True

    def _dispatch_window(
        self,
        window: npt.NDArray[np.float32],
        is_speech: bool,
        timestamp: float,
    ) -> None:
        """Dispatch a VAD-processed audio window to consumers.

        ALL audio is forwarded during recording regardless of VAD state.
        VAD is used only for the speech_active indicator (visual feedback).
        This ensures the STT engine receives complete audio without gaps.
        """
        if self._mode == PipelineMode.STREAMING:
            self._enqueue_chunk(AudioChunk(
                samples=window.copy(),
                timestamp=timestamp,
                is_speech=is_speech,
            ))
        else:
            # Batch mode: accumulate all frames
            self._batch_accumulator.append(window.copy())

    def _enqueue_chunk(self, chunk: AudioChunk) -> None:
        """Thread-safe enqueue of a chunk to the async consumer queue."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._chunk_queue.put_nowait, chunk)
        except RuntimeError:
            # Event loop is shutting down
            pass
        except asyncio.QueueFull:
            logger.debug("Audio chunk queue full — dropping chunk")

    # -- Utilities -----------------------------------------------------------

    def get_pre_roll(self, seconds: float = 0.5) -> npt.NDArray[np.float32]:
        """Read the last *seconds* of audio from the ring buffer without consuming.

        Useful for including audio that was captured *before* the hotkey was
        pressed (pre-roll buffer) so the STT engine doesn't miss the first
        syllable.
        """
        n_samples = int(self._sample_rate * seconds)
        available = self._ring.available()
        n = min(n_samples, available)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        return self._ring.peek_recent(n)

    def play_start_tone(self) -> None:
        """Manually trigger the recording-start feedback tone."""
        if self._feedback_enabled:
            play_tone(self._start_tone, self._sample_rate)

    def play_stop_tone(self) -> None:
        """Manually trigger the recording-stop feedback tone."""
        if self._feedback_enabled:
            play_tone(self._stop_tone, self._sample_rate)
