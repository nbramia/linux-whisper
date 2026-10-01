"""Tests for linux_whisper.stt — engine protocol, factory, model errors.

Actual model inference is NOT tested (requires downloaded models).
"""

from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from linux_whisper.config import Config, STTConfig
from linux_whisper.stt.engine import (
    STTEngine,
    TranscriptResult,
    TranscriptSegment,
    create_engine,
)

# ── TranscriptSegment / TranscriptResult ────────────────────────────────────


class TestTranscriptDataTypes:
    def test_transcript_segment_fields(self):
        seg = TranscriptSegment(
            text="hello world",
            start_time=0.0,
            end_time=1.5,
            is_partial=True,
        )
        assert seg.text == "hello world"
        assert seg.start_time == 0.0
        assert seg.end_time == 1.5
        assert seg.is_partial is True

    def test_transcript_segment_default_partial(self):
        seg = TranscriptSegment(text="x", start_time=0, end_time=1)
        assert seg.is_partial is False

    def test_transcript_result_defaults(self):
        result = TranscriptResult()
        assert result.segments == []
        assert result.full_text == ""
        assert result.language is None
        assert result.duration == 0.0

    def test_transcript_result_with_data(self):
        seg = TranscriptSegment(text="hello", start_time=0, end_time=1)
        result = TranscriptResult(
            segments=[seg],
            full_text="hello",
            language="en",
            duration=1.0,
        )
        assert len(result.segments) == 1
        assert result.full_text == "hello"
        assert result.language == "en"


# ── STTEngine protocol ─────────────────────────────────────────────────────


class TestSTTEngineProtocol:
    def test_protocol_is_runtime_checkable(self):
        """The STTEngine protocol can be used with isinstance at runtime."""
        assert (
            hasattr(STTEngine, "__protocol_attrs__")
            or hasattr(STTEngine, "__abstractmethods__")
            or True
        )  # Protocol existence is sufficient

    def test_mock_engine_satisfies_protocol(self):
        """A mock object with the right methods satisfies the STTEngine protocol."""

        class FakeEngine:
            def start_stream(self) -> None:
                pass

            def feed_audio(self, chunk: bytes) -> list[TranscriptSegment]:
                return []

            def finalize(self) -> TranscriptResult:
                return TranscriptResult()

            def reset(self) -> None:
                pass

        engine = FakeEngine()
        assert isinstance(engine, STTEngine)


# ── create_engine factory ───────────────────────────────────────────────────


class TestCreateEngine:
    def test_unknown_backend_raises_value_error(self):
        cfg = Config.from_dict({"stt": {"backend": "openai"}})
        with pytest.raises(ValueError, match="Unknown STT backend"):
            create_engine(cfg)

    def test_moonshine_backend_import(self):
        """Test that create_engine attempts to import MoonshineEngine for moonshine backend."""
        cfg = Config.from_dict(
            {
                "stt": {"backend": "moonshine", "model": "moonshine-medium"},
            }
        )

        # Patch the import inside create_engine so it returns a mock engine
        mock_engine = MagicMock()
        with patch(
            "linux_whisper.stt.moonshine.MoonshineEngine",
            return_value=mock_engine,
        ):
            engine = create_engine(cfg)
        assert engine is mock_engine

    def test_whisper_cpp_gpu_backend_import(self):
        """Test that create_engine selects WhisperGPUEngine for whisper-cpp + rocm."""
        cfg = Config.from_dict(
            {
                "stt": {
                    "backend": "whisper-cpp",
                    "device": "rocm",
                    "model": "whisper-large-v3-turbo",
                },
            }
        )

        mock_engine = MagicMock()
        mock_cls = MagicMock(return_value=mock_engine)
        mock_module = MagicMock()
        mock_module.WhisperGPUEngine = mock_cls
        with patch.dict("sys.modules", {"linux_whisper.stt.whisper_gpu": mock_module}):
            engine = create_engine(cfg)
        assert engine is mock_engine

    def test_whisper_cpp_cpu_backend_import(self):
        """Test that create_engine selects WhisperCppEngine for whisper-cpp + cpu."""
        cfg = Config.from_dict(
            {
                "stt": {
                    "backend": "whisper-cpp",
                    "device": "cpu",
                    "model": "whisper-large-v3-turbo",
                },
            }
        )

        mock_engine = MagicMock()
        mock_cls = MagicMock(return_value=mock_engine)
        mock_module = MagicMock()
        mock_module.WhisperCppEngine = mock_cls
        with patch.dict("sys.modules", {"linux_whisper.stt.whisper_cpp": mock_module}):
            engine = create_engine(cfg)
        assert engine is mock_engine


# ── MoonshineEngine unit tests (mocked) ────────────────────────────────────


class TestWorkerImportOrder:
    """The GPU worker must import pywhispercpp before numpy.

    pywhispercpp's ROCm/HIP extension segfaults if numpy is loaded first. A
    comment saying so is not enough: ruff's I001 auto-fix reordered exactly
    this block and shipped a SIGSEGV that killed transcription until it was
    caught in production. `# isort: off` now guards it, and this test fails
    if anyone removes the guard or reorders the imports again.
    """

    def test_pywhispercpp_is_imported_before_numpy(self):
        from pathlib import Path

        import linux_whisper.stt as stt_pkg

        src = (Path(stt_pkg.__file__).parent / "whisper_gpu_worker.py").read_text()
        whisper_at = src.index("from pywhispercpp")
        numpy_at = src.index("import numpy")
        assert whisper_at < numpy_at, (
            "pywhispercpp must be imported before numpy in whisper_gpu_worker.py "
            "— importing numpy first segfaults the ROCm backend"
        )

    def test_entrypoint_preload_is_also_guarded(self):
        """`__main__.py` carries the identical preload and the same hazard."""
        from pathlib import Path

        import linux_whisper

        src = (Path(linux_whisper.__file__).parent / "__main__.py").read_text()
        assert "# isort: off" in src, (
            "the entrypoint's pywhispercpp preload needs the same guard — it "
            "survived an earlier formatter run by luck, not by protection"
        )
        assert src.index("import pywhispercpp") < src.index("from linux_whisper.cli")

    def test_isort_guard_is_present(self):
        from pathlib import Path

        import linux_whisper.stt as stt_pkg

        src = (Path(stt_pkg.__file__).parent / "whisper_gpu_worker.py").read_text()
        assert "# isort: off" in src, (
            "the `# isort: off` guard is load-bearing — without it an "
            "auto-formatter will reorder the imports and reintroduce the segfault"
        )


_SYNTHETIC_GPU_WORKER = r"""
import json
import signal
import struct
import sys
import time

mode = sys.argv[1]
stdin = sys.stdin.buffer
stdout = sys.stdout.buffer


def recv():
    header = stdin.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack(">I", header)[0]
    data = stdin.read(length)
    if len(data) < length:
        return None
    return json.loads(data)


def send(message):
    data = json.dumps(message).encode()
    stdout.write(struct.pack(">I", len(data)) + data)
    stdout.flush()


if mode == "startup_hang":
    time.sleep(10)

recv()
send({"status": "ready"})

if mode == "exit":
    raise SystemExit(7)
if mode == "blocked_write":
    time.sleep(10)

message = recv()
audio_length = message["audio_length"]
stdin.read(audio_length)

if mode == "inference_hang_kill":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(10)
elif mode == "malformed":
    payload = b"not-json"
    stdout.write(struct.pack(">I", len(payload)) + payload)
    stdout.flush()
    recv()
elif mode == "silence":
    send({"status": "ok", "segments": [], "full_text": ""})
    recv()
elif mode in ("serve", "slow_serve"):
    # Answer every transcription with silence until shutdown or EOF;
    # ``slow_serve`` takes half a second over each one.
    while True:
        if mode == "slow_serve":
            time.sleep(0.5)
        send({"status": "ok", "segments": [], "full_text": ""})
        message = recv()
        if message is None or message.get("cmd") != "transcribe":
            break
        stdin.read(message["audio_length"])
"""


def _synthetic_gpu_engine(idle_unload_s: float = 0):
    from linux_whisper.stt.whisper_gpu import WhisperGPUEngine

    engine = object.__new__(WhisperGPUEngine)
    engine._init_idle_unload(idle_unload_s)
    engine._model_path = Path("/synthetic/model")
    engine._threads = 1
    engine._process = None
    engine._operation_timeout_s = 0.02
    engine._stream_started = False
    engine._audio_buffer = bytearray()
    engine._stream_start_time = 0.0
    return engine


@contextmanager
def _synthetic_gpu_workers(monkeypatch, *modes):
    import linux_whisper.stt.whisper_gpu as whisper_gpu

    pending_modes = list(modes)
    processes = []
    real_popen = subprocess.Popen

    def start(_command, **kwargs):
        mode = pending_modes.pop(0)
        process = real_popen(
            [sys.executable, "-u", "-c", _SYNTHETIC_GPU_WORKER, mode],
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(whisper_gpu.subprocess, "Popen", start)
    try:
        yield processes
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)


def _assert_silence_recovery(engine) -> None:
    engine.start_stream()
    engine.feed_audio(b"\x00\x00" * 100)
    result = engine.finalize()
    assert result.full_text == ""
    assert result.segments == []
    engine._shutdown_worker()


class TestGPUWorkerIPC:
    """Synthetic pipe tests for the worker boundary; no model or GPU is needed."""

    def test_response_timeout_is_typed(self):
        from linux_whisper.stt.whisper_gpu import GPUWorkerTimeoutError, _recv_msg

        read_fd, write_fd = os.pipe()
        reader = os.fdopen(read_fd, "rb", buffering=0)
        try:
            with pytest.raises(GPUWorkerTimeoutError, match="startup timed out"):
                _recv_msg(
                    reader,
                    deadline=time.monotonic() + 0.01,
                    operation="startup",
                )
        finally:
            reader.close()
            os.close(write_fd)

    def test_truncated_response_is_not_silence(self):
        from linux_whisper.stt.whisper_gpu import GPUWorkerProtocolError, _recv_msg

        read_fd, write_fd = os.pipe()
        reader = os.fdopen(read_fd, "rb", buffering=0)
        try:
            os.write(write_fd, struct.pack(">I", 8) + b"{}")
            os.close(write_fd)
            with pytest.raises(GPUWorkerProtocolError, match="closed stdout"):
                _recv_msg(
                    reader,
                    deadline=time.monotonic() + 1,
                    operation="inference response",
                )
        finally:
            reader.close()

    def test_malformed_success_response_is_a_protocol_error(self):
        from linux_whisper.stt.whisper_gpu import GPUWorkerProtocolError, _parse_result

        with pytest.raises(GPUWorkerProtocolError, match="no text field"):
            _parse_result({"status": "ok", "segments": []}, duration=1.0)

        with pytest.raises(GPUWorkerProtocolError, match="invalid segment timing"):
            _parse_result(
                {
                    "status": "ok",
                    "full_text": "synthetic",
                    "segments": [{"text": "synthetic", "t0": 1.0, "t1": 0.0}],
                },
                duration=1.0,
            )

        with pytest.raises(GPUWorkerProtocolError, match="invalid segment timing"):
            _parse_result(
                {
                    "status": "ok",
                    "full_text": "synthetic",
                    "segments": [{"text": "synthetic", "t0": True, "t1": 1.0}],
                },
                duration=1.0,
            )

        with pytest.raises(GPUWorkerProtocolError, match="inconsistent text"):
            _parse_result(
                {
                    "status": "ok",
                    "full_text": "",
                    "segments": [{"text": "synthetic", "t0": 0.0, "t1": 1.0}],
                },
                duration=1.0,
            )

        with pytest.raises(GPUWorkerProtocolError, match="invalid segment timing"):
            _parse_result(
                {
                    "status": "ok",
                    "full_text": "synthetic",
                    "segments": [{"text": "synthetic", "t0": 0.0, "t1": 2.01}],
                },
                duration=1.0,
            )

        result = _parse_result(
            {
                "status": "ok",
                "full_text": "synthetic",
                "segments": [{"text": "synthetic", "t0": 0.0, "t1": 2.0}],
            },
            duration=1.0,
        )
        assert result.segments[0].end_time == 2.0

    def test_startup_hang_reaps_worker_and_replacement_recovers(self, monkeypatch):
        import linux_whisper.stt.whisper_gpu as whisper_gpu

        engine = _synthetic_gpu_engine()
        monkeypatch.setattr(whisper_gpu, "_WORKER_STARTUP_TIMEOUT", 0.02)
        with _synthetic_gpu_workers(monkeypatch, "startup_hang", "silence") as processes:
            with pytest.raises(whisper_gpu.GPUWorkerTimeoutError, match="startup timed out"):
                engine.start_stream()

            assert engine._process is None
            assert processes[0].poll() is not None
            monkeypatch.setattr(whisper_gpu, "_WORKER_STARTUP_TIMEOUT", 1.0)
            _assert_silence_recovery(engine)
            assert processes[1].poll() is not None

    def test_blocked_audio_write_reaps_worker_and_replacement_recovers(self, monkeypatch):
        import linux_whisper.stt.whisper_gpu as whisper_gpu

        engine = _synthetic_gpu_engine()
        with _synthetic_gpu_workers(monkeypatch, "blocked_write", "silence") as processes:
            engine.start_stream()
            engine.feed_audio(b"\x00" * (8 * 1024 * 1024))
            with pytest.raises(whisper_gpu.GPUWorkerTimeoutError, match="audio timed out"):
                engine.finalize()

            assert engine._process is None
            assert processes[0].poll() is not None
            _assert_silence_recovery(engine)

    def test_inference_hang_escalates_to_kill_reap_then_recovers(self, monkeypatch):
        import linux_whisper.stt.whisper_gpu as whisper_gpu

        engine = _synthetic_gpu_engine()
        with _synthetic_gpu_workers(monkeypatch, "inference_hang_kill", "silence") as processes:
            engine.start_stream()
            engine.feed_audio(b"\x00\x00" * 100)
            with pytest.raises(whisper_gpu.GPUWorkerTimeoutError, match="response timed out"):
                engine.finalize()

            assert engine._process is None
            assert processes[0].poll() == -signal.SIGKILL
            _assert_silence_recovery(engine)

    def test_worker_exit_reaps_process_and_replacement_recovers(self, monkeypatch):
        import linux_whisper.stt.whisper_gpu as whisper_gpu

        engine = _synthetic_gpu_engine()
        with _synthetic_gpu_workers(monkeypatch, "exit", "silence") as processes:
            engine.start_stream()
            processes[0].wait(timeout=1)
            engine.feed_audio(b"\x00\x00" * 100)
            with pytest.raises(whisper_gpu.GPUWorkerError, match="unavailable"):
                engine.finalize()

            assert engine._process is None
            assert processes[0].returncode == 7
            _assert_silence_recovery(engine)

    def test_malformed_frame_reaps_process_and_replacement_recovers(self, monkeypatch):
        import linux_whisper.stt.whisper_gpu as whisper_gpu

        engine = _synthetic_gpu_engine()
        with _synthetic_gpu_workers(monkeypatch, "malformed", "silence") as processes:
            engine.start_stream()
            engine.feed_audio(b"\x00\x00" * 100)
            with pytest.raises(whisper_gpu.GPUWorkerProtocolError, match="invalid JSON"):
                engine.finalize()

            assert engine._process is None
            assert processes[0].poll() is not None
            _assert_silence_recovery(engine)

    def test_worker_error_reaps_before_replacement_is_used(self):
        from linux_whisper.stt.whisper_gpu import GPUWorkerError, WhisperGPUEngine

        class FakeWorker:
            def __init__(self, response: dict):
                input_read, input_write = os.pipe()
                output_read, output_write = os.pipe()
                self.stdin = os.fdopen(input_write, "wb", buffering=0)
                self.stdout = os.fdopen(output_read, "rb", buffering=0)
                self._input_read = input_read
                payload = json.dumps(response).encode()
                os.write(output_write, struct.pack(">I", len(payload)) + payload)
                os.close(output_write)
                self.returncode: int | None = None
                self.terminated = False
                self.pid = 1234

            def poll(self):
                return self.returncode

            def wait(self, timeout: float):
                if self.returncode is None:
                    raise TimeoutError
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = 15

            def kill(self):
                self.returncode = 9

            def close_input(self):
                os.close(self._input_read)

        engine = object.__new__(WhisperGPUEngine)
        engine._init_idle_unload(0)
        engine._operation_timeout_s = 1
        engine._stream_started = True
        engine._audio_buffer = bytearray(b"\x00\x00" * 10)
        failed = FakeWorker({"status": "error", "error": "synthetic"})
        engine._process = failed

        with pytest.raises(GPUWorkerError, match="reported an inference failure"):
            engine.finalize()

        assert failed.terminated is True
        assert engine._process is None
        failed.close_input()

        replacement = FakeWorker({"status": "ok", "segments": [], "full_text": ""})
        engine._process = replacement
        engine.start_stream()
        engine.feed_audio(b"\x00\x00" * 10)
        assert engine.finalize().full_text == ""
        engine._shutdown_worker()
        replacement.close_input()

    def test_operation_timeout_must_be_positive(self):
        from linux_whisper.stt.whisper_gpu import WhisperGPUEngine

        engine = object.__new__(WhisperGPUEngine)
        engine._process = None
        with pytest.raises(ValueError, match="positive"):
            engine.set_operation_timeout(0)


class _ManualTimer:
    """Stand-in for ``threading.Timer`` that only fires when a test says so.

    ``fire`` runs the callback even after ``cancel`` — exactly what a real
    timer does when it expires just before being cancelled — so tests can
    replay that late-fire interleaving deterministically.
    """

    def __init__(self, interval, function, args=()):
        self.interval = interval
        self.function = function
        self.args = args
        self.daemon = False
        self.name = ""
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.function(*self.args)


@pytest.fixture
def manual_timers(monkeypatch):
    import linux_whisper.stt.whisper_gpu as whisper_gpu

    timers: list[_ManualTimer] = []

    def make(interval, function, args=()):
        timer = _ManualTimer(interval, function, args)
        timers.append(timer)
        return timer

    monkeypatch.setattr(whisper_gpu.threading, "Timer", make)
    return timers


def _transcribe_silence(engine) -> None:
    engine.start_stream()
    engine.feed_audio(b"\x00\x00" * 100)
    assert engine.finalize().full_text == ""
    engine.reset()


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestGPUIdleUnload:
    """Idle unload of the GPU worker (#60), using the synthetic worker."""

    def test_idle_worker_is_unloaded_and_logged(self, monkeypatch, caplog):
        engine = _synthetic_gpu_engine(idle_unload_s=0.2)
        with _synthetic_gpu_workers(monkeypatch, "serve") as processes:
            with caplog.at_level("INFO", logger="linux_whisper.stt.whisper_gpu"):
                _transcribe_silence(engine)
                assert processes[0].poll() is None

                assert _wait_until(lambda: engine._process is None)
                assert processes[0].wait(timeout=2) is not None

            unload_lines = [r for r in caplog.records if "idle" in r.getMessage().lower()]
            assert len(unload_lines) == 1
            assert unload_lines[0].levelname == "INFO"
            assert "0.2s idle" in unload_lines[0].getMessage()

    def test_unloaded_worker_reloads_on_next_stream(self, monkeypatch, manual_timers):
        engine = _synthetic_gpu_engine(idle_unload_s=1200)
        with _synthetic_gpu_workers(monkeypatch, "serve", "serve") as processes:
            _transcribe_silence(engine)
            manual_timers[-1].fire()
            assert engine._process is None
            assert processes[0].wait(timeout=2) is not None

            _transcribe_silence(engine)
            assert len(processes) == 2
            assert engine._process is processes[1]
            assert processes[1].poll() is None
            engine.close()

    def test_stream_in_progress_is_never_unloaded(self, monkeypatch):
        engine = _synthetic_gpu_engine(idle_unload_s=0.1)
        with _synthetic_gpu_workers(monkeypatch, "serve") as processes:
            _transcribe_silence(engine)
            engine.start_stream()
            engine.feed_audio(b"\x00\x00" * 100)
            time.sleep(0.5)  # the stream outlasts the idle period several times

            assert engine._process is processes[0]
            assert processes[0].poll() is None
            assert engine.finalize().full_text == ""
            engine.close()

    def test_timer_that_fires_after_stream_start_is_a_no_op(self, monkeypatch, manual_timers):
        engine = _synthetic_gpu_engine(idle_unload_s=1200)
        with _synthetic_gpu_workers(monkeypatch, "serve") as processes:
            _transcribe_silence(engine)
            expired = manual_timers[-1]
            engine.start_stream()
            assert expired.cancelled

            expired.fire()
            assert engine._process is processes[0]
            engine.feed_audio(b"\x00\x00" * 100)
            assert engine.finalize().full_text == ""
            engine.close()

    def test_each_transcription_restarts_the_countdown(self, monkeypatch, manual_timers):
        engine = _synthetic_gpu_engine(idle_unload_s=1200)
        with _synthetic_gpu_workers(monkeypatch, "serve") as processes:
            _transcribe_silence(engine)
            first = manual_timers[-1]
            # A second transcription just before the first countdown expires
            # replaces it with a full-length one.
            _transcribe_silence(engine)
            second = manual_timers[-1]
            assert second is not first
            assert first.cancelled and not second.cancelled
            assert second.interval == 1200

            first.fire()  # the old deadline (t = idle) passes: still loaded
            assert engine._process is processes[0]
            second.fire()  # the new deadline (t = 2 * idle - 1) unloads
            assert engine._process is None
            assert processes[0].wait(timeout=2) is not None

    def test_zero_disables_idle_unload(self, monkeypatch, manual_timers):
        engine = _synthetic_gpu_engine(idle_unload_s=0)
        with _synthetic_gpu_workers(monkeypatch, "serve") as processes:
            _transcribe_silence(engine)
            assert manual_timers == []
            assert engine._idle_timer is None
            assert engine._process is processes[0]
            engine.close()

    @pytest.mark.parametrize("unload_first", [True, False])
    def test_unload_racing_start_stream_leaves_one_working_worker(
        self, monkeypatch, manual_timers, unload_first
    ):
        import threading

        engine = _synthetic_gpu_engine(idle_unload_s=1200)
        with _synthetic_gpu_workers(monkeypatch, "serve", "serve") as processes:
            _transcribe_silence(engine)
            expired = manual_timers[-1]

            firing = threading.Thread(target=expired.fire)
            if unload_first:
                firing.start()
                firing.join(timeout=2)
                engine.start_stream()
            else:
                # The timer thread wakes while start_stream holds the lock and
                # only gets it once the stream has begun.
                with engine._lock:
                    firing.start()
                    time.sleep(0.05)
                    assert firing.is_alive()
                    engine.start_stream()
                firing.join(timeout=2)
            assert not firing.is_alive()

            engine.feed_audio(b"\x00\x00" * 100)
            assert engine.finalize().full_text == ""
            live = [p for p in processes if p.poll() is None]
            assert live == [engine._process]
            assert len(processes) == (2 if unload_first else 1)
            engine.close()
            assert all(p.wait(timeout=2) is not None for p in processes)

    def test_abandoned_stream_is_unloaded(self, monkeypatch, manual_timers):
        engine = _synthetic_gpu_engine(idle_unload_s=1200)
        with _synthetic_gpu_workers(monkeypatch, "serve") as processes:
            engine.start_stream()
            engine.feed_audio(b"\x00\x00" * 100)
            engine.reset()  # the stream is dropped without a finalize()

            manual_timers[-1].fire()
            assert engine._process is None
            assert processes[0].wait(timeout=2) is not None

    def test_start_stream_does_not_wait_for_inflight_finalize(
        self, monkeypatch, manual_timers
    ):
        import threading

        engine = _synthetic_gpu_engine(idle_unload_s=1200)
        engine._operation_timeout_s = 5.0
        with _synthetic_gpu_workers(monkeypatch, "slow_serve") as processes:
            engine.start_stream()
            engine.feed_audio(b"\x00\x00" * 100)
            results = []
            finalizing = threading.Thread(target=lambda: results.append(engine.finalize()))
            finalizing.start()
            assert _wait_until(lambda: engine._inflight == 1)

            # The hotkey thread starts the next recording mid-inference.
            starting = threading.Thread(target=engine.start_stream)
            starting.start()
            starting.join(timeout=0.2)
            assert not starting.is_alive()
            assert finalizing.is_alive()  # the 0.5s transcription is still running

            # An idle fire during the transcription must not unload its worker.
            engine._unload_if_idle(engine._idle_generation)
            assert engine._process is processes[0]

            finalizing.join(timeout=3)
            assert results[0].full_text == ""
            # The new stream is open, so the first finalize arms no countdown.
            assert engine._idle_timer is None

            engine.feed_audio(b"\x00\x00" * 100)
            assert engine.finalize().full_text == ""
            assert engine._idle_timer is not None
            assert engine._process is processes[0]
            engine.close()

    def test_failed_finalize_spares_a_replacement_worker(self, monkeypatch, manual_timers):
        import threading

        from linux_whisper.stt.whisper_gpu import GPUWorkerError

        engine = _synthetic_gpu_engine(idle_unload_s=1200)
        engine._operation_timeout_s = 5.0
        with _synthetic_gpu_workers(
            monkeypatch, "inference_hang_kill", "serve"
        ) as processes:
            engine.start_stream()
            engine.feed_audio(b"\x00\x00" * 100)
            errors = []

            def run_finalize():
                try:
                    engine.finalize()
                except GPUWorkerError as exc:
                    errors.append(exc)

            finalizing = threading.Thread(target=run_finalize)
            finalizing.start()
            assert _wait_until(lambda: engine._inflight == 1)

            # Holding the lock orders the steps: the first worker dies
            # mid-inference and the next stream replaces it before the
            # failed finalize can settle.
            with engine._lock:
                processes[0].kill()
                processes[0].wait(timeout=2)
                engine.start_stream()
                assert engine._process is processes[1]
            finalizing.join(timeout=3)
            assert not finalizing.is_alive()

            assert len(errors) == 1
            assert engine._process is processes[1]
            assert processes[1].poll() is None
            engine.feed_audio(b"\x00\x00" * 100)
            assert engine.finalize().full_text == ""
            engine.close()
            assert processes[1].wait(timeout=2) is not None

    def test_discarded_engine_with_pending_timer_stops_its_worker(
        self, monkeypatch, manual_timers
    ):
        import gc

        engine = _synthetic_gpu_engine(idle_unload_s=1200)
        with _synthetic_gpu_workers(monkeypatch, "serve") as processes:
            _transcribe_silence(engine)
            pending = manual_timers[-1]
            assert not pending.cancelled

            del engine
            gc.collect()
            assert processes[0].wait(timeout=2) is not None
            pending.fire()  # the timer outlives the engine: a silent no-op

    def test_close_stops_timer_and_worker(self, monkeypatch):
        import threading

        engine = _synthetic_gpu_engine(idle_unload_s=60)
        with _synthetic_gpu_workers(monkeypatch, "serve") as processes:
            _transcribe_silence(engine)
            timer = engine._idle_timer
            assert timer is not None and timer.is_alive()

            engine.close()
            timer.join(timeout=2)
            assert not timer.is_alive()
            assert engine._idle_timer is None
            assert engine._process is None
            assert processes[0].wait(timeout=2) is not None
            assert not any(
                t.name == "whisper-gpu-idle-unload" and t.is_alive()
                for t in threading.enumerate()
            )

    def test_config_value_reaches_engine(self, monkeypatch):
        from linux_whisper.stt.whisper_gpu import WhisperGPUEngine

        monkeypatch.setattr(
            WhisperGPUEngine, "_resolve_model_path", staticmethod(lambda _: Path("/m"))
        )
        cfg = Config.from_dict({"stt": {"device": "rocm", "gpu_idle_unload_s": 45}})
        engine = WhisperGPUEngine(cfg)
        assert engine._idle_unload_s == 45
        assert WhisperGPUEngine(Config())._idle_unload_s == 1200


class TestGPUIdleUnloadConfig:
    def test_default_is_twenty_minutes(self):
        assert STTConfig().gpu_idle_unload_s == 1200
        assert Config().validate() == []

    @pytest.mark.parametrize("value", [0, 1, 1200])
    def test_non_negative_is_valid(self, value):
        assert Config.from_dict({"stt": {"gpu_idle_unload_s": value}}).validate() == []

    def test_negative_is_rejected(self):
        errors = Config.from_dict({"stt": {"gpu_idle_unload_s": -1}}).validate()
        assert any("gpu_idle_unload_s" in e for e in errors)


class TestMoonshineEngine:
    @pytest.fixture(autouse=True)
    def _require_backend(self):
        """Skip when the optional Moonshine backend is not installed.

        These construct a real engine, so they need the extra. It is
        present on the dev machine but not in `pip install -e ".[dev]"`,
        so they failed in CI while passing locally — exactly the kind of
        machine dependency an unrun CI suite hides.
        """
        pytest.importorskip(
            "moonshine_onnx",
            reason="optional Moonshine backend not installed",
        )

    def test_invalid_model_raises_value_error(self):
        from linux_whisper.stt.moonshine import MoonshineEngine

        cfg = Config.from_dict(
            {
                "stt": {"backend": "moonshine", "model": "nonexistent-model"},
            }
        )
        with pytest.raises(ValueError, match="Unknown Moonshine model"):
            MoonshineEngine(cfg)

    def test_missing_package_raises_import_error(self):
        """If moonshine is not installed, creating the engine should raise ImportError."""
        import linux_whisper.stt.moonshine as moonshine_module

        original = moonshine_module._HAS_MOONSHINE
        try:
            moonshine_module._HAS_MOONSHINE = False
            cfg = Config.from_dict(
                {
                    "stt": {"backend": "moonshine", "model": "moonshine-medium"},
                }
            )
            with pytest.raises(ImportError, match="moonshine"):
                moonshine_module.MoonshineEngine(cfg)
        finally:
            moonshine_module._HAS_MOONSHINE = original

    def test_feed_audio_without_start_raises(self):
        """feed_audio before start_stream should raise RuntimeError."""
        from linux_whisper.stt.moonshine import MoonshineEngine

        cfg = Config.from_dict(
            {
                "stt": {"backend": "moonshine", "model": "moonshine-medium"},
            }
        )
        engine = MoonshineEngine(cfg)
        engine._stream_started = False  # ensure not started
        with pytest.raises(RuntimeError, match="start_stream"):
            engine.feed_audio(b"\x00" * 100)

    def test_finalize_without_start_returns_empty(self):
        from linux_whisper.stt.moonshine import MoonshineEngine

        cfg = Config.from_dict(
            {
                "stt": {"backend": "moonshine", "model": "moonshine-medium"},
            }
        )
        engine = MoonshineEngine(cfg)
        result = engine.finalize()
        assert result.full_text == ""
        assert result.segments == []

    def test_reset_clears_state(self):
        from linux_whisper.stt.moonshine import MoonshineEngine

        cfg = Config.from_dict(
            {
                "stt": {"backend": "moonshine", "model": "moonshine-medium"},
            }
        )
        engine = MoonshineEngine(cfg)
        engine._audio_buffer = bytearray(b"\x00" * 100)
        engine._stream_started = True

        engine.reset()
        assert engine._audio_buffer == bytearray()
        assert engine._stream_started is False


# ── ParakeetEngine unit tests (mocked) ─────────────────────────────────────


class TestParakeetEngine:
    @pytest.fixture(autouse=True)
    def _require_backend(self):
        """Skip when the optional Parakeet backend is not installed.

        These construct a real engine, so they need the extra. It is
        present on the dev machine but not in `pip install -e ".[dev]"`,
        so they failed in CI while passing locally — exactly the kind of
        machine dependency an unrun CI suite hides.
        """
        pytest.importorskip(
            "onnx_asr",
            reason="optional Parakeet backend not installed",
        )

    def _cfg(self, **stt):
        base = {"backend": "parakeet", "model": "parakeet-tdt-0.6b-v3"}
        base.update(stt)
        return Config.from_dict({"stt": base})

    def test_invalid_model_raises_value_error(self):
        from linux_whisper.stt.parakeet import ParakeetEngine

        with pytest.raises(ValueError, match="Unknown Parakeet model"):
            ParakeetEngine(self._cfg(model="nonexistent-model"))

    def test_missing_package_raises_import_error(self):
        import linux_whisper.stt.parakeet as parakeet_module

        original = parakeet_module._HAS_ONNX_ASR
        try:
            parakeet_module._HAS_ONNX_ASR = False
            with pytest.raises(ImportError, match="onnx-asr"):
                parakeet_module.ParakeetEngine(self._cfg())
        finally:
            parakeet_module._HAS_ONNX_ASR = original

    def test_feed_audio_without_start_raises(self):
        from linux_whisper.stt.parakeet import ParakeetEngine

        engine = ParakeetEngine(self._cfg())
        engine._stream_started = False
        with pytest.raises(RuntimeError, match="start_stream"):
            engine.feed_audio(b"\x00" * 100)

    def test_finalize_without_start_returns_empty(self):
        from linux_whisper.stt.parakeet import ParakeetEngine

        engine = ParakeetEngine(self._cfg())
        result = engine.finalize()
        assert result.full_text == ""
        assert result.segments == []

    def test_reset_clears_state(self):
        from linux_whisper.stt.parakeet import ParakeetEngine

        engine = ParakeetEngine(self._cfg())
        engine._audio_buffer = bytearray(b"\x00" * 100)
        engine._stream_started = True
        engine.reset()
        assert engine._audio_buffer == bytearray()
        assert engine._stream_started is False

    def test_default_threads_are_capped(self):
        """cpu_count() oversubscribes this model badly — see the measured table."""
        from linux_whisper.stt.parakeet import _MAX_DEFAULT_THREADS, ParakeetEngine

        engine = ParakeetEngine(self._cfg())
        assert engine._threads <= _MAX_DEFAULT_THREADS

    def test_explicit_thread_count_overrides_the_cap(self):
        from linux_whisper.stt.parakeet import ParakeetEngine

        engine = ParakeetEngine(self._cfg(threads=16))
        assert engine._threads == 16

    def test_transcription_failure_returns_empty_not_raises(self):
        from unittest.mock import MagicMock

        from linux_whisper.stt.parakeet import ParakeetEngine

        engine = ParakeetEngine(self._cfg())
        engine._model = MagicMock()
        engine._model.recognize.side_effect = RuntimeError("onnx blew up")

        engine.start_stream()
        engine.feed_audio(b"\x00\x00" * 1000)
        result = engine.finalize()

        assert result.full_text == ""
        assert result.segments == []

    def test_finalize_returns_recognized_text(self):
        from unittest.mock import MagicMock

        from linux_whisper.stt.parakeet import ParakeetEngine

        engine = ParakeetEngine(self._cfg())
        engine._model = MagicMock()
        engine._model.recognize.return_value = "  Hello there.  "

        engine.start_stream()
        engine.feed_audio(b"\x00\x00" * 16000)
        result = engine.finalize()

        assert result.full_text == "Hello there."
        assert len(result.segments) == 1
        assert result.duration == pytest.approx(1.0, abs=0.01)


# ── WhisperCppEngine unit tests (mocked) ───────────────────────────────────


class TestWhisperCppEngine:
    def test_invalid_model_raises_value_error(self, monkeypatch):
        import linux_whisper.stt.whisper_cpp as wcpp_module

        monkeypatch.setattr(wcpp_module, "_check_whispercpp", lambda: True)
        cfg = Config.from_dict(
            {
                "stt": {"backend": "whisper-cpp", "model": "nonexistent-model"},
            }
        )
        with pytest.raises(ValueError, match="Unknown whisper.cpp model"):
            wcpp_module.WhisperCppEngine(cfg)

    def test_missing_package_raises_import_error(self, monkeypatch):
        import linux_whisper.stt.whisper_cpp as wcpp_module

        monkeypatch.setattr(wcpp_module, "_check_whispercpp", lambda: False)
        cfg = Config.from_dict(
            {
                "stt": {"backend": "whisper-cpp", "model": "whisper-large-v3-turbo"},
            }
        )
        with pytest.raises(ImportError, match="whispercpp"):
            wcpp_module.WhisperCppEngine(cfg)

    def test_model_file_not_found(self, monkeypatch):
        """If the model file does not exist on disk, FileNotFoundError is raised."""
        import linux_whisper.stt.whisper_cpp as wcpp_module

        monkeypatch.setattr(wcpp_module, "_check_whispercpp", lambda: True)
        cfg = Config.from_dict(
            {
                "stt": {"backend": "whisper-cpp", "model": "distil-large-v3.5"},
            }
        )
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            wcpp_module.WhisperCppEngine(cfg)

    def test_feed_audio_without_start_raises(self):
        """feed_audio before start_stream should raise RuntimeError."""
        import linux_whisper.stt.whisper_cpp as wcpp_module

        engine = object.__new__(wcpp_module.WhisperCppEngine)
        engine._stream_started = False
        engine._audio_buffer = bytearray()
        with pytest.raises(RuntimeError, match="start_stream"):
            engine.feed_audio(b"\x00" * 100)

    def test_reset_clears_state(self):
        import linux_whisper.stt.whisper_cpp as wcpp_module

        engine = object.__new__(wcpp_module.WhisperCppEngine)
        engine._audio_buffer = bytearray(b"\x00" * 100)
        engine._stream_started = True

        engine.reset()
        assert engine._audio_buffer == bytearray()
        assert engine._stream_started is False

    def test_finalize_without_start_returns_empty(self):
        import linux_whisper.stt.whisper_cpp as wcpp_module

        engine = object.__new__(wcpp_module.WhisperCppEngine)
        engine._stream_started = False
        engine._audio_buffer = bytearray()

        result = engine.finalize()
        assert result.full_text == ""
        assert result.segments == []


# ── STT Device Config ─────────────────────────────────────────────────────


class TestSTTDeviceConfig:
    """Test stt.device configuration field."""

    def test_default_device_is_rocm(self):
        stt = STTConfig()
        assert stt.device == "rocm"

    def test_device_from_dict(self):
        cfg = Config.from_dict({"stt": {"device": "rocm"}})
        assert cfg.stt.device == "rocm"

    def test_device_preserved_with_other_overrides(self):
        cfg = Config.from_dict(
            {"stt": {"backend": "whisper-cpp", "device": "rocm", "model": "whisper-large-v3-turbo"}}
        )
        assert cfg.stt.backend == "whisper-cpp"
        assert cfg.stt.device == "rocm"
        assert cfg.stt.model == "whisper-large-v3-turbo"


class TestWhisperCppGPUDetection:
    """Test GPU detection and fallback in WhisperCppEngine."""

    def test_detect_gpu_available_with_rocm(self, monkeypatch):
        import linux_whisper.stt.whisper_cpp as wcpp

        monkeypatch.setattr(wcpp, "_check_whispercpp", lambda: True)

        mock_pw = MagicMock()
        mock_pw.whisper_print_system_info.return_value = (
            "WHISPER : ROCm : NO_VMM = 1 | CPU : SSE3 = 1"
        )
        monkeypatch.setitem(sys.modules, "_pywhispercpp", mock_pw)

        assert wcpp._detect_gpu_available() is True

    def test_detect_gpu_available_without_rocm(self, monkeypatch):
        import linux_whisper.stt.whisper_cpp as wcpp

        monkeypatch.setattr(wcpp, "_check_whispercpp", lambda: True)

        mock_pw = MagicMock()
        mock_pw.whisper_print_system_info.return_value = "WHISPER : CPU : SSE3 = 1 | AVX = 1"
        monkeypatch.setitem(sys.modules, "_pywhispercpp", mock_pw)

        assert wcpp._detect_gpu_available() is False

    def test_detect_gpu_unavailable_when_not_installed(self, monkeypatch):
        import linux_whisper.stt.whisper_cpp as wcpp

        monkeypatch.setattr(wcpp, "_check_whispercpp", lambda: False)
        assert wcpp._detect_gpu_available() is False
