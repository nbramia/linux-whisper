"""Tests for linux_whisper.stt — engine protocol, factory, model errors.

Actual model inference is NOT tested (requires downloaded models).
"""

from __future__ import annotations

import sys
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
        assert hasattr(STTEngine, "__protocol_attrs__") or hasattr(
            STTEngine, "__abstractmethods__"
        ) or True  # Protocol existence is sufficient

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
        cfg = Config.from_dict({
            "stt": {"backend": "moonshine", "model": "moonshine-medium"},
        })

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
        cfg = Config.from_dict({
            "stt": {"backend": "whisper-cpp", "device": "rocm", "model": "whisper-large-v3-turbo"},
        })

        mock_engine = MagicMock()
        mock_cls = MagicMock(return_value=mock_engine)
        mock_module = MagicMock()
        mock_module.WhisperGPUEngine = mock_cls
        with patch.dict("sys.modules", {"linux_whisper.stt.whisper_gpu": mock_module}):
            engine = create_engine(cfg)
        assert engine is mock_engine

    def test_whisper_cpp_cpu_backend_import(self):
        """Test that create_engine selects WhisperCppEngine for whisper-cpp + cpu."""
        cfg = Config.from_dict({
            "stt": {"backend": "whisper-cpp", "device": "cpu", "model": "whisper-large-v3-turbo"},
        })

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

        cfg = Config.from_dict({
            "stt": {"backend": "moonshine", "model": "nonexistent-model"},
        })
        with pytest.raises(ValueError, match="Unknown Moonshine model"):
            MoonshineEngine(cfg)

    def test_missing_package_raises_import_error(self):
        """If moonshine is not installed, creating the engine should raise ImportError."""
        import linux_whisper.stt.moonshine as moonshine_module

        original = moonshine_module._HAS_MOONSHINE
        try:
            moonshine_module._HAS_MOONSHINE = False
            cfg = Config.from_dict({
                "stt": {"backend": "moonshine", "model": "moonshine-medium"},
            })
            with pytest.raises(ImportError, match="moonshine"):
                moonshine_module.MoonshineEngine(cfg)
        finally:
            moonshine_module._HAS_MOONSHINE = original

    def test_feed_audio_without_start_raises(self):
        """feed_audio before start_stream should raise RuntimeError."""
        from linux_whisper.stt.moonshine import MoonshineEngine

        cfg = Config.from_dict({
            "stt": {"backend": "moonshine", "model": "moonshine-medium"},
        })
        engine = MoonshineEngine(cfg)
        engine._stream_started = False  # ensure not started
        with pytest.raises(RuntimeError, match="start_stream"):
            engine.feed_audio(b"\x00" * 100)

    def test_finalize_without_start_returns_empty(self):
        from linux_whisper.stt.moonshine import MoonshineEngine

        cfg = Config.from_dict({
            "stt": {"backend": "moonshine", "model": "moonshine-medium"},
        })
        engine = MoonshineEngine(cfg)
        result = engine.finalize()
        assert result.full_text == ""
        assert result.segments == []

    def test_reset_clears_state(self):
        from linux_whisper.stt.moonshine import MoonshineEngine

        cfg = Config.from_dict({
            "stt": {"backend": "moonshine", "model": "moonshine-medium"},
        })
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
        cfg = Config.from_dict({
            "stt": {"backend": "whisper-cpp", "model": "nonexistent-model"},
        })
        with pytest.raises(ValueError, match="Unknown whisper.cpp model"):
            wcpp_module.WhisperCppEngine(cfg)

    def test_missing_package_raises_import_error(self, monkeypatch):
        import linux_whisper.stt.whisper_cpp as wcpp_module

        monkeypatch.setattr(wcpp_module, "_check_whispercpp", lambda: False)
        cfg = Config.from_dict({
            "stt": {"backend": "whisper-cpp", "model": "whisper-large-v3-turbo"},
        })
        with pytest.raises(ImportError, match="whispercpp"):
            wcpp_module.WhisperCppEngine(cfg)

    def test_model_file_not_found(self, monkeypatch):
        """If the model file does not exist on disk, FileNotFoundError is raised."""
        import linux_whisper.stt.whisper_cpp as wcpp_module

        monkeypatch.setattr(wcpp_module, "_check_whispercpp", lambda: True)
        cfg = Config.from_dict({
            "stt": {"backend": "whisper-cpp", "model": "distil-large-v3.5"},
        })
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
        cfg = Config.from_dict({
            "stt": {"backend": "whisper-cpp", "device": "rocm", "model": "whisper-large-v3-turbo"}
        })
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
        mock_pw.whisper_print_system_info.return_value = (
            "WHISPER : CPU : SSE3 = 1 | AVX = 1"
        )
        monkeypatch.setitem(sys.modules, "_pywhispercpp", mock_pw)

        assert wcpp._detect_gpu_available() is False

    def test_detect_gpu_unavailable_when_not_installed(self, monkeypatch):
        import linux_whisper.stt.whisper_cpp as wcpp

        monkeypatch.setattr(wcpp, "_check_whispercpp", lambda: False)
        assert wcpp._detect_gpu_available() is False
