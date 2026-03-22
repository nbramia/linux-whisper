"""Shared pytest fixtures for the Linux Whisper test suite.

Provides helpers for temporary config files, mock audio data, and
import-guarded fixtures that work even when optional dependencies
(evdev, sounddevice, pystray, moonshine, llama_cpp, onnxruntime)
are not installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Ensure optional heavy dependencies can be faked during test collection.
# We insert lightweight stubs into sys.modules BEFORE any linux_whisper code
# tries to import them, so that import-time guards (`try: import X`) see a
# usable (mock) module rather than raising ImportError.
# ---------------------------------------------------------------------------

_OPTIONAL_DEPS = [
    "evdev",
    "sounddevice",
    "pystray",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "faster-whisper",
    "llama_cpp",
    "whispercpp",
    # onnxruntime is used in disfluency/punctuation but we test regex fallback
    # so we do NOT stub it here — the code's own try/except handles ImportError.
]


def _ensure_stub(name: str) -> None:
    """Insert a MagicMock into sys.modules if the real package is missing."""
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = MagicMock()


for _dep in _OPTIONAL_DEPS:
    _ensure_stub(_dep)

# After stubbing, make sure evdev has the constants the hotkey module needs
_evdev_stub = sys.modules.get("evdev")
if isinstance(_evdev_stub, MagicMock):
    # Create a proper ecodes sub-attribute with real integer constants
    ecodes = types.SimpleNamespace(
        EV_KEY=1,
        KEY_LEFTCTRL=29,
        KEY_RIGHTCTRL=97,
        KEY_LEFTSHIFT=42,
        KEY_RIGHTSHIFT=54,
        KEY_LEFTALT=56,
        KEY_RIGHTALT=100,
        KEY_LEFTMETA=125,
        KEY_RIGHTMETA=126,
        KEY_A=30,
        KEY_B=48,
        KEY_E=18,
        KEY_F1=59,
        KEY_ENTER=28,
        KEY_SPACE=57,
    )
    _evdev_stub.ecodes = ecodes
    _evdev_stub.InputDevice = MagicMock
    _evdev_stub.InputEvent = MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Return a temporary directory suitable for config files."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    return cfg_dir


@pytest.fixture()
def tmp_config_file(tmp_config_dir: Path) -> Path:
    """Write a minimal YAML config and return its path."""
    cfg = {
        "hotkey": "fn",
        "mode": "auto",
        "stt": {"backend": "faster-whisper", "model": "large-v3-turbo"},
        "polish": {"enabled": True},
        "audio": {"sample_rate": 16000},
        "inject": {"method": "auto"},
        "tray": {"enabled": False},
    }
    path = tmp_config_dir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


@pytest.fixture()
def empty_config_file(tmp_config_dir: Path) -> Path:
    """Write an empty YAML config and return its path."""
    path = tmp_config_dir / "config.yaml"
    path.write_text("")
    return path


@pytest.fixture()
def mock_audio_f32():
    """Return a factory that creates numpy float32 audio arrays."""
    import numpy as np

    def _make(n_samples: int = 1600, frequency: float = 440.0, sr: int = 16000):
        t = np.linspace(0, n_samples / sr, n_samples, endpoint=False, dtype=np.float32)
        return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    return _make


@pytest.fixture()
def mock_audio_pcm_bytes(mock_audio_f32):
    """Return a factory that creates 16-bit PCM audio bytes."""
    import numpy as np

    def _make(n_samples: int = 1600, **kwargs):
        audio_f32 = mock_audio_f32(n_samples=n_samples, **kwargs)
        audio_i16 = (audio_f32 * 32767).astype(np.int16)
        return audio_i16.tobytes()

    return _make
