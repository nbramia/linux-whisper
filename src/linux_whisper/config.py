"""Configuration loading, validation, and defaults."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "linux-whisper"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
CACHE_DIR = Path.home() / ".cache" / "linux-whisper"
MODELS_DIR = CACHE_DIR / "models"


@dataclass(frozen=True)
class STTConfig:
    backend: str = "moonshine"
    model: str = "moonshine-medium"
    threads: int = 0  # 0 = auto

    VALID_BACKENDS = ("moonshine", "whisper-cpp")
    VALID_MODELS = (
        "moonshine-tiny",
        "moonshine-medium",
        "whisper-large-v3-turbo",
        "distil-large-v3.5",
    )


@dataclass(frozen=True)
class PolishConfig:
    enabled: bool = True
    disfluency: bool = True
    punctuation: bool = True
    llm: bool = True
    llm_always: bool = False
    llm_backend: str = "llama-cpp"
    llm_model: str = "Qwen3-4B-Q4_K_M"
    llm_threads: int = 0  # 0 = auto


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    silence_timeout: float = 2.0
    feedback_sounds: bool = True
    buffer_size: int = 512  # samples per chunk


@dataclass(frozen=True)
class InjectConfig:
    method: str = "auto"
    typing_delay: int = 0  # ms between keystrokes

    VALID_METHODS = ("auto", "xdotool", "ydotool", "wtype", "clipboard")


@dataclass(frozen=True)
class TrayConfig:
    enabled: bool = True
    show_preview: bool = False


@dataclass(frozen=True)
class Config:
    hotkey: str = "fn"
    mode: str = "hold"  # hold | toggle | vad-auto
    stt: STTConfig = field(default_factory=STTConfig)
    polish: PolishConfig = field(default_factory=PolishConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)
    tray: TrayConfig = field(default_factory=TrayConfig)

    VALID_MODES = ("hold", "toggle", "vad-auto")

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a Config from a raw dictionary (parsed YAML)."""
        return cls(
            hotkey=data.get("hotkey", cls.hotkey),
            mode=data.get("mode", cls.mode),
            stt=_merge_dataclass(STTConfig, data.get("stt", {})),
            polish=_merge_dataclass(PolishConfig, data.get("polish", {})),
            audio=_merge_dataclass(AudioConfig, data.get("audio", {})),
            inject=_merge_dataclass(InjectConfig, data.get("inject", {})),
            tray=_merge_dataclass(TrayConfig, data.get("tray", {})),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        """Load config from YAML file, falling back to defaults."""
        path = path or CONFIG_PATH
        if path.exists():
            logger.info("Loading config from %s", path)
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        logger.info("No config file found at %s, using defaults", path)
        return cls()

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = valid)."""
        errors: list[str] = []
        if self.mode not in self.VALID_MODES:
            errors.append(f"Invalid mode '{self.mode}', must be one of {self.VALID_MODES}")
        if self.stt.backend not in STTConfig.VALID_BACKENDS:
            errors.append(
                f"Invalid stt.backend '{self.stt.backend}', "
                f"must be one of {STTConfig.VALID_BACKENDS}"
            )
        if self.inject.method not in InjectConfig.VALID_METHODS:
            errors.append(
                f"Invalid inject.method '{self.inject.method}', "
                f"must be one of {InjectConfig.VALID_METHODS}"
            )
        if self.audio.sample_rate not in (8000, 16000, 22050, 44100, 48000):
            errors.append(f"Unusual sample_rate {self.audio.sample_rate}")
        if not 0.0 < self.audio.vad_threshold < 1.0:
            errors.append(f"vad_threshold must be between 0 and 1, got {self.audio.vad_threshold}")
        return errors

    def save_default(self, path: Path | None = None) -> None:
        """Write the default config file if it doesn't exist."""
        path = path or CONFIG_PATH
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _dataclass_to_dict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info("Wrote default config to %s", path)


def _merge_dataclass[T](cls: type[T], overrides: dict) -> T:
    """Create a dataclass instance, merging overrides with defaults."""
    defaults = cls()
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for name in fields:
        if name.startswith("VALID_"):
            continue
        kwargs[name] = overrides.get(name, getattr(defaults, name))
    return cls(**kwargs)


def _dataclass_to_dict(obj: object) -> dict:
    """Recursively convert a dataclass to a dict, skipping class-level constants."""
    from dataclasses import fields, is_dataclass

    if not is_dataclass(obj):
        return obj  # type: ignore[return-value]
    result = {}
    for f in fields(obj):
        if f.name.startswith("VALID_"):
            continue
        val = getattr(obj, f.name)
        if is_dataclass(val):
            result[f.name] = _dataclass_to_dict(val)
        else:
            result[f.name] = val
    return result
