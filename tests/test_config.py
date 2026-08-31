"""Tests for linux_whisper.config — loading, validation, defaults, edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from linux_whisper.config import (
    AudioConfig,
    Config,
    InjectConfig,
    OverlayConfig,
    PolishConfig,
    STTConfig,
    TrayConfig,
    _dataclass_to_dict,
    _merge_dataclass,
)


# ── Defaults ────────────────────────────────────────────────────────────────


class TestDefaults:
    """Verify that default-constructed configs have the expected values."""

    def test_config_defaults(self):
        cfg = Config()
        assert cfg.hotkey == "fn"
        assert cfg.mode == "auto"
        assert cfg.snippets == {}

    def test_stt_defaults(self):
        stt = STTConfig()
        assert stt.backend == "whisper-cpp"
        assert stt.model == "whisper-large-v3-turbo"
        assert stt.device == "rocm"
        assert stt.threads == 0

    def test_polish_defaults(self):
        p = PolishConfig()
        assert p.enabled is True
        assert p.disfluency is True
        assert p.punctuation is True
        assert p.formatting is True
        assert p.llm is True
        assert p.llm_always is False
        assert p.context_awareness is True
        assert p.llm_device == "rocm"

    def test_audio_defaults(self):
        a = AudioConfig()
        assert a.sample_rate == 16000
        assert a.vad_threshold == 0.6
        assert a.silence_timeout == 0.5
        assert a.feedback_sounds is True
        assert a.buffer_size == 512
        assert a.auto_gain is True

    def test_inject_defaults(self):
        inj = InjectConfig()
        assert inj.method == "auto"
        assert inj.typing_delay == 0

    def test_tray_defaults(self):
        t = TrayConfig()
        assert t.enabled is True
        assert t.show_preview is False

    def test_overlay_defaults(self):
        o = OverlayConfig()
        assert o.enabled is True
        # bottom-center, not center: a pill parked in the middle of the screen
        # sits on top of whatever you are dictating into.
        assert o.position == "bottom-center"


# ── from_dict ───────────────────────────────────────────────────────────────


class TestFromDict:
    """Test Config.from_dict with various input dictionaries."""

    def test_empty_dict_gives_defaults(self):
        cfg = Config.from_dict({})
        assert cfg == Config()

    def test_partial_override(self):
        cfg = Config.from_dict({"hotkey": "alt+d", "mode": "toggle"})
        assert cfg.hotkey == "alt+d"
        assert cfg.mode == "toggle"
        # Nested defaults still hold
        assert cfg.stt.backend == "whisper-cpp"

    def test_nested_override(self):
        cfg = Config.from_dict({
            "stt": {"backend": "whisper-cpp", "model": "distil-large-v3.5"},
            "audio": {"vad_threshold": 0.3},
        })
        assert cfg.stt.backend == "whisper-cpp"
        assert cfg.stt.model == "distil-large-v3.5"
        assert cfg.stt.threads == 0  # default preserved
        assert cfg.audio.vad_threshold == 0.3

    def test_unknown_keys_in_nested_ignored(self):
        # Extra keys in a nested dict should not cause errors because
        # _merge_dataclass only looks at known fields.
        cfg = Config.from_dict({
            "stt": {"backend": "faster-whisper", "nonexistent_key": 42},
        })
        assert cfg.stt.backend == "faster-whisper"

    def test_inject_method_override(self):
        cfg = Config.from_dict({"inject": {"method": "clipboard", "typing_delay": 10}})
        assert cfg.inject.method == "clipboard"
        assert cfg.inject.typing_delay == 10

    def test_snippets_from_dict(self):
        cfg = Config.from_dict({
            "snippets": {"my email": "test@example.com", "hi": "hello"}
        })
        assert cfg.snippets == {"my email": "test@example.com", "hi": "hello"}

    def test_snippets_none_treated_as_empty(self):
        cfg = Config.from_dict({"snippets": None})
        assert cfg.snippets == {}

    def test_snippets_missing_defaults_to_empty(self):
        cfg = Config.from_dict({})
        assert cfg.snippets == {}

    def test_overlay_from_dict(self):
        cfg = Config.from_dict({"overlay": {"enabled": False, "position": "top-center"}})
        assert cfg.overlay.enabled is False
        assert cfg.overlay.position == "top-center"

    def test_overlay_missing_defaults(self):
        cfg = Config.from_dict({})
        assert cfg.overlay == OverlayConfig()

    def test_polish_all_disabled(self):
        cfg = Config.from_dict({
            "polish": {
                "enabled": False,
                "disfluency": False,
                "punctuation": False,
                "llm": False,
            }
        })
        assert cfg.polish.enabled is False
        assert cfg.polish.disfluency is False


# ── Validation ──────────────────────────────────────────────────────────────


class TestValidation:
    """Test Config.validate() with valid and invalid configurations."""

    def test_default_config_is_valid(self):
        errors = Config().validate()
        assert errors == []

    def test_invalid_mode(self):
        cfg = Config.from_dict({"mode": "push-to-talk"})
        errors = cfg.validate()
        assert any("mode" in e.lower() for e in errors)

    def test_invalid_stt_backend(self):
        cfg = Config.from_dict({"stt": {"backend": "openai"}})
        errors = cfg.validate()
        assert any("stt.backend" in e for e in errors)

    def test_invalid_inject_method(self):
        cfg = Config.from_dict({"inject": {"method": "magic"}})
        errors = cfg.validate()
        assert any("inject.method" in e for e in errors)

    def test_unusual_sample_rate(self):
        cfg = Config.from_dict({"audio": {"sample_rate": 12345}})
        errors = cfg.validate()
        assert any("sample_rate" in e for e in errors)

    def test_valid_sample_rates_accepted(self):
        for sr in (8000, 16000, 22050, 44100, 48000):
            cfg = Config.from_dict({"audio": {"sample_rate": sr}})
            errors = cfg.validate()
            assert not any("sample_rate" in e for e in errors), f"sample_rate {sr} rejected"

    def test_vad_threshold_out_of_range_low(self):
        cfg = Config.from_dict({"audio": {"vad_threshold": 0.0}})
        errors = cfg.validate()
        assert any("vad_threshold" in e for e in errors)

    def test_vad_threshold_out_of_range_high(self):
        cfg = Config.from_dict({"audio": {"vad_threshold": 1.0}})
        errors = cfg.validate()
        assert any("vad_threshold" in e for e in errors)

    def test_vad_threshold_valid(self):
        cfg = Config.from_dict({"audio": {"vad_threshold": 0.5}})
        errors = cfg.validate()
        assert not any("vad_threshold" in e for e in errors)

    def test_invalid_overlay_position(self):
        cfg = Config.from_dict({"overlay": {"position": "middle"}})
        errors = cfg.validate()
        assert any("overlay.position" in e for e in errors)

    @pytest.mark.parametrize("position", ["center", "bottom-center", "top-center"])
    def test_valid_overlay_positions_accepted(self, position):
        cfg = Config.from_dict({"overlay": {"position": position}})
        errors = cfg.validate()
        assert not any("overlay.position" in e for e in errors), f"position {position} rejected"

    def test_multiple_errors(self):
        cfg = Config.from_dict({
            "mode": "invalid",
            "stt": {"backend": "invalid"},
            "inject": {"method": "invalid"},
        })
        errors = cfg.validate()
        assert len(errors) >= 3


# ── Loading from file ───────────────────────────────────────────────────────


class TestLoad:
    """Test Config.load() with real files."""

    def test_load_from_yaml(self, tmp_config_file: Path):
        cfg = Config.load(tmp_config_file)
        assert cfg.hotkey == "fn"
        assert cfg.mode == "auto"
        # The fixture file pins whisper-cpp explicitly, so this asserts the
        # file wins over the default rather than asserting the default itself.
        assert cfg.stt.backend == "whisper-cpp"

    def test_load_empty_file_gives_defaults(self, empty_config_file: Path):
        cfg = Config.load(empty_config_file)
        assert cfg == Config()

    def test_load_missing_file_gives_defaults(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.yaml"
        cfg = Config.load(missing)
        assert cfg == Config()

    def test_load_partial_yaml(self, tmp_config_dir: Path):
        path = tmp_config_dir / "partial.yaml"
        with open(path, "w") as f:
            yaml.dump({"mode": "vad-auto"}, f)
        cfg = Config.load(path)
        assert cfg.mode == "vad-auto"
        assert cfg.hotkey == "fn"  # default

    def test_overlay_round_trips_through_yaml(self, tmp_config_dir: Path):
        path = tmp_config_dir / "overlay.yaml"
        with open(path, "w") as f:
            yaml.dump({"overlay": {"enabled": False, "position": "top-center"}}, f)
        cfg = Config.load(path)
        assert cfg.overlay.enabled is False
        assert cfg.overlay.position == "top-center"
        assert cfg.validate() == []

    def test_bare_overlay_section_gives_defaults_instead_of_crashing(
        self, tmp_config_dir: Path
    ):
        # `overlay:` with nothing indented under it parses to
        # {"overlay": None}, not {"overlay": {}} — this used to raise
        # AttributeError: 'NoneType' object has no attribute 'get' before
        # validation ever ran.
        path = tmp_config_dir / "bare_overlay.yaml"
        path.write_text("overlay:\nhotkey: fn\n")
        cfg = Config.load(path)
        assert cfg.overlay == OverlayConfig()
        assert cfg.validate() == []


# ── save_default ────────────────────────────────────────────────────────────


class TestSaveDefault:
    """Test Config.save_default()."""

    def test_creates_file_if_missing(self, tmp_path: Path):
        path = tmp_path / "subdir" / "config.yaml"
        assert not path.exists()
        Config().save_default(path)
        assert path.exists()

        # Read back and verify
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["hotkey"] == "fn"
        assert data["mode"] == "auto"

    def test_does_not_overwrite_existing(self, tmp_config_file: Path):
        # Modify the file
        with open(tmp_config_file, "w") as f:
            yaml.dump({"hotkey": "custom"}, f)

        # save_default should NOT overwrite
        Config().save_default(tmp_config_file)

        with open(tmp_config_file) as f:
            data = yaml.safe_load(f)
        assert data["hotkey"] == "custom"


# ── _merge_dataclass ────────────────────────────────────────────────────────


class TestMergeDataclass:
    """Test the _merge_dataclass helper."""

    def test_empty_overrides(self):
        result = _merge_dataclass(STTConfig, {})
        assert result == STTConfig()

    def test_partial_overrides(self):
        result = _merge_dataclass(STTConfig, {"backend": "whisper-cpp"})
        assert result.backend == "whisper-cpp"
        assert result.model == "whisper-large-v3-turbo"  # untouched default

    def test_skips_VALID_constants(self):
        # VALID_BACKENDS is a class variable, not a constructor param.
        # _merge_dataclass must skip it.
        result = _merge_dataclass(STTConfig, {"VALID_BACKENDS": ("fake",)})
        assert result.backend == "whisper-cpp"

    def test_none_overrides_falls_back_to_defaults(self):
        # A YAML section written with no mapping under it (e.g. a bare
        # "overlay:" key) parses to None, not {} — this must not crash.
        result = _merge_dataclass(STTConfig, None)
        assert result == STTConfig()

    def test_falsey_non_mapping_override_raises_instead_of_silently_enabling(self):
        # `overlay: false` in YAML parses to `{"overlay": False}`, not
        # `{"overlay": None}` or `{"overlay": {}}`. `False or {}` == `{}`,
        # so treating every falsey value the same as "absent" would merge
        # in every default — including OverlayConfig's `enabled=True` — the
        # opposite of what a user writing `overlay: false` almost certainly
        # meant, and silently (it forces GDK_BACKEND=x11 and starts a GTK
        # thread nobody asked for). Must raise instead of guessing.
        with pytest.raises(ValueError, match="OverlayConfig"):
            _merge_dataclass(OverlayConfig, False)

    def test_other_non_mapping_overrides_also_raise(self):
        # Any non-mapping value, not just False, is malformed input the
        # same way — a bare scalar or a list can't sensibly become
        # per-field overrides.
        with pytest.raises(ValueError, match="TrayConfig"):
            _merge_dataclass(TrayConfig, "off")
        with pytest.raises(ValueError, match="TrayConfig"):
            _merge_dataclass(TrayConfig, [1, 2, 3])

    def test_config_from_dict_surfaces_the_falsey_section_error(self):
        # The same danger, exercised through the real entry point: loading
        # a config with `overlay: false` must not quietly produce
        # OverlayConfig(enabled=True).
        with pytest.raises(ValueError, match="OverlayConfig"):
            Config.from_dict({"overlay": False})


# ── _dataclass_to_dict ──────────────────────────────────────────────────────


class TestDataclassToDict:
    """Test the _dataclass_to_dict helper."""

    def test_simple_dataclass(self):
        d = _dataclass_to_dict(AudioConfig())
        assert d == {
            "sample_rate": 16000,
            "vad_threshold": 0.6,
            "silence_timeout": 0.5,
            "feedback_sounds": True,
            "buffer_size": 512,
            "auto_gain": True,
        }

    def test_nested_config(self):
        d = _dataclass_to_dict(Config())
        assert "hotkey" in d
        assert "stt" in d
        assert isinstance(d["stt"], dict)
        assert d["stt"]["backend"] == "whisper-cpp"
        # VALID_ keys should not appear
        assert "VALID_BACKENDS" not in d["stt"]
        assert "VALID_MODES" not in d

    def test_snippets_round_trip(self):
        snippets = {"my email": "test@example.com", "hi": "hello"}
        cfg = Config.from_dict({"snippets": snippets})
        d = _dataclass_to_dict(cfg)
        assert d["snippets"] == snippets

    def test_non_dataclass_passthrough(self):
        assert _dataclass_to_dict(42) == 42
        assert _dataclass_to_dict("hello") == "hello"

    def test_overlay_round_trip(self):
        cfg = Config.from_dict({"overlay": {"enabled": False, "position": "bottom-center"}})
        d = _dataclass_to_dict(cfg)
        assert d["overlay"] == {"enabled": False, "position": "bottom-center"}


# ── Frozen dataclass ───────────────────────────────────────────────────────


class TestFrozen:
    """Verify that config dataclasses are immutable."""

    def test_config_is_frozen(self):
        cfg = Config()
        with pytest.raises(AttributeError):
            cfg.hotkey = "other"  # type: ignore[misc]

    def test_stt_config_is_frozen(self):
        stt = STTConfig()
        with pytest.raises(AttributeError):
            stt.backend = "other"  # type: ignore[misc]
