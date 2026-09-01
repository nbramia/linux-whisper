"""Tests for linux_whisper.cli — argument parsing, config subcommands.

These tests verify the CLI interface without actually starting the app
(the ``run`` subcommand is not exercised because it requires the full
runtime).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from linux_whisper.cli import main

# ── Argument parsing ────────────────────────────────────────────────────────


class TestCliParsing:
    """Test that the CLI parser handles various argument combinations."""

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out

    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "linux-whisper" in captured.out

    def test_default_command_is_run(self):
        """With no subcommand, the CLI defaults to 'run'."""
        # We patch _cmd_run to avoid actually starting the app
        with patch("linux_whisper.cli._cmd_run", return_value=0) as mock_run:
            result = main([])
        mock_run.assert_called_once()
        assert result == 0

    def test_verbose_single(self):
        with patch("linux_whisper.cli._cmd_run", return_value=0):
            result = main(["-v"])
        assert result == 0

    def test_verbose_double(self):
        with patch("linux_whisper.cli._cmd_run", return_value=0):
            result = main(["-vv"])
        assert result == 0

    def test_custom_config_path(self, tmp_path):
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("hotkey: alt+d\n")

        with patch("linux_whisper.cli._cmd_run", return_value=0) as mock_run:
            result = main(["--config", str(config_file)])
        assert result == 0
        # The point of this test is that --config reaches run, so check it
        # rather than only the exit code — the binding was previously unused,
        # which meant a regression here would have gone unnoticed.
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0].config == config_file


# ── Config subcommands ──────────────────────────────────────────────────────


class TestConfigSubcommands:

    def test_config_path(self, capsys):
        result = main(["config", "path"])
        assert result == 0
        captured = capsys.readouterr()
        assert "config.yaml" in captured.out

    def test_config_show(self, capsys):
        result = main(["config", "show"])
        assert result == 0
        captured = capsys.readouterr()
        # Output should be valid YAML containing known keys
        assert "hotkey" in captured.out
        assert "mode" in captured.out

    def test_config_validate_default(self, capsys):
        result = main(["config", "validate"])
        assert result == 0
        captured = capsys.readouterr()
        assert "valid" in captured.out.lower()

    def test_config_validate_reports_malformed_section_cleanly(
        self, tmp_path, capsys, monkeypatch
    ):
        # `overlay: false` (a bare non-mapping value) is caught by
        # _merge_dataclass and raises ValueError rather than being loaded as
        # OverlayConfig(enabled=True). `config validate`'s whole purpose is
        # reporting config problems without a crash — it must turn that
        # into the same "Validation errors:" output as any other invalid
        # value, not an unhandled traceback.
        fake_path = tmp_path / "config.yaml"
        fake_path.write_text("overlay: false\n")
        monkeypatch.setattr("linux_whisper.cli.CONFIG_PATH", fake_path)
        monkeypatch.setattr("linux_whisper.config.CONFIG_PATH", fake_path)

        result = main(["config", "validate"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Validation errors:" in captured.out
        assert "OverlayConfig" in captured.out

    def test_config_init(self, tmp_path, capsys, monkeypatch):
        # Patch CONFIG_PATH so init writes to our temp dir
        fake_path = tmp_path / "config.yaml"
        monkeypatch.setattr("linux_whisper.cli.CONFIG_PATH", fake_path)
        monkeypatch.setattr("linux_whisper.config.CONFIG_PATH", fake_path)

        result = main(["config", "init"])
        assert result == 0
        assert fake_path.exists()

    def test_config_no_subcommand(self, capsys):
        result = main(["config"])
        assert result == 1
        captured = capsys.readouterr()
        assert "Usage" in captured.err or "usage" in captured.err.lower()


# ── Config subcommands honour --config (issue #49) ─────────────────────────


class TestConfigFlagPassthrough:
    """`config show`/`validate`/`path`/`init` must honour a global --config,
    the same way `run` already does.

    Each test asserts which file's content actually shows up in the output,
    not just the exit code — an exit-code-only assertion passes even with
    the pre-fix bug, because the default config also happens to be valid.
    """

    def _patch_default_config_path(self, monkeypatch, path):
        monkeypatch.setattr("linux_whisper.cli.CONFIG_PATH", path)
        monkeypatch.setattr("linux_whisper.config.CONFIG_PATH", path)

    def test_show_without_config_flag_reads_default_path(self, tmp_path, capsys, monkeypatch):
        default_path = tmp_path / "default.yaml"
        default_path.write_text("hotkey: ctrl+alt+d\n")
        self._patch_default_config_path(monkeypatch, default_path)

        result = main(["config", "show"])

        assert result == 0
        captured = capsys.readouterr()
        assert "ctrl+alt+d" in captured.out

    def test_show_with_config_flag_reads_that_file_not_default(
        self, tmp_path, capsys, monkeypatch
    ):
        default_path = tmp_path / "default.yaml"
        default_path.write_text("hotkey: ctrl+alt+d\n")
        self._patch_default_config_path(monkeypatch, default_path)

        custom_path = tmp_path / "custom.yaml"
        custom_path.write_text("hotkey: super+space\n")

        result = main(["--config", str(custom_path), "config", "show"])

        assert result == 0
        captured = capsys.readouterr()
        assert "super+space" in captured.out
        assert "ctrl+alt+d" not in captured.out

    def test_validate_without_config_flag_reads_default_path(
        self, tmp_path, capsys, monkeypatch
    ):
        default_path = tmp_path / "default.yaml"
        default_path.write_text("mode: bogus-mode\n")
        self._patch_default_config_path(monkeypatch, default_path)

        result = main(["config", "validate"])

        assert result == 1
        captured = capsys.readouterr()
        assert "bogus-mode" in captured.out

    def test_validate_with_config_flag_reads_that_file_not_default(
        self, tmp_path, capsys, monkeypatch
    ):
        # The default path is a *valid* config; the custom path is invalid.
        # Pre-fix, `validate` ignores --config and reports on the (valid)
        # default, so this fails before the fix and passes after.
        default_path = tmp_path / "default.yaml"
        default_path.write_text("mode: auto\n")
        self._patch_default_config_path(monkeypatch, default_path)

        custom_path = tmp_path / "custom.yaml"
        custom_path.write_text("mode: bogus-mode\n")

        result = main(["--config", str(custom_path), "config", "validate"])

        assert result == 1
        captured = capsys.readouterr()
        assert "bogus-mode" in captured.out

    def test_path_without_config_flag_prints_default_path(self, tmp_path, capsys, monkeypatch):
        default_path = tmp_path / "default.yaml"
        self._patch_default_config_path(monkeypatch, default_path)

        result = main(["config", "path"])

        assert result == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == str(default_path)

    def test_path_with_config_flag_prints_that_path_not_default(
        self, tmp_path, capsys, monkeypatch
    ):
        default_path = tmp_path / "default.yaml"
        self._patch_default_config_path(monkeypatch, default_path)
        custom_path = tmp_path / "custom.yaml"

        result = main(["--config", str(custom_path), "config", "path"])

        assert result == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == str(custom_path)

    def test_init_with_config_flag_writes_that_path_not_default(
        self, tmp_path, capsys, monkeypatch
    ):
        default_path = tmp_path / "default.yaml"
        self._patch_default_config_path(monkeypatch, default_path)
        custom_path = tmp_path / "sub" / "custom.yaml"

        result = main(["--config", str(custom_path), "config", "init"])

        assert result == 0
        assert custom_path.exists()
        assert not default_path.exists()
        captured = capsys.readouterr()
        assert str(custom_path) in captured.out

    def test_show_with_nonexistent_config_flag_errors_clearly(
        self, tmp_path, capsys, monkeypatch
    ):
        # A typo'd --config path must not silently fall back to the
        # (possibly different) default config — report it instead.
        default_path = tmp_path / "default.yaml"
        default_path.write_text("hotkey: ctrl+alt+d\n")
        self._patch_default_config_path(monkeypatch, default_path)

        missing_path = tmp_path / "does-not-exist.yaml"

        result = main(["--config", str(missing_path), "config", "show"])

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()
        assert str(missing_path) in captured.err
        # Must not have silently fallen back and printed the default's content.
        assert "ctrl+alt+d" not in captured.out

    def test_validate_with_nonexistent_config_flag_errors_clearly(
        self, tmp_path, capsys, monkeypatch
    ):
        default_path = tmp_path / "default.yaml"
        default_path.write_text("mode: auto\n")
        self._patch_default_config_path(monkeypatch, default_path)

        missing_path = tmp_path / "does-not-exist.yaml"

        result = main(["--config", str(missing_path), "config", "validate"])

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()
        assert str(missing_path) in captured.err
        assert "valid" not in captured.out.lower()


# ── Models subcommand ───────────────────────────────────────────────────────


class TestModelsSubcommand:

    def test_models_list(self, capsys):
        result = main(["models", "list"])
        assert result == 0
        captured = capsys.readouterr()
        assert "moonshine-tiny" in captured.out
        assert "moonshine-medium" in captured.out

    def test_models_download(self, capsys):
        result = main(["models", "download", "moonshine-tiny"])
        assert result == 0
        captured = capsys.readouterr()
        assert "moonshine-tiny" in captured.out

    def test_models_default(self, capsys):
        result = main(["models", "default", "moonshine-medium"])
        assert result == 0
        captured = capsys.readouterr()
        assert "moonshine-medium" in captured.out

    def test_models_no_subcommand(self, capsys):
        result = main(["models"])
        assert result == 1
        captured = capsys.readouterr()
        assert "Usage" in captured.err or "usage" in captured.err.lower()


# ── Run subcommand (mocked) ────────────────────────────────────────────────


class TestRunSubcommand:

    def test_run_with_no_tray(self):
        """The --no-tray flag should disable the tray in the config."""
        with patch("linux_whisper.cli.asyncio.run") as mock_asyncio_run:
            result = main(["run", "--no-tray"])
        assert result == 0
        mock_asyncio_run.assert_called_once()

    def test_run_subcommand_explicit(self):
        with patch("linux_whisper.cli._cmd_run", return_value=0) as mock_run:
            result = main(["run"])
        mock_run.assert_called_once()
        assert result == 0
