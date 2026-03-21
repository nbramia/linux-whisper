"""Tests for linux_whisper.cli — argument parsing, config subcommands.

These tests verify the CLI interface without actually starting the app
(the ``run`` subcommand is not exercised because it requires the full
runtime).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

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
