"""CLI entry point for Linux Whisper."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from linux_whisper import __version__
from linux_whisper.config import CONFIG_PATH, MODELS_DIR, Config


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="linux-whisper",
        description="Local voice dictation for Linux",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Config file path (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase log verbosity (-v=INFO, -vv=DEBUG)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # `linux-whisper run` (default)
    run_parser = subparsers.add_parser("run", help="Start the dictation service")
    run_parser.add_argument("--no-tray", action="store_true", help="Disable system tray")

    # `linux-whisper models`
    models_parser = subparsers.add_parser("models", help="Manage models")
    models_sub = models_parser.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="List available and downloaded models")
    dl_parser = models_sub.add_parser("download", help="Download a model")
    dl_parser.add_argument("model_id", help="Model identifier to download")
    default_parser = models_sub.add_parser("default", help="Set the default model")
    default_parser.add_argument("model_id", help="Model identifier to set as default")

    # `linux-whisper config`
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_sub.add_parser("init", help="Create default config file")
    config_sub.add_parser("show", help="Show current config")
    config_sub.add_parser("path", help="Show config file path")
    config_sub.add_parser("validate", help="Validate current config")

    args = parser.parse_args(argv)

    # Setup logging
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Route to command
    command = args.command or "run"

    if command == "run":
        return _cmd_run(args)
    elif command == "models":
        return _cmd_models(args)
    elif command == "config":
        return _cmd_config(args)
    else:
        parser.print_help()
        return 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Start the dictation service."""
    from linux_whisper.app import run_app

    config = Config.load(args.config)

    if getattr(args, "no_tray", False):
        # Override tray setting
        from linux_whisper.config import TrayConfig

        config = Config(
            hotkey=config.hotkey,
            mode=config.mode,
            stt=config.stt,
            polish=config.polish,
            audio=config.audio,
            inject=config.inject,
            tray=TrayConfig(enabled=False),
        )

    try:
        asyncio.run(run_app(config))
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    """Handle model management commands."""
    models_command = getattr(args, "models_command", None)

    if models_command == "list":
        return _models_list()
    elif models_command == "download":
        return _models_download(args.model_id)
    elif models_command == "default":
        return _models_default(args.model_id)
    else:
        print("Usage: linux-whisper models {list|download|default}", file=sys.stderr)
        return 1


def _models_list() -> int:
    """List available and downloaded models."""
    available = {
        "moonshine-tiny": {
            "params": "33.6M",
            "type": "streaming",
            "wer": "12.01%",
            "ram": "~150MB",
        },
        "moonshine-medium": {
            "params": "244.9M",
            "type": "streaming",
            "wer": "6.65%",
            "ram": "~500MB",
        },
        "whisper-large-v3-turbo": {
            "params": "809M",
            "type": "batch",
            "wer": "7.25%",
            "ram": "~4GB (Q8)",
        },
        "distil-large-v3.5": {
            "params": "756M",
            "type": "batch",
            "wer": "7.10%",
            "ram": "~3.5GB (Q8)",
        },
    }

    print("Available STT models:\n")
    for model_id, info in available.items():
        downloaded = (MODELS_DIR / model_id).exists()
        status = "[downloaded]" if downloaded else "[not downloaded]"
        print(f"  {model_id:<30} {status}")
        print(f"    Params: {info['params']}, WER: {info['wer']}, RAM: {info['ram']}, Type: {info['type']}")
        print()

    return 0


def _models_download(model_id: str) -> int:
    """Download a model."""
    print(f"Downloading {model_id}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Model download logic would use huggingface_hub
    # For now, print instructions
    print(f"Model download not yet implemented. Models will be downloaded automatically on first use.")
    print(f"Models are cached in: {MODELS_DIR}")
    return 0


def _models_default(model_id: str) -> int:
    """Set the default model in config."""
    config = Config.load()
    print(f"To set the default model, edit {CONFIG_PATH}:")
    print(f"  stt:")
    print(f"    model: {model_id}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    """Handle config commands."""
    config_command = getattr(args, "config_command", None)

    if config_command == "init":
        config = Config()
        config.save_default()
        print(f"Config written to {CONFIG_PATH}")
        return 0
    elif config_command == "show":
        config = Config.load()
        import yaml
        from linux_whisper.config import _dataclass_to_dict

        print(yaml.dump(_dataclass_to_dict(config), default_flow_style=False, sort_keys=False))
        return 0
    elif config_command == "path":
        print(CONFIG_PATH)
        return 0
    elif config_command == "validate":
        config = Config.load()
        errors = config.validate()
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  - {e}")
            return 1
        print("Config is valid")
        return 0
    else:
        print("Usage: linux-whisper config {init|show|path|validate}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
