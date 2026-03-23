"""Console script entry point with GPU backend preloading.

The pywhispercpp C extension (ROCm/HIP) segfaults if numpy or
sounddevice (portaudio) is loaded first, due to shared library
symbol conflicts with libamdhip64.  This module ensures the
preload happens before any other import.
"""


def main() -> None:
    from linux_whisper.cli import main as cli_main

    raise SystemExit(cli_main())
