"""Entry point for ``python -m linux_whisper``.

Preloads pywhispercpp before numpy/sounddevice to avoid a ROCm/HIP
segfault caused by shared library symbol conflicts with libamdhip64.
"""

# This MUST happen before any other import that pulls in numpy.
try:
    import pywhispercpp.model  # noqa: F401
except ImportError:
    pass

from linux_whisper.cli import main

raise SystemExit(main())
