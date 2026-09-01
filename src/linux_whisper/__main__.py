"""Entry point for ``python -m linux_whisper``.

Preloads pywhispercpp before numpy/sounddevice to avoid a ROCm/HIP
segfault caused by shared library symbol conflicts with libamdhip64.
"""

# This MUST happen before any other import that pulls in numpy.
# Deliberately a bare try/except rather than contextlib.suppress: this block
# exists precisely to control import order, and importing contextlib to tidy
# it would put another import ahead of the preload it is guarding. The risk
# is a documented ROCm/libamdhip64 segfault, which is not worth a style point.
try:  # noqa: SIM105
    import pywhispercpp.model  # noqa: F401
except ImportError:
    pass

from linux_whisper.cli import main

raise SystemExit(main())
