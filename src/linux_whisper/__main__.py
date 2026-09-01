"""Entry point for ``python -m linux_whisper``.

Preloads pywhispercpp before numpy/sounddevice to avoid a ROCm/HIP
segfault caused by shared library symbol conflicts with libamdhip64.
"""

# This MUST happen before any other import that pulls in numpy.
#
# The `isort: off` guard below is load-bearing. The identical preload in
# stt/whisper_gpu_worker.py was silently reordered by ruff's I001 fix and
# shipped a SIGSEGV; this block survived only because its try/except shape
# happened to defeat the sorter, which is luck, not protection.
# Deliberately a bare try/except rather than contextlib.suppress: this block
# exists precisely to control import order, and importing contextlib to tidy
# it would put another import ahead of the preload it is guarding. The risk
# is a documented ROCm/libamdhip64 segfault, which is not worth a style point.
# isort: off
try:  # noqa: SIM105
    import pywhispercpp.model  # noqa: F401
except ImportError:
    pass

from linux_whisper.cli import main
# isort: on

raise SystemExit(main())
