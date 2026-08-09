"""Model A/B benchmark harness.

Not part of the default pytest run — this package loads real models and real
audio, which ``CLAUDE.md`` forbids in CI.  Only the pure scoring functions in
:mod:`tests.benchmarks.metrics` are exercised by ``tests/test_benchmarks.py``.

See ``tests/benchmarks/README.md`` for usage.
"""
