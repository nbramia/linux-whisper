"""Model A/B benchmark runner.

Scores the real pipeline — real models, real audio — and writes a JSON result
that can be diffed against a committed baseline.  Deliberately outside the
default pytest run: ``CLAUDE.md`` requires CI tests to mock every model, and
this does the opposite on purpose.

Usage::

    # Capture a baseline of whatever is currently configured
    python -m tests.benchmarks.run --suite all --label baseline \\
        --out tests/benchmarks/baseline/current.json

    # Score a candidate and diff it against that baseline
    python -m tests.benchmarks.run --suite all --label qwen35 --out /tmp/cand.json
    python -m tests.benchmarks.run --compare tests/benchmarks/baseline/current.json /tmp/cand.json

``--compare`` exits non-zero when the candidate regresses, so it can gate a merge.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from linux_whisper.config import Config
from tests.benchmarks import metrics
from tests.benchmarks.fixtures import (
    DEFAULT_AUDIO_FIXTURE_COUNT,
    AudioFixture,
    TextFixture,
    load_audio,
    load_audio_fixtures,
    to_pcm16,
)
from tests.benchmarks.text_fixtures import load_text_fixtures

logger = logging.getLogger("benchmark")

SCHEMA_VERSION = 1

# Audio is fed to STT backends in the same 512-sample chunks the live capture
# path uses, so buffering behaviour is exercised the same way.
CHUNK_SAMPLES = 512

# Below this fraction of speech frames on clean read speech, the VAD is not
# merely tuned conservatively — it is broken.  A mis-sized window scores ~0.
MIN_HEALTHY_SPEECH_RATE = 0.30


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# STT suite
# ---------------------------------------------------------------------------


def run_stt_suite(config: Config, fixtures: list[AudioFixture]) -> dict[str, Any]:
    """Transcribe every fixture and score WER plus per-utterance latency."""
    from linux_whisper.stt.engine import create_engine

    logger.info(
        "STT suite: backend=%s model=%s device=%s over %d fixtures",
        config.stt.backend,
        config.stt.model,
        config.stt.device,
        len(fixtures),
    )

    engine = create_engine(config)

    # One warm-up pass so model load and any GPU kernel compilation are not
    # charged to the first fixture's latency.
    if fixtures:
        _transcribe(engine, fixtures[0])
        logger.info("Warm-up transcription complete")

    pairs: list[tuple[str, str]] = []
    latencies: list[float] = []
    per_fixture: list[dict[str, Any]] = []
    audio_seconds = 0.0

    for fixture in fixtures:
        hypothesis, elapsed_ms, duration_s = _transcribe(engine, fixture)
        pairs.append((fixture.reference, hypothesis))
        latencies.append(elapsed_ms)
        audio_seconds += duration_s

        wer = metrics.word_error_rate(fixture.reference, hypothesis)
        per_fixture.append(
            {
                "id": fixture.id,
                "reference": fixture.reference,
                "hypothesis": hypothesis,
                "wer": wer.wer,
                "latency_ms": elapsed_ms,
                "audio_seconds": duration_s,
            }
        )
        logger.debug("%s: wer=%.3f %.0fms", fixture.id, wer.wer, elapsed_ms)

    total = metrics.corpus_wer(pairs)
    latency = metrics.summarize_latency(latencies)
    total_ms = sum(latencies)

    # Punctuation/capitalisation are only meaningful when the reference has
    # them.  LibriSpeech references are unpunctuated upper-case, so this is
    # reported but naturally near-zero for that corpus.
    punct = metrics.punctuation_f1(
        " ".join(f.reference for f in fixtures),
        " ".join(p[1] for p in pairs),
    )

    return {
        "metrics": {
            "stt": {**total.to_dict(), "fixtures": len(fixtures)},
            "punctuation": punct,
            "latency": {"stt": latency.to_dict()},
            "rtfx": (audio_seconds / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        },
        "per_fixture": per_fixture,
    }


def _transcribe(engine: Any, fixture: AudioFixture) -> tuple[str, float, float]:
    """Run one fixture through the engine.  Returns (text, elapsed_ms, duration_s)."""
    audio = load_audio(fixture.path)
    duration_s = len(audio) / 16_000.0
    pcm = to_pcm16(audio)

    engine.start_stream()
    start = time.perf_counter()
    for offset in range(0, len(pcm), CHUNK_SAMPLES * 2):
        engine.feed_audio(pcm[offset : offset + CHUNK_SAMPLES * 2])
    result = engine.finalize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    engine.reset()

    return result.full_text.strip(), elapsed_ms, duration_s


# ---------------------------------------------------------------------------
# Polish suite
# ---------------------------------------------------------------------------


def run_polish_suite(config: Config, fixtures: list[TextFixture]) -> dict[str, Any]:
    """Run the polish stages individually so each one's latency is attributed.

    Mirrors ``PolishPipeline.process`` ordering exactly (4a → 4b → 4d →
    conditional 4c); reimplemented here only so per-stage timings can be
    collected without instrumenting production code.
    """
    from linux_whisper.polish.disfluency import DisfluencyRemover
    from linux_whisper.polish.formatting import SpokenFormFormatter
    from linux_whisper.polish.llm import LLMCorrector
    from linux_whisper.polish.punctuation import PunctuationRestorer

    polish_config = config.polish
    logger.info(
        "Polish suite: llm_model=%s device=%s over %d fixtures",
        polish_config.llm_model,
        polish_config.llm_device,
        len(fixtures),
    )

    disfluency = DisfluencyRemover() if polish_config.disfluency else None
    punctuation = PunctuationRestorer() if polish_config.punctuation else None
    formatting = SpokenFormFormatter() if polish_config.formatting else None
    llm = LLMCorrector(config=polish_config) if polish_config.llm else None

    if llm is not None:
        logger.info("LLM available: %s", llm.available)

    stage_latencies: dict[str, list[float]] = {
        "polish_4a": [],
        "polish_4b": [],
        "polish_4d": [],
        "polish_4c": [],
        "polish_total": [],
    }
    pairs: list[tuple[str, str]] = []
    per_fixture: list[dict[str, Any]] = []
    exact_matches = 0
    llm_invocations = 0
    thinking_leaks = 0

    for fixture in fixtures:
        total_start = time.perf_counter()
        current = fixture.raw
        has_self_corrections = False

        if disfluency is not None:
            t = time.perf_counter()
            result = disfluency.process(current)
            stage_latencies["polish_4a"].append((time.perf_counter() - t) * 1000.0)
            current = result.text
            has_self_corrections = result.has_self_corrections

        if punctuation is not None:
            t = time.perf_counter()
            current = punctuation.process(current)
            stage_latencies["polish_4b"].append((time.perf_counter() - t) * 1000.0)

        if formatting is not None:
            t = time.perf_counter()
            current = formatting.process(current)
            stage_latencies["polish_4d"].append((time.perf_counter() - t) * 1000.0)

        ran_llm = False
        if (
            llm is not None
            and llm.available
            and (has_self_corrections or polish_config.llm_always)
        ):
            t = time.perf_counter()
            corrected = llm.process(current)
            stage_latencies["polish_4c"].append((time.perf_counter() - t) * 1000.0)
            ran_llm = True
            llm_invocations += 1
            if corrected and corrected.strip():
                current = corrected

        stage_latencies["polish_total"].append((time.perf_counter() - total_start) * 1000.0)

        # A hybrid-reasoning model that ignores its instructions leaks thinking
        # markers into the output.  Catching this is the whole point of Phase 1.
        if _has_thinking_leak(current):
            thinking_leaks += 1
            logger.warning("Thinking trace leaked on fixture %s: %r", fixture.id, current[:120])

        pairs.append((fixture.expected, current))
        if current.strip() == fixture.expected.strip():
            exact_matches += 1

        per_fixture.append(
            {
                "id": fixture.id,
                "tags": list(fixture.tags),
                "raw": fixture.raw,
                "expected": fixture.expected,
                "actual": current,
                "wer": metrics.word_error_rate(fixture.expected, current).wer,
                "exact": current.strip() == fixture.expected.strip(),
                "llm_invoked": ran_llm,
                "self_corrections_detected": has_self_corrections,
            }
        )

    total = metrics.corpus_wer(pairs)
    expected_all = " ".join(f.expected for f in fixtures)
    actual_all = " ".join(p[1] for p in pairs)

    return {
        "metrics": {
            "polish": {
                **total.to_dict(),
                "fixtures": len(fixtures),
                "exact_match": exact_matches / len(fixtures) if fixtures else 0.0,
                "llm_invocations": llm_invocations,
                "thinking_leaks": thinking_leaks,
            },
            "punctuation": metrics.punctuation_f1(expected_all, actual_all),
            "capitalization": {
                "accuracy": metrics.capitalization_accuracy(expected_all, actual_all)
            },
            "latency": {
                stage: metrics.summarize_latency(samples).to_dict()
                for stage, samples in stage_latencies.items()
                if samples
            },
        },
        "per_fixture": per_fixture,
    }


# ---------------------------------------------------------------------------
# VAD suite
# ---------------------------------------------------------------------------


def run_vad_suite(config: Config, fixtures: list[AudioFixture]) -> dict[str, Any]:
    """Score voice activity detection on speech, silence, and quiet noise.

    Exists because a silently mis-sized VAD window is invisible everywhere
    else: the model accepts the wrong window length without error and simply
    returns ~0 for every frame, so dictation still "works" in hold-to-talk mode
    while VAD-driven auto-stop never fires.  This suite fails loudly instead.
    """
    from linux_whisper.audio import (
        SILERO_MODEL_PATH,
        VAD_WINDOW_SAMPLES,
        SileroVAD,
    )

    logger.info("VAD suite: window=%d samples over %d fixtures", VAD_WINDOW_SAMPLES, len(fixtures))
    vad = SileroVAD(SILERO_MODEL_PATH)
    threshold = config.audio.vad_threshold

    def frame_probabilities(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        probabilities: list[float] = []
        for offset in range(0, len(audio) - VAD_WINDOW_SAMPLES, VAD_WINDOW_SAMPLES):
            probabilities.append(vad(audio[offset : offset + VAD_WINDOW_SAMPLES]))
        vad.reset_state()
        return np.asarray(probabilities, dtype=np.float32)

    speech_rates: list[float] = []
    per_fixture: list[dict[str, Any]] = []
    window_latencies: list[float] = []

    for fixture in fixtures:
        audio = load_audio(fixture.path)
        start = time.perf_counter()
        probabilities = frame_probabilities(audio)
        if len(probabilities):
            window_latencies.append(
                (time.perf_counter() - start) * 1000.0 / len(probabilities)
            )

        rate = float(np.mean(probabilities > threshold)) if len(probabilities) else 0.0
        speech_rates.append(rate)
        per_fixture.append(
            {
                "id": fixture.id,
                "speech_frame_rate": rate,
                "mean_probability": float(probabilities.mean()) if len(probabilities) else 0.0,
                "windows": len(probabilities),
            }
        )

    # Negative controls: neither digital silence nor low-level noise may register
    # as speech, or the "detection" above is just a stuck-high output.
    silence = np.zeros(16_000 * 3, dtype=np.float32)
    noise = (np.random.default_rng(0).standard_normal(16_000 * 3) * 0.01).astype(np.float32)
    silence_rate = float(np.mean(frame_probabilities(silence) > threshold))
    noise_rate = float(np.mean(frame_probabilities(noise) > threshold))

    mean_speech_rate = sum(speech_rates) / len(speech_rates) if speech_rates else 0.0
    logger.info(
        "VAD: speech %.1f%% | silence %.1f%% | noise %.1f%%",
        mean_speech_rate * 100,
        silence_rate * 100,
        noise_rate * 100,
    )

    return {
        "metrics": {
            "vad": {
                "window_samples": VAD_WINDOW_SAMPLES,
                "threshold": threshold,
                "speech_frame_rate": mean_speech_rate,
                "silence_false_positive_rate": silence_rate,
                "noise_false_positive_rate": noise_rate,
                "fixtures": len(fixtures),
            },
            "latency": {"vad_window": metrics.summarize_latency(window_latencies).to_dict()},
        },
        "per_fixture": per_fixture,
    }


_THINKING_MARKERS = ("<think>", "</think>", "<thinking>", "Let me think", "Okay, so the user")


def _has_thinking_leak(text: str) -> bool:
    """True if *text* contains reasoning-trace markers that should never ship."""
    return any(marker.lower() in text.lower() for marker in _THINKING_MARKERS)


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *extra* into *base*, returning a new dict."""
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def describe_stack(config: Config) -> dict[str, Any]:
    """Record exactly which models produced a result, for provenance in the JSON."""
    return {
        "stt_backend": config.stt.backend,
        "stt_model": config.stt.model,
        "stt_device": config.stt.device,
        "llm_model": config.polish.llm_model,
        "llm_device": config.polish.llm_device,
        "polish_stages": {
            "disfluency": config.polish.disfluency,
            "punctuation": config.polish.punctuation,
            "formatting": config.polish.formatting,
            "llm": config.polish.llm,
            "llm_always": config.polish.llm_always,
        },
        "vad_threshold": config.audio.vad_threshold,
    }


def build_result(label: str, config: Config, suites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine suite outputs into the on-disk result document."""
    combined_metrics: dict[str, Any] = {}
    per_suite: dict[str, Any] = {}
    for name, suite in suites.items():
        combined_metrics = _deep_merge(combined_metrics, suite.get("metrics", {}))
        per_suite[name] = suite.get("per_fixture", [])

    return {
        "schema": SCHEMA_VERSION,
        "label": label,
        "timestamp": datetime.now(UTC).isoformat(),
        "stack": describe_stack(config),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "metrics": combined_metrics,
        "peak_rss_mb": peak_rss_mb(),
        "per_fixture": per_suite,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable digest of a run to stdout."""
    metrics = result.get("metrics", {})
    print(f"\n=== {result.get('label', 'run')} ===")
    stack = result.get("stack", {})
    print(f"  STT:   {stack.get('stt_backend')} / {stack.get('stt_model')} "
          f"({stack.get('stt_device')})")
    print(f"  LLM:   {stack.get('llm_model')} ({stack.get('llm_device')})")

    if "stt" in metrics:
        stt = metrics["stt"]
        print(f"\n  STT WER:          {stt['wer']:.4f}  "
              f"({stt['errors']}/{stt['reference_words']} words, "
              f"{stt['fixtures']} fixtures)")
        if "rtfx" in metrics:
            print(f"  RTFx:             {metrics['rtfx']:.1f}")

    if "polish" in metrics:
        polish = metrics["polish"]
        print(f"\n  Polish WER:       {polish['wer']:.4f}")
        print(f"  Exact match:      {polish['exact_match']:.1%}")
        print(f"  LLM invocations:  {polish['llm_invocations']}")
        if polish.get("thinking_leaks"):
            print(f"  !! thinking leaks: {polish['thinking_leaks']}")

    if "vad" in metrics:
        vad = metrics["vad"]
        print(f"\n  VAD window:       {vad['window_samples']} samples "
              f"(threshold {vad['threshold']})")
        print(f"  Speech detected:  {vad['speech_frame_rate']:.1%} of frames")
        print(f"  False positives:  silence {vad['silence_false_positive_rate']:.1%}, "
              f"noise {vad['noise_false_positive_rate']:.1%}")
        if vad["speech_frame_rate"] < MIN_HEALTHY_SPEECH_RATE:
            print("  !! VAD detects almost no speech — check VAD_WINDOW_SAMPLES")

    if "punctuation" in metrics:
        print(f"  Punctuation F1:   {metrics['punctuation']['f1']:.4f}")
    if "capitalization" in metrics:
        print(f"  Capitalisation:   {metrics['capitalization']['accuracy']:.4f}")

    latency = metrics.get("latency", {})
    if latency:
        print("\n  Latency (ms)      p50      p95      max")
        for stage, values in sorted(latency.items()):
            print(f"    {stage:<14} {values['p50_ms']:>8.1f} "
                  f"{values['p95_ms']:>8.1f} {values['max_ms']:>8.1f}")

    print(f"\n  Peak RSS:         {result.get('peak_rss_mb', 0):.0f} MB")


def run_compare(baseline_path: Path, candidate_path: Path, thresholds: metrics.Thresholds) -> int:
    """Diff two result files.  Returns a process exit code."""
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())

    print(f"\nbaseline:  {baseline.get('label')}  ({baseline_path})")
    print(f"candidate: {candidate.get('label')}  ({candidate_path})")

    _print_metric_deltas(baseline.get("metrics", {}), candidate.get("metrics", {}))

    regressions = metrics.compare_runs(baseline, candidate, thresholds)
    if regressions:
        print(f"\n❌ {len(regressions)} regression(s):")
        for regression in regressions:
            print(f"   - {regression.format()}")
        return 1

    print("\n✅ no regressions — candidate is safe to adopt")
    return 0


def _print_metric_deltas(base: dict[str, Any], cand: dict[str, Any]) -> None:
    """Print a side-by-side table of the headline metrics."""
    rows: list[tuple[str, float, float, bool]] = []

    for path, lower_better in (
        ("stt.wer", True),
        ("polish.wer", True),
        ("polish.exact_match", False),
        ("punctuation.f1", False),
        ("capitalization.accuracy", False),
        ("vad.speech_frame_rate", False),
        ("vad.silence_false_positive_rate", True),
        ("vad.noise_false_positive_rate", True),
    ):
        b, c = _lookup(base, path), _lookup(cand, path)
        if b is not None and c is not None:
            rows.append((path, b, c, lower_better))

    for stage in sorted(set(base.get("latency", {})) | set(cand.get("latency", {}))):
        b = _lookup(base, f"latency.{stage}.p95_ms")
        c = _lookup(cand, f"latency.{stage}.p95_ms")
        if b is not None and c is not None:
            rows.append((f"latency.{stage}.p95_ms", b, c, True))

    if not rows:
        return

    print(f"\n  {'metric':<32} {'baseline':>10} {'candidate':>10} {'delta':>10}")
    for name, b, c, lower_better in rows:
        delta = c - b
        arrow = "" if abs(delta) < 1e-9 else ("↓" if delta < 0 else "↑")
        good = (delta <= 0) if lower_better else (delta >= 0)
        mark = " " if abs(delta) < 1e-9 else ("✓" if good else "✗")
        print(f"  {name:<32} {b:>10.4f} {c:>10.4f} {delta:>9.4f}{arrow} {mark}")


def _lookup(data: dict[str, Any], path: str) -> float | None:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, int | float) else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.benchmarks.run",
        description="Benchmark the linux-whisper model stack and diff runs.",
    )
    parser.add_argument(
        "--suite",
        choices=("stt", "polish", "vad", "all"),
        default="all",
        help="Which suite to run (default: all)",
    )
    parser.add_argument("--config", type=Path, help="Config YAML (default: installed config)")
    parser.add_argument("--label", default="run", help="Name recorded in the result file")
    parser.add_argument("--out", type=Path, help="Write the JSON result here")
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        help="Directory of <name>.wav + <name>.txt pairs (default: LibriSpeech test-clean)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_AUDIO_FIXTURE_COUNT,
        help=f"Audio fixtures to score (default: {DEFAULT_AUDIO_FIXTURE_COUNT})",
    )
    parser.add_argument("--stt-backend", help="Override stt.backend")
    parser.add_argument("--stt-model", help="Override stt.model")
    parser.add_argument("--stt-device", help="Override stt.device")
    parser.add_argument("--llm-model", help="Override polish.llm_model")
    parser.add_argument(
        "--llm-always",
        action="store_true",
        help="Run stage 4c on every fixture, not only self-corrections",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASELINE", "CANDIDATE"),
        help="Diff two result files and exit non-zero on regression",
    )
    parser.add_argument(
        "--wer-tolerance",
        type=float,
        default=metrics.DEFAULT_WER_TOLERANCE,
        help=f"Absolute WER regression tolerance (default: {metrics.DEFAULT_WER_TOLERANCE})",
    )
    parser.add_argument(
        "--latency-tolerance",
        type=float,
        default=metrics.DEFAULT_LATENCY_RATIO,
        help=f"p95 latency slowdown ratio allowed (default: {metrics.DEFAULT_LATENCY_RATIO})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI model overrides onto a loaded config."""
    stt = config.stt
    if args.stt_backend:
        stt = replace(stt, backend=args.stt_backend)
    if args.stt_model:
        stt = replace(stt, model=args.stt_model)
    if args.stt_device:
        stt = replace(stt, device=args.stt_device)

    polish = config.polish
    if args.llm_model:
        polish = replace(polish, llm_model=args.llm_model)
    if args.llm_always:
        polish = replace(polish, llm_always=True)

    return replace(config, stt=stt, polish=polish)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    thresholds = metrics.Thresholds(
        wer_abs=args.wer_tolerance,
        latency_ratio=args.latency_tolerance,
    )

    if args.compare:
        return run_compare(args.compare[0], args.compare[1], thresholds)

    config = apply_overrides(Config.load(args.config), args)

    suites: dict[str, dict[str, Any]] = {}
    if args.suite in ("stt", "all"):
        fixtures = load_audio_fixtures(args.fixtures_dir, args.count)
        if not fixtures:
            logger.error("No audio fixtures found — cannot run the STT suite")
            return 2
        suites["stt"] = run_stt_suite(config, fixtures)

    if args.suite in ("vad", "all"):
        vad_fixtures = load_audio_fixtures(args.fixtures_dir, min(args.count, 10))
        if not vad_fixtures:
            logger.error("No audio fixtures found — cannot run the VAD suite")
            return 2
        suites["vad"] = run_vad_suite(config, vad_fixtures)

    if args.suite in ("polish", "all"):
        suites["polish"] = run_polish_suite(config, load_text_fixtures())

    result = build_result(args.label, config, suites)
    print_summary(result)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
