"""Tests for the benchmark harness's scoring logic.

Only the pure functions are covered here — these run in CI with no models, no
audio devices, and no network.  The suite runners themselves are exercised by
running the harness for real (see ``tests/benchmarks/README.md``).
"""

from __future__ import annotations

import pytest

from tests.benchmarks import metrics

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalizeText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Hello, World!", "hello world"),
            ("  spaced   out  ", "spaced out"),
            ("", ""),
            ("...", ""),
            ("don't stop", "don't stop"),
            ("Café", "cafe"),
            ("MIXED Case TEXT", "mixed case text"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert metrics.normalize_text(raw) == expected

    def test_apostrophes_are_preserved(self) -> None:
        # Contractions are real words; stripping the apostrophe would merge
        # "we'll" and "well" into the same token.
        assert metrics.normalize_text("we'll") == "we'll"

    @pytest.mark.parametrize(
        ("left", "right"),
        [("okay", "OK"), ("Mr Smith", "mister smith"), ("50 %", "50 percent")],
    )
    def test_known_equivalents_collapse(self, left: str, right: str) -> None:
        assert metrics.normalize_text(left) == metrics.normalize_text(right)

    def test_tokenize_empty_is_empty_list(self) -> None:
        assert metrics.tokenize("") == []
        assert metrics.tokenize("!!!") == []


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------


class TestWordErrorRate:
    def test_identical_is_zero(self) -> None:
        result = metrics.word_error_rate("the cat sat on the mat", "the cat sat on the mat")
        assert result.wer == 0.0
        assert result.errors == 0

    def test_single_substitution(self) -> None:
        result = metrics.word_error_rate("the cat sat", "the dog sat")
        assert result.substitutions == 1
        assert result.deletions == 0
        assert result.insertions == 0
        assert result.wer == pytest.approx(1 / 3)

    def test_single_deletion(self) -> None:
        result = metrics.word_error_rate("the cat sat", "the sat")
        assert result.deletions == 1
        assert result.substitutions == 0
        assert result.insertions == 0
        assert result.wer == pytest.approx(1 / 3)

    def test_single_insertion(self) -> None:
        result = metrics.word_error_rate("the cat sat", "the big cat sat")
        assert result.insertions == 1
        assert result.substitutions == 0
        assert result.deletions == 0
        assert result.wer == pytest.approx(1 / 3)

    def test_mixed_operations(self) -> None:
        # ref: the quick brown fox jumps
        # hyp: the quick red fox leaps over
        #        =     =    S    =    S    I
        result = metrics.word_error_rate(
            "the quick brown fox jumps", "the quick red fox leaps over"
        )
        assert result.reference_words == 5
        assert result.substitutions == 2
        assert result.insertions == 1
        assert result.deletions == 0
        assert result.wer == pytest.approx(3 / 5)

    def test_punctuation_and_case_are_ignored(self) -> None:
        result = metrics.word_error_rate("The cat sat.", "the CAT sat")
        assert result.wer == 0.0

    def test_completely_wrong(self) -> None:
        result = metrics.word_error_rate("alpha beta", "gamma delta")
        assert result.substitutions == 2
        assert result.wer == 1.0

    def test_empty_hypothesis_is_all_deletions(self) -> None:
        result = metrics.word_error_rate("the cat sat", "")
        assert result.deletions == 3
        assert result.wer == 1.0

    def test_empty_reference_and_hypothesis(self) -> None:
        assert metrics.word_error_rate("", "").wer == 0.0

    def test_empty_reference_with_output_is_maximally_wrong(self) -> None:
        # No reference words to divide by — anything emitted is pure insertion.
        result = metrics.word_error_rate("", "unexpected words")
        assert result.insertions == 2
        assert result.wer == 1.0

    def test_wer_can_exceed_one(self) -> None:
        result = metrics.word_error_rate("hello", "hello hello hello hello")
        assert result.wer > 1.0

    def test_edit_counts_sum_to_edit_distance(self) -> None:
        # Backtracking must attribute every edit exactly once.
        result = metrics.word_error_rate(
            "the quick brown fox jumps over the lazy dog",
            "a quick brown cat jumped over lazy dog today",
        )
        assert result.errors == result.substitutions + result.deletions + result.insertions
        assert result.errors > 0


class TestCorpusWer:
    def test_pools_errors_across_utterances(self) -> None:
        pairs = [("a b c d", "a b c d"), ("e f", "x y")]
        result = metrics.corpus_wer(pairs)
        assert result.reference_words == 6
        assert result.errors == 2
        assert result.wer == pytest.approx(2 / 6)

    def test_long_utterances_dominate(self) -> None:
        # Corpus WER pools before dividing, so one perfect long utterance
        # outweighs one wrong short one — unlike averaging per-utterance WER.
        long_utterance = "one two three four five six seven eight"
        pairs = [(long_utterance, long_utterance), ("nine", "wrong")]
        assert metrics.corpus_wer(pairs).wer == pytest.approx(1 / 9)

    def test_empty_corpus(self) -> None:
        assert metrics.corpus_wer([]).wer == 0.0


# ---------------------------------------------------------------------------
# Punctuation and capitalisation
# ---------------------------------------------------------------------------


class TestPunctuationF1:
    def test_perfect_match(self) -> None:
        scores = metrics.punctuation_f1("Hello. How are you?", "Hello. How are you?")
        assert scores["f1"] == pytest.approx(1.0)

    def test_no_punctuation_emitted_scores_zero(self) -> None:
        scores = metrics.punctuation_f1("Hello. How are you?", "hello how are you")
        assert scores["recall"] == 0.0
        assert scores["f1"] == 0.0

    def test_both_unpunctuated_is_perfect(self) -> None:
        scores = metrics.punctuation_f1("hello there", "hello there")
        assert scores["f1"] == pytest.approx(1.0)

    def test_over_punctuation_hurts_precision(self) -> None:
        scores = metrics.punctuation_f1("Hello there.", "Hello, there. Really?")
        assert scores["precision"] < 1.0


class TestCapitalizationAccuracy:
    def test_perfect(self) -> None:
        assert metrics.capitalization_accuracy("The Cat Sat", "The Cat Sat") == 1.0

    def test_all_lowercase_against_capitalized(self) -> None:
        assert metrics.capitalization_accuracy("The Cat", "the cat") == 0.0

    def test_partial(self) -> None:
        assert metrics.capitalization_accuracy("The cat Sat", "The cat sat") == pytest.approx(2 / 3)

    def test_length_mismatch_compares_common_prefix(self) -> None:
        # Length differences are already penalised by WER; don't double-count.
        assert metrics.capitalization_accuracy("The Cat", "The Cat Sat Down") == 1.0

    def test_both_empty(self) -> None:
        assert metrics.capitalization_accuracy("", "") == 1.0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


class TestLatency:
    def test_percentile_nearest_rank(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert metrics.percentile(samples, 0.5) == 3.0
        assert metrics.percentile(samples, 1.0) == 5.0

    def test_percentile_empty(self) -> None:
        assert metrics.percentile([], 0.95) == 0.0

    def test_percentile_single_sample(self) -> None:
        assert metrics.percentile([42.0], 0.95) == 42.0

    def test_summarize(self) -> None:
        summary = metrics.summarize_latency([10.0, 20.0, 30.0, 40.0])
        assert summary.count == 4
        assert summary.mean_ms == 25.0
        assert summary.max_ms == 40.0

    def test_summarize_empty(self) -> None:
        summary = metrics.summarize_latency([])
        assert summary.count == 0
        assert summary.p95_ms == 0.0

    def test_p95_tracks_the_tail(self) -> None:
        samples = [10.0] * 99 + [500.0]
        assert metrics.summarize_latency(samples).p95_ms == 10.0
        assert metrics.summarize_latency(samples).max_ms == 500.0


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def _run(**metrics: object) -> dict[str, object]:
    return {"metrics": metrics}


class TestCompareRuns:
    def test_identical_runs_have_no_regressions(self) -> None:
        run = _run(stt={"wer": 0.05}, latency={"stt": {"p95_ms": 300.0}})
        assert metrics.compare_runs(run, run) == []

    def test_wer_increase_beyond_tolerance_is_a_regression(self) -> None:
        baseline = _run(stt={"wer": 0.050})
        candidate = _run(stt={"wer": 0.070})
        regressions = metrics.compare_runs(baseline, candidate)
        assert len(regressions) == 1
        assert regressions[0].metric == "stt.wer"

    def test_wer_increase_within_tolerance_passes(self) -> None:
        baseline = _run(stt={"wer": 0.050})
        candidate = _run(stt={"wer": 0.053})
        assert metrics.compare_runs(baseline, candidate) == []

    def test_wer_improvement_is_never_a_regression(self) -> None:
        baseline = _run(stt={"wer": 0.080})
        candidate = _run(stt={"wer": 0.050})
        assert metrics.compare_runs(baseline, candidate) == []

    def test_latency_slowdown_beyond_ratio_is_a_regression(self) -> None:
        baseline = _run(latency={"stt": {"p95_ms": 300.0}})
        candidate = _run(latency={"stt": {"p95_ms": 400.0}})
        regressions = metrics.compare_runs(baseline, candidate)
        assert [r.metric for r in regressions] == ["latency.stt.p95_ms"]

    def test_latency_slowdown_within_ratio_passes(self) -> None:
        baseline = _run(latency={"stt": {"p95_ms": 300.0}})
        candidate = _run(latency={"stt": {"p95_ms": 320.0}})
        assert metrics.compare_runs(baseline, candidate) == []

    def test_quality_metric_drop_is_a_regression(self) -> None:
        baseline = _run(punctuation={"f1": 0.90})
        candidate = _run(punctuation={"f1": 0.80})
        regressions = metrics.compare_runs(baseline, candidate)
        assert [r.metric for r in regressions] == ["punctuation.f1"]

    def test_quality_metric_improvement_passes(self) -> None:
        baseline = _run(polish={"exact_match": 0.50})
        candidate = _run(polish={"exact_match": 0.70})
        assert metrics.compare_runs(baseline, candidate) == []

    def test_missing_metrics_are_skipped_not_failed(self) -> None:
        # An older baseline lacking a newly added metric must not fail a run.
        baseline = _run(stt={"wer": 0.05})
        candidate = _run(stt={"wer": 0.05}, punctuation={"f1": 0.9})
        assert metrics.compare_runs(baseline, candidate) == []

    def test_multiple_regressions_are_all_reported(self) -> None:
        baseline = _run(
            stt={"wer": 0.05},
            punctuation={"f1": 0.95},
            latency={"stt": {"p95_ms": 300.0}},
        )
        candidate = _run(
            stt={"wer": 0.09},
            punctuation={"f1": 0.70},
            latency={"stt": {"p95_ms": 600.0}},
        )
        assert len(metrics.compare_runs(baseline, candidate)) == 3

    def test_custom_thresholds_are_respected(self) -> None:
        baseline = _run(stt={"wer": 0.050})
        candidate = _run(stt={"wer": 0.070})
        loose = metrics.Thresholds(wer_abs=0.05)
        assert metrics.compare_runs(baseline, candidate, loose) == []

    def test_zero_baseline_latency_is_not_a_division_trap(self) -> None:
        baseline = _run(latency={"stt": {"p95_ms": 0.0}})
        candidate = _run(latency={"stt": {"p95_ms": 50.0}})
        assert metrics.compare_runs(baseline, candidate) == []

    def test_regression_formats_readably(self) -> None:
        regression = metrics.Regression("stt.wer", 0.05, 0.09, "+0.0050 abs")
        formatted = regression.format()
        assert "stt.wer" in formatted
        assert "0.0500" in formatted and "0.0900" in formatted


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestRunnerCli:
    def test_parser_builds_without_models_installed(self) -> None:
        from tests.benchmarks.run import build_parser

        args = build_parser().parse_args(["--suite", "stt", "--count", "5"])
        assert args.suite == "stt"
        assert args.count == 5

    def test_compare_mode_parses_two_paths(self) -> None:
        from tests.benchmarks.run import build_parser

        args = build_parser().parse_args(["--compare", "a.json", "b.json"])
        assert len(args.compare) == 2

    def test_tolerance_defaults_are_real_numbers(self) -> None:
        # Thresholds is a slots dataclass, so `Thresholds.wer_abs` is a member
        # descriptor rather than the default value.  Reading it as an argparse
        # default silently produced a non-numeric threshold that only blew up
        # once --compare ran.  The defaults must come from module constants.
        from tests.benchmarks.run import build_parser

        args = build_parser().parse_args([])
        assert isinstance(args.wer_tolerance, float)
        assert isinstance(args.latency_tolerance, float)
        assert metrics.Thresholds(
            wer_abs=args.wer_tolerance, latency_ratio=args.latency_tolerance
        ).wer_abs == metrics.DEFAULT_WER_TOLERANCE

    def test_thresholds_instance_exposes_numeric_defaults(self) -> None:
        thresholds = metrics.Thresholds()
        assert isinstance(thresholds.wer_abs, float)
        assert isinstance(thresholds.latency_ratio, float)
        assert isinstance(thresholds.punctuation_f1_abs, float)

    def test_thinking_leak_detection(self) -> None:
        from tests.benchmarks.run import _has_thinking_leak

        assert _has_thinking_leak("<think>hmm</think> Ship it.")
        assert _has_thinking_leak("Let me think about this.")
        assert not _has_thinking_leak("Ship it.")

    def test_deep_merge_combines_nested_suites(self) -> None:
        from tests.benchmarks.run import _deep_merge

        merged = _deep_merge(
            {"latency": {"stt": {"p95_ms": 1.0}}, "stt": {"wer": 0.1}},
            {"latency": {"polish_4a": {"p95_ms": 2.0}}},
        )
        assert merged["latency"]["stt"]["p95_ms"] == 1.0
        assert merged["latency"]["polish_4a"]["p95_ms"] == 2.0
        assert merged["stt"]["wer"] == 0.1
