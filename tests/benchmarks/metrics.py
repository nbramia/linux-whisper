"""Scoring primitives for the model A/B harness — WER, latency, regressions.

Every function here is pure and dependency-free so it can be unit-tested in CI
without downloading a model or touching an audio device.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Characters stripped before word-level comparison.  Punctuation is scored
# separately (see :func:`punctuation_f1`) because an ASR backend that emits
# punctuation natively should not be penalised against one that does not.
_PUNCT_RE = re.compile(r"[^\w\s']")
_WS_RE = re.compile(r"\s+")

# Symbols an ASR backend may render either way.  Substituted *before*
# punctuation stripping, since the symbol forms would otherwise be deleted.
# "$40" vs "40 dollars" is deliberately absent: the symbol precedes the number
# and the word follows it, so substituting would turn one alignment error into
# two.  Currency needs real reordering, which is more than this belongs doing.
_SYMBOL_EQUIVALENTS: tuple[tuple[str, str], ...] = (
    ("%", " percent "),
    ("&", " and "),
)

# Spoken/written word variants that must not count as errors.  Kept
# deliberately small — an aggressive normaliser hides real regressions.
_EQUIVALENTS: dict[str, str] = {
    "okay": "ok",
    "alright": "all right",
    "cannot": "can not",
    "gonna": "going to",
    "wanna": "want to",
    "mr": "mister",
    "mrs": "missus",
    "dr": "doctor",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and accents, collapse whitespace.

    Applied to both reference and hypothesis before word-level scoring.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    for symbol, word in _SYMBOL_EQUIVALENTS:
        text = text.replace(symbol, word)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()

    if not text:
        return ""

    words = [_EQUIVALENTS.get(w, w) for w in text.split(" ")]
    return " ".join(words)


def tokenize(text: str) -> list[str]:
    """Normalise *text* and split into comparison tokens."""
    normalized = normalize_text(text)
    return normalized.split(" ") if normalized else []


@dataclass(frozen=True, slots=True)
class WerResult:
    """Word error rate broken down by edit operation."""

    reference_words: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        """Errors per reference word.  0.0 for an empty reference and hypothesis."""
        if self.reference_words == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.reference_words

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_words": self.reference_words,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "wer": self.wer,
        }


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Levenshtein word error rate between *reference* and *hypothesis*.

    Uses the standard dynamic-programming alignment with unit cost for
    substitution, deletion, and insertion, then backtracks to attribute the
    edit distance to each operation type.
    """
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)

    n, m = len(ref), len(hyp)

    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ref_word = ref[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            if ref_word == hyp[j - 1]:
                row[j] = prev_row[j - 1]
            else:
                row[j] = 1 + min(prev_row[j - 1], prev_row[j], row[j - 1])

    # Backtrack to classify each edit.
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return WerResult(reference_words=n, substitutions=subs, deletions=dels, insertions=ins)


def corpus_wer(pairs: list[tuple[str, str]]) -> WerResult:
    """Aggregate WER over (reference, hypothesis) pairs.

    Errors and reference lengths are pooled before dividing, which is the
    standard corpus-level definition — averaging per-utterance WER would
    over-weight short utterances.
    """
    total = WerResult(0, 0, 0, 0)
    for reference, hypothesis in pairs:
        r = word_error_rate(reference, hypothesis)
        total = WerResult(
            reference_words=total.reference_words + r.reference_words,
            substitutions=total.substitutions + r.substitutions,
            deletions=total.deletions + r.deletions,
            insertions=total.insertions + r.insertions,
        )
    return total


# ---------------------------------------------------------------------------
# Punctuation / capitalisation scoring
# ---------------------------------------------------------------------------


def punctuation_f1(reference: str, hypothesis: str) -> dict[str, float]:
    """Token-level F1 over sentence-ending and comma punctuation.

    Scored on the multiset of punctuation marks rather than their positions —
    positional scoring needs an alignment that word-level normalisation has
    already discarded, and mark counts catch the failure we care about
    (a backend that emits no punctuation at all).
    """
    marks = ".,?!;:"
    ref_counts = {m: reference.count(m) for m in marks}
    hyp_counts = {m: hypothesis.count(m) for m in marks}

    true_positive = sum(min(ref_counts[m], hyp_counts[m]) for m in marks)
    predicted = sum(hyp_counts.values())
    actual = sum(ref_counts.values())

    precision = true_positive / predicted if predicted else (1.0 if actual == 0 else 0.0)
    recall = true_positive / actual if actual else (1.0 if predicted == 0 else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def capitalization_accuracy(reference: str, hypothesis: str) -> float:
    """Fraction of aligned word positions whose leading-capital state matches.

    Compares only up to the shorter of the two token sequences; a length
    mismatch is already penalised by WER.
    """
    ref_words = [w for w in re.split(r"\s+", reference.strip()) if w]
    hyp_words = [w for w in re.split(r"\s+", hypothesis.strip()) if w]
    if not ref_words or not hyp_words:
        return 1.0 if not ref_words and not hyp_words else 0.0

    compared = min(len(ref_words), len(hyp_words))
    matches = sum(
        1
        for i in range(compared)
        if ref_words[i][:1].isupper() == hyp_words[i][:1].isupper()
    )
    return matches / compared


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Distribution summary for a set of stage timings, in milliseconds."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
        }


def percentile(samples: list[float], fraction: float) -> float:
    """Nearest-rank percentile of *samples* (0.0 <= fraction <= 1.0)."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if fraction <= 0:
        return ordered[0]
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return ordered[rank - 1]


def summarize_latency(samples: list[float]) -> LatencySummary:
    """Reduce raw millisecond timings to a distribution summary."""
    if not samples:
        return LatencySummary(count=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, max_ms=0.0)
    return LatencySummary(
        count=len(samples),
        mean_ms=sum(samples) / len(samples),
        p50_ms=percentile(samples, 0.50),
        p95_ms=percentile(samples, 0.95),
        max_ms=max(samples),
    )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


# Defaults live at module scope, not only as dataclass field defaults: with
# ``slots=True`` the class attributes become member descriptors, so
# ``Thresholds.wer_abs`` is not the number — callers wanting the default value
# (argparse, docs) must read these constants instead.
DEFAULT_WER_TOLERANCE = 0.005
DEFAULT_LATENCY_RATIO = 1.10
DEFAULT_QUALITY_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class Thresholds:
    """How much a candidate may drift from baseline before it counts as a regression.

    Absolute WER tolerance is expressed in WER points (0.005 == 0.5 points), so
    a candidate may be marginally worse on a small fixture set without failing.
    Latency tolerance is a ratio: 1.10 allows a 10% slowdown.
    """

    wer_abs: float = DEFAULT_WER_TOLERANCE
    latency_ratio: float = DEFAULT_LATENCY_RATIO
    punctuation_f1_abs: float = DEFAULT_QUALITY_TOLERANCE
    capitalization_abs: float = DEFAULT_QUALITY_TOLERANCE


@dataclass(frozen=True, slots=True)
class Regression:
    """A single metric that moved the wrong way against baseline."""

    metric: str
    baseline: float
    candidate: float
    tolerance: str

    def format(self) -> str:
        return (
            f"{self.metric}: {self.baseline:.4f} → {self.candidate:.4f} "
            f"(tolerance {self.tolerance})"
        )


def _get(data: dict[str, Any], path: str) -> float | None:
    """Fetch a dotted path from a nested result dict, or None if absent."""
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, int | float) else None


# Metrics where a *higher* value is better.
_HIGHER_IS_BETTER = ("punctuation.f1", "capitalization.accuracy", "polish.exact_match")


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: Thresholds | None = None,
) -> list[Regression]:
    """Return every metric where *candidate* regressed against *baseline*.

    An empty list means the candidate is safe to adopt.  Metrics absent from
    either run are skipped rather than treated as regressions, so adding a new
    metric does not invalidate an older baseline.
    """
    thresholds = thresholds or Thresholds()
    regressions: list[Regression] = []

    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})

    # --- WER: lower is better, absolute tolerance in WER points -------------
    for path in ("stt.wer", "polish.wer"):
        b, c = _get(base_metrics, path), _get(cand_metrics, path)
        if b is None or c is None:
            continue
        if c > b + thresholds.wer_abs:
            regressions.append(
                Regression(path, b, c, f"+{thresholds.wer_abs:.4f} abs")
            )

    # --- Latency: lower is better, ratio tolerance --------------------------
    for stage, stage_data in cand_metrics.get("latency", {}).items():
        base_stage = base_metrics.get("latency", {}).get(stage)
        if not isinstance(base_stage, dict) or not isinstance(stage_data, dict):
            continue
        b, c = base_stage.get("p95_ms"), stage_data.get("p95_ms")
        if not isinstance(b, int | float) or not isinstance(c, int | float):
            continue
        if b > 0 and c > b * thresholds.latency_ratio:
            regressions.append(
                Regression(
                    f"latency.{stage}.p95_ms",
                    float(b),
                    float(c),
                    f"x{thresholds.latency_ratio:.2f}",
                )
            )

    # --- Quality metrics: higher is better ----------------------------------
    tolerance_for = {
        "punctuation.f1": thresholds.punctuation_f1_abs,
        "capitalization.accuracy": thresholds.capitalization_abs,
        "polish.exact_match": thresholds.punctuation_f1_abs,
    }
    for path in _HIGHER_IS_BETTER:
        b, c = _get(base_metrics, path), _get(cand_metrics, path)
        if b is None or c is None:
            continue
        tol = tolerance_for[path]
        if c < b - tol:
            regressions.append(Regression(path, b, c, f"-{tol:.4f} abs"))

    return regressions
