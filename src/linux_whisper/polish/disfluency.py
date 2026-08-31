"""Stage 4a: Disfluency removal via BERT token classification (ONNX).

Removes filler words, repetitions, and false starts from raw STT transcripts.
Uses a BERT token classifier when the ONNX model is available; otherwise falls
back to a robust regex-based approach.

The module also detects self-correction patterns (e.g. "X... actually Y") and
flags them so the downstream LLM stage (4c) knows to activate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from linux_whisper.config import MODELS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing ONNX Runtime — optional dependency
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    logger.debug(
        "onnxruntime or numpy not available; "
        "DisfluencyRemover will use regex fallback"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = MODELS_DIR / "disfluency-bert"
_MODEL_FILENAME = "model.onnx"
_VOCAB_FILENAME = "vocab.txt"

# Labels emitted by the BERT token classifier (BIO scheme)
_LABEL_KEEP = 0  # O  — keep the token
_LABEL_REMOVE = 1  # B-RM / I-RM — filler / repetition / false start
_LABEL_REPAIR = 2  # B-RP / I-RP — self-correction repair marker

# ---------------------------------------------------------------------------
# Regex fallback patterns
# ---------------------------------------------------------------------------

# Filler words and discourse markers (matched as whole words, case-insensitive).
# Order matters: longer phrases first to avoid partial matches.
_FILLER_PHRASES: list[str] = [
    r"you\s+know\s+what\s+I\s+mean",
    r"you\s+know",
    r"I\s+mean",
    r"kind\s+of",
    r"sort\s+of",
    r"at\s+the\s+end\s+of\s+the\s+day",
    r"to\s+be\s+honest",
]

# Unambiguous fillers: not English words, always safe to strip anywhere.
_FILLER_WORDS: list[str] = [
    "um+",
    "uh+",
    "ah+",
    "eh+",
    "er+",
    "hmm+",
    "hm+",
    "mm+",
    "mhm+",
    "erm+",
]

# Ambiguous fillers: ordinary English words that *sometimes* function as
# discourse fillers ("So, I think..." / "It was, like, really fast.") but are
# just as often plain content ("I like this design.", "Turn right at the
# light."). Whether a given occurrence is a filler is a contextual judgement
# — see `_remove_ambiguous_fillers()` below and issue #43. This mirrors the
# `_is_literal_token()` precedent in `polish/punctuation.py`: a rule that is
# right in general and wrong on a recognisable subset gets a predicate, not a
# flat pattern.
_AMBIGUOUS_FILLERS: list[str] = [
    "like",
    "basically",
    "actually",
    "literally",
    "right",
    "so",
    "well",
    "anyway",
    "anyways",
]
_AMBIGUOUS_FILLER_SET: frozenset[str] = frozenset(w.lower() for w in _AMBIGUOUS_FILLERS)

# Characters stripped from a token's edges to compare its bare word form.
_EDGE_PUNCT = ".,!?;:\"'()[]{}"

# Build compiled patterns.
_phrase_alts = "|".join(_FILLER_PHRASES)
_word_alts = "|".join(_FILLER_WORDS)
_FILLER_PHRASE_RE = re.compile(rf"(?<!['\w])(?:{_phrase_alts})(?!['\w])", re.IGNORECASE)
_FILLER_WORD_RE = re.compile(rf"(?<!['\w])(?:{_word_alts})(?!['\w])", re.IGNORECASE)
_UNAMBIGUOUS_WORD_RE = re.compile(rf"^(?:{_word_alts})$", re.IGNORECASE)

# Word-level repetitions: "I I I think" → "I think", "the the" → "the".
_REPETITION_RE = re.compile(
    r"\b(\w+)(?:\s+\1){1,}\b",
    re.IGNORECASE,
)

# Self-correction patterns — the speaker backtracks and rephrases.
_SELF_CORRECTION_PATTERNS: list[re.Pattern[str]] = [
    # "X actually Y" / "X wait Y" / "X no Y" / "X sorry Y" / "X I mean Y"
    re.compile(
        r"(?P<reparandum>\b.{2,40}?)\s+"
        r"(?:actually|wait|no|sorry|I\s+mean|rather|or\s+rather)\s+"
        r"(?P<repair>.+)",
        re.IGNORECASE,
    ),
    # "X... Y" — dash/ellipsis mid-sentence restart
    re.compile(
        r"(?P<reparandum>\b.{2,40}?)\s*"
        r"(?:--|—|\.\.\.)\s*"
        r"(?P<repair>.+)",
    ),
    # "X, no, Y" — comma-separated correction
    re.compile(
        r"(?P<reparandum>\b.{2,40}?)\s*,\s*"
        r"(?:no|wait|sorry|actually)\s*,\s*"
        r"(?P<repair>.+)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DisfluencyResult:
    """Output of the disfluency removal stage."""

    text: str
    has_self_corrections: bool


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DisfluencyRemover:
    """Remove disfluencies (fillers, repetitions, false starts) from text.

    When a trained BERT ONNX model is available under *model_dir*, it is used
    for token-level classification.  Otherwise a regex-based heuristic provides
    a reasonable fallback that handles the most common English disfluencies.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._session: ort.InferenceSession | None = None  # type: ignore[name-defined]
        self._vocab: dict[str, int] = {}
        self._id_to_label: dict[int, int] = {}
        self._using_onnx = False

        self._try_load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to load the ONNX model; fall back to regex silently."""
        if not _ONNX_AVAILABLE:
            logger.info("ONNX Runtime unavailable — using regex fallback")
            return

        model_path = self._model_dir / _MODEL_FILENAME
        vocab_path = self._model_dir / _VOCAB_FILENAME

        if not model_path.exists():
            logger.info(
                "Disfluency ONNX model not found at %s — using regex fallback",
                model_path,
            )
            return

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.inter_op_num_threads = 1
            sess_opts.intra_op_num_threads = 2
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._vocab = self._load_vocab(vocab_path)
            self._using_onnx = True
            logger.info("Loaded disfluency BERT model from %s", model_path)
        except Exception:
            logger.exception("Failed to load disfluency ONNX model")
            self._session = None

    @staticmethod
    def _load_vocab(path: Path) -> dict[str, int]:
        """Load a WordPiece vocab.txt into a token→id mapping."""
        vocab: dict[str, int] = {}
        if not path.exists():
            logger.warning("vocab.txt not found at %s", path)
            return vocab
        with open(path) as f:
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx
        return vocab

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> DisfluencyResult:
        """Remove disfluencies from *text*.

        Returns a :class:`DisfluencyResult` with the cleaned text and a flag
        indicating whether self-corrections were detected (which signals the
        LLM stage to activate).
        """
        if not text or not text.strip():
            return DisfluencyResult(text="", has_self_corrections=False)

        if self._using_onnx and self._session is not None:
            return self._process_onnx(text)
        return self._process_regex(text)

    # ------------------------------------------------------------------
    # ONNX path
    # ------------------------------------------------------------------

    def _process_onnx(self, text: str) -> DisfluencyResult:
        """Run the BERT token classifier on *text*."""
        assert self._session is not None

        tokens = text.split()
        if not tokens:
            return DisfluencyResult(text="", has_self_corrections=False)

        # Tokenise with WordPiece (simplified — real tokeniser would handle
        # subwords; this assumes pre-tokenised input aligned to the vocab).
        input_ids: list[int] = [self._vocab.get("[CLS]", 101)]
        token_map: list[int] = []  # maps each wordpiece position → word idx
        for word_idx, word in enumerate(tokens):
            wp_id = self._vocab.get(word.lower(), self._vocab.get("[UNK]", 100))
            input_ids.append(wp_id)
            token_map.append(word_idx)
        input_ids.append(self._vocab.get("[SEP]", 102))

        ids_array = np.array([input_ids], dtype=np.int64)
        attn_mask = np.ones_like(ids_array, dtype=np.int64)
        token_type = np.zeros_like(ids_array, dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids": ids_array,
                "attention_mask": attn_mask,
                "token_type_ids": token_type,
            },
        )
        # outputs[0] shape: (1, seq_len, num_labels)
        logits = outputs[0][0]
        # Strip [CLS] and [SEP]
        logits = logits[1 : len(token_map) + 1]

        has_self_corrections = False
        kept: list[str] = []

        for i, word_idx in enumerate(token_map):
            label = int(np.argmax(logits[i]))
            if label == _LABEL_KEEP:
                kept.append(tokens[word_idx])
            elif label == _LABEL_REPAIR:
                kept.append(tokens[word_idx])
                has_self_corrections = True
            # _LABEL_REMOVE: skip the token

        cleaned = " ".join(kept)
        cleaned = _normalise_whitespace(cleaned)
        return DisfluencyResult(text=cleaned, has_self_corrections=has_self_corrections)

    # ------------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------------

    def _process_regex(self, text: str) -> DisfluencyResult:
        """Heuristic disfluency removal using regex patterns."""
        has_self_corrections = _detect_self_corrections(text)
        cleaned = _remove_fillers(text)
        cleaned = _remove_repetitions(cleaned)
        cleaned = _normalise_whitespace(cleaned)
        return DisfluencyResult(text=cleaned, has_self_corrections=has_self_corrections)


# ---------------------------------------------------------------------------
# Regex helpers (module-level so they can be reused / tested independently)
# ---------------------------------------------------------------------------


def _detect_self_corrections(text: str) -> bool:
    """Return True if *text* contains a self-correction pattern."""
    for pattern in _SELF_CORRECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _bare_word(token: str) -> str:
    """Strip leading/trailing punctuation from *token* for word comparison."""
    return token.strip(_EDGE_PUNCT)


def _is_unambiguous_filler(bare: str) -> bool:
    """True if *bare* is one of the non-word fillers (um, uh, hmm, ...)."""
    return bool(bare) and _UNAMBIGUOUS_WORD_RE.fullmatch(bare) is not None


def _remove_ambiguous_fillers(text: str) -> str:
    """Strip an ambiguous filler only where context marks it as disfluent.

    An ambiguous word (e.g. "like", "so", "right") is content by default —
    stripping it unconditionally deletes ordinary English ("I like this
    design." -> "I this design.", issue #43). It is only treated as a filler
    when *any* of these context cues hold:

    - it is utterance-initial and followed by a comma ("So, I think...");
    - it is bounded by commas on both sides ("..., like, ...");
    - it is immediately adjacent to an unambiguous filler ("um, like, ...");
    - it is utterance-final and preceded by a comma ("..., right.").

    This operates on the ORIGINAL token list (before unambiguous fillers are
    stripped) so the adjacency cue still sees "um"/"uh"/etc. as a neighbour.
    """
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return text

    bares = [_bare_word(t) for t in tokens]
    ends_comma = [t.endswith(",") for t in tokens]

    remove = [False] * n
    clear_preceding_comma = [False] * n

    for i in range(n):
        if bares[i].lower() not in _AMBIGUOUS_FILLER_SET:
            continue

        preceded_by_comma = i > 0 and ends_comma[i - 1]
        followed_by_comma = ends_comma[i]
        prev_bare = bares[i - 1] if i > 0 else ""
        next_bare = bares[i + 1] if i + 1 < n else ""
        adjacent_unambiguous = _is_unambiguous_filler(prev_bare) or _is_unambiguous_filler(
            next_bare
        )
        is_initial = i == 0
        is_final = i == n - 1

        is_filler = (
            (is_initial and followed_by_comma)
            or (preceded_by_comma and followed_by_comma)
            or adjacent_unambiguous
            or (is_final and preceded_by_comma)
        )
        if is_filler:
            remove[i] = True
            if preceded_by_comma:
                clear_preceding_comma[i] = True

    if not any(remove):
        return text

    output: list[str] = []
    for i, token in enumerate(tokens):
        if remove[i]:
            # The comma pairing with this filler was marking the aside it
            # introduced ("It was, like, really fast."); with the filler
            # gone the comma is an orphan and must go too, or the sentence
            # reads as if it were cut off ("It was, really fast.").
            if clear_preceding_comma[i] and output and output[-1].endswith(","):
                output[-1] = output[-1][:-1]
            continue
        output.append(token)

    return " ".join(output)


def _has_word_char(text: str) -> bool:
    """True if *text* contains at least one alphanumeric character."""
    return any(ch.isalnum() for ch in text)


def _remove_fillers(text: str) -> str:
    """Strip filler words / discourse markers from *text*.

    Unambiguous fillers (um, uh, hmm, ...) are stripped anywhere.  Ambiguous
    fillers (like, so, right, ...) are stripped only when context marks them
    as disfluent — see `_remove_ambiguous_fillers()`.

    Hard backstop: if removal would reduce non-empty, non-punctuation-only
    input to empty or punctuation-only text, the input is returned unchanged.
    No dictation may ever inject nothing when the STT produced words.
    """
    without_phrases = _FILLER_PHRASE_RE.sub("", text)
    without_ambiguous = _remove_ambiguous_fillers(without_phrases)
    cleaned = _FILLER_WORD_RE.sub("", without_ambiguous)

    if _has_word_char(text) and not _has_word_char(cleaned):
        return text
    return cleaned


# Repeated number words are almost never a stammer.  Spoken years repeat by
# construction ("twenty twenty six" is 2026, "twenty twenty" is 2020), and
# "fifty fifty" is an idiom.  Collapsing them silently destroyed every dictated
# year in this decade before stage 4d ever saw the text.
_NUMBER_WORDS_NO_COLLAPSE: frozenset[str] = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion",
})


def _remove_repetitions(text: str) -> str:
    """Collapse consecutive repeated words: 'I I I think' → 'I think'.

    Number words are exempt — see :data:`_NUMBER_WORDS_NO_COLLAPSE`.
    """

    def _collapse(match: re.Match[str]) -> str:
        word = match.group(1)
        if word.lower() in _NUMBER_WORDS_NO_COLLAPSE:
            return match.group(0)
        return word

    return _REPETITION_RE.sub(_collapse, text)


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip leading/trailing space."""
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()
