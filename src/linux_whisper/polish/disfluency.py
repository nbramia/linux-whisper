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
    "like",
    "basically",
    "actually",
    "literally",
    "right",
    "so",
    "well",
    "anyway",
    "anyways",
    "okay",
    "ok",
]

# Build a single compiled pattern for fillers.
_phrase_alts = "|".join(_FILLER_PHRASES)
_word_alts = "|".join(_FILLER_WORDS)
_FILLER_RE = re.compile(
    rf"(?<!['\w])(?:{_phrase_alts}|{_word_alts})(?!['\w])",
    re.IGNORECASE,
)

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


def _remove_fillers(text: str) -> str:
    """Strip filler words / discourse markers from *text*."""
    return _FILLER_RE.sub("", text)


def _remove_repetitions(text: str) -> str:
    """Collapse consecutive repeated words: 'I I I think' → 'I think'."""
    return _REPETITION_RE.sub(r"\1", text)


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip leading/trailing space."""
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()
