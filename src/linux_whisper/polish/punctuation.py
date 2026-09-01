"""Stage 4b: Punctuation and capitalisation restoration via ELECTRA (ONNX).

Adds punctuation (periods, commas, question marks, exclamation marks) and
fixes capitalisation on cleaned transcript text.  Uses a pair of ELECTRA-small
token classifiers when ONNX models are available; otherwise applies a
rule-based heuristic that handles the most common cases.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from linux_whisper.config import MODELS_DIR

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ONNX import
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    logger.debug(
        "onnxruntime or numpy not available; "
        "PunctuationRestorer will use rule-based fallback"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = MODELS_DIR / "punctuation-electra"
_PUNCT_MODEL = "punct_model.onnx"
_CAPS_MODEL = "caps_model.onnx"
_VOCAB_FILENAME = "vocab.txt"

# Punctuation labels emitted by the ELECTRA classifier
_PUNCT_LABELS: dict[int, str] = {
    0: "",       # NONE
    1: ",",      # COMMA
    2: ".",      # PERIOD
    3: "?",      # QUESTION
    4: "!",      # EXCLAMATION
    5: ":",      # COLON
    6: ";",      # SEMICOLON
}

# Capitalisation labels
_CAP_LABELS: dict[int, str] = {
    0: "lower",
    1: "upper_first",  # Capitalise first letter
    2: "all_upper",    # ALL CAPS (acronyms, etc.)
}

# ---------------------------------------------------------------------------
# Rule-based fallback helpers
# ---------------------------------------------------------------------------

# Words / phrases that strongly suggest the sentence is a question.
_QUESTION_STARTERS: set[str] = {
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "am",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "shall",
    "may",
    "might",
    "have",
    "has",
    "had",
    "isn't",
    "aren't",
    "don't",
    "doesn't",
    "didn't",
    "can't",
    "couldn't",
    "won't",
    "wouldn't",
    "shouldn't",
}

# Sentence-ending punctuation characters.
_TERMINAL_PUNCT = frozenset(".?!")

# Abbreviations / acronyms that should be capitalised.
_ALWAYS_UPPER: set[str] = {
    "i",  # The pronoun "I"
}

# ---------------------------------------------------------------------------
# Literal (code-like) tokens
# ---------------------------------------------------------------------------

# Dictating code means the transcript contains tokens that are *not* prose and
# must survive byte-identical.  Prose rules actively corrupt them:
#
#   "server-test.sh"        -> "Server-test.sh."   capitalised + period appended
#   "... parakeet.py"       -> "... parakeet.py."  period breaks the path
#   "--no-cache and --verbose" -> "--no-cache, and --verbose"
#
# On a case-sensitive filesystem the first is a different file, and the others
# are no longer runnable.  A token matching any pattern below is treated as
# literal: never capitalised, never given a trailing period, and never used as
# a comma anchor.
_LITERAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/"),                        # paths: src/foo.py, /var/log
    re.compile(r"_"),                        # snake_case, __dunder__
    re.compile(r"^-{1,2}[A-Za-z]"),          # CLI flags: -v, --no-cache
    re.compile(r"^\.[A-Za-z]"),              # dotfiles: .env, .gitignore
    re.compile(r"^[\w.-]+\.[A-Za-z]{1,5}$"), # filenames: server-test.sh
    re.compile(r"^\d+(\.\d+)+$"),            # versions: 0.3.34
    re.compile(r"[a-z][A-Z]"),               # camelCase: getUserById
)


def _is_literal_token(word: str) -> bool:
    """True if *word* is code-like and must not be reshaped by prose rules."""
    if not word:
        return False
    # Strip punctuation the sentence splitter may have left attached, but keep
    # a leading dot — that is what makes ".env" a dotfile rather than prose.
    bare = word.rstrip(",;:!?")
    if not bare:
        return False
    return any(pattern.search(bare) for pattern in _LITERAL_PATTERNS)

# Common proper nouns / names are hard to detect without NER — we only handle
# the pronoun "I" and sentence-initial capitalisation in the rule-based mode.

# Clause boundary markers — words that often start a new clause/sentence in
# dictated speech (used for rule-based comma / period insertion).
_CLAUSE_MARKERS: set[str] = {
    "but",
    "and",
    "or",
    "so",
    "because",
    "however",
    "although",
    "though",
    "while",
    "whereas",
    "since",
    "unless",
    "if",
    "then",
    "therefore",
    "meanwhile",
    "furthermore",
    "moreover",
    "nevertheless",
    "nonetheless",
    "otherwise",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PunctuationRestorer:
    """Restore punctuation and capitalisation on unpunctuated text.

    Loads a pair of ELECTRA-small ONNX models (one for punctuation labels, one
    for capitalisation labels) when available.  Falls back to a deterministic
    rule-based approach that capitalises sentence starts, adds terminal
    punctuation, and inserts commas before common clause-boundary words.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._punct_session: ort.InferenceSession | None = None  # type: ignore[name-defined]
        self._caps_session: ort.InferenceSession | None = None  # type: ignore[name-defined]
        self._vocab: dict[str, int] = {}
        self._using_onnx = False

        self._try_load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_models(self) -> None:
        if not _ONNX_AVAILABLE:
            logger.info("ONNX Runtime unavailable — using rule-based punctuation")
            return

        punct_path = self._model_dir / _PUNCT_MODEL
        caps_path = self._model_dir / _CAPS_MODEL
        vocab_path = self._model_dir / _VOCAB_FILENAME

        if not punct_path.exists() or not caps_path.exists():
            logger.info(
                "Punctuation ELECTRA models not found at %s — using rule-based fallback",
                self._model_dir,
            )
            return

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.inter_op_num_threads = 1
            sess_opts.intra_op_num_threads = 2
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            providers = ["CPUExecutionProvider"]

            self._punct_session = ort.InferenceSession(
                str(punct_path), sess_options=sess_opts, providers=providers
            )
            self._caps_session = ort.InferenceSession(
                str(caps_path), sess_options=sess_opts, providers=providers
            )
            self._vocab = self._load_vocab(vocab_path)
            self._using_onnx = True
            logger.info("Loaded punctuation ELECTRA models from %s", self._model_dir)
        except Exception:
            logger.exception("Failed to load punctuation ONNX models")
            self._punct_session = None
            self._caps_session = None

    @staticmethod
    def _load_vocab(path: Path) -> dict[str, int]:
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

    def process(self, text: str) -> str:
        """Add punctuation and fix capitalisation on *text*.

        Returns the fully punctuated and capitalised string.
        """
        if not text or not text.strip():
            return ""

        if self._using_onnx and self._punct_session and self._caps_session:
            return self._process_onnx(text)
        return self._process_rules(text)

    # ------------------------------------------------------------------
    # ONNX path
    # ------------------------------------------------------------------

    def _process_onnx(self, text: str) -> str:
        assert self._punct_session is not None
        assert self._caps_session is not None

        words = text.split()
        if not words:
            return ""

        # Tokenise
        input_ids: list[int] = [self._vocab.get("[CLS]", 101)]
        token_map: list[int] = []
        for word_idx, word in enumerate(words):
            wp_id = self._vocab.get(word.lower(), self._vocab.get("[UNK]", 100))
            input_ids.append(wp_id)
            token_map.append(word_idx)
        input_ids.append(self._vocab.get("[SEP]", 102))

        ids_arr = np.array([input_ids], dtype=np.int64)
        attn = np.ones_like(ids_arr, dtype=np.int64)
        ttype = np.zeros_like(ids_arr, dtype=np.int64)

        feed = {
            "input_ids": ids_arr,
            "attention_mask": attn,
            "token_type_ids": ttype,
        }

        punct_logits = self._punct_session.run(None, feed)[0][0]
        caps_logits = self._caps_session.run(None, feed)[0][0]

        # Strip [CLS] / [SEP]
        punct_logits = punct_logits[1 : len(token_map) + 1]
        caps_logits = caps_logits[1 : len(token_map) + 1]

        result_tokens: list[str] = []
        for i, word_idx in enumerate(token_map):
            word = words[word_idx]

            # Code-like tokens bypass the model entirely.  The model was trained
            # on prose and will happily capitalise a filename or append a period
            # to a path, which breaks it — exactly as the rule path did before
            # the same guard was added there.  A model prediction is not more
            # trustworthy than a rule when the input is not prose.
            if _is_literal_token(word):
                result_tokens.append(word)
                continue

            # Capitalisation
            cap_label = int(np.argmax(caps_logits[i]))
            cap_kind = _CAP_LABELS.get(cap_label, "lower")
            if cap_kind == "upper_first":
                word = word[0].upper() + word[1:] if word else word
            elif cap_kind == "all_upper":
                word = word.upper()
            # else: keep as-is (lower)

            # Punctuation — appended directly after the word
            punct_label = int(np.argmax(punct_logits[i]))
            punct_char = _PUNCT_LABELS.get(punct_label, "")
            result_tokens.append(word + punct_char)

        return " ".join(result_tokens)

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _process_rules(self, text: str) -> str:
        """Deterministic rule-based punctuation and capitalisation."""
        text = text.strip()
        if not text:
            return ""

        # Split into crude sentences on existing terminal punctuation (if any)
        # then process each. This handles partially-punctuated input gracefully.
        sentences = _split_into_sentences(text)
        processed: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence = _insert_commas(sentence)
            sentence = _capitalise_sentence(sentence)
            sentence = _ensure_terminal_punctuation(sentence)
            processed.append(sentence)

        return " ".join(processed)


# ---------------------------------------------------------------------------
# Rule-based helper functions
# ---------------------------------------------------------------------------


def _split_into_sentences(text: str) -> list[str]:
    """Split *text* on existing sentence boundaries, preserving the delimiter.

    If the text has no terminal punctuation at all, it is returned as a single
    element list.
    """
    # Split on .?! followed by space or end-of-string
    parts = re.split(r"(?<=[.?!])\s+", text)
    if len(parts) <= 1:
        return [text]
    return parts


def _insert_commas(text: str) -> str:
    """Insert commas before clause-boundary conjunctions/adverbs.

    Only inserts a comma when the word is preceded by at least two words
    (avoids adding commas at the very start or after a single word).
    """
    words = text.split()
    if len(words) <= 3:
        return text

    result: list[str] = [words[0]]
    for i in range(1, len(words)):
        lower = words[i].lower()
        if (
            lower in _CLAUSE_MARKERS
            and i >= 2
            # Don't add a comma if the preceding token already ends with one
            and not result[-1].endswith(",")
            and not result[-1].endswith(";")
            # "--no-cache and --verbose" is a command line, not a clause
            # boundary.  A comma either side of the conjunction breaks it.
            and not _is_literal_token(result[-1])
            and not (i + 1 < len(words) and _is_literal_token(words[i + 1]))
            # Adjacent markers are one boundary, not two: "and then" produced
            # "..., and, then ..." before this guard.
            and result[-1].lower().rstrip(",;") not in _CLAUSE_MARKERS
        ):
            # Append a comma to the preceding word
            result[-1] = result[-1] + ","
        result.append(words[i])

    return " ".join(result)


def _capitalise_sentence(text: str) -> str:
    """Capitalise the first letter + the pronoun 'I' throughout."""
    if not text:
        return text

    words = text.split()
    result: list[str] = []
    capitalise_next = True  # first word

    for word in words:
        bare = word.rstrip(".,?!;:")
        trailing = word[len(bare):]

        if _is_literal_token(word):
            # Leave code-like tokens exactly as dictated.  The pending
            # capitalisation is consumed rather than carried forward — a
            # filename genuinely opened this sentence.
            capitalise_next = False
        elif bare.lower() in _ALWAYS_UPPER:
            bare = bare.upper()
            capitalise_next = False  # first-word cap consumed
        elif capitalise_next and bare:
            bare = bare[0].upper() + bare[1:]
            capitalise_next = False

        # After sentence-ending punctuation, capitalise the next word
        if trailing and trailing[-1] in _TERMINAL_PUNCT:
            capitalise_next = True

        result.append(bare + trailing)

    return " ".join(result)


def _ensure_terminal_punctuation(text: str) -> str:
    """Append a period or question mark if the sentence lacks one."""
    text = text.rstrip()
    if not text:
        return text

    if text[-1] in _TERMINAL_PUNCT:
        return text

    # A sentence ending in a path, filename, or flag must not gain a period —
    # "parakeet.py." and "--verbose." are both broken.  Prefer a missing full
    # stop over a corrupted token.
    if _is_literal_token(text.split()[-1]):
        return text

    # Determine if the sentence is likely a question
    first_word = text.split()[0].lower().rstrip(".,?!;:")
    if first_word in _QUESTION_STARTERS:
        return text + "?"

    return text + "."
