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
from typing import TYPE_CHECKING

from linux_whisper.config import MODELS_DIR

if TYPE_CHECKING:
    from pathlib import Path

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
# — see `_classify_ambiguous_fillers()` below and issue #43. This mirrors the
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
# Includes unicode curly quotes and the single-character ellipsis (…) in
# addition to their ASCII equivalents — STT output uses both, and a token
# like "um…" must strip down to the bare word "um" just as "um..." does
# (issue #43 review, P3). Without these, a filler glued to unicode
# punctuation neither gets recognised as a filler nor leaves an orphan,
# because the whole raw token (word + attached punctuation) is dropped as a
# unit once it IS recognised — see `_remove_fillers()`.
_EDGE_PUNCT = ".,!?;:\"'()[]{}‘’“”…"

# Build compiled patterns.
_phrase_alts = "|".join(_FILLER_PHRASES)
_word_alts = "|".join(_FILLER_WORDS)
_FILLER_PHRASE_RE = re.compile(rf"(?<!['\w])(?:{_phrase_alts})(?!['\w])", re.IGNORECASE)
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
        self._warned_unknown_label = False

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

        # Feed only what this model actually declares. Hardcoding a
        # BERT-style signature makes any DistilBERT-based checkpoint fail
        # outright with `InvalidArgument: Invalid input name: token_type_ids`
        # — DistilBERT has no segment embeddings — which silently rules out
        # a whole family of otherwise-suitable models.
        available = {
            "input_ids": ids_array,
            "attention_mask": attn_mask,
            "token_type_ids": token_type,
        }
        feed = {i.name: available[i.name] for i in self._session.get_inputs()
                if i.name in available}
        missing = {i.name for i in self._session.get_inputs()} - feed.keys()
        if missing:
            logger.warning(
                "Disfluency model expects unsupported inputs %s — using regex fallback",
                sorted(missing),
            )
            self._using_onnx = False
            return self._process_regex(text)

        outputs = self._session.run(None, feed)
        # outputs[0] shape: (1, seq_len, num_labels)
        logits = outputs[0][0]
        # Strip [CLS] and [SEP]
        logits = logits[1 : len(token_map) + 1]

        # The model is trained to make exactly the contextual judgement issue
        # #43 wants, but a token classifier is not more trustworthy than a
        # cheap, hard rule when the two disagree on the model's own known
        # blind spot — the same reasoning `_is_literal_token()` applies for
        # code-like tokens in `polish/punctuation.py`. For the fixed set of
        # ambiguous vocabulary, a REMOVE prediction is only honoured when the
        # same context cue the regex fallback requires is also present;
        # otherwise the word is kept as content regardless of the model's
        # prediction (issue #43 review, finding 3).
        bares = [_bare_word(t) for t in tokens]
        context_marks_filler, clear_preceding_comma = _classify_ambiguous_fillers(
            tokens, bares
        )

        has_self_corrections = False
        kept: list[str] = []

        for i, word_idx in enumerate(token_map):
            word = tokens[word_idx]
            bare = bares[word_idx]

            # Unambiguous fillers (um, uh, hmm, ...) are always stripped,
            # anywhere, on the regex path — no context cue required. The
            # ONNX path must give the same guarantee rather than deferring
            # to the model: a KEEP label (or a tokenisation quirk that
            # dropped the attached punctuation from the vocab lookup) must
            # not let one through (issue #43 review, O2).
            if _is_unambiguous_filler(bare):
                continue

            label = int(np.argmax(logits[i]))
            if label == _LABEL_KEEP:
                kept.append(word)
                continue
            if label == _LABEL_REPAIR:
                kept.append(word)
                has_self_corrections = True
                continue

            if label != _LABEL_REMOVE:
                # An unrecognised label is NOT a licence to delete. Model
                # label spaces differ: a candidate evaluated for #36 uses
                # {0: KEEP, 1: DELETE, 2: KEEP_STRIP_COMMA, 3: KEEP_CAPITALIZE},
                # where 2 and 3 are *keep* variants. Treating anything
                # unknown as REMOVE would silently drop those words. Keep
                # the token and say so once, rather than eating the input.
                if not self._warned_unknown_label:
                    self._warned_unknown_label = True
                    logger.warning(
                        "Disfluency model emitted unrecognised label %d — keeping "
                        "the token. Expected 0=KEEP, 1=REMOVE, 2=REPAIR.",
                        label,
                    )
                kept.append(word)
                continue

            # _LABEL_REMOVE: the model wants to drop this token. Override
            # that for ambiguous vocabulary with no context cue — otherwise
            # this is the exact bug issue #43 fixes on the regex path,
            # still live here.
            if bare.lower() in _AMBIGUOUS_FILLER_SET and not context_marks_filler[word_idx]:
                kept.append(word)
                continue

            # Genuine removal. If this was a comma-bounded ambiguous filler
            # ("It was, like, really fast."), the comma on the previously
            # kept word was only pairing with it and is now an orphan —
            # same fix as the regex path (issue #43 review, O1).
            if clear_preceding_comma[word_idx] and kept and kept[-1].endswith(","):
                kept[-1] = kept[-1][:-1]
            # else: genuine removal — skip the token

        cleaned = " ".join(kept)
        cleaned = _normalise_whitespace(cleaned)

        # Empty-output backstop — mirrors the regex fallback's guarantee that
        # no dictation ever injects nothing when the STT produced words
        # (issue #43 review, finding 3).
        if _has_word_char(text) and not _has_word_char(cleaned):
            return DisfluencyResult(text=text, has_self_corrections=has_self_corrections)

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
    return any(pattern.search(text) for pattern in _SELF_CORRECTION_PATTERNS)


def _bare_word(token: str) -> str:
    """Strip leading/trailing punctuation from *token* for word comparison."""
    return token.strip(_EDGE_PUNCT)


def _is_unambiguous_filler(bare: str) -> bool:
    """True if *bare* is one of the non-word fillers (um, uh, hmm, ...)."""
    return bool(bare) and _UNAMBIGUOUS_WORD_RE.fullmatch(bare) is not None


def _trailing_punct(token: str) -> str:
    """Return the run of edge-punctuation characters trailing *token*.

    A plain ``token.endswith(",")`` misses a comma hidden behind a closing
    quote or bracket — ``'"Well,"'`` or ``'"like,"'`` — and silently treats
    quoted or bracketed fillers as uncued (issue #43 review, finding 5).
    """
    i = len(token)
    while i > 0 and token[i - 1] in _EDGE_PUNCT:
        i -= 1
    return token[i:]


def _classify_ambiguous_fillers(
    tokens: list[str], bares: list[str]
) -> tuple[list[bool], list[bool]]:
    """Decide which ambiguous-filler tokens context marks as disfluent.

    An ambiguous word (e.g. "like", "so", "right") is content by default —
    stripping it unconditionally deletes ordinary English ("I like this
    design." -> "I this design.", issue #43). It is only treated as a filler
    when *any* of these context cues hold:

    - it is utterance-initial and followed by a comma ("So, I think...");
    - it is bounded by commas on both sides ("..., like, ...");
    - it is utterance-final and preceded by a comma ("..., right.").

    An earlier revision of this rule also fired on plain adjacency to a
    *leading* run of unambiguous fillers when the following word looked like
    a clause-starting pronoun ("um so I was..." -> "so" stripped). Issue #43
    review, P1/P2 dropped that cue: it deleted genuine content whenever a
    pronoun happened to follow ("um literally I translated it word for
    word" stripped "literally", a real adverb, not a filler), and it was
    inconsistent whenever one didn't ("um so we should go" stripped "so" but
    "um so the build is broken" kept it — the exact same construction,
    different outcome, for no reason a user could predict). A regex cannot
    reliably tell "so" the filler from "so" the adverb by checking what part
    of speech the next word merely looks like; that is a job for the BERT
    classifier (issue #36), not another hand-tuned word list layered on this
    one. Until then the conservative behaviour — keep the word — is
    correct, and every ambiguous word now goes through the exact same three
    comma-based cues regardless of what precedes it.

    *tokens* and *bares* MUST be the ORIGINAL token stream — before filler
    phrases such as "you know" are removed. Classifying against
    already-phrase-stripped text fabricates adjacency that never existed in
    the input: "um you know right turn..." would otherwise see "right" as
    adjacent to "um" once "you know" is gone (issue #43 review, finding 2).

    Every cue above depends on a comma already being present. Stage 4a runs
    BEFORE stage 4b (punctuation restoration), so on fully unpunctuated STT
    output ("well I think this works", "so I said no") none of the
    comma-based cues can fire and these words are always kept as content.
    This is intentional (issue #43 review, finding 7) — loosening the
    predicate to fire without a comma would reintroduce the exact
    content-word deletion this module exists to prevent. In production,
    whisper.cpp does emit its own punctuation ("It's now the next day. How
    did the sink go last night?"), so real dictation exercises these cues
    far more than the deliberately unpunctuated benchmark fixtures do.

    Returns ``(remove, clear_preceding_comma)``, parallel to *tokens*:
    `remove[i]` is True if `tokens[i]` should be dropped; `clear_preceding_comma[i]`
    is True if the comma on the previously-kept token was only there to pair
    with this filler and must go with it ("It was, like, really fast." ->
    "It was really fast.", not the orphaned "It was, really fast.").
    """
    n = len(tokens)
    remove = [False] * n
    clear_preceding_comma = [False] * n
    if n == 0:
        return remove, clear_preceding_comma

    ends_comma = ["," in _trailing_punct(t) for t in tokens]

    for i in range(n):
        if bares[i].lower() not in _AMBIGUOUS_FILLER_SET:
            continue

        preceded_by_comma = i > 0 and ends_comma[i - 1]
        followed_by_comma = ends_comma[i]
        is_initial = i == 0
        is_final = i == n - 1

        is_filler = (
            (is_initial and followed_by_comma)
            or (preceded_by_comma and followed_by_comma)
            or (is_final and preceded_by_comma)
        )
        if is_filler:
            remove[i] = True
            if preceded_by_comma:
                clear_preceding_comma[i] = True

    return remove, clear_preceding_comma


def _phrase_removed_mask(tokens: list[str]) -> tuple[list[bool], dict[int, str]]:
    """Mark tokens that fall inside a matched filler phrase ("you know", ...).

    Matches against ``" ".join(tokens)``, the canonical single-spaced form of
    the ORIGINAL token stream, so character offsets map onto token
    boundaries.

    A phrase match does not always *start* on a token boundary: the regex's
    lookbehind only requires the preceding character not be a word character
    or apostrophe, so punctuation glued to the previous word with no space
    ("I was,you know thinking") lets the match begin mid-token, inside
    "was,you". The previous version of this function assumed every match
    started at a token boundary and recovered the token index by re-running
    `str.split()` on the string prefix before the match — which (a) silently
    mis-locates a mid-token match (it treats the partial "was," as if it
    were the complete token, off-by-one against the real tokens that
    follow), deleting a content token instead of the filler while leaving
    the filler word itself untouched, and (b) re-splits that growing prefix
    on *every* match, making phrase removal O(n^2) on inputs with many
    matches (issue #43 review, R1 and R2).

    Token start offsets are computed once in a single linear pass instead,
    and a match landing mid-token has its retained (non-phrase) substring
    recorded in the returned *partial* map rather than dropping the whole
    token — so "was,you" correctly becomes "was," (the "you" consumed by
    the phrase, the comma and "was" kept) rather than losing "was," outright
    or leaving "you" behind.

    Returns ``(mask, partial)``: `mask[i]` is True if `tokens[i]` falls
    *entirely* inside a matched phrase and should be dropped; `partial`
    maps the index of any token only *partially* covered by a match to the
    substring of that token which survives.
    """
    n = len(tokens)
    mask = [False] * n
    partial: dict[int, str] = {}
    if n == 0:
        return mask, partial

    starts = [0] * n
    pos = 0
    for i, t in enumerate(tokens):
        starts[i] = pos
        pos += len(t) + 1  # token text plus the joining space

    joined = " ".join(tokens)

    def _token_at(offset: int) -> int:
        """Binary search: index of the token whose span contains *offset*."""
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    for m in _FILLER_PHRASE_RE.finditer(joined):
        start_tok = _token_at(m.start())
        end_tok = _token_at(m.end() - 1)

        if m.start() > starts[start_tok]:
            # Match starts mid-token — keep the token's pre-match prefix.
            prefix = tokens[start_tok][: m.start() - starts[start_tok]]
            partial[start_tok] = prefix
        else:
            mask[start_tok] = True

        tok_end = starts[end_tok] + len(tokens[end_tok])
        if m.end() < tok_end:
            # Match ends mid-token — keep the token's post-match suffix.
            suffix = tokens[end_tok][m.end() - starts[end_tok] :]
            partial[end_tok] = partial.get(end_tok, "") + suffix
        elif end_tok not in partial:
            mask[end_tok] = True

        for i in range(start_tok + 1, end_tok):
            mask[i] = True

    return mask, partial


def _has_word_char(text: str) -> bool:
    """True if *text* contains at least one alphanumeric character."""
    return any(ch.isalnum() for ch in text)


def _remove_fillers(text: str) -> str:
    """Strip filler words / discourse markers from *text*.

    Unambiguous fillers (um, uh, hmm, ...) and filler phrases (you know, I
    mean, ...) are stripped anywhere. Ambiguous fillers (like, so, right,
    ...) are stripped only when context marks them as disfluent — see
    `_classify_ambiguous_fillers()`. All three are classified against the
    ORIGINAL token stream in a single pass, so removing a phrase can never
    fabricate adjacency for a word that follows it (issue #43 review,
    finding 2).

    Hard backstop: if removal would reduce non-empty, non-punctuation-only
    input to empty or punctuation-only text, the input is returned unchanged.
    No dictation may ever inject nothing when the STT produced words.
    """
    tokens = text.split()
    if not tokens:
        return text

    bares = [_bare_word(t) for t in tokens]
    phrase_removed, phrase_partial = _phrase_removed_mask(tokens)
    ambiguous_remove, clear_preceding_comma = _classify_ambiguous_fillers(tokens, bares)

    output: list[str] = []
    for i, token in enumerate(tokens):
        if i in phrase_partial:
            # Only part of this token fell inside a matched phrase — keep
            # the retained substring rather than the whole token or nothing
            # (issue #43 review, R1).
            output.append(phrase_partial[i])
            continue
        if phrase_removed[i] or _is_unambiguous_filler(bares[i]) or ambiguous_remove[i]:
            # The comma pairing with a removed ambiguous filler was marking
            # the aside it introduced ("It was, like, really fast."); with
            # the filler gone the comma is an orphan and must go too, or the
            # sentence reads as if it were cut off ("It was, really fast.").
            if clear_preceding_comma[i] and output and output[-1].endswith(","):
                output[-1] = output[-1][:-1]
            continue
        output.append(token)

    cleaned = " ".join(output)
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
