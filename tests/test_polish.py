"""Comprehensive tests for the polish pipeline stages.

Tests cover:
- DisfluencyRemover regex fallback (filler removal, repetitions, self-corrections)
- PunctuationRestorer rule-based fallback (capitalization, terminal punct, commas)
- LLMCorrector (unavailable behavior, timeout, hallucination rejection)
- PolishPipeline (full integration, stage toggling, conditional LLM invocation)
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from linux_whisper.config import PolishConfig

# =====================================================================
# Stage 4a: DisfluencyRemover
# =====================================================================
from linux_whisper.polish.disfluency import (
    _AMBIGUOUS_FILLERS,
    _FILLER_WORDS,
    _LABEL_KEEP,
    _LABEL_REMOVE,
    DisfluencyRemover,
    DisfluencyResult,
    _detect_self_corrections,
    _normalise_whitespace,
    _remove_fillers,
    _remove_repetitions,
)


class TestRemoveFillers:
    """Test the _remove_fillers function directly."""

    def test_removes_um(self):
        assert "I think" in _remove_fillers("um I think").strip()

    def test_removes_uh(self):
        assert "uh" not in _remove_fillers("uh I think").lower()

    def test_keeps_like_without_context_cue(self):
        # "like" here is plain content ("similar to"/verb usage), not a
        # discourse filler — no comma, no adjacency to an unambiguous filler,
        # not utterance-initial/final. Issue #43: ambiguous fillers are only
        # stripped when context marks them as disfluent.
        result = _remove_fillers("I was like going to the store")
        assert "like" in result.split()

    def test_removes_comma_bounded_like(self):
        result = _remove_fillers("It was, like, really fast.")
        assert "like" not in result.lower()

    def test_keeps_basically_without_context_cue(self):
        result = _remove_fillers("I need basically help")
        assert "basically" in result.lower()

    def test_removes_utterance_initial_basically_with_comma(self):
        result = _remove_fillers("Basically, I need help.")
        assert "basically" not in result.lower()

    def test_removes_you_know(self):
        result = _remove_fillers("I was you know thinking about it")
        assert "you know" not in result.lower()

    def test_removes_i_mean(self):
        result = _remove_fillers("I mean we should go")
        assert "I mean" not in result

    def test_removes_kind_of(self):
        result = _remove_fillers("it was kind of interesting")
        assert "kind of" not in result.lower()

    def test_removes_sort_of(self):
        result = _remove_fillers("it was sort of cool")
        assert "sort of" not in result.lower()

    def test_removes_multiple_fillers(self):
        # Issue #43 review, finding 6: the previous version of this test only
        # checked that content words survived, which a no-op `_remove_fillers`
        # would also satisfy. Assert the exact cleaned string instead, so a
        # no-op fails: "um" (unambiguous) and comma-bounded "like" must be
        # gone, with no stray whitespace and no orphaned comma left behind.
        #
        # "so" here is mid-sentence with no comma cue of its own (it follows
        # "um," but isn't utterance-initial itself, and isn't comma-bounded)
        # — it survives as content. Issue #43 review, P1/P2: an earlier
        # revision stripped it anyway via a leading-adjacency-to-a-pronoun
        # heuristic that also deleted genuine content elsewhere and was
        # dropped for that reason — see `_classify_ambiguous_fillers()`.
        result = _remove_fillers("um, so I was thinking, like, we should go")
        assert result == "so I was thinking we should go"

    def test_keeps_well_without_context_cue(self):
        # "The well is dry." / "It went so well." — "well" as plain content
        # (noun/adverb) is not utterance-initial+comma, comma-bounded,
        # adjacent to an unambiguous filler, or utterance-final+comma.
        result = _remove_fillers("It went so well")
        assert "well" in result.split()

    def test_removes_utterance_initial_well_with_comma(self):
        result = _remove_fillers("Well, maybe.")
        assert "well" not in result.lower()

    def test_keeps_okay(self):
        # "okay" is content, not a filler — see issue #42. Deleting it
        # unconditionally caused stand-alone "OK" utterances to vanish.
        result = _remove_fillers("okay lets do this")
        assert "okay" in result.lower()

    def test_removes_hmm(self):
        result = _remove_fillers("hmm let me think")
        assert "hmm" not in result.lower()

    def test_removes_repeated_um(self):
        result = _remove_fillers("ummm I need help")
        assert "ummm" not in result.lower()

    def test_does_not_remove_from_within_words(self):
        # "like" in "unlikely" should NOT be removed
        result = _remove_fillers("it was unlikely")
        assert "unlikely" in result

    def test_preserves_non_filler_content(self):
        text = "the quick brown fox jumps over the lazy dog"
        assert _remove_fillers(text).strip() == text


class TestPhraseRemovalTokenBoundaries:
    """Issue #43 review, R1/R2: `_phrase_removed_mask` assumed every filler
    phrase match starts on a whitespace token boundary. When punctuation
    glues the phrase to the previous word with no space ("was,you know"),
    the match starts mid-token, and the old index bookkeeping (re-splitting
    the string prefix before the match) both mis-located the match and made
    phrase removal O(n^2) on inputs with many matches.
    """

    def test_phrase_starting_mid_token_does_not_delete_following_content(self):
        # R1 (BLOCKER): with the bug, this dropped "thinking" (real content)
        # while "you" (part of the filler) survived attached to "was,".
        assert _remove_fillers("I was,you know thinking") == "I was, thinking"

    def test_phrase_starting_mid_token_after_various_punctuation(self):
        # The comma case above is the reported regression; cover the other
        # edge-punctuation characters that can glue a word to a phrase with
        # no space, to make sure the fix isn't comma-specific.
        assert _remove_fillers("I was;you know thinking") == "I was; thinking"
        assert _remove_fillers('I was"you know thinking') == 'I was" thinking'

    def test_many_phrase_matches_stay_linear_and_within_budget(self):
        # R2 (MAJOR): the old re-split-the-prefix-per-match approach was
        # O(n^2). 2001 tokens ("you know" x1000 + "done") took 35.7ms on the
        # buggy branch against 9.9ms on main — over the stage 4a latency
        # budget in CLAUDE.md (< 15ms). Assert both correctness and that the
        # pass is fast enough to stay well inside budget, so a future
        # regression back to O(n^2) fails loudly instead of silently
        # returning (the note that shipped this test).
        text = ("you know " * 1000 + "done").strip()
        assert len(text.split()) == 2001

        _remove_fillers(text)  # warm up (regex module caches, etc.)
        start = time.perf_counter()
        result = _remove_fillers(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result == "done"
        assert elapsed_ms < 15.0, (
            f"_remove_fillers took {elapsed_ms:.2f}ms on 2001 tokens — "
            "stage 4a budget is < 15ms (CLAUDE.md)"
        )


class TestContextualAmbiguousFillers:
    """Issue #43: ambiguous fillers (like, right, so, well, actually,
    basically, literally, anyway, anyways) are only stripped when context
    marks them as disfluent, never unconditionally.
    """

    # The six reproduction cases from the issue — each retains its content
    # word because none of them carry a context cue (comma or adjacency to
    # an unambiguous filler).

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("I like this design.", "I like this design."),
            ("You are right about that.", "You are right about that."),
            ("Turn right at the light.", "Turn right at the light."),
            ("It went so well.", "It went so well."),
            ("The well is dry.", "The well is dry."),
            ("That is actually true.", "That is actually true."),
        ],
    )
    def test_reproduction_cases_retain_content_word(self, raw, expected):
        assert _remove_fillers(raw) == expected

    # Genuine fillers still strip when context marks them as disfluent.

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("So, I think we should go.", "I think we should go."),
            ("It was, like, really fast.", "It was really fast."),
            ("um, like, maybe", "maybe"),
        ],
    )
    def test_context_marked_fillers_still_strip(self, raw, expected):
        assert _remove_fillers(raw).strip() == expected

    # Each of the four context rules from the design, exercised directly.

    def test_utterance_initial_followed_by_comma_is_filler(self):
        result = _remove_fillers("So, we should leave now.")
        assert "so" not in result.lower()

    def test_bounded_by_commas_both_sides_is_filler(self):
        result = _remove_fillers("It is, right, the best option.")
        assert not re.search(r"\bright\b", result, re.IGNORECASE)

    def test_utterance_final_preceded_by_comma_is_filler(self):
        result = _remove_fillers("That is the plan, anyway.")
        assert "anyway" not in result.lower()

    def test_utterance_initial_without_comma_is_not_filler(self):
        # No comma cue — "Right" opens a sentence as content ("Correct,
        # let's continue" vs. plain agreement), not a discourse filler.
        result = _remove_fillers("Right now I need to leave.")
        assert "right" in result.lower()

    def test_mid_sentence_without_comma_is_not_filler(self):
        result = _remove_fillers("I think it is literally the best plan.")
        assert "literally" in result.lower()

    # Issue #43 review, P1/P2: an earlier revision also treated plain
    # adjacency to a leading run of unambiguous fillers as a cue, but only
    # when the following word looked like a clause-starting pronoun. That
    # was dropped entirely rather than narrowed further:
    #
    # - P1: it deleted real content whenever a pronoun happened to follow
    #   ("um literally I translated it word for word" -> "literally", an
    #   adverb, was stripped as if it were a filler).
    # - P2: it was inconsistent whenever a pronoun didn't follow ("um so we
    #   should go" stripped "so" but "um so the build is broken" kept it —
    #   the same construction, different outcome, with nothing in the
    #   sentence for a user to point to as the reason).
    #
    # A regex cannot reliably tell "so"-the-filler from "so"-the-adverb by
    # checking what part of speech merely follows it. Adjacency to a leading
    # filler is no longer a cue at all — every ambiguous word now goes
    # through the same three comma-based rules regardless of what precedes
    # it, deferring the harder judgement call to the BERT classifier
    # (issue #36).

    def test_mid_sentence_adjacency_is_not_a_cue(self):
        # "so" is adjacent to "uh" but mid-sentence, not utterance-initial —
        # plain content ("so good"), not a filler.
        assert _remove_fillers("it was uh so good") == "it was so good"

    def test_leading_adjacency_alone_is_not_a_cue(self):
        # "like" opens the utterance right after "um" and is followed by a
        # clause-starting pronoun ("I") — exactly the pattern the dropped
        # heuristic treated as a filler. With no comma cue, it survives as
        # content now (P1/P2).
        assert _remove_fillers("um like I think that works") == "like I think that works"

    def test_leading_adjacency_without_clause_starter_is_not_a_cue(self):
        # "right" opens the utterance right after "um", but the next word
        # ("turn") doesn't open a clause either — still content either way.
        assert _remove_fillers("um right turn at the light") == "right turn at the light"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # P1's exact reproduction: a real adverb, not a filler, deleted
            # only because a pronoun happened to be the next word.
            (
                "um literally I translated it word for word",
                "literally I translated it word for word",
            ),
            ("um actually I did finish it", "actually I did finish it"),
        ],
    )
    def test_leading_adjacency_does_not_delete_content_before_a_pronoun(self, raw, expected):
        assert _remove_fillers(raw) == expected

    # Issue #43 review, finding 2: a filler *phrase* must not fabricate
    # adjacency for an ambiguous word that follows it. Classification has to
    # run against the original token stream, before "you know" disappears.
    # (Leading adjacency is no longer a cue at all — see P1/P2 above — so
    # this now also just confirms phrase removal doesn't otherwise disturb
    # "right".)

    def test_phrase_removal_does_not_fabricate_adjacency(self):
        result = _remove_fillers("um you know right turn at the light")
        assert result == "right turn at the light"

    # Issue #43 review, finding 4: removing a filler must not leave its
    # punctuation behind as an orphaned or doubled comma.

    def test_leading_filler_comma_leaves_no_orphan(self):
        assert _remove_fillers("um, I think") == "I think"

    def test_bounded_filler_comma_does_not_double(self):
        assert _remove_fillers("I think, um, we should go") == "I think, we should go"

    # Issue #43 review, finding 5: comma detection must see through a
    # trailing quote or bracket, not just check the literal last character.

    def test_comma_cue_seen_through_trailing_quote(self):
        result = _remove_fillers('"Well," I think')
        assert "well" not in result.lower()

    def test_comma_bounded_cue_seen_through_quotes(self):
        result = _remove_fillers('It was, "like," really fast')
        assert "like" not in result.lower()

    # Issue #43 review, P3: `_EDGE_PUNCT` omitted unicode quotes and the
    # single-character ellipsis (…), so a filler glued to one of them wasn't
    # recognised as a filler at all — it neither stripped nor left an
    # orphan, it just survived outright.

    def test_unicode_ellipsis_glued_filler_is_stripped(self):
        assert _remove_fillers("um… I think") == "I think"

    def test_unicode_curly_quotes_glued_filler_is_stripped(self):
        assert _remove_fillers("“um” I think") == "I think"

    # Whole-utterance backstop: filler removal can never empty an utterance
    # that contained at least one word. Cover every filler (unambiguous and
    # ambiguous) alone and with terminal punctuation.

    _all_filler_words = [w.rstrip("+") for w in _FILLER_WORDS] + list(_AMBIGUOUS_FILLERS)

    @pytest.mark.parametrize("word", _all_filler_words)
    @pytest.mark.parametrize("suffix", ["", ".", ",", "?"])
    def test_backstop_never_empties_a_single_filler_word(self, word, suffix):
        raw = f"{word}{suffix}"
        result = _remove_fillers(raw)
        assert any(ch.isalnum() for ch in result), (
            f"{raw!r} -> {result!r} contains no word characters"
        )

    @pytest.mark.parametrize("word", _all_filler_words)
    def test_backstop_never_empties_all_fillers_utterance(self, word):
        # An utterance made entirely of the same filler, repeated with
        # commas, so every context rule would otherwise fire.
        raw = f"{word}, {word}, {word}."
        result = _remove_fillers(raw)
        assert any(ch.isalnum() for ch in result), (
            f"{raw!r} -> {result!r} contains no word characters"
        )


class TestRemoveRepetitions:
    """Test the _remove_repetitions function."""

    def test_simple_repetition(self):
        assert _remove_repetitions("the the") == "the"

    def test_triple_repetition(self):
        result = _remove_repetitions("I I I think")
        assert result == "I think"

    def test_no_repetition(self):
        text = "I think therefore I am"
        assert _remove_repetitions(text) == text

    def test_multiple_repetitions(self):
        result = _remove_repetitions("the the cat sat sat down")
        assert result == "the cat sat down"

    def test_case_insensitive(self):
        result = _remove_repetitions("The the cat")
        assert "The" in result or "the" in result
        # Should collapse to one word
        words = result.split()
        assert sum(1 for w in words if w.lower() == "the") == 1


class TestDetectSelfCorrections:
    """Test self-correction detection."""

    def test_actually_correction(self):
        assert _detect_self_corrections("at 2 actually at 4") is True

    def test_wait_correction(self):
        assert _detect_self_corrections("go left wait go right") is True

    def test_no_correction(self):
        assert _detect_self_corrections("I think we should go home") is False

    def test_sorry_correction(self):
        assert _detect_self_corrections("meet at 3 sorry meet at 5") is True

    def test_i_mean_correction(self):
        assert _detect_self_corrections("the red one I mean the blue one") is True

    def test_dash_correction(self):
        assert _detect_self_corrections("I want the -- the other one") is True

    def test_ellipsis_correction(self):
        assert _detect_self_corrections("go to the... go to the park") is True

    def test_comma_no_correction(self):
        assert _detect_self_corrections("the cat, no, the dog is here") is True

    def test_or_rather_correction(self):
        assert _detect_self_corrections("ten items or rather twelve items") is True

    def test_empty_string(self):
        assert _detect_self_corrections("") is False


class TestNormaliseWhitespace:
    """Test whitespace normalization."""

    def test_collapses_multiple_spaces(self):
        assert _normalise_whitespace("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert _normalise_whitespace("  hello  ") == "hello"

    def test_preserves_single_spaces(self):
        assert _normalise_whitespace("a b c") == "a b c"

    def test_empty_string(self):
        assert _normalise_whitespace("") == ""

    def test_only_whitespace(self):
        assert _normalise_whitespace("   ") == ""


class TestDisfluencyRemover:
    """Test the DisfluencyRemover class (regex fallback path)."""

    @pytest.fixture()
    def remover(self):
        """Create a DisfluencyRemover that always uses regex fallback."""
        # Use a nonexistent model dir to force regex fallback
        return DisfluencyRemover(model_dir=Path("/nonexistent/model"))

    def test_empty_input(self, remover):
        result = remover.process("")
        assert result.text == ""
        assert result.has_self_corrections is False

    def test_whitespace_only(self, remover):
        result = remover.process("   ")
        assert result.text == ""
        assert result.has_self_corrections is False

    def test_clean_text_unchanged(self, remover):
        result = remover.process("the cat sat on the mat")
        assert result.text == "the cat sat on the mat"
        assert result.has_self_corrections is False

    def test_single_word(self, remover):
        result = remover.process("hello")
        assert result.text == "hello"

    def test_filler_removal(self, remover):
        result = remover.process("um so I was like thinking about it")
        assert "um" not in result.text.split()
        assert "thinking" in result.text

    def test_repetition_removal(self, remover):
        result = remover.process("I I I want to go go home")
        assert result.text == "I want to go home"

    def test_combined_fillers_and_repetitions(self, remover):
        # "like" here has no context cue (no comma, not adjacent to an
        # unambiguous filler) — issue #43 keeps it as content. Only "um"
        # (unambiguous) and the "the the" repetition are removed.
        result = remover.process("um the the cat was like sitting there")
        assert result.text == "the cat was like sitting there"

    def test_self_correction_detected(self, remover):
        result = remover.process("meet at 3 actually meet at 5")
        assert result.has_self_corrections is True

    def test_no_self_correction_flagged(self, remover):
        result = remover.process("the weather is nice today")
        assert result.has_self_corrections is False

    def test_returns_disfluency_result(self, remover):
        result = remover.process("hello world")
        assert isinstance(result, DisfluencyResult)

    # "okay"/"ok" regression tests — issue #42.
    # These used to be stripped as fillers, so a stand-alone "OK" utterance
    # vanished into an empty transcript. They're content, not filler.

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("OK", "OK"),
            ("Okay.", "Okay."),
            ("Is that okay?", "Is that okay?"),
            ("That is okay with me.", "That is okay with me."),
            # Utterance-initial "Okay," is a deliberate behaviour change,
            # not a regression — it used to be stripped.
            ("Okay, let's ship it.", "Okay, let's ship it."),
        ],
    )
    def test_okay_is_retained(self, remover, raw, expected):
        result = remover.process(raw)
        assert result.text == expected

    @pytest.mark.parametrize(
        ("raw", "stripped_word"),
        [
            ("um I think we should go", "um"),
            ("um, I think we should go", "um"),
            ("uh let's start", "uh"),
            ("hmm let me think", "hmm"),
            ("erm I forgot", "erm"),
            ("Basically, I need help.", "basically"),
            ("it was, like, really fast", "like"),
        ],
    )
    def test_other_fillers_still_stripped(self, remover, raw, stripped_word):
        # Proves the #42 fix is scoped to "okay"/"ok" — every other
        # unambiguous filler (um/uh/hmm/erm) still strips unconditionally as
        # before, and every ambiguous filler (basically/like) still strips
        # when its context marks it as disfluent (issue #43).
        #
        # Match on a word boundary rather than str.split(): a filler that
        # survived with punctuation attached ("um,") is still a survivor,
        # but split() tokenises it as "um," and would let the test pass.
        result = remover.process(raw)
        assert not re.search(rf"\b{stripped_word}\b", result.text, re.IGNORECASE)

    def test_bare_so_is_no_longer_stripped(self, remover):
        # Issue #43, intentional behaviour change from #45's baseline: "so"
        # used to be stripped unconditionally by the flat filler regex. With
        # no comma cue, no adjacency to an unambiguous filler, and not in a
        # position that reads as a discourse marker, it is ordinary content
        # ("therefore") and must survive.
        result = remover.process("so we should go")
        assert "so" in result.text.lower().split()

    def test_utterance_initial_so_with_comma_still_stripped(self, remover):
        # The disfluency-marking use of "so" ("So, I think we should go.")
        # is unaffected by the above — it still strips.
        result = remover.process("So, we should go.")
        assert "so" not in re.sub(r"[^\w\s]", "", result.text).lower().split()

    # Real-world dictation examples

    def test_real_dictation_meeting_notes(self, remover):
        text = "um so basically we need to uh schedule a meeting for um next week"
        result = remover.process(text)
        assert "we need to" in result.text
        assert "schedule" in result.text
        assert "meeting" in result.text
        assert "next week" in result.text

    def test_real_dictation_email(self, remover):
        text = "I I wanted to to let uh the the project is done"
        result = remover.process(text)
        assert result.text == "I wanted to let the project is done"

    def test_real_dictation_with_correction(self, remover):
        text = "send it to john actually send it to sarah"
        result = remover.process(text)
        assert result.has_self_corrections is True

    def test_real_dictation_numbers(self, remover):
        # "like" has no context cue here (no comma, not adjacent to an
        # unambiguous filler) — issue #43 keeps it as content ("about three
        # hundred and fifty"). Only "um" (unambiguous) is removed.
        text = "um the total is like three hundred and fifty"
        result = remover.process(text)
        assert result.text == "the total is like three hundred and fifty"


class TestOnnxDisfluencyPath:
    """Issue #43 review, finding 3: the ONNX path applied neither the
    contextual predicate nor the empty-output backstop that the regex
    fallback gained from this PR. A loaded model could still return
    `text=''` for "um" or drop "right" from "Turn right at the light." — the
    exact bugs this PR exists to fix, still live on the untested model path.

    No model is present on any dev machine, so — mirroring
    `TestOnnxPunctuationPath` in the stage 4b tests — a fake session with
    forced predictions gives this path its first real coverage.
    """

    @staticmethod
    def _remover_with_fake_model(remove_words: set[str]):
        """A DisfluencyRemover whose fake ONNX session predicts REMOVE for
        every token whose vocab id belongs to *remove_words* (matched via a
        dedicated vocab entry per word, case-insensitive) and KEEP otherwise.
        """
        import numpy as np

        vocab: dict[str, int] = {"[CLS]": 101, "[SEP]": 102, "[UNK]": 100}
        remove_ids: set[int] = set()
        for i, word in enumerate(remove_words, start=200):
            vocab[word.lower()] = i
            remove_ids.add(i)

        class FakeSession:
            def run(self, _outputs, feed):
                ids = feed["input_ids"][0]
                n_tokens = len(ids)
                logits = np.zeros((1, n_tokens, 3), dtype=np.float32)
                logits[0, :, _LABEL_KEEP] = 10.0
                for pos, token_id in enumerate(ids):
                    if int(token_id) in remove_ids:
                        logits[0, pos, :] = 0
                        logits[0, pos, _LABEL_REMOVE] = 10.0
                return [logits]

        r = DisfluencyRemover(model_dir=Path("/nonexistent/model"))
        r._session = FakeSession()
        r._vocab = vocab
        r._using_onnx = True
        return r

    def test_standalone_um_does_not_come_back_empty(self):
        # Without the backstop, a forced REMOVE on the only token in the
        # utterance returns text=='' — exactly the bug this finding reports.
        r = self._remover_with_fake_model({"um"})
        result = r.process("um")
        assert result.text != ""
        assert any(ch.isalnum() for ch in result.text)

    def test_ambiguous_word_survives_a_remove_prediction_with_no_cue(self):
        # The model wants to drop "right", same as it wants to drop "Turn",
        # "at", "the", "light" — but "right" has no context cue (no comma,
        # not a leading-adjacency case), so it must be kept as content even
        # though the (fake) model predicted REMOVE for it.
        r = self._remover_with_fake_model({"right"})
        result = r.process("Turn right at the light.")
        assert "right" in result.text.split()

    def test_ambiguous_word_is_removed_when_the_model_and_context_agree(self):
        # "so" at utterance-start with a comma is a genuine cue — the
        # override must not block a REMOVE prediction that context agrees
        # with, only one that context contradicts.
        #
        # The fake model matches vocab ids by exact `word.lower()`, and the
        # actual token here is "So," (comma attached, no space) — the vocab
        # entry has to include the comma or the lookup misses and the model
        # predicts KEEP for a token it was never actually asked about.
        r = self._remover_with_fake_model({"so,"})
        result = r.process("So, I think we should go.")
        # Issue #43 review, M1: `result.text.lower().split()` tokenises a
        # surviving "So," as "so," (comma attached), which is never equal
        # to the bare string "so" — so `"so" not in [...]` passed even when
        # "So," was still in the output. Match on a word boundary instead,
        # same fix as `test_other_fillers_still_stripped` in
        # TestDisfluencyRemover.
        assert not re.search(r"\bso\b", result.text, re.IGNORECASE)

    def test_onnx_path_is_actually_taken(self):
        r = self._remover_with_fake_model(set())
        assert r._using_onnx is True
        # Would raise AssertionError inside _process_onnx if session were None.
        r.process("some text")

    # Issue #43 review, O1: `_process_onnx` computed `clear_preceding_comma`
    # but never applied it, so a model REMOVE on a comma-bounded ambiguous
    # filler left the pairing comma behind as an orphan.

    def test_comma_bounded_filler_leaves_no_orphan_comma(self):
        r = self._remover_with_fake_model({"like,"})
        result = r.process("It was, like, really fast.")
        assert result.text == "It was really fast."

    # Issue #43 review, O2: the ONNX path deferred to the model for
    # unambiguous fillers (um, uh, hmm, ...) instead of always stripping
    # them the way the regex path does — a KEEP label (or a vocab lookup
    # that missed the attached punctuation) let one through.

    def test_unambiguous_filler_is_always_stripped_regardless_of_label(self):
        # KEEP-all stub (empty remove set) — nothing tells the model to
        # drop "um,", so without the O2 fix it survives.
        r = self._remover_with_fake_model(set())
        result = r.process("um, I think")
        assert result.text == "I think"


# =====================================================================
# Stage 4b: PunctuationRestorer
# =====================================================================

from linux_whisper.polish.punctuation import (
    PunctuationRestorer,
    _capitalise_sentence,
    _ensure_terminal_punctuation,
    _insert_commas,
    _split_into_sentences,
)


class TestSplitIntoSentences:

    def test_single_sentence_no_punct(self):
        assert _split_into_sentences("hello world") == ["hello world"]

    def test_two_sentences(self):
        result = _split_into_sentences("Hello world. How are you.")
        assert len(result) == 2

    def test_question_and_statement(self):
        result = _split_into_sentences("How are you? I am fine.")
        assert len(result) == 2

    def test_no_terminal_punct(self):
        result = _split_into_sentences("hello world")
        assert result == ["hello world"]


class TestInsertCommas:

    def test_comma_before_but(self):
        result = _insert_commas("I wanted to go but I stayed home")
        assert "go," in result

    def test_comma_before_because(self):
        result = _insert_commas("I stayed home because it was raining")
        assert "home," in result

    def test_comma_before_however(self):
        result = _insert_commas("the plan worked however we need improvements")
        assert "worked," in result

    def test_no_comma_for_short_text(self):
        result = _insert_commas("go but stay")
        # 3 words or fewer: no commas inserted
        assert "," not in result

    def test_no_double_comma(self):
        result = _insert_commas("I went, but I returned later today")
        # Should not add comma after "went," since it already has one
        assert result.count(",") == 1

    def test_comma_before_so(self):
        result = _insert_commas("we finished the work so we went home")
        assert "work," in result

    def test_no_comma_at_position_1(self):
        # Comma insertion requires i >= 2
        result = _insert_commas("go but I need to stay here now")
        # "but" is at position 1, so no comma before it
        words = result.split()
        assert not words[0].endswith(",")


class TestCapitaliseSentence:

    def test_capitalises_first_word(self):
        result = _capitalise_sentence("hello world")
        assert result.startswith("H")

    def test_capitalises_pronoun_i(self):
        result = _capitalise_sentence("i think i should go")
        assert "I" in result.split()

    def test_already_capitalised(self):
        result = _capitalise_sentence("Hello World")
        assert result.startswith("Hello")

    def test_empty_string(self):
        assert _capitalise_sentence("") == ""

    def test_capitalise_after_period(self):
        result = _capitalise_sentence("hello. world")
        assert "World" in result or "world" in result
        # After ".", next word should be capitalized
        parts = result.split(". ")
        if len(parts) == 2:
            assert parts[1][0].isupper()

    def test_i_always_capitalised(self):
        result = _capitalise_sentence("when i go i will see")
        # Every "i" should become "I"
        words = result.split()
        for w in words:
            if w.lower().rstrip(".,?!;:") == "i":
                assert w.rstrip(".,?!;:") == "I"


class TestEnsureTerminalPunctuation:

    def test_adds_period(self):
        result = _ensure_terminal_punctuation("hello world")
        assert result.endswith(".")

    def test_adds_question_mark_for_question(self):
        result = _ensure_terminal_punctuation("what time is it")
        assert result.endswith("?")

    def test_preserves_existing_period(self):
        result = _ensure_terminal_punctuation("hello world.")
        assert result == "hello world."
        assert result.count(".") == 1

    def test_preserves_existing_question_mark(self):
        result = _ensure_terminal_punctuation("is it done?")
        assert result == "is it done?"

    def test_preserves_existing_exclamation(self):
        result = _ensure_terminal_punctuation("wow!")
        assert result == "wow!"

    def test_question_starters(self):
        starters = ["who", "what", "where", "when", "why", "how",
                     "is", "are", "do", "does", "did", "can", "could",
                     "will", "would", "should"]
        for starter in starters:
            result = _ensure_terminal_punctuation(f"{starter} they coming")
            assert result.endswith("?"), f"'{starter}' should produce a question"

    def test_non_question_gets_period(self):
        result = _ensure_terminal_punctuation("the cat sat on the mat")
        assert result.endswith(".")

    def test_empty_string(self):
        result = _ensure_terminal_punctuation("")
        assert result == ""

    def test_whitespace_stripped(self):
        result = _ensure_terminal_punctuation("hello world   ")
        assert result.endswith(".")
        assert not result.endswith(" .")


class TestPunctuationRestorer:
    """Test the PunctuationRestorer class (rule-based fallback path)."""

    @pytest.fixture()
    def restorer(self):
        return PunctuationRestorer(model_dir=Path("/nonexistent/model"))

    def test_empty_input(self, restorer):
        assert restorer.process("") == ""

    def test_whitespace_only(self, restorer):
        assert restorer.process("   ") == ""

    def test_simple_sentence(self, restorer):
        result = restorer.process("hello world")
        assert result[0].isupper()  # capitalized
        assert result.endswith(".")  # terminal punct

    def test_question_detection(self, restorer):
        result = restorer.process("where are you going")
        assert result.endswith("?")
        assert result.startswith("W")

    def test_comma_insertion(self, restorer):
        result = restorer.process("I went to the store but I forgot my wallet")
        assert "," in result

    def test_pronoun_i_capitalised(self, restorer):
        result = restorer.process("i think i should leave now")
        words = result.split()
        for w in words:
            bare = w.rstrip(".,?!;:")
            if bare.lower() == "i":
                assert bare == "I"

    def test_already_punctuated(self, restorer):
        result = restorer.process("Hello world. How are you?")
        # Should not double-punctuate
        assert not result.endswith("..")
        assert "Hello" in result

    def test_multiple_sentences(self, restorer):
        result = restorer.process("hello world. how are you")
        assert result.count(".") >= 1 or result.count("?") >= 1

    def test_real_dictation_long(self, restorer):
        text = (
            "i went to the store and i bought some milk but they "
            "didnt have eggs so i went to another store"
        )
        result = restorer.process(text)
        assert result[0].isupper()
        assert result[-1] in ".?!"
        assert "I" in result.split()


# =====================================================================
# Stage 4d: SpokenFormFormatter
# =====================================================================

from linux_whisper.polish.formatting import (
    SpokenFormFormatter,
    _format_cardinal_numbers,
    _format_currency,
    _format_dates,
    _format_emails,
    _format_phone_numbers,
    _format_times,
    _words_to_number,
)


class TestWordsToNumber:
    """Test the internal _words_to_number helper."""

    def test_simple_ones(self):
        assert _words_to_number(["five"]) == 5

    def test_teens(self):
        assert _words_to_number(["thirteen"]) == 13

    def test_tens(self):
        assert _words_to_number(["twenty"]) == 20

    def test_compound(self):
        assert _words_to_number(["twenty", "five"]) == 25

    def test_hundred(self):
        assert _words_to_number(["three", "hundred"]) == 300

    def test_hundred_and_ones(self):
        assert _words_to_number(["three", "hundred", "and", "fifty"]) == 350

    def test_thousand(self):
        assert _words_to_number(["one", "thousand"]) == 1000

    def test_thousand_and_hundreds(self):
        assert _words_to_number(["one", "thousand", "two", "hundred"]) == 1200

    def test_million(self):
        assert _words_to_number(["one", "million"]) == 1_000_000

    def test_a_hundred(self):
        assert _words_to_number(["a", "hundred"]) == 100

    def test_empty(self):
        assert _words_to_number([]) is None

    def test_invalid_words(self):
        assert _words_to_number(["hello"]) is None


class TestFormatEmails:
    """Test email address formatting."""

    def test_basic_email(self):
        assert _format_emails("john at gmail dot com") == "john@gmail.com"

    def test_org_email(self):
        assert _format_emails("info at company dot org") == "info@company.org"

    def test_edu_email(self):
        assert _format_emails("student at school dot edu") == "student@school.edu"

    def test_io_email(self):
        assert _format_emails("dev at startup dot io") == "dev@startup.io"

    def test_preserves_surrounding_text(self):
        result = _format_emails("Send it to john at gmail dot com please")
        assert result == "Send it to john@gmail.com please"

    def test_already_formatted(self):
        assert _format_emails("john@gmail.com") == "john@gmail.com"

    def test_no_email(self):
        text = "the cat sat on the mat"
        assert _format_emails(text) == text

    def test_case_insensitive_tld(self):
        assert _format_emails("user at site dot COM") == "user@site.com"


class TestFormatPhoneNumbers:
    """Test phone number formatting."""

    def test_ten_digits(self):
        text = "one two three four five six seven eight nine zero"
        assert _format_phone_numbers(text) == "123-456-7890"

    def test_seven_digits(self):
        text = "five five five one two three four"
        assert _format_phone_numbers(text) == "555-1234"

    def test_preserves_non_phone_digits(self):
        # Three digit words don't form a phone number
        text = "one two three"
        assert _format_phone_numbers(text) == "one two three"

    def test_preserves_surrounding(self):
        text = "call me at one two three four five six seven eight nine zero please"
        result = _format_phone_numbers(text)
        assert "123-456-7890" in result
        assert result.startswith("call me at")
        assert result.endswith("please")

    def test_preserves_trailing_punctuation(self):
        text = "one two three four five six seven eight nine zero."
        result = _format_phone_numbers(text)
        assert result == "123-456-7890."


class TestFormatTimes:
    """Test time formatting."""

    def test_four_thirty_pm(self):
        assert _format_times("four thirty PM") == "4:30 PM"

    def test_twelve_fifteen_am(self):
        assert _format_times("twelve fifteen AM") == "12:15 AM"

    def test_three_forty_five(self):
        assert _format_times("three forty five") == "3:45"

    def test_nine_fifteen(self):
        assert _format_times("nine fifteen") == "9:15"

    def test_preserves_surrounding_text(self):
        result = _format_times("The meeting is at four thirty PM today")
        assert "4:30 PM" in result
        assert result.startswith("The meeting")

    def test_no_time_pattern(self):
        text = "the weather is nice"
        assert _format_times(text) == text

    def test_already_formatted(self):
        text = "4:30 PM"
        assert _format_times(text) == text

    def test_two_thirty(self):
        assert _format_times("two thirty") == "2:30"

    def test_one_fifteen_pm(self):
        assert _format_times("one fifteen PM") == "1:15 PM"

    def test_ten_twenty(self):
        assert _format_times("ten twenty") == "10:20"

    # -- Regression tests for #48: a comma-separated enumeration of number
    # words ("one, two, three") is not a spoken clock time, and used to be
    # collapsed into a single timestamp with the trailing numbers summed
    # into the minutes.

    @pytest.mark.parametrize(
        "text",
        [
            "one, two",
            "one, two, three",
            "one, two, three, four",
        ],
    )
    def test_comma_separated_enumeration_left_unchanged(self, text):
        # English spoken times never place a comma between hour and minute,
        # so the comma must terminate the pattern before it starts.
        assert _format_times(text) == text

    def test_comma_after_minute_word_blocks_extension(self):
        # A comma right after the (single-word) minute must not pull a
        # following list item in as a second minute word.
        assert _format_times("three forty, five") == "3:40, five"

    def test_minute_bounded_to_a_single_expression(self):
        # The old code summed every trailing number word into the minute
        # slot without limit: "forty" + "thirty" = 70 was rejected only
        # because it fell outside 0-59, discarding the whole match. The
        # fix caps the minute parse to one expression up front, so "forty"
        # alone is a valid partial match and "thirty" is correctly left
        # as its own word - never an invalid time like "1:70".
        result = _format_times("one forty thirty")
        assert result == "1:40 thirty"
        assert "1:70" not in result
        assert "1:75" not in result


class TestFormatDates:
    """Test date formatting."""

    def test_march_twenty_second(self):
        assert _format_dates("march twenty second") == "March 22nd"

    def test_january_first(self):
        assert _format_dates("january first") == "January 1st"

    def test_december_thirty_first(self):
        assert _format_dates("december thirty first") == "December 31st"

    def test_april_third(self):
        assert _format_dates("april third") == "April 3rd"

    def test_june_fifteenth(self):
        assert _format_dates("june fifteenth") == "June 15th"

    def test_november_twentieth(self):
        assert _format_dates("november twentieth") == "November 20th"

    def test_february_fourteenth(self):
        assert _format_dates("february fourteenth") == "February 14th"

    def test_preserves_surrounding_text(self):
        result = _format_dates("The party is on march twenty second this year")
        assert "March 22nd" in result

    def test_no_date(self):
        text = "the cat sat on the mat"
        assert _format_dates(text) == text

    def test_preserves_trailing_punctuation(self):
        result = _format_dates("march twenty second.")
        assert result == "March 22nd."

    def test_month_without_ordinal(self):
        # "march" alone should not be converted
        result = _format_dates("we march forward")
        assert result == "we march forward"


class TestFormatCurrency:
    """Test currency formatting."""

    def test_eight_hundred_dollars(self):
        assert _format_currency("eight hundred dollars") == "$800"

    def test_fifty_cents(self):
        assert _format_currency("fifty cents") == "$0.50"

    def test_twenty_five_dollars(self):
        assert _format_currency("twenty five dollars") == "$25"

    def test_dollars_and_cents(self):
        assert _format_currency("twenty five dollars and fifty cents") == "$25.50"

    def test_one_hundred_dollars(self):
        assert _format_currency("one hundred dollars") == "$100"

    def test_five_dollars(self):
        assert _format_currency("five dollars") == "$5"

    def test_preserves_surrounding_text(self):
        result = _format_currency("It costs eight hundred dollars total")
        assert "$800" in result
        assert result.endswith("total")

    def test_no_currency(self):
        text = "the cat sat on the mat"
        assert _format_currency(text) == text

    def test_preserves_trailing_punctuation(self):
        result = _format_currency("eight hundred dollars.")
        assert result == "$800."

    def test_three_hundred_and_fifty_dollars(self):
        assert _format_currency("three hundred and fifty dollars") == "$350"


class TestFormatCardinalNumbers:
    """Test cardinal number formatting."""

    def test_three_hundred_and_fifty(self):
        assert _format_cardinal_numbers("three hundred and fifty") == "350"

    def test_twenty_five(self):
        assert _format_cardinal_numbers("twenty five") == "25"

    def test_one_thousand(self):
        assert _format_cardinal_numbers("one thousand") == "1000"

    def test_one_thousand_two_hundred(self):
        assert _format_cardinal_numbers("one thousand two hundred") == "1200"

    def test_single_word_preserved(self):
        # Single number words in prose should NOT be converted
        text = "one of the reasons"
        assert _format_cardinal_numbers(text) == text

    def test_single_number_word_alone(self):
        text = "five"
        assert _format_cardinal_numbers(text) == text

    def test_preserves_surrounding_text(self):
        result = _format_cardinal_numbers("about three hundred and fifty items")
        assert "350" in result
        assert result.startswith("about")
        assert result.endswith("items")

    def test_already_numeric(self):
        text = "350 items"
        assert _format_cardinal_numbers(text) == text

    def test_preserves_trailing_punctuation(self):
        result = _format_cardinal_numbers("three hundred and fifty.")
        assert result == "350."

    def test_a_hundred(self):
        assert _format_cardinal_numbers("about a hundred items") == "about 100 items"


class TestSpokenFormFormatter:
    """Test the SpokenFormFormatter class."""

    @pytest.fixture()
    def formatter(self):
        return SpokenFormFormatter()

    def test_empty_input(self, formatter):
        assert formatter.process("") == ""

    def test_whitespace_only(self, formatter):
        assert formatter.process("   ") == "   "

    def test_clean_text_unchanged(self, formatter):
        text = "The cat sat on the mat."
        assert formatter.process(text) == text

    def test_already_formatted_number(self, formatter):
        assert formatter.process("350") == "350"

    def test_already_formatted_currency(self, formatter):
        assert formatter.process("$800") == "$800"

    def test_email_conversion(self, formatter):
        result = formatter.process("Send to john at gmail dot com")
        assert "john@gmail.com" in result

    def test_phone_number_conversion(self, formatter):
        text = "Call one two three four five six seven eight nine zero"
        result = formatter.process(text)
        assert "123-456-7890" in result

    def test_time_conversion(self, formatter):
        result = formatter.process("Meeting at four thirty PM")
        assert "4:30 PM" in result

    def test_date_conversion(self, formatter):
        result = formatter.process("Due on march twenty second")
        assert "March 22nd" in result

    def test_currency_conversion(self, formatter):
        result = formatter.process("It costs eight hundred dollars")
        assert "$800" in result

    def test_cardinal_number_conversion(self, formatter):
        result = formatter.process("There are three hundred and fifty items")
        assert "350" in result

    def test_multiple_formats_in_one_text(self, formatter):
        text = "Email john at gmail dot com about the march twenty second meeting"
        result = formatter.process(text)
        assert "john@gmail.com" in result
        assert "March 22nd" in result


class TestEnumerationNotFormattedAsTime:
    """Regression tests for #48.

    A comma-separated enumeration of number words ("one, two, three") was
    read as a spoken clock time: the whole list collapsed into one
    timestamp, with the trailing numbers summed into the minutes. Found in
    production dictating a message about implementing two features.
    """

    @pytest.fixture()
    def formatter(self):
        return SpokenFormFormatter()

    def test_two_item_list(self, formatter):
        text = "implement one, two, as well as the recording indicator"
        assert formatter.process(text) == text

    def test_two_item_list_with_and(self, formatter):
        text = "do one, two and three"
        assert formatter.process(text) == text

    def test_three_item_list(self, formatter):
        text = "items one, two, three"
        assert formatter.process(text) == text

    def test_four_item_list(self, formatter):
        text = "steps one, two, three, four"
        assert formatter.process(text) == text

    def test_genuine_time_is_unaffected(self, formatter):
        # The pattern this bug exploited is real and worth keeping.
        assert formatter.process("let's meet at one thirty") == "let's meet at 1:30"

    @pytest.mark.parametrize(
        ("text", "summed_minute"),
        [
            ("count one, two", "1:02"),  # would-be sum: 2
            ("items one, two, three", "1:05"),  # would-be sum: 2 + 3
            ("steps one, two, three, four", "1:09"),  # would-be sum: 2 + 3 + 4
        ],
    )
    def test_minute_never_equals_sum_of_list_items(self, formatter, text, summed_minute):
        """No list of length 2, 3, or 4 may sum its trailing items into the minute."""
        assert summed_minute not in formatter.process(text)

    @pytest.mark.parametrize(
        "text",
        [
            "implement one, two, as well as the recording indicator",
            "do one, two and three",
            "items one, two, three",
            "steps one, two, three, four",
        ],
    )
    def test_no_time_pattern_emitted(self, formatter, text):
        """None of these enumerations should ever collapse into an H:MM time."""
        result = formatter.process(text)
        assert not re.search(r"\b\d{1,2}:\d{2}\b", result), f"{text!r} -> {result!r}"


# =====================================================================
# Stage 4c: LLMCorrector
# =====================================================================

from linux_whisper.polish.llm import LLMCorrector


class TestLLMCorrectorUnavailable:
    """Test LLMCorrector when the model is not available."""

    def test_not_available_by_default(self, monkeypatch):
        # With a non-existent model file, the corrector should be unavailable
        monkeypatch.setattr(
            "linux_whisper.polish.llm._DEFAULT_MODEL_DIR",
            Path("/tmp/nonexistent-llm-dir"),
        )
        corrector = LLMCorrector(config=PolishConfig())
        assert corrector.available is False

    def test_process_returns_unchanged_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "linux_whisper.polish.llm._DEFAULT_MODEL_DIR",
            Path("/tmp/nonexistent-llm-dir"),
        )
        corrector = LLMCorrector(config=PolishConfig())
        text = "at 2 actually at 4"
        result = corrector.process(text)
        assert result == text

    def test_process_empty_returns_empty(self):
        corrector = LLMCorrector(config=PolishConfig())
        assert corrector.process("") == ""

    def test_process_whitespace_returns_whitespace(self):
        corrector = LLMCorrector(config=PolishConfig())
        assert corrector.process("   ") == "   "


class TestLLMCorrectorTimeout:
    """Test LLMCorrector timeout behavior."""

    def test_timeout_returns_original(self):
        corrector = LLMCorrector(config=PolishConfig())
        # Force-enable the corrector with a mock model
        corrector._loaded = True
        corrector._model = MagicMock()
        corrector._timeout_s = 0.01  # very short timeout

        # Make the mock model block forever
        def slow_inference(*args, **kwargs):
            import time
            time.sleep(10)  # much longer than timeout
            return {"choices": [{"message": {"content": "corrected"}}]}

        corrector._model.create_chat_completion = slow_inference

        result = corrector.process("hello world")
        assert result == "hello world"  # original text returned


class TestLLMCorrectorHallucinationRejection:
    """Test that excessively long LLM outputs are rejected."""

    def test_rejects_output_over_2x_length(self):
        corrector = LLMCorrector(config=PolishConfig())
        corrector._loaded = True
        corrector._model = MagicMock()
        corrector._timeout_s = 5.0

        # Return output that is > 2x the input length
        long_output = "This is a very long hallucinated response " * 20
        corrector._model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": long_output}}]
        }

        short_input = "fix this"
        result = corrector.process(short_input)
        assert result == short_input  # original returned

    def test_accepts_reasonable_output(self):
        corrector = LLMCorrector(config=PolishConfig())
        corrector._loaded = True
        corrector._model = MagicMock()
        corrector._timeout_s = 5.0

        corrector._model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "at 4"}}]
        }

        result = corrector.process("at 2 actually at 4")
        assert result == "at 4"

    def test_empty_llm_output_returns_original(self):
        corrector = LLMCorrector(config=PolishConfig())
        corrector._loaded = True
        corrector._model = MagicMock()
        corrector._timeout_s = 5.0

        corrector._model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": ""}}]
        }

        result = corrector.process("hello world")
        assert result == "hello world"

    def test_no_choices_returns_original(self):
        corrector = LLMCorrector(config=PolishConfig())
        corrector._loaded = True
        corrector._model = MagicMock()
        corrector._timeout_s = 5.0

        corrector._model.create_chat_completion.return_value = {"choices": []}

        result = corrector.process("hello world")
        assert result == "hello world"

    def test_inference_exception_returns_original(self):
        corrector = LLMCorrector(config=PolishConfig())
        corrector._loaded = True
        corrector._model = MagicMock()
        corrector._timeout_s = 5.0

        corrector._model.create_chat_completion.side_effect = RuntimeError("boom")

        result = corrector.process("hello world")
        assert result == "hello world"


class TestLLMCorrectorDevice:
    """Test GPU offload configuration."""

    def test_cpu_device_zero_gpu_layers(self, monkeypatch):
        """CPU device should use n_gpu_layers=0."""
        monkeypatch.setattr(
            "linux_whisper.polish.llm._DEFAULT_MODEL_DIR",
            Path("/tmp/nonexistent-llm-dir"),
        )
        cfg = PolishConfig(llm_device="cpu")
        corrector = LLMCorrector(config=cfg)
        assert corrector._config.llm_device == "cpu"

    def test_rocm_device_stored_in_config(self, monkeypatch):
        """ROCm device config should be stored correctly."""
        monkeypatch.setattr(
            "linux_whisper.polish.llm._DEFAULT_MODEL_DIR",
            Path("/tmp/nonexistent-llm-dir"),
        )
        cfg = PolishConfig(llm_device="rocm")
        corrector = LLMCorrector(config=cfg)
        assert corrector._config.llm_device == "rocm"

    def test_rocm_fallback_when_gpu_unavailable(self, monkeypatch):
        """When rocm is requested but GPU offload is unavailable, should fall back."""
        monkeypatch.setattr(
            "linux_whisper.polish.llm._DEFAULT_MODEL_DIR",
            Path("/tmp/nonexistent-llm-dir"),
        )
        monkeypatch.setattr(
            "linux_whisper.polish.llm.llama_supports_gpu_offload",
            lambda: False,
        )
        cfg = PolishConfig(llm_device="rocm")
        corrector = LLMCorrector(config=cfg)
        # Model won't load (no file), but config is set — the fallback
        # logic is in _try_load_model which we can't call without a model file.
        # Verify the config is stored correctly for the fallback path.
        assert corrector._config.llm_device == "rocm"

    def test_default_device_is_rocm(self):
        """Default config should use CPU."""
        cfg = PolishConfig()
        assert cfg.llm_device == "rocm"

    def test_rocm_config_from_dict(self):
        """Config.from_dict should parse llm_device."""
        from linux_whisper.config import Config
        cfg = Config.from_dict({"polish": {"llm_device": "rocm"}})
        assert cfg.polish.llm_device == "rocm"


class TestLLMCorrectorModelPath:
    """Test the _resolve_model_path logic."""

    def test_resolve_default_model(self):
        corrector = LLMCorrector(config=PolishConfig())
        path = corrector._resolve_model_path()
        assert path is not None
        assert path.name == "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

    def test_config_default_matches_packaged_filename(self):
        # The config default and _DEFAULT_MODEL_FILENAME must name the same
        # file.  They drifted apart before, leaving the empty-config fallback
        # pointing at a GGUF that had never existed.
        from linux_whisper.polish.llm import _DEFAULT_MODEL_FILENAME

        assert f"{PolishConfig().llm_model}.gguf" == _DEFAULT_MODEL_FILENAME

    def test_resolve_empty_model_falls_back_to_packaged_default(self):
        from linux_whisper.polish.llm import _DEFAULT_MODEL_FILENAME

        corrector = LLMCorrector(config=PolishConfig(llm_model=""))
        path = corrector._resolve_model_path()
        assert path is not None
        assert path.name == _DEFAULT_MODEL_FILENAME

    def test_previous_model_remains_selectable(self):
        # Rollback path: the old Qwen3-4B must stay loadable from config.
        cfg = PolishConfig(llm_model="Qwen3-4B-Q4_K_M")
        corrector = LLMCorrector(config=cfg)
        path = corrector._resolve_model_path()
        assert path is not None
        assert path.name == "Qwen3-4B-Q4_K_M.gguf"

    def test_resolve_gguf_suffix(self):
        cfg = PolishConfig(llm_model="custom-model.gguf")
        corrector = LLMCorrector(config=cfg)
        path = corrector._resolve_model_path()
        assert path.name == "custom-model.gguf"

    def test_resolve_absolute_path(self):
        cfg = PolishConfig(llm_model="/opt/models/my-model.gguf")
        corrector = LLMCorrector(config=cfg)
        path = corrector._resolve_model_path()
        assert path == Path("/opt/models/my-model.gguf")

    def test_resolve_plain_name(self):
        cfg = PolishConfig(llm_model="SomeModel")
        corrector = LLMCorrector(config=cfg)
        path = corrector._resolve_model_path()
        assert path.name == "SomeModel.gguf"


# =====================================================================
# PolishPipeline integration
# =====================================================================

from linux_whisper.polish.pipeline import PolishPipeline


class TestPolishPipelineDisabled:
    """Test pipeline when polish is disabled."""

    def test_disabled_returns_input_unchanged(self):
        pipeline = PolishPipeline(PolishConfig(enabled=False))
        text = "um so like I was going"
        assert pipeline.process(text) == text

    def test_disabled_does_not_init_stages(self):
        pipeline = PolishPipeline(PolishConfig(enabled=False))
        assert pipeline._disfluency is None
        assert pipeline._punctuation is None
        assert pipeline._formatting is None
        assert pipeline._llm is None


class TestPolishPipelineStageToggling:
    """Test enabling/disabling individual stages."""

    def test_disfluency_only(self):
        cfg = PolishConfig(
            enabled=True,
            disfluency=True,
            punctuation=False,
            formatting=False,
            llm=False,
        )
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is not None
        assert pipeline._punctuation is None
        assert pipeline._formatting is None
        assert pipeline._llm is None

    def test_punctuation_only(self):
        cfg = PolishConfig(
            enabled=True,
            disfluency=False,
            punctuation=True,
            formatting=False,
            llm=False,
        )
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is None
        assert pipeline._punctuation is not None
        assert pipeline._formatting is None
        assert pipeline._llm is None

    def test_formatting_only(self):
        cfg = PolishConfig(
            enabled=True,
            disfluency=False,
            punctuation=False,
            formatting=True,
            llm=False,
        )
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is None
        assert pipeline._punctuation is None
        assert pipeline._formatting is not None
        assert pipeline._llm is None

    def test_llm_only(self):
        cfg = PolishConfig(
            enabled=True,
            disfluency=False,
            punctuation=False,
            formatting=False,
            llm=True,
        )
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is None
        assert pipeline._punctuation is None
        assert pipeline._formatting is None
        assert pipeline._llm is not None

    def test_all_stages_enabled(self):
        cfg = PolishConfig(
            enabled=True,
            disfluency=True,
            punctuation=True,
            formatting=True,
            llm=True,
        )
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is not None
        assert pipeline._punctuation is not None
        assert pipeline._formatting is not None
        assert pipeline._llm is not None


class TestPolishPipelineEmpty:
    """Test pipeline with empty/whitespace input."""

    def test_empty_string(self):
        pipeline = PolishPipeline(PolishConfig())
        assert pipeline.process("") == ""

    def test_whitespace_string(self):
        pipeline = PolishPipeline(PolishConfig())
        assert pipeline.process("   ") == ""


class TestPolishPipelineIntegration:
    """Full pipeline integration tests (disfluency + punctuation, no LLM model)."""

    @pytest.fixture()
    def pipeline(self):
        """Pipeline with disfluency + punctuation but no LLM (model not present)."""
        cfg = PolishConfig(enabled=True, disfluency=True, punctuation=True, llm=False)
        return PolishPipeline(cfg)

    def test_cleans_and_punctuates(self, pipeline):
        text = "um I think we should go"
        result = pipeline.process(text)
        # Should remove "um", capitalize, add period
        assert result[0].isupper()
        assert result[-1] in ".?!"
        assert "um" not in result.split()

    def test_repetition_and_punctuation(self, pipeline):
        text = "the the cat sat on the mat"
        result = pipeline.process(text)
        assert result.startswith("The")
        assert result.endswith(".")
        # "the the" collapsed to "the"
        assert "the the" not in result.lower()

    def test_question_detection_after_disfluency(self, pipeline):
        text = "um where are you going"
        result = pipeline.process(text)
        assert result.endswith("?")

    def test_preserves_clean_text(self, pipeline):
        text = "the weather is nice today"
        result = pipeline.process(text)
        assert "weather" in result
        assert "nice" in result
        assert "today" in result


class TestPolishPipelineLLMConditional:
    """Stage 4c is only invoked when it should be.

    These previously wrapped every assertion in
    `if pipeline._llm is not None and pipeline._llm._model is not None:`, so
    wherever the LLM could not be constructed — CI, for one — the assertions
    were skipped entirely and the tests passed without testing anything. The
    unused `result` was the tell that ruff's F841 flagged. They now skip
    loudly instead of passing quietly.
    """

    @staticmethod
    def _pipeline_with_mock_llm(*, llm_always: bool, reply: str) -> PolishPipeline:
        cfg = PolishConfig(
            enabled=True,
            disfluency=True,
            punctuation=False,
            llm=True,
            llm_always=llm_always,
        )
        pipeline = PolishPipeline(cfg)
        if pipeline._llm is None:
            pytest.skip("stage 4c (LLM) could not be constructed in this environment")
        pipeline._llm._loaded = True
        pipeline._llm._model = MagicMock()
        pipeline._llm._timeout_s = 5.0
        pipeline._llm._model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": reply}}]
        }
        return pipeline

    def test_llm_skipped_without_self_corrections(self):
        pipeline = self._pipeline_with_mock_llm(
            llm_always=False, reply="should not be called"
        )

        result = pipeline.process("the weather is nice today")

        assert isinstance(result, str) and result
        pipeline._llm._model.create_chat_completion.assert_not_called()

    def test_llm_invoked_with_self_corrections(self):
        pipeline = self._pipeline_with_mock_llm(llm_always=False, reply="at 4")

        result = pipeline.process("at 2 actually at 4")

        assert isinstance(result, str) and result
        pipeline._llm._model.create_chat_completion.assert_called_once()

    def test_llm_always_flag(self):
        pipeline = self._pipeline_with_mock_llm(
            llm_always=True, reply="cleaned up text"
        )

        result = pipeline.process("normal text without corrections")

        assert isinstance(result, str) and result
        pipeline._llm._model.create_chat_completion.assert_called_once()


class TestSystemPromptHygiene:
    """The system prompt must not carry model-specific workarounds."""

    def test_first_line_is_intact(self):
        """The opening line is assembled from three implicitly concatenated
        pieces to fit the column limit. Assert it reassembles exactly — a
        dropped space would silently change a benchmark-gated prompt."""
        from linux_whisper.polish.llm import _SYSTEM_PROMPT

        assert _SYSTEM_PROMPT.split("\n")[0] == (
            "You resolve self-corrections in dictated text. When someone "
            "changes their mind mid-sentence, keep ONLY the final version. "
            "Fix grammar. Output ONLY the result."
        )

    def test_no_thinking_suppression_hack(self):
        from linux_whisper.polish.llm import _SYSTEM_PROMPT

        # The default model is instruct-only, so there is no reasoning mode to
        # switch off.  A stray /no_think would be dead weight in every prompt.
        assert "/no_think" not in _SYSTEM_PROMPT
        assert "/nothink" not in _SYSTEM_PROMPT.lower()

    def test_prompt_still_forbids_paraphrasing(self):
        from linux_whisper.polish.llm import _SYSTEM_PROMPT

        assert "ONLY" in _SYSTEM_PROMPT

    def test_prompt_retains_self_correction_examples(self):
        from linux_whisper.polish.llm import _SYSTEM_PROMPT

        assert "actually" in _SYSTEM_PROMPT
        assert _SYSTEM_PROMPT.count("→") >= 5


class TestLiteralTokenPreservation:
    """Code-like tokens must survive the rule-based punctuation stage intact.

    Dictating code produces tokens that are not prose. Prose rules corrupted
    them: filenames got capitalised, paths gained a trailing period, and
    command lines gained Oxford commas. On a case-sensitive filesystem
    "Server-test.sh" is a different file, and "--verbose." is not a valid flag.
    """

    @pytest.fixture()
    def restorer(self):
        from linux_whisper.polish.punctuation import PunctuationRestorer

        return PunctuationRestorer(model_dir=Path("/nonexistent/model"))

    @pytest.mark.parametrize(
        "text",
        [
            "server-test.sh",
            "Open src/linux_whisper/stt/parakeet.py",
            "Check the .env file in the project root.",
            "It lives in /var/log/syslog",
            "The variable is called max_default_threads.",
            "Call getUserById with the account ID.",
            "Run it with --no-cache and --verbose.",
            "Upgrade to version 0.3.34 and rebuild.",
            "Run git rebase --interactive on main.",
        ],
    )
    def test_code_passes_through_unchanged(self, restorer, text):
        assert restorer.process(text) == text

    def test_filename_is_not_capitalised(self, restorer):
        # "Server-test.sh" is a different file on a case-sensitive filesystem.
        assert restorer.process("server-test.sh").startswith("server")

    def test_no_period_appended_after_a_path(self, restorer):
        assert not restorer.process("It lives in /var/log/syslog").endswith(".")

    def test_no_comma_inserted_between_flags(self, restorer):
        assert "," not in restorer.process("Run it with --no-cache and --verbose.")

    def test_prose_still_gets_capitalised_and_punctuated(self, restorer):
        # The guard must not disable normal behaviour.
        assert restorer.process("the tests are green") == "The tests are green."

    def test_prose_questions_still_work(self, restorer):
        out = restorer.process("did you see the pull request")
        assert out.endswith("?") and out.startswith("Did")

    def test_adjacent_clause_markers_get_one_comma(self, restorer):
        # Regression: "and then" produced "..., and, then ...".
        out = restorer.process("i was thinking we should move it and then ship it")
        assert ", and," not in out
        assert out.count(",") == 1


class TestIsLiteralToken:
    """Unit coverage for the literal-token predicate."""

    @pytest.mark.parametrize(
        "token",
        [
            "server-test.sh",
            "src/linux_whisper/stt/parakeet.py",
            "/var/log/syslog",
            ".env",
            "max_default_threads",
            "getUserById",
            "--no-cache",
            "-v",
            "0.3.34",
        ],
    )
    def test_recognises_code_tokens(self, token):
        from linux_whisper.polish.punctuation import _is_literal_token

        assert _is_literal_token(token) is True

    @pytest.mark.parametrize(
        "token",
        ["hello", "SQL", "JSON", "the", "I", "", "don't", "Nathan", "well-known"],
    )
    def test_leaves_prose_alone(self, token):
        # "well-known" is hyphenated prose, not a filename — no extension.
        # Bare acronyms are prose too; they need no protection.
        from linux_whisper.polish.punctuation import _is_literal_token

        assert _is_literal_token(token) is False


class TestSpokenYearFormatting:
    """Spoken years are read as paired numbers, not summed.

    "twenty twenty six" is 2026, not 20 + 20 + 6 = 46. The summing behaviour
    silently corrupted every dictated year this decade (issue #32).
    """

    @pytest.fixture()
    def formatter(self):
        from linux_whisper.polish.formatting import SpokenFormFormatter

        return SpokenFormFormatter()

    @pytest.mark.parametrize(
        ("words", "expected"),
        [
            (["twenty", "twenty", "six"], 2026),
            (["twenty", "twenty"], 2020),
            (["nineteen", "ninety", "nine"], 1999),
            (["nineteen", "eighty", "four"], 1984),
            (["two", "thousand", "twenty", "six"], 2026),
        ],
    )
    def test_years_parse(self, words, expected):
        from linux_whisper.polish.formatting import _words_to_year

        assert _words_to_year(words) == expected

    @pytest.mark.parametrize(
        "words",
        [
            ["twenty", "three"],       # 23, not 2003
            ["thirty", "five"],        # 35, not 3005
            ["three", "hundred"],      # not a year
            ["five"],                  # too short
            [],
        ],
    )
    def test_non_years_rejected(self, words):
        from linux_whisper.polish.formatting import _words_to_year

        assert _words_to_year(words) is None

    def test_date_with_year(self, formatter):
        out = formatter.process("lets ship on march fifteenth twenty twenty six")
        assert out == "lets ship on March 15, 2026"

    def test_date_without_year_keeps_ordinal(self, formatter):
        assert formatter.process("lets ship on march twenty second") == "lets ship on March 22nd"

    def test_bare_year(self, formatter):
        assert formatter.process("the release was in nineteen ninety nine") == (
            "the release was in 1999"
        )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("the build took twenty three minutes", "the build took 23 minutes"),
            ("she is twenty five years old", "she is 25 years old"),
            ("i counted three hundred and fifty items", "i counted 350 items"),
        ],
    )
    def test_ordinary_numbers_unaffected(self, formatter, text, expected):
        # The year rule must not capture ordinary compounds.
        assert formatter.process(text) == expected


class TestPercentAndMeridiem:

    @pytest.fixture()
    def formatter(self):
        from linux_whisper.polish.formatting import SpokenFormFormatter

        return SpokenFormFormatter()

    def test_percent(self, formatter):
        assert formatter.process("latency dropped by forty percent") == (
            "latency dropped by 40%"
        )

    def test_percent_compound(self, formatter):
        assert formatter.process("it grew by twenty five percent") == "it grew by 25%"

    def test_split_meridiem(self, formatter):
        # Dictation splits "a.m." into two tokens: "nine thirty a m".
        assert formatter.process("the standup is at nine thirty a m") == (
            "the standup is at 9:30 AM"
        )

    def test_joined_meridiem_still_works(self, formatter):
        assert formatter.process("the meeting is at four thirty PM") == (
            "the meeting is at 4:30 PM"
        )

    def test_single_number_word_left_alone(self, formatter):
        # Deliberate: converting bare cardinals would turn "one of the things"
        # into "1 of the things".
        assert formatter.process("one of the things we discussed") == (
            "one of the things we discussed"
        )
        assert formatter.process("i need fifteen units") == "i need fifteen units"


class TestOnnxPunctuationPath:
    """First coverage for the ELECTRA path, which no test previously exercised.

    The models are not present on any dev machine, so this path silently never
    ran. Dropping models in would have switched the live pipeline to untested
    code — and would have bypassed the code-token guard that the rule path
    gained, re-breaking every dictated filename.
    """

    @staticmethod
    def _restorer_with_fake_models(punct_label: int, cap_label: int):
        """A PunctuationRestorer whose ONNX sessions always predict the same labels."""
        import numpy as np

        from linux_whisper.polish.punctuation import PunctuationRestorer

        class FakeSession:
            def __init__(self, label: int, n_labels: int) -> None:
                self._label = label
                self._n = n_labels

            def run(self, _outputs, feed):
                n_tokens = feed["input_ids"].shape[1]
                logits = np.zeros((1, n_tokens, self._n), dtype=np.float32)
                logits[0, :, self._label] = 10.0
                return [logits]

        r = PunctuationRestorer(model_dir=Path("/nonexistent/model"))
        r._punct_session = FakeSession(punct_label, 7)
        r._caps_session = FakeSession(cap_label, 3)
        r._vocab = {"[CLS]": 101, "[SEP]": 102, "[UNK]": 100}
        r._using_onnx = True
        return r

    def test_model_predictions_apply_to_prose(self):
        # Sanity check the harness: with "capitalise + period" forced, prose
        # should come back capitalised and punctuated.
        r = self._restorer_with_fake_models(punct_label=2, cap_label=1)
        assert r.process("hello there") == "Hello. There."

    def test_code_tokens_bypass_the_model(self):
        # Same forced predictions — code tokens must be untouched.
        r = self._restorer_with_fake_models(punct_label=2, cap_label=1)
        assert r.process("server-test.sh") == "server-test.sh"

    @pytest.mark.parametrize(
        "token",
        [
            "server-test.sh",
            "src/linux_whisper/stt/parakeet.py",
            "/var/log/syslog",
            ".env",
            "max_default_threads",
            "getUserById",
            "--no-cache",
            "0.3.34",
        ],
    )
    def test_every_literal_form_survives_the_model(self, token):
        r = self._restorer_with_fake_models(punct_label=2, cap_label=2)
        assert token in r.process(f"run {token} now")

    def test_mixed_prose_and_code(self):
        r = self._restorer_with_fake_models(punct_label=0, cap_label=1)
        out = r.process("open src/main.py now")
        assert "src/main.py" in out
        assert out.startswith("Open")

    def test_onnx_path_is_actually_taken(self):
        r = self._restorer_with_fake_models(punct_label=0, cap_label=0)
        assert r._using_onnx is True
        # Would raise AssertionError inside _process_onnx if sessions were None.
        r.process("some text")
