"""Comprehensive tests for the polish pipeline stages.

Tests cover:
- DisfluencyRemover regex fallback (filler removal, repetitions, self-corrections)
- PunctuationRestorer rule-based fallback (capitalization, terminal punct, commas)
- LLMCorrector (unavailable behavior, timeout, hallucination rejection)
- PolishPipeline (full integration, stage toggling, conditional LLM invocation)
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

from linux_whisper.config import PolishConfig


# =====================================================================
# Stage 4a: DisfluencyRemover
# =====================================================================

from linux_whisper.polish.disfluency import (
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

    def test_removes_like(self):
        result = _remove_fillers("I was like going to the store")
        assert "like" not in result.split()

    def test_removes_basically(self):
        result = _remove_fillers("basically I need help")
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
        result = _remove_fillers("um so like I was basically going there")
        # Only meaningful words should remain
        assert "going" in result
        assert "there" in result

    def test_removes_well(self):
        result = _remove_fillers("well I think so")
        assert "well" not in result.split()

    def test_removes_okay(self):
        result = _remove_fillers("okay lets do this")
        assert "okay" not in result.lower()

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
        assert "I want to go home" == result.text

    def test_combined_fillers_and_repetitions(self, remover):
        result = remover.process("um the the cat was like sitting there")
        assert "the cat was sitting there" == result.text

    def test_self_correction_detected(self, remover):
        result = remover.process("meet at 3 actually meet at 5")
        assert result.has_self_corrections is True

    def test_no_self_correction_flagged(self, remover):
        result = remover.process("the weather is nice today")
        assert result.has_self_corrections is False

    def test_returns_disfluency_result(self, remover):
        result = remover.process("hello world")
        assert isinstance(result, DisfluencyResult)

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
        assert "I wanted to let the project is done" == result.text

    def test_real_dictation_with_correction(self, remover):
        text = "send it to john actually send it to sarah"
        result = remover.process(text)
        assert result.has_self_corrections is True

    def test_real_dictation_numbers(self, remover):
        text = "um the total is like three hundred and fifty"
        result = remover.process(text)
        assert "the total is three hundred and fifty" == result.text


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
        text = "i went to the store and i bought some milk but they didnt have eggs so i went to another store"
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

    def test_default_device_is_cpu(self):
        """Default config should use CPU."""
        cfg = PolishConfig()
        assert cfg.llm_device == "cpu"

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
        cfg = PolishConfig(enabled=True, disfluency=True, punctuation=False, formatting=False, llm=False)
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is not None
        assert pipeline._punctuation is None
        assert pipeline._formatting is None
        assert pipeline._llm is None

    def test_punctuation_only(self):
        cfg = PolishConfig(enabled=True, disfluency=False, punctuation=True, formatting=False, llm=False)
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is None
        assert pipeline._punctuation is not None
        assert pipeline._formatting is None
        assert pipeline._llm is None

    def test_formatting_only(self):
        cfg = PolishConfig(enabled=True, disfluency=False, punctuation=False, formatting=True, llm=False)
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is None
        assert pipeline._punctuation is None
        assert pipeline._formatting is not None
        assert pipeline._llm is None

    def test_llm_only(self):
        cfg = PolishConfig(enabled=True, disfluency=False, punctuation=False, formatting=False, llm=True)
        pipeline = PolishPipeline(cfg)
        assert pipeline._disfluency is None
        assert pipeline._punctuation is None
        assert pipeline._formatting is None
        assert pipeline._llm is not None

    def test_all_stages_enabled(self):
        cfg = PolishConfig(enabled=True, disfluency=True, punctuation=True, formatting=True, llm=True)
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
    """Test that LLM is only invoked when appropriate."""

    def test_llm_skipped_without_self_corrections(self):
        cfg = PolishConfig(enabled=True, disfluency=True, punctuation=False, llm=True, llm_always=False)
        pipeline = PolishPipeline(cfg)

        # Mock the LLM
        if pipeline._llm is not None:
            pipeline._llm._loaded = True
            pipeline._llm._model = MagicMock()
            pipeline._llm._timeout_s = 5.0
            pipeline._llm._model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "should not be called"}}]
            }

        result = pipeline.process("the weather is nice today")
        # LLM should NOT have been called (no self-corrections)
        if pipeline._llm is not None and pipeline._llm._model is not None:
            pipeline._llm._model.create_chat_completion.assert_not_called()

    def test_llm_invoked_with_self_corrections(self):
        cfg = PolishConfig(enabled=True, disfluency=True, punctuation=False, llm=True, llm_always=False)
        pipeline = PolishPipeline(cfg)

        if pipeline._llm is not None:
            pipeline._llm._loaded = True
            pipeline._llm._model = MagicMock()
            pipeline._llm._timeout_s = 5.0
            pipeline._llm._model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "at 4"}}]
            }

        result = pipeline.process("at 2 actually at 4")
        # LLM should have been called because self-correction was detected
        if pipeline._llm is not None and pipeline._llm._model is not None:
            pipeline._llm._model.create_chat_completion.assert_called_once()

    def test_llm_always_flag(self):
        cfg = PolishConfig(enabled=True, disfluency=True, punctuation=False, llm=True, llm_always=True)
        pipeline = PolishPipeline(cfg)

        if pipeline._llm is not None:
            pipeline._llm._loaded = True
            pipeline._llm._model = MagicMock()
            pipeline._llm._timeout_s = 5.0
            pipeline._llm._model.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "cleaned up text"}}]
            }

        result = pipeline.process("normal text without corrections")
        # LLM should still be called because llm_always is True
        if pipeline._llm is not None and pipeline._llm._model is not None:
            pipeline._llm._model.create_chat_completion.assert_called_once()
