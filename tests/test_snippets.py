"""Tests for linux_whisper.snippets — SnippetMatcher fuzzy matching."""

from __future__ import annotations

import pytest

from linux_whisper.snippets import SnippetMatcher

# ── Fixture ────────────────────────────────────────────────────────────────


@pytest.fixture()
def matcher() -> SnippetMatcher:
    """A SnippetMatcher with typical snippet configurations."""
    return SnippetMatcher({
        "my email": "nathan@example.com",
        "meeting followup": (
            "Hi team,\n\n"
            "Following up on our meeting, here are the action items:\n"
            "- "
        ),
        "home address": "123 Main St, City, State 12345",
    })


# ── Exact match ────────────────────────────────────────────────────────────


class TestExactMatch:
    """Test exact (normalized) matching."""

    def test_exact_match(self, matcher):
        assert matcher.match("my email") == "nathan@example.com"

    def test_exact_match_address(self, matcher):
        assert matcher.match("home address") == "123 Main St, City, State 12345"

    def test_exact_match_multiline(self, matcher):
        result = matcher.match("meeting followup")
        assert result is not None
        assert "Hi team," in result
        assert "\n" in result

    def test_case_insensitive(self, matcher):
        assert matcher.match("My Email") == "nathan@example.com"

    def test_case_insensitive_upper(self, matcher):
        assert matcher.match("MY EMAIL") == "nathan@example.com"

    def test_whitespace_stripping(self, matcher):
        assert matcher.match("  my email  ") == "nathan@example.com"

    def test_extra_internal_whitespace(self, matcher):
        assert matcher.match("my   email") == "nathan@example.com"


# ── Fuzzy match ────────────────────────────────────────────────────────────


class TestFuzzyMatch:
    """Test fuzzy matching via SequenceMatcher."""

    def test_minor_variation(self, matcher):
        # "meeting follow up" (with space) vs "meeting followup" (no space)
        result = matcher.match("meeting follow up")
        assert result is not None
        assert "Hi team," in result

    def test_slight_typo(self):
        m = SnippetMatcher({"thank you": "Thanks for your help!"})
        # "thank yu" has high similarity to "thank you"
        result = m.match("thank yu")
        assert result == "Thanks for your help!"

    def test_stt_variation_hyphens(self):
        m = SnippetMatcher({"follow up": "Following up on our last conversation."})
        # STT might produce "followup" (one word)
        result = m.match("followup")
        assert result is not None


# ── No match ───────────────────────────────────────────────────────────────


class TestNoMatch:
    """Test that non-matching transcriptions return None."""

    def test_completely_different(self, matcher):
        assert matcher.match("the weather is nice today") is None

    def test_empty_string(self, matcher):
        assert matcher.match("") is None

    def test_whitespace_only(self, matcher):
        assert matcher.match("   ") is None

    def test_no_partial_match(self, matcher):
        # "my email and more stuff" should NOT match "my email" trigger
        assert matcher.match("my email and more stuff after it") is None

    def test_below_threshold(self):
        m = SnippetMatcher({"hello world": "hi"}, threshold=0.99)
        # "hello worl" similarity is below 0.99
        assert m.match("hello worl") is None


# ── Empty snippets ─────────────────────────────────────────────────────────


class TestEmptySnippets:
    """Test SnippetMatcher with no configured snippets."""

    def test_empty_dict_returns_none(self):
        m = SnippetMatcher({})
        assert m.match("anything") is None

    def test_empty_dict_triggers_empty(self):
        m = SnippetMatcher({})
        assert m.triggers == []


# ── Triggers property ──────────────────────────────────────────────────────


class TestTriggersProperty:
    """Test the triggers property."""

    def test_returns_original_casing(self, matcher):
        triggers = matcher.triggers
        assert "my email" in triggers
        assert "meeting followup" in triggers
        assert "home address" in triggers

    def test_count(self, matcher):
        assert len(matcher.triggers) == 3


# ── Threshold boundary ─────────────────────────────────────────────────────


class TestThresholdBoundary:
    """Test behavior near the matching threshold."""

    def test_custom_low_threshold_matches_more(self):
        m = SnippetMatcher({"hello": "world"}, threshold=0.5)
        # With a very low threshold, even moderately similar strings match
        result = m.match("helo")
        assert result == "world"

    def test_custom_high_threshold_matches_less(self):
        m = SnippetMatcher({"hello world": "hi"}, threshold=0.99)
        # Even close matches rejected at very high threshold
        assert m.match("hello worl") is None

    def test_exact_match_always_works(self):
        m = SnippetMatcher({"test": "result"}, threshold=1.0)
        # Exact match bypasses SequenceMatcher entirely
        assert m.match("test") == "result"
