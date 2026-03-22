"""Voice snippets — trigger phrases that expand to saved text.

Matches full STT transcriptions against user-configured trigger phrases
using fuzzy matching.  When a match is found, the snippet text is returned
directly, bypassing the polish pipeline entirely.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.85


class SnippetMatcher:
    """Matches full STT transcriptions against configured snippet triggers.

    Parameters
    ----------
    snippets:
        Mapping of trigger phrases to replacement text.
    threshold:
        Minimum ``SequenceMatcher.ratio()`` for a fuzzy match (0.0–1.0).
    """

    def __init__(
        self, snippets: dict[str, str], threshold: float = _DEFAULT_THRESHOLD
    ) -> None:
        self._threshold = threshold

        # Store original trigger→text mapping and a normalized lookup.
        self._snippets = dict(snippets)
        self._normalized: dict[str, str] = {}
        for trigger, text in snippets.items():
            self._normalized[self._normalize(trigger)] = text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, transcription: str) -> str | None:
        """Return snippet text if *transcription* matches a trigger, else ``None``.

        Matching is case-insensitive, strips whitespace, and tolerates
        minor STT variations via fuzzy matching.  Only full-transcription
        matches are considered — no partial/substring matching.
        """
        if not self._normalized:
            return None

        norm = self._normalize(transcription)
        if not norm:
            return None

        # Fast path: exact match after normalization
        if norm in self._normalized:
            return self._normalized[norm]

        # Slow path: fuzzy match against all triggers
        best_ratio = 0.0
        best_text: str | None = None

        for trigger, text in self._normalized.items():
            ratio = SequenceMatcher(None, norm, trigger).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_text = text

        if best_ratio >= self._threshold:
            return best_text

        return None

    @property
    def triggers(self) -> list[str]:
        """Return the list of configured trigger phrases (original casing)."""
        return list(self._snippets.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip, and collapse whitespace."""
        return " ".join(text.lower().split())
