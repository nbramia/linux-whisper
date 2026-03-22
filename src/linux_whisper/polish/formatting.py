"""Stage 4d: Spoken-form-to-written-form formatting via regex/dict rules.

Converts spoken numbers, dates, times, currency, email addresses, and phone
numbers into their conventional written forms.  Runs after punctuation (4b)
and before the LLM correction stage (4c).

This stage is entirely rule-based — no model loading required.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Word-to-number mappings
# ---------------------------------------------------------------------------

_ONES: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}

_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_SCALES: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

# All number words for boundary detection.
_NUMBER_WORDS: set[str] = {*_ONES, *_TENS, *_SCALES, "and", "a"}

_ORDINAL_WORD_MAP: dict[str, tuple[int, str]] = {
    "first": (1, "1st"), "second": (2, "2nd"), "third": (3, "3rd"),
    "fourth": (4, "4th"), "fifth": (5, "5th"), "sixth": (6, "6th"),
    "seventh": (7, "7th"), "eighth": (8, "8th"), "ninth": (9, "9th"),
    "tenth": (10, "10th"), "eleventh": (11, "11th"), "twelfth": (12, "12th"),
    "thirteenth": (13, "13th"), "fourteenth": (14, "14th"),
    "fifteenth": (15, "15th"), "sixteenth": (16, "16th"),
    "seventeenth": (17, "17th"), "eighteenth": (18, "18th"),
    "nineteenth": (19, "19th"), "twentieth": (20, "20th"),
    "thirtieth": (30, "30th"),
}

# Compound ordinals: "twenty first" -> "21st"
_ORDINAL_ONES: dict[str, tuple[int, str]] = {
    "first": (1, "st"), "second": (2, "nd"), "third": (3, "rd"),
    "fourth": (4, "th"), "fifth": (5, "th"), "sixth": (6, "th"),
    "seventh": (7, "th"), "eighth": (8, "th"), "ninth": (9, "th"),
}

_MONTHS: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_MONTH_NAMES: set[str] = set(_MONTHS)

# ---------------------------------------------------------------------------
# Number word parser
# ---------------------------------------------------------------------------


def _words_to_number(words: list[str]) -> int | None:
    """Convert a sequence of number words to an integer.

    Returns None if the words do not form a valid number.
    Handles patterns like: "three hundred and fifty", "one thousand two hundred",
    "twenty five", "a hundred".
    """
    if not words:
        return None

    # Filter out "and" which is just a connector
    tokens = [w for w in words if w != "and"]
    if not tokens:
        return None

    result = 0
    current = 0

    for token in tokens:
        if token == "a":
            current = 1
        elif token in _ONES:
            current += _ONES[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token == "hundred":
            current = (current if current else 1) * 100
        elif token in ("thousand", "million", "billion"):
            current = (current if current else 1) * _SCALES[token]
            result += current
            current = 0
        else:
            return None

    return result + current


def _ordinal_suffix(n: int) -> str:
    """Return the ordinal suffix for a number (e.g. 1 -> '1st')."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    remainder = n % 10
    if remainder == 1:
        return f"{n}st"
    if remainder == 2:
        return f"{n}nd"
    if remainder == 3:
        return f"{n}rd"
    return f"{n}th"


# ---------------------------------------------------------------------------
# Sub-formatters (each takes and returns a string)
# ---------------------------------------------------------------------------


def _format_emails(text: str) -> str:
    """Convert 'user at domain dot tld' to 'user@domain.tld'."""
    pattern = re.compile(
        r"\b(\w+)\s+at\s+(\w+)\s+dot\s+(com|org|net|edu|io|co|gov|info)\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: f"{m[1]}@{m[2]}.{m[3].lower()}", text)


def _format_phone_numbers(text: str) -> str:
    """Convert sequences of 10 spoken digits to phone number format.

    'one two three four five six seven eight nine zero' -> '123-456-7890'
    Also handles 7-digit sequences: 'five five five one two three four' -> '555-1234'
    """
    digit_words = (
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
    )
    digit_map = {w: str(i) for i, w in enumerate(digit_words)}

    words = text.split()
    i = 0
    result_parts: list[str] = []

    while i < len(words):
        # Try to match a run of digit words
        run_start = i
        digits: list[str] = []
        while i < len(words) and words[i].lower().rstrip(".,?!;:") in digit_map:
            bare = words[i].rstrip(".,?!;:")
            trailing = words[i][len(bare):]
            digits.append(digit_map[bare.lower()])
            i += 1

        if len(digits) == 10:
            phone = f"{''.join(digits[:3])}-{''.join(digits[3:6])}-{''.join(digits[6:])}"
            # Preserve trailing punctuation from last digit word
            last_word = words[i - 1]
            bare = last_word.rstrip(".,?!;:")
            trailing = last_word[len(bare):]
            result_parts.append(phone + trailing)
        elif len(digits) == 7:
            phone = f"{''.join(digits[:3])}-{''.join(digits[3:])}"
            last_word = words[i - 1]
            bare = last_word.rstrip(".,?!;:")
            trailing = last_word[len(bare):]
            result_parts.append(phone + trailing)
        else:
            # Not a phone number — put the words back
            for j in range(run_start, i):
                result_parts.append(words[j])
            if run_start == i:
                result_parts.append(words[i])
                i += 1

    return " ".join(result_parts)


def _format_times(text: str) -> str:
    """Convert spoken times to written form.

    'four thirty PM' -> '4:30 PM'
    'twelve fifteen' -> '12:15'
    """
    # Hour words that can appear in time expressions
    hour_words = {**_ONES, **_TENS}

    # Pattern: <hour-word> <minute-word(s)> [AM|PM]
    # Minute words: "fifteen", "thirty", "forty five", etc.
    # We match: <number-word> <number-word(s)> [AM/PM]
    # where the first number is 1-12 (hour) and the second is 0-59 (minutes).

    words = text.split()
    result: list[str] = []
    i = 0

    while i < len(words):
        matched = False

        if i < len(words):
            bare = words[i].lower().rstrip(".,?!;:")

            # Check if this word is a valid hour (1-12)
            hour_val = hour_words.get(bare)
            if hour_val is not None and 1 <= hour_val <= 12:
                # Look ahead for minute words
                j = i + 1
                minute_words_found: list[str] = []
                while j < len(words):
                    next_bare = words[j].lower().rstrip(".,?!;:")
                    if next_bare in hour_words and hour_words[next_bare] <= 59:
                        minute_words_found.append(next_bare)
                        j += 1
                    else:
                        break

                if minute_words_found:
                    minute_val = _words_to_number(minute_words_found)
                    if minute_val is not None and 0 <= minute_val <= 59:
                        # Check for AM/PM
                        am_pm = ""
                        trailing = ""
                        if j < len(words):
                            ampm_bare = words[j].upper().rstrip(".,?!;:")
                            ampm_trailing = words[j][len(words[j].rstrip(".,?!;:")):]
                            if ampm_bare in ("AM", "PM"):
                                am_pm = f" {ampm_bare}"
                                trailing = ampm_trailing
                                j += 1

                        if not trailing:
                            # Get trailing punct from last consumed word
                            last = words[j - 1]
                            bare_last = last.rstrip(".,?!;:")
                            trailing = last[len(bare_last):]

                        time_str = f"{hour_val}:{minute_val:02d}{am_pm}{trailing}"
                        result.append(time_str)
                        i = j
                        matched = True

        if not matched:
            result.append(words[i])
            i += 1

    return " ".join(result)


def _format_dates(text: str) -> str:
    """Convert 'month ordinal' patterns to 'Month Nth' form.

    'march twenty second' -> 'March 22nd'
    'january first' -> 'January 1st'
    """
    words = text.split()
    result: list[str] = []
    i = 0

    while i < len(words):
        bare = words[i].lower().rstrip(".,?!;:")

        if bare in _MONTH_NAMES:
            # Look ahead for ordinal day
            month_cap = bare.capitalize()
            j = i + 1

            # Try compound ordinal: "twenty second", "thirty first"
            if j + 1 < len(words):
                tens_bare = words[j].lower().rstrip(".,?!;:")
                ones_bare = words[j + 1].lower().rstrip(".,?!;:")
                trailing = words[j + 1][len(words[j + 1].rstrip(".,?!;:")):]

                if tens_bare in _TENS and ones_bare in _ORDINAL_ONES:
                    day = _TENS[tens_bare] + _ORDINAL_ONES[ones_bare][0]
                    result.append(f"{month_cap} {_ordinal_suffix(day)}{trailing}")
                    i = j + 2
                    continue

            # Try simple ordinal: "first", "tenth", "twentieth"
            if j < len(words):
                ord_bare = words[j].lower().rstrip(".,?!;:")
                trailing = words[j][len(words[j].rstrip(".,?!;:")):]
                if ord_bare in _ORDINAL_WORD_MAP:
                    _, day_str = _ORDINAL_WORD_MAP[ord_bare]
                    result.append(f"{month_cap} {day_str}{trailing}")
                    i = j + 1
                    continue

            result.append(words[i])
            i += 1
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)


def _format_currency(text: str) -> str:
    """Convert spoken currency to symbols.

    'eight hundred dollars' -> '$800'
    'fifty cents' -> '$0.50'
    'twenty five dollars and fifty cents' -> '$25.50'
    """
    words = text.split()
    result: list[str] = []
    i = 0

    while i < len(words):
        # Try to find a number phrase followed by "dollars" [and <n> "cents"]
        j = i
        number_words: list[str] = []

        while j < len(words):
            bare = words[j].lower().rstrip(".,?!;:")
            if bare in _NUMBER_WORDS:
                number_words.append(bare)
                j += 1
            else:
                break

        if number_words and j < len(words):
            bare_next = words[j].lower().rstrip(".,?!;:")
            trailing_next = words[j][len(words[j].rstrip(".,?!;:")):]

            if bare_next == "dollars":
                dollar_amount = _words_to_number(number_words)
                if dollar_amount is not None:
                    # Check for "and X cents"
                    cents_amount = 0
                    k = j + 1
                    trailing_final = trailing_next

                    if k < len(words) and words[k].lower().rstrip(".,?!;:") == "and":
                        # Look for cents
                        m = k + 1
                        cents_words: list[str] = []
                        while m < len(words):
                            cbare = words[m].lower().rstrip(".,?!;:")
                            if cbare in _NUMBER_WORDS:
                                cents_words.append(cbare)
                                m += 1
                            else:
                                break

                        if cents_words and m < len(words):
                            cbare = words[m].lower().rstrip(".,?!;:")
                            if cbare == "cents":
                                parsed_cents = _words_to_number(cents_words)
                                if parsed_cents is not None:
                                    cents_amount = parsed_cents
                                    trailing_final = words[m][len(words[m].rstrip(".,?!;:")):]
                                    k = m + 1

                    if cents_amount > 0:
                        result.append(f"${dollar_amount}.{cents_amount:02d}{trailing_final}")
                    else:
                        result.append(f"${dollar_amount}{trailing_final}")
                    i = k
                    continue

            elif bare_next == "cents":
                cents_val = _words_to_number(number_words)
                if cents_val is not None:
                    result.append(f"$0.{cents_val:02d}{trailing_next}")
                    i = j + 1
                    continue

        # No currency match — emit current word and advance
        if i == j:
            result.append(words[i])
            i += 1
        else:
            # Number words were consumed but no "dollars"/"cents" — put them back
            for idx in range(i, j):
                result.append(words[idx])
            i = j

    return " ".join(result)


def _format_cardinal_numbers(text: str) -> str:
    """Convert multi-word spoken numbers to digits.

    Only converts sequences of 2+ number words to avoid changing natural prose
    like 'one of the things'.

    'three hundred and fifty' -> '350'
    'twenty five' -> '25'
    """
    words = text.split()
    result: list[str] = []
    i = 0

    while i < len(words):
        bare = words[i].lower().rstrip(".,?!;:")

        # "a" can start a number phrase only if followed by a scale word
        is_number_start = bare in _NUMBER_WORDS and bare != "and"
        if bare == "a":
            next_bare = words[i + 1].lower().rstrip(".,?!;:") if i + 1 < len(words) else ""
            is_number_start = next_bare in _SCALES

        if is_number_start:
            # Start collecting number words
            j = i
            number_words: list[str] = []

            while j < len(words):
                nbare = words[j].lower().rstrip(".,?!;:")
                if nbare in _NUMBER_WORDS:
                    number_words.append(nbare)
                    j += 1
                else:
                    break

            # Only convert if it looks like a real number phrase:
            # either 2+ meaningful words, or contains a scale word ("a hundred")
            meaningful = [w for w in number_words if w not in ("and", "a")]
            has_scale = any(w in _SCALES for w in meaningful)
            if len(meaningful) >= 2 or (has_scale and len(number_words) >= 2):
                parsed = _words_to_number(number_words)
                if parsed is not None:
                    trailing = words[j - 1][len(words[j - 1].rstrip(".,?!;:")):]
                    result.append(f"{parsed}{trailing}")
                    i = j
                    continue

            # Not enough words to convert — emit them as-is
            for idx in range(i, j):
                result.append(words[idx])
            i = j
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SpokenFormFormatter:
    """Convert spoken forms of numbers, dates, times, etc. to written form.

    This stage is entirely rule-based and requires no model loading.
    It processes text through a chain of sub-formatters:

    1. Email addresses
    2. Phone numbers
    3. Times
    4. Dates
    5. Currency
    6. Cardinal numbers
    """

    def process(self, text: str) -> str:
        """Format spoken forms in *text* and return the result."""
        if not text or not text.strip():
            return text

        current = text
        current = _format_emails(current)
        current = _format_phone_numbers(current)
        current = _format_times(current)
        current = _format_dates(current)
        current = _format_currency(current)
        current = _format_cardinal_numbers(current)
        return current
