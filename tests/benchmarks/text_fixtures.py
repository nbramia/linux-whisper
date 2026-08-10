"""Raw-transcript → expected-output pairs for scoring the polish pipeline.

These are the fixtures that judge stages 4a (disfluency), 4b (punctuation),
4d (formatting), and 4c (LLM self-correction).  Raw text stands in for STT
output, so the polish stages can be scored without running a model.

Tags drive fixture selection and per-category reporting:

``filler``
    Filled pauses and discourse markers stage 4a must delete.
``repetition`` / ``false-start``
    Stammers and abandoned phrases stage 4a must collapse.
``self-correction``
    Reparandum + repair — the **only** category that should wake stage 4c.
``formatting``
    Numbers, dates, times, and URLs handled by stage 4d.
``punctuation``
    Sentence boundaries and question marks stage 4b must insert.
``passthrough``
    Already-clean input.  These are the guard rails: any edit here is the
    pipeline paraphrasing, which ``vision.md`` treats as a correctness bug,
    not a style preference.
"""

from __future__ import annotations

from tests.benchmarks.fixtures import TextFixture

POLISH_FIXTURES: list[TextFixture] = [
    # ── Filler words and discourse markers ────────────────────────────────
    TextFixture(
        id="filler-basic",
        raw="um so i was thinking we should probably move the meeting",
        expected="So I was thinking we should probably move the meeting.",
        tags=("filler", "punctuation"),
    ),
    TextFixture(
        id="filler-discourse-markers",
        raw="you know i mean the deploy basically just needs another review",
        expected="The deploy just needs another review.",
        tags=("filler",),
    ),
    TextFixture(
        id="filler-trailing",
        raw="lets ship it uh yeah lets ship it",
        expected="Let's ship it.",
        tags=("filler", "repetition"),
    ),
    # ── Repetitions and false starts ──────────────────────────────────────
    TextFixture(
        id="repetition-stammer",
        raw="i i i think the the cache is stale",
        expected="I think the cache is stale.",
        tags=("repetition",),
    ),
    TextFixture(
        id="false-start-abandoned",
        raw="we should go to the lets stay here instead",
        expected="Let's stay here instead.",
        tags=("false-start",),
    ),
    TextFixture(
        id="repetition-phrase",
        raw="can you can you send me the link when youre done",
        expected="Can you send me the link when you're done?",
        tags=("repetition", "punctuation"),
    ),
    # ── Self-corrections — these should wake stage 4c ─────────────────────
    TextFixture(
        id="self-correction-time",
        # Same reason as self-correction-quantity: bare "four" is left alone.
        raw="lets meet at two actually make it four",
        expected="Let's meet at four.",
        tags=("self-correction",),
    ),
    TextFixture(
        id="self-correction-name",
        raw="send it to sarah no sorry send it to rachel",
        expected="Send it to Rachel.",
        tags=("self-correction",),
    ),
    TextFixture(
        id="self-correction-quantity",
        # "fifteen" stays a word on purpose.  Single number words are not
        # converted — see _format_cardinal_numbers — because "one of the things"
        # would become "1 of the things".  The self-correction is what is being
        # scored here, not the digit form.
        raw="we need about fifty units i mean fifteen units",
        expected="We need about fifteen units.",
        tags=("self-correction",),
    ),
    TextFixture(
        id="self-correction-day",
        raw="the release is on tuesday sorry wednesday",
        expected="The release is on Wednesday.",
        tags=("self-correction",),
    ),
    TextFixture(
        id="self-correction-negation",
        raw="deploy to staging no wait deploy to production",
        expected="Deploy to production.",
        tags=("self-correction",),
    ),
    # ── Number, date, and URL formatting ──────────────────────────────────
    TextFixture(
        id="formatting-numbers",
        raw="the build took twenty three minutes and cost forty dollars",
        expected="The build took 23 minutes and cost $40.",
        tags=("formatting",),
    ),
    TextFixture(
        id="formatting-date",
        raw="lets ship on march fifteenth twenty twenty six",
        expected="Let's ship on March 15, 2026.",
        tags=("formatting",),
    ),
    TextFixture(
        id="formatting-time",
        raw="the standup is at nine thirty a m",
        expected="The standup is at 9:30 AM.",
        tags=("formatting",),
    ),
    TextFixture(
        id="formatting-percent",
        raw="latency dropped by forty percent after the change",
        expected="Latency dropped by 40% after the change.",
        tags=("formatting",),
    ),
    # ── Punctuation and capitalisation ────────────────────────────────────
    TextFixture(
        id="punctuation-question",
        raw="did you get a chance to look at the pull request",
        expected="Did you get a chance to look at the pull request?",
        tags=("punctuation",),
    ),
    TextFixture(
        id="punctuation-multi-sentence",
        raw="the tests are green i pushed the fix can you review it",
        expected="The tests are green. I pushed the fix. Can you review it?",
        tags=("punctuation",),
    ),
    TextFixture(
        id="punctuation-proper-nouns",
        raw="i talked to nathan about the linux whisper project on friday",
        expected="I talked to Nathan about the Linux Whisper project on Friday.",
        tags=("punctuation",),
    ),
    # ── Passthrough guards — any edit here is a paraphrasing bug ──────────
    TextFixture(
        id="passthrough-clean-statement",
        raw="The deployment finished successfully.",
        expected="The deployment finished successfully.",
        tags=("passthrough",),
    ),
    TextFixture(
        id="passthrough-technical",
        raw="Run kubectl get pods in the staging namespace.",
        expected="Run kubectl get pods in the staging namespace.",
        tags=("passthrough",),
    ),
    TextFixture(
        id="passthrough-already-punctuated",
        raw="Can you review the PR? I pushed a fix for the cache bug.",
        expected="Can you review the PR? I pushed a fix for the cache bug.",
        tags=("passthrough",),
    ),
    TextFixture(
        id="passthrough-no-invented-content",
        raw="Ship it.",
        expected="Ship it.",
        tags=("passthrough",),
    ),
    # ── Code and filenames must survive the pipeline untouched ────────────
    # These are passthrough guards with teeth.  A grammar-fixing LLM has every
    # incentive to "correct" a filename into prose, expand an acronym, or add a
    # space after a hyphen — each of which silently breaks dictated code.
    TextFixture(
        id="passthrough-filename",
        raw="server-test.sh",
        expected="server-test.sh",
        tags=("passthrough", "filename"),
    ),
    TextFixture(
        id="passthrough-path",
        raw="Open src/linux_whisper/stt/parakeet.py",
        expected="Open src/linux_whisper/stt/parakeet.py",
        tags=("passthrough", "filename"),
    ),
    TextFixture(
        id="passthrough-dotfile",
        raw="Check the .env file in the project root.",
        expected="Check the .env file in the project root.",
        tags=("passthrough", "filename"),
    ),
    TextFixture(
        id="passthrough-snake-case",
        raw="The variable is called max_default_threads.",
        expected="The variable is called max_default_threads.",
        tags=("passthrough", "code"),
    ),
    TextFixture(
        id="passthrough-camel-case",
        raw="Call getUserById with the account ID.",
        expected="Call getUserById with the account ID.",
        tags=("passthrough", "code"),
    ),
    TextFixture(
        id="passthrough-flags",
        raw="Run it with --no-cache and --verbose.",
        expected="Run it with --no-cache and --verbose.",
        tags=("passthrough", "code"),
    ),
    TextFixture(
        id="passthrough-sql",
        raw="Write a SQL query against the users table.",
        expected="Write a SQL query against the users table.",
        tags=("passthrough", "code"),
    ),
    TextFixture(
        id="passthrough-acronyms",
        raw="The API returns JSON, but the config is YAML.",
        expected="The API returns JSON, but the config is YAML.",
        tags=("passthrough", "code"),
    ),
    TextFixture(
        id="passthrough-git-command",
        raw="Run git rebase --interactive on main.",
        expected="Run git rebase --interactive on main.",
        tags=("passthrough", "code"),
    ),
    TextFixture(
        id="passthrough-version",
        raw="Upgrade to version 0.3.34 and rebuild.",
        expected="Upgrade to version 0.3.34 and rebuild.",
        tags=("passthrough", "code"),
    ),
]


def load_text_fixtures(tags: tuple[str, ...] | None = None) -> list[TextFixture]:
    """Return polish fixtures, optionally filtered to those carrying any of *tags*."""
    if not tags:
        return list(POLISH_FIXTURES)
    wanted = set(tags)
    return [f for f in POLISH_FIXTURES if wanted & set(f.tags)]
