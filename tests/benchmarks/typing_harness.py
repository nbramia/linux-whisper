"""Guarded typing harness: measure text injection into a window this script owns.

``ydotool`` types into whatever window has focus, so it must never be pointed at
the desktop during development.  This harness opens its own window (a GTK text
view, a VTE terminal, or a headed Chromium page), waits until that window is
active, types through ``ydotool`` or the repo's own ``YdotoolInjector``, and
then reads the text back and diffs it against what was sent.  It aborts
without typing if its window is not the active one, and it types a single
space first and aborts unless that space arrives in its own window, so a wrong
focus judgement costs one space rather than a paragraph.

Not part of ``pytest tests/``: it needs a Wayland session, ``ydotool`` and, for
``chromium``, Playwright with a Chromium build.

Examples::

    python -m tests.benchmarks.typing_harness gtk --key-delay 0 --chars 350
    python -m tests.benchmarks.typing_harness vte --start-delay 25 --chars 60
    python -m tests.benchmarks.typing_harness chromium --injector --chars 1200
    python -m tests.benchmarks.typing_harness gtk --successive 3

``--successive N`` types N fixture sentences one injection at a time, putting
the app's leading space in front of each as ``App._inject_text`` would with no
key or click in between, and expects them joined by single spaces.

The text comes from the ``.txt`` references in ``--fixtures-dir`` (default
``~/dictation-fixtures``), joined and truncated to ``--chars``.  Output is one
``RESULT`` line: mode, delays, character count, wall time, and ``ok`` or the
number of lost and extra characters.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

RESULT = (
    "RESULT mode={mode} start_delay={start} key_delay={key} chars={chars} ms={ms:.0f} {verdict}"
)


def _fixture_text(fixtures_dir: Path, chars: int) -> str:
    parts = [p.read_text().strip() for p in sorted(fixtures_dir.glob("*.txt"))]
    if not parts:
        sys.exit(f"no .txt fixtures in {fixtures_dir}")
    return " ".join(parts).replace("\n", " ")[:chars]


def _fixture_sentences(fixtures_dir: Path, count: int, chars: int) -> list[str]:
    """The first *count* fixtures, one sentence each, each cut to *chars*."""
    parts = [
        " ".join(p.read_text().split())[:chars].strip()
        for p in sorted(fixtures_dir.glob("*.txt"))
    ]
    parts = [p for p in parts if p]
    if len(parts) < count:
        sys.exit(f"need {count} .txt fixtures in {fixtures_dir}, found {len(parts)}")
    return parts[:count]


def _compare(sent: str, got: str) -> str:
    if got == sent:
        return "ok"
    matcher = difflib.SequenceMatcher(None, sent, got, autojunk=False)
    opcodes = matcher.get_opcodes()
    lost = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in ("delete", "replace"))
    extra = sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in ("insert", "replace"))
    return f"MISMATCH lost={lost} extra={extra} got={got[:80]!r}"


PROBE = " "  # typed first; the real text goes only if this character arrives in our window


def _probe_arrived(read_back: Callable[[], str]) -> bool:
    return read_back().endswith(PROBE)


def _type_with_ydotool(text: str, start_delay: int, key_delay: int) -> subprocess.Popen:
    cmd = ["ydotool", "type", "--delay", str(start_delay), "--key-delay", str(key_delay)]
    return subprocess.Popen([*cmd, "--", text])


def _type_with_injector(text: str, key_delay: int) -> None:
    """Drive the repo's own ydotool injector, in a thread with its own event loop."""
    from linux_whisper.config import InjectConfig
    from linux_whisper.inject.injector import YdotoolInjector

    injector = YdotoolInjector(InjectConfig(typing_delay=key_delay))
    outcome: dict[str, bool] = {}

    def run() -> None:
        outcome["ok"] = asyncio.run(injector.inject(text))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if not outcome.get("ok"):
        print("injector reported failure", file=sys.stderr)


def _type_successive(sentences: list[str], key_delay: int) -> None:
    """Type each sentence as its own injection, spaced the way the app spaces them."""
    from linux_whisper.inject.injector import leading_space

    previous_end: str | None = None
    for sentence in sentences:
        _type_with_injector(leading_space(sentence, previous_end, False) + sentence, key_delay)
        previous_end = sentence[-1]


def _type_text(args: argparse.Namespace, text: str) -> subprocess.Popen | None:
    """Start typing *text* the way *args* asks; a ``Popen`` if it runs on, else ``None``."""
    if args.successive:
        _type_successive(args.sentences, args.key_delay)
        return None
    if args.injector:
        _type_with_injector(text, args.key_delay)
        return None
    return _type_with_ydotool(text, args.start_delay, args.key_delay)


def _run_gtk(args: argparse.Namespace, text: str) -> None:
    import gi

    gi.require_version("Gtk", "3.0")
    from gi.repository import GLib, Gtk

    window = Gtk.Window(title="linux-whisper typing harness (do not touch)")
    window.set_default_size(900, 300)
    window.set_keep_above(True)

    read_back: Callable[[], str]
    if args.mode == "gtk":
        view = Gtk.TextView()
        view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        window.add(view)

        def read_back() -> str:
            buf = view.get_buffer()
            return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), True)

    else:
        gi.require_version("Vte", "2.91")
        from gi.repository import Vte

        view = Vte.Terminal()
        view.spawn_async(
            Vte.PtyFlags.DEFAULT, None, ["/bin/cat"], None, GLib.SpawnFlags.DEFAULT,
            None, None, -1, None, None,
        )
        window.add(view)

        def read_back() -> str:
            # ``cat`` echoes what was typed; join the wrapped rows back together.
            # Only newlines are stripped: the probe is a trailing space.
            try:
                txt = view.get_text_format(Vte.Format.TEXT)
            except AttributeError:
                txt = view.get_text(None)[0]
            return "".join((txt or "").rstrip("\n").split("\n"))

    def pump(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            while Gtk.events_pending():
                Gtk.main_iteration_do(False)
            time.sleep(0.005)

    def active() -> bool:
        return bool(window.is_active() and view.has_focus())

    def go() -> bool:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not active():
            pump(0.05)
        if not active():
            print(_result(args, text, 0, "ABORT window-not-active"))
            Gtk.main_quit()
            return False
        pump(0.2)
        if not active():
            print(_result(args, text, 0, "ABORT focus-lost"))
            Gtk.main_quit()
            return False
        probe = _type_with_ydotool(PROBE, args.start_delay, 0)
        while probe.poll() is None:
            pump(0.002)
        pump(0.3)
        if not _probe_arrived(read_back):
            print(_result(args, text, 0, "ABORT probe-did-not-arrive"))
            Gtk.main_quit()
            return False
        t0 = time.perf_counter()
        proc = _type_text(args, text)
        while proc is not None and proc.poll() is None:
            pump(0.002)  # keep dispatching Wayland events while typing
        ms = (time.perf_counter() - t0) * 1000
        pump(0.6)
        print(_result(args, text, ms, _compare(text, read_back().removeprefix(PROBE))))
        Gtk.main_quit()
        return False

    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    view.grab_focus()
    GLib.timeout_add(300, go)
    Gtk.main()


def _run_chromium(args: argparse.Namespace, text: str) -> None:
    from playwright.sync_api import sync_playwright

    html = (
        "<html><body style='margin:0'><textarea id=t autofocus "
        "style='width:98vw;height:90vh;font-size:16px'></textarea></body></html>"
    )
    focused = "document.hasFocus() && document.activeElement.id === 't'"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            # A native Wayland window: under XWayland the page reports focus it
            # does not have and the probe below aborts the run.
            args=["--window-size=900,400", "--ozone-platform=wayland"],
        )
        page = browser.new_page()
        page.set_content(html)
        page.focus("#t")
        page.bring_to_front()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not page.evaluate(focused):
            time.sleep(0.05)
        if not page.evaluate(focused):
            print(_result(args, text, 0, "ABORT window-not-active"))
            browser.close()
            return
        time.sleep(0.2)
        if not page.evaluate("document.hasFocus()"):
            print(_result(args, text, 0, "ABORT focus-lost"))
            browser.close()
            return
        _type_with_ydotool(PROBE, args.start_delay, 0).wait()
        time.sleep(0.3)
        if not _probe_arrived(lambda: page.evaluate("document.getElementById('t').value")):
            print(_result(args, text, 0, "ABORT probe-did-not-arrive"))
            browser.close()
            return
        t0 = time.perf_counter()
        proc = _type_text(args, text)
        if proc is not None:
            proc.wait()
        ms = (time.perf_counter() - t0) * 1000
        time.sleep(0.6)
        got = page.evaluate("document.getElementById('t').value").removeprefix(PROBE)
        print(_result(args, text, ms, _compare(text, got)))
        browser.close()


def _result(args: argparse.Namespace, text: str, ms: float, verdict: str) -> str:
    start = "injector" if args.injector or args.successive else args.start_delay
    mode = f"{args.mode}-successive{args.successive}" if args.successive else args.mode
    return RESULT.format(
        mode=mode, start=start, key=args.key_delay, chars=len(text), ms=ms, verdict=verdict
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("gtk", "vte", "chromium"))
    parser.add_argument("--chars", type=int, default=350)
    parser.add_argument(
        "--key-delay", type=int, default=0, help="ydotool --key-delay (ms per char)"
    )
    parser.add_argument(
        "--start-delay", type=int, default=60,
        help="ydotool --delay before the first key (daemon-less 0.1.8 needs >= 50)",
    )
    parser.add_argument(
        "--injector", action="store_true",
        help="type through linux_whisper's YdotoolInjector instead of calling ydotool directly",
    )
    parser.add_argument(
        "--successive", type=int, default=0, metavar="N",
        help="type N fixture sentences as separate injections with the app's spacing rule "
        "(--chars then limits each sentence)",
    )
    parser.add_argument("--fixtures-dir", type=Path, default=Path.home() / "dictation-fixtures")
    args = parser.parse_args(argv)

    if args.successive:
        args.sentences = _fixture_sentences(args.fixtures_dir, args.successive, args.chars)
        text = " ".join(args.sentences)
    else:
        text = _fixture_text(args.fixtures_dir, args.chars)
    if "\n" in text:
        sys.exit("refusing to type a newline")
    if args.mode == "chromium":
        _run_chromium(args, text)
    else:
        _run_gtk(args, text)


if __name__ == "__main__":
    main()
