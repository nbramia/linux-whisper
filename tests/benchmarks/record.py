"""Record your own dictation fixtures for the benchmark harness.

LibriSpeech is read audiobook prose: no fillers, no stammers, no self-corrections,
studio microphones, and utterances far longer than a typical dictated sentence.
It ranks models sensibly but it cannot tell you how a backend handles *your*
voice saying the things you actually dictate — which is the only question that
matters when picking a default.

This records that set. Each prompt is spoken once and saved as a WAV plus a
``.txt`` reference, in the layout ``--fixtures-dir`` expects::

    python -m tests.benchmarks.record --out ~/dictation-fixtures
    python -m tests.benchmarks.run --suite stt --fixtures-dir ~/dictation-fixtures

**Say the prompt naturally — do not perform it.** If you would normally say "um"
or restart a sentence, do that. The disfluency prompts exist precisely to catch
backends that transcribe fillers verbatim or delete real words along with them.

The reference transcript saved is the *cleaned* text — what the full pipeline
should ultimately produce, not a literal transcription of what you said. That
makes the file usable for end-to-end scoring; for raw STT WER, edit the ``.txt``
to match your actual words.
"""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from pathlib import Path

logger = logging.getLogger("record")

SAMPLE_RATE = 16_000

# Prompts chosen to cover what LibriSpeech cannot: dictation-length utterances,
# disfluencies, self-corrections, technical vocabulary, numbers, and questions.
PROMPTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("clean-short", "The deployment finished successfully.", ("clean",)),
    (
        "clean-medium",
        "Can you review the pull request when you get a chance?",
        ("clean", "question"),
    ),
    (
        "clean-long",
        "I spent most of the morning tracking down a race condition in the audio "
        "callback and it turned out to be a buffer size mismatch.",
        ("clean", "long"),
    ),
    ("filler-um", "So I was thinking we should probably move the meeting.", ("filler",)),
    ("filler-discourse", "The deploy just needs another review.", ("filler",)),
    ("repetition", "I think the cache is stale.", ("repetition",)),
    ("self-correction-time", "Let's meet at four.", ("self-correction",)),
    ("self-correction-name", "Send it to Rachel.", ("self-correction",)),
    ("self-correction-place", "Deploy to production.", ("self-correction",)),
    ("numbers", "The build took 23 minutes and cost $40.", ("formatting",)),
    ("date-time", "The standup is at 9:30 AM on March 15th.", ("formatting",)),
    ("technical", "Run kubectl get pods in the staging namespace.", ("technical",)),
    ("technical-2", "The ONNX runtime session needs an explicit thread count.", ("technical",)),
    ("proper-nouns", "I talked to Nathan about the Linux Whisper project on Friday.", ("names",)),
    ("question", "Did the ROCm build finish, or is it still compiling?", ("question",)),
]

# Spoken hints for the prompts whose point is the disfluency, not the words.
DELIVERY_HINTS: dict[str, str] = {
    "filler-um": 'start with "um" and a false start, e.g. "um so I was— I was thinking..."',
    "filler-discourse": 'work in "you know" or "I mean" somewhere',
    "repetition": 'stammer the opening, e.g. "I I I think..."',
    "self-correction-time": 'say a wrong time first: "let\'s meet at two, actually four"',
    "self-correction-name": 'say a wrong name first: "send it to Sarah, no sorry, Rachel"',
    "self-correction-place": 'say the wrong target first: "deploy to staging, no wait, production"',
}


def _require_sounddevice():  # noqa: ANN202
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise ImportError(
            "The 'sounddevice' package is required to record fixtures.  "
            "Install it with:\n    pip install sounddevice\n"
        ) from exc
    return sd


def write_wav(path: Path, audio, sample_rate: int = SAMPLE_RATE) -> None:
    """Write float32 audio in [-1, 1] as a 16-bit mono WAV."""
    import numpy as np

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def record_one(sd, max_seconds: float) -> object:
    """Record until the user presses Enter, or *max_seconds* elapses.

    Callback-driven rather than read-driven: a blocking ``input()`` alongside
    ``stream.read()`` would let the device buffer overflow and drop audio, which
    is exactly the artefact a fixture must not contain.
    """
    import numpy as np

    frames: list = []
    max_frames = int(SAMPLE_RATE * max_seconds)
    captured = 0

    def callback(indata, _frames, _time, status) -> None:
        nonlocal captured
        if status:
            logger.debug("input status: %s", status)
        if captured < max_frames:
            frames.append(indata.copy())
            captured += len(indata)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    ):
        input()

    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames).flatten()[:max_frames]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.benchmarks.record",
        description="Record your own dictation fixtures for the benchmark harness.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory to write fixtures into")
    parser.add_argument(
        "--seconds",
        type=float,
        default=15.0,
        help="Maximum length of a single take (default: 15)",
    )
    parser.add_argument("--only", help="Record just this prompt id (re-take a bad one)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sd = _require_sounddevice()

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = [p for p in PROMPTS if not args.only or p[0] == args.only]
    if not prompts:
        print(f"No prompt with id '{args.only}'. Known ids:", file=sys.stderr)
        for pid, _, _ in PROMPTS:
            print(f"  {pid}", file=sys.stderr)
        return 2

    print(f"\nRecording {len(prompts)} fixture(s) into {args.out}")
    print("For each: press Enter to start, speak, press Enter again to stop.")
    print("Say it the way you would actually dictate it — fillers and all.\n")

    for index, (fixture_id, reference, tags) in enumerate(prompts, start=1):
        wav_path = args.out / f"{fixture_id}.wav"
        print(f"[{index}/{len(prompts)}] {fixture_id}  ({', '.join(tags)})")
        print(f'    say: "{reference}"')
        if hint := DELIVERY_HINTS.get(fixture_id):
            print(f"    hint: {hint}")
        print("    Enter to start... ", end="", flush=True)
        input()
        print("    recording — Enter to stop... ", end="", flush=True)

        audio = record_one(sd, args.seconds)
        duration = len(audio) / SAMPLE_RATE

        if duration < 0.3:
            print(f"    ⚠ only {duration:.1f}s captured — skipping, re-run with "
                  f"--only {fixture_id}")
            continue

        write_wav(wav_path, audio)
        (args.out / f"{fixture_id}.txt").write_text(reference + "\n", encoding="utf-8")
        print(f"    ✓ {duration:.1f}s → {wav_path.name}\n")

    print(f"Done. Score a backend against these with:\n"
          f"  python -m tests.benchmarks.run --suite stt --fixtures-dir {args.out}\n")
    print("Note: the .txt references are the *cleaned* target text. For raw STT "
          "WER, edit them to match the words you actually said.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
