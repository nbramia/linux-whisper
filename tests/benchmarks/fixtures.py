"""Fixture sourcing for the model A/B harness.

Two independent fixture families:

* **Audio fixtures** — real recorded speech with reference transcripts, used to
  score STT backends and the VAD.  Sourced from LibriSpeech ``test-clean``
  (public domain, the corpus behind the Open ASR Leaderboard numbers) and
  downloaded into the cache on demand.  Nothing large is committed to git.
* **Text fixtures** — raw-transcript → expected-output pairs used to score the
  polish pipeline.  These need no audio at all, which makes stage 4a/4b/4c
  scoring deterministic and fast.

Users can point the harness at their own dictation clips with ``--fixtures-dir``;
that is the highest-signal option, since LibriSpeech is read audiobook speech and
under-represents the disfluencies real dictation contains.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from linux_whisper.config import CACHE_DIR

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

BENCH_DIR = CACHE_DIR / "benchmarks"
LIBRISPEECH_DIR = BENCH_DIR / "librispeech"

# LibriSpeech splits.  ``test-clean`` is studio-quality read prose; ``test-other``
# is the deliberately harder split — accents, faster delivery, worse recordings.
# Neither contains disfluencies or self-corrections, so both understate how a
# model handles real dictation.  Use ``--fixtures-dir`` with your own clips for
# that; see ``record.py``.
LIBRISPEECH_SPLITS: dict[str, str] = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}
DEFAULT_SPLIT = "test-clean"

SAMPLE_RATE = 16_000

# How many LibriSpeech utterances to score by default.  Large enough that a
# 0.5-point WER difference is meaningful, small enough to iterate on.
DEFAULT_AUDIO_FIXTURE_COUNT = 40


@dataclass(frozen=True, slots=True)
class AudioFixture:
    """A speech clip with a known reference transcript."""

    id: str
    path: Path
    reference: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TextFixture:
    """A raw transcript with the polished output the pipeline should produce."""

    id: str
    raw: str
    expected: str
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to decode benchmark fixtures but was not found on "
            "PATH.  Install it with:  sudo apt install ffmpeg"
        )
    return ffmpeg


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> npt.NDArray[np.float32]:
    """Decode *path* to mono float32 PCM at *sample_rate* via ffmpeg.

    ffmpeg is used rather than a Python audio library so the harness adds no
    dependency to ``pyproject.toml``.
    """
    ffmpeg = _require_ffmpeg()
    result = subprocess.run(
        [
            ffmpeg, "-nostdin", "-loglevel", "error",
            "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(sample_rate),
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}: {stderr}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def to_pcm16(audio: npt.NDArray[np.float32]) -> bytes:
    """Convert float32 [-1, 1] audio to the 16-bit PCM bytes STT backends expect."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# LibriSpeech
# ---------------------------------------------------------------------------


def ensure_librispeech(split: str = DEFAULT_SPLIT, force: bool = False) -> Path:
    """Download and extract a LibriSpeech *split* into the cache.

    Returns the directory containing the extracted ``LibriSpeech/<split>`` tree.
    Roughly a 330-346 MB download per split, performed once.
    """
    if split not in LIBRISPEECH_SPLITS:
        raise ValueError(
            f"Unknown LibriSpeech split '{split}'. Valid: {list(LIBRISPEECH_SPLITS)}"
        )

    extracted = LIBRISPEECH_DIR / "LibriSpeech" / split
    if extracted.is_dir() and not force:
        return extracted

    LIBRISPEECH_DIR.mkdir(parents=True, exist_ok=True)
    url = LIBRISPEECH_SPLITS[split]
    archive = LIBRISPEECH_DIR / f"{split}.tar.gz"

    if not archive.exists() or force:
        logger.info("Downloading LibriSpeech %s (~340 MB) from %s", split, url)
        tmp = archive.with_suffix(".partial")
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp.rename(archive)

    logger.info("Extracting %s", archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(LIBRISPEECH_DIR, filter="data")

    if not extracted.is_dir():
        raise RuntimeError(f"LibriSpeech extraction did not produce {extracted}")
    return extracted


def load_librispeech_fixtures(
    count: int = DEFAULT_AUDIO_FIXTURE_COUNT,
    split: str = DEFAULT_SPLIT,
) -> list[AudioFixture]:
    """Return the first *count* LibriSpeech utterances in deterministic order.

    Sorted by utterance id so the same fixture set is scored on every run —
    a shuffled set would make run-to-run WER differences meaningless.
    """
    root = ensure_librispeech(split)

    fixtures: list[AudioFixture] = []
    for transcript_file in sorted(root.rglob("*.trans.txt")):
        for line in transcript_file.read_text(encoding="utf-8").splitlines():
            utterance_id, _, reference = line.partition(" ")
            if not reference:
                continue
            audio_path = transcript_file.parent / f"{utterance_id}.flac"
            if not audio_path.exists():
                continue
            fixtures.append(
                AudioFixture(
                    id=utterance_id,
                    path=audio_path,
                    reference=reference.strip(),
                    tags=("librispeech", split, "read-speech"),
                )
            )
            if len(fixtures) >= count:
                return fixtures
    return fixtures


def load_user_fixtures(directory: Path) -> list[AudioFixture]:
    """Load user-supplied clips from *directory*.

    Expects each ``<name>.wav`` (or ``.flac``/``.mp3``/``.ogg``) to sit beside a
    ``<name>.txt`` holding its reference transcript.  Clips without a reference
    are skipped with a warning rather than silently scored against nothing.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Fixture directory not found: {directory}")

    fixtures: list[AudioFixture] = []
    for audio_path in sorted(directory.iterdir()):
        if audio_path.suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
            continue
        reference_path = audio_path.with_suffix(".txt")
        if not reference_path.exists():
            logger.warning("Skipping %s — no matching %s", audio_path.name, reference_path.name)
            continue
        fixtures.append(
            AudioFixture(
                id=audio_path.stem,
                path=audio_path,
                reference=reference_path.read_text(encoding="utf-8").strip(),
                tags=("user",),
            )
        )
    return fixtures


def load_audio_fixtures(
    directory: Path | None = None,
    count: int = DEFAULT_AUDIO_FIXTURE_COUNT,
    split: str = DEFAULT_SPLIT,
) -> list[AudioFixture]:
    """Load user fixtures if *directory* is given, otherwise LibriSpeech."""
    if directory is not None:
        return load_user_fixtures(directory)
    return load_librispeech_fixtures(count, split)
