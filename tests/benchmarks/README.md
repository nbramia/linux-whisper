# Model Benchmark Harness

Measures whether a model change is actually an improvement. Every model swap in
the pipeline is gated on a before/after run from this harness.

This code is **not** part of `pytest tests/`. It loads real models and real
audio, which the CI rules in `CLAUDE.md` forbid. Only the pure scoring functions
in `metrics.py` are unit-tested (`tests/test_benchmarks.py`).

## Quick start

```bash
# One-time: fetch fixture audio (~346 MB LibriSpeech test-clean into ~/.cache)
python -m tests.benchmarks.run --suite stt --count 5 --label smoke

# Capture a baseline of the currently configured stack
python -m tests.benchmarks.run --suite all --label baseline \
    --out tests/benchmarks/baseline/current.json

# Score a candidate, then gate it against the baseline
python -m tests.benchmarks.run --suite all --label candidate --out /tmp/cand.json
python -m tests.benchmarks.run --compare tests/benchmarks/baseline/current.json /tmp/cand.json
```

`--compare` exits **non-zero** when the candidate regresses, so it works as a
merge gate.

## Suites

| Suite | Fixtures | Scores |
|-------|----------|--------|
| `stt` | LibriSpeech `test-clean` audio, or your own clips | WER, per-utterance latency, RTFx |
| `polish` | Text pairs in `text_fixtures.py` — no audio needed | WER vs expected, exact-match rate, punctuation F1, capitalisation, per-stage latency (4a/4b/4c/4d) |
| `vad` | Audio fixtures plus synthetic silence and noise | Speech-frame detection rate, false-positive rates, per-window latency |
| `all` | Both | Everything above |

Splitting them this way is deliberate: the polish stages are text-in/text-out, so
scoring them through an STT backend would just add noise to the measurement.

## Fixtures

**Audio** defaults to the first N LibriSpeech `test-clean` utterances in sorted
order — deterministic, so run-to-run WER differences mean something. Downloaded
to `~/.cache/linux-whisper/benchmarks/` on first use.

`--split test-other` selects LibriSpeech's harder split — accents, faster
delivery, worse recordings. Worth running when a backend looks good on
`test-clean`, since the ranking can move: Moonshine v2-small sits 1.5x behind v1
on `test-clean` but only 1.13x behind on `test-other`.

**Both splits are read audiobook prose.** Neither contains fillers, stammers, or
self-corrections, and both use utterances far longer than a dictated sentence.
They rank models sensibly but cannot tell you how a backend handles *your* voice
saying what you actually dictate — the only question that decides a default.

Record that set with the bundled helper:

```bash
python -m tests.benchmarks.record --out ~/dictation-fixtures
python -m tests.benchmarks.run --suite stt --fixtures-dir ~/dictation-fixtures
```

It walks 15 prompts covering fillers, stammers, self-corrections, numbers,
dates, technical vocabulary, and questions, writing the `clip.wav` + `clip.txt`
pairs `--fixtures-dir` expects. Re-take a bad one with `--only <prompt-id>`.
Any directory of `.wav`/`.txt` pairs works, so hand-made fixtures are fine too.

**Text** fixtures live in `text_fixtures.py`, tagged by category: `filler`,
`repetition`, `false-start`, `self-correction`, `formatting`, `punctuation`, and
`passthrough`.

The `passthrough` cases are the guard rails. They are already-clean input that
must come out **byte-identical**. Any edit there means the pipeline is
paraphrasing, which `vision.md` treats as a correctness bug rather than a style
preference — worth failing a model over even if every other metric improved.

Only `self-correction` fixtures should wake stage 4c. `llm_invocations` in the
output tells you whether that gating still holds; a model change that quietly
starts invoking the LLM on everything would blow the latency budget.

## Reading the output

```
  STT WER:          0.0512  (61/1191 words, 40 fixtures)
  RTFx:             298.4
  Polish WER:       0.1043
  Exact match:      54.5%
  LLM invocations:  5
  Punctuation F1:   0.9231

  Latency (ms)      p50      p95      max
    polish_4a         1.2      2.1      3.4
    polish_4c       187.3    241.8    260.1
    stt             284.1    331.7    402.9
```

The `vad` suite exists because a mis-sized VAD window is invisible everywhere
else. Silero v6 accepts v5's 512-sample window without error and returns ~0.001
for every frame, so detection dies silently while hold-to-talk dictation keeps
working and nothing logs a complaint. The suite scores speech detection against
negative controls (digital silence, low-level noise) so a stuck-low **or**
stuck-high VAD fails. A speech rate below 30% prints a warning; a collapse
against baseline fails the compare gate.

`thinking_leaks` appears only when it is non-zero. It counts outputs containing
reasoning-trace markers (`<think>` and friends) — a hybrid-reasoning model
ignoring its instructions. Any non-zero value fails the run regardless of what
the other numbers say.

## Regression thresholds

| Metric | Direction | Default tolerance |
|--------|-----------|-------------------|
| `stt.wer`, `polish.wer` | lower better | +0.005 absolute (0.5 WER points) |
| `latency.*.p95_ms` | lower better | ×1.10 (10% slowdown) |
| `punctuation.f1`, `capitalization.accuracy`, `polish.exact_match` | higher better | −0.05 absolute |
| `vad.speech_frame_rate` | higher better | −0.10 absolute |
| `vad.*_false_positive_rate` | lower better | +0.10 absolute |

Override with `--wer-tolerance` and `--latency-tolerance`.

Metrics missing from either run are skipped rather than treated as regressions,
so adding a new metric does not invalidate an older baseline.

## Re-baselining

Only after a candidate has passed its gate and merged:

```bash
python -m tests.benchmarks.run --suite all --label baseline-<change> \
    --out tests/benchmarks/baseline/current.json
git add tests/benchmarks/baseline/current.json
```

Keep the superseded baseline file — the history of what each model swap actually
bought is the point of the harness.

## Caveats

- Latency numbers are hardware-specific. A baseline captured on one machine
  cannot gate a candidate run on another.
- **Ambient system load moves the numbers a lot.** A whisper.cpp baseline taken
  while a `llama-server` was resident measured p95 968ms; the same stack on a
  quiet machine measured 383ms. Capture the baseline and the candidate
  **back to back**, and do not compare against a baseline from another session.
  This hits GPU backends hardest, since they contend for the shared iGPU.
- The first transcription is a discarded warm-up so model load and GPU kernel
  compilation are not charged to fixture 1.
- `peak_rss_mb` is the whole process, so on a GPU backend it excludes VRAM and
  on the subprocess-isolated whisper worker it excludes the worker.
