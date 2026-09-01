# Linux Whisper — Agent Operating Rules

Local voice dictation for Linux. 6-stage pipeline: hotkey (evdev) → audio capture (sounddevice + Silero VAD) → STT (whisper.cpp on GPU) → polish (BERT disfluency + ELECTRA punctuation + formatting + conditional LLM) → text injection (xdotool/wtype/ydotool). Runs entirely on-device, targets < 800ms end-to-end.

Read `architecture.md` for pipeline details, `vision.md` for design principles, `pyproject.toml` for dependencies.

## Development Conventions

- Python 3.12+. Use `X | Y` unions, `match` where appropriate.
- Type hints on all function signatures, return types, and class attributes.
- `@dataclass(frozen=True, slots=True)` for config and value objects.
- `typing.Protocol` for interfaces — not ABC.
- `asyncio` for coordination. CPU-bound work goes in `asyncio.to_thread`.
- `logging.getLogger(__name__)` — never `print`.
- No PyTorch. All inference via ONNX Runtime, whisper.cpp (`pywhispercpp`), or llama-cpp.
- Ruff: rules `E, F, I, N, W, UP, B, SIM, TCH`. Line length 100.
- Tests: pytest, use fixtures, `@pytest.mark.parametrize`, markers `slow` and `integration`.
- Mock external dependencies in tests (audio devices, evdev, display servers, ONNX/GGUF models). Use `conftest.py` stubs.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- Branch naming: `<type>/<issue>-<short-description>`.
- **Maintain documentation as you go.** A change that alters behaviour, models, latency, or config updates `architecture.md`, `README.md`, and this file in the *same* PR — never as a follow-up.

## Pull Requests and Merging

- **Verification gate before opening a PR:** `python -m pytest tests/`,
  `ruff check src/ tests/`, and `python -c "import linux_whisper"`. A change to
  the CLI also runs `linux-whisper --help` and `linux-whisper config validate`.
- **PR body:** a summary of what and why, a bullet list of changes,
  `Closes #N` / `Refs #N`, the pasted test and lint output, and a
  latency/memory impact line — state "no impact on latency budgets or memory
  usage" explicitly when there is none.
- **Size:** under 200 changed lines across 1–3 files is the target, 200–500 is
  acceptable, and 500+ lines or 8+ files needs a justification in the
  description. A new pipeline stage, a refactor that has to be atomic, or
  generated code are good justifications.
- **Issue acceptance criteria** always include the pytest and ruff gates, and
  add a latency or memory criterion for anything touching audio, STT, or the
  polish pipeline.
- **Merging:** squash-merge and delete the branch. Use a regular merge only when
  the individual commits tell a story worth keeping. Never merge a draft PR, a
  PR with failing CI, or a branch whose suite does not pass — rebase on `main`
  and re-run pytest and ruff after resolving conflicts.

## Dependency Rebuilds

`llama-cpp-python` is built **from source with HIP** for gfx1151, not installed from a wheel:

```bash
cp -r ~/.pyenv/versions/3.12.12/lib/python3.12/site-packages/llama_cpp/lib ~/llama_cpp_lib.bak
CMAKE_BUILD_PARALLEL_LEVEL=8 \
CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151" FORCE_CMAKE=1 \
  pip install --no-binary :all: --force-reinstall llama-cpp-python==<version>
```

Back up `llama_cpp/lib` first — rollback is then a file copy rather than a 20-minute rebuild.

**`--force-reinstall` upgrades transitive dependencies whether or not they need it.** The 0.3.34 rebuild silently moved numpy 2.4.6 → 2.5.2, which broke `numba` ("needs NumPy 2.4 or less") for every other project in this interpreter. llama-cpp-python does not even pin numpy. After any rebuild, check the pip conflict block and re-pin what moved:

```bash
python -c "import numba, scipy, onnxruntime, onnx_asr, pywhispercpp, llama_cpp"
```

LifeOS is unaffected by all of this — it lives in `~/.venvs/lifeos/`, never imports `llama_cpp`, and reaches its Gemma model over HTTP on port 8080. Do not point linux-whisper at that server: it serves a different model and LifeOS depends on it.

## Benchmarking Model Changes

Any change that swaps a model, quantisation, or inference backend must be gated on `tests/benchmarks/` — never merged on the assumption it is better.

```bash
python -m tests.benchmarks.run --suite all --label candidate --out /tmp/cand.json
python -m tests.benchmarks.run --compare tests/benchmarks/baseline/current.json /tmp/cand.json
```

`--compare` exits non-zero on regression. Rules:

- The harness needs real models and real audio, so it is **excluded from the default pytest run**. Only the pure scoring functions are unit-tested (`tests/test_benchmarks.py`).
- Latency numbers are hardware-specific — a baseline captured on one machine cannot gate a run from another.
- Re-baseline only *after* a candidate passes and merges. Keep the superseded baseline file.
- `passthrough` text fixtures must come out byte-identical. An edit there is the pipeline paraphrasing, which `vision.md` treats as a correctness bug — fail the candidate even if every other metric improved.
- Any non-zero `thinking_leaks` fails a run outright, regardless of other metrics.

## Escalation Rules

The agent can autonomously plan, implement, test, review, and merge changes. It **must stop and ask a human** before proceeding if any of these apply:

| Trigger | Why |
|---------|-----|
| Changes touch audio pipeline timing or latency-critical paths | The real-time audio callback runs on a dedicated thread at 32ms intervals. Latency regressions are silent and hard to diagnose. |
| Changes modify the `STTEngine` protocol in `stt/engine.py` | This is the contract across all 4 backends (whisper-cpp GPU, whisper-cpp CPU, faster-whisper, moonshine). Changing it requires updating all simultaneously. |
| Changes affect hotkey handling or evdev integration | Hotkey runs in a kernel-level input thread. Bugs here freeze the entire application or miss keypresses system-wide. |
| Changes modify state machine transitions in `state.py` | The state machine guards against illegal transitions (e.g., recording while processing). Incorrect changes cause silent pipeline corruption. |
| Adding a new dependency to `pyproject.toml` | Dependencies affect install size, startup time, and cross-platform compatibility. ROCm shared-library conflicts are real (see pywhispercpp/onnxruntime isolation). |
| Changes to the CLI interface or config schema | Config changes affect all existing users' YAML files. CLI changes affect the systemd service and documentation. |
| Any acceptance criterion is ambiguous or untestable | Implementing against vague criteria wastes effort and produces code that can't be verified. |
| Tests require real audio devices or model downloads | Tests must run in CI without hardware. Mock everything external. |

## Skill System

Development is driven by an orchestrated skill system. The `/implement` skill coordinates the full lifecycle:

```
/implement (orchestrator — does NOT write code itself)
  |
  |-- /implement-plan    Phase 1: Explore codebase, write plan with risk assessment
  |-- /implement-code    Phase 2: Branch, implement, test, lint, open PR
  |-- review             Phase 3: Three passes — correctness, testing, architecture
  |-- /implement-address Phase 3b: Fix findings (can push back with justification)
  \-- /merge-pr          Phase 4: Pre-merge checks, squash merge, cleanup
```

Review is three separate passes:
1. **Correctness** — Does it do what the issue asks? Edge cases? Error paths?
2. **Testing** — Adequate coverage? Testing behavior not implementation? Appropriate markers?
3. **Architecture** — Follows project patterns? Right module? Coupling? Latency impact?

The address skill can **decline review findings** with justification — it has engineering judgment, not just blind compliance. Valid pushback: latency budget would be violated, project patterns differ from the suggestion, test would require real hardware.

Sub-skills run in forked agent contexts (own working memory). Max 3 review/address iterations before escalating.

### Standalone skills

| Skill | Purpose |
|-------|---------|
| `/draft-issue` | Create well-structured issues optimized for `/implement` |
| `/pr-check` | Validate a PR against project standards before review |
| `/mine-for-ideas` | Analyze a topic grounded in architecture constraints |
| `/catchup` | Synthesize recent PR/commit activity |
| `/standup` | Daily summary: shipped, in progress, blocked, next |
| `/stale` | Find stale PRs, orphan branches, stale issues |

## Hotkey Latency and Keyboard Remappers

Before investigating hotkey or overlay latency in this codebase, check for a
keyboard remapper. Toshy/xwaykeyz, keyd, kanata and similar tools `EVIOCGRAB`
the real keyboard and re-emit through a virtual uinput device; a **modifier**
hotkey then gets held while the remapper disambiguates a possible combo.

On the development machine this put `fn` about a second behind the keypress.
It is not observable from inside the app: evdev timestamps come from the
virtual device and are stamped at replay, so `kernel timestamp -> handler`
measures ~0.2ms while the real delay sits upstream of it. The 0.75s audio
pre-roll independently masks the recording half of the symptom, leaving only
a visibly late overlay and pointing the investigation at the wrong component.

The decisive test is external: stop the remapper's service, try the hotkey,
start it again. Prefer a non-modifier hotkey (`capslock`, a function key, or a
modifier *combo*) over a bare modifier.

## Latency Budgets

Referenced during planning to assess whether a change is safe.

| Stage | GPU (default) | CPU fallback |
|-------|--------------|-------------|
| 1. Hotkey detection | < 5ms | < 5ms |
| 2. Audio + VAD + AGC | < 10ms | < 10ms |
| 3. STT (whisper.cpp large-v3-turbo) | ~285ms | ~2.5s |
| 4a. Disfluency removal | < 15ms | < 15ms |
| 4b. Punctuation | < 15ms | < 15ms |
| 4d. Number/date formatting | < 1ms | < 1ms |
| 4c. LLM correction (conditional) | ~150ms | ~370ms |
| 5. Text injection | < 20ms | < 20ms |
| **Total (simple)** | **~340ms** | **~2.6s** |
| **Total (with LLM)** | **~490ms** | **~2.9s** |

The default STT backend is whisper.cpp large-v3-turbo on the ROCm GPU, in a subprocess worker (the pywhispercpp/onnxruntime `libamdhip64` conflict is why).

**Benchmark corpora do not decide the default; recorded dictation does.** Parakeet TDT v3 beat whisper on LibriSpeech (0.54% vs 1.48% WER, 191ms vs 285ms p50) and was made default on that basis — then lost 49.3% to 21.5% on 28 real dictation clips. LibriSpeech contains no digits, symbols, or filenames, so it never tested inverse text normalisation, which is most of what dictation is. Always confirm an STT change against `--fixtures-dir` recordings before touching the default.

## Key Files

| File | Role |
|------|------|
| `app.py` | Main orchestrator — wires all pipeline stages, manages async lifecycle |
| `config.py` | Frozen dataclass config, YAML loading, validation |
| `state.py` | Async state machine: IDLE → RECORDING → PROCESSING → IDLE |
| `hotkey.py` | evdev global hotkey daemon, 4 modes (auto/hold/toggle/vad-auto) |
| `audio.py` | Ring buffer, Silero VAD, AGC, feedback tones, sounddevice capture |
| `stt/engine.py` | `STTEngine` protocol + factory. All backends implement this. |
| `stt/whisper_gpu.py` | GPU STT — subprocess isolation, pipe-based IPC with worker |
| `polish/pipeline.py` | Four-stage orchestrator: disfluency → punctuation → formatting → LLM |
| `polish/llm.py` | Qwen3 4B via llama-cpp, ROCm GPU offload, context-aware prompts |
| `focus.py` | Focused app detection (X11/Sway/Hyprland) for tone adaptation |
| `snippets.py` | Voice snippet matching — fuzzy, bypasses polish pipeline |
| `inject/injector.py` | Display server detection, 4 injection backends |
| `overlay.py` | GTK3 (via XWayland) floating recording pill — primary recording indicator, never takes focus |
