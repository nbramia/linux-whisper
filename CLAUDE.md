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

## Latency Budgets

Referenced during planning to assess whether a change is safe.

| Stage | GPU (default) | CPU fallback |
|-------|--------------|-------------|
| 1. Hotkey detection | < 5ms | < 5ms |
| 2. Audio + VAD + AGC | < 10ms | < 10ms |
| 3. STT (whisper.cpp) | ~300ms | ~2.5s |
| 4a. Disfluency removal | < 15ms | < 15ms |
| 4b. Punctuation | < 15ms | < 15ms |
| 4d. Number/date formatting | < 1ms | < 1ms |
| 4c. LLM correction (conditional) | ~200ms | ~370ms |
| 5. Text injection | < 20ms | < 20ms |
| **Total (simple)** | **~350ms** | **~2.6s** |
| **Total (with LLM)** | **~550ms** | **~2.9s** |

GPU STT runs in a subprocess worker (pywhispercpp/onnxruntime ROCm conflict requires process isolation).

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
