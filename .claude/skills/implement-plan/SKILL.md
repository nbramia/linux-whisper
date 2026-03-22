---
name: implement-plan
description: Explore codebase and plan implementation for implement workflow
context: fork
agent: general-purpose
argument-hint: <task description and optional instructions>
---

# Skill: Implement Plan

Create a concrete implementation plan for a GitHub issue.

## Trigger

Delegated from the **implement** orchestrator, or when the user asks to plan an implementation.

## Instructions

### 1. Understand the Issue

Read the issue thoroughly. Identify:
- What needs to change (features, bugs, refactors)
- Acceptance criteria (what "done" looks like)
- Constraints (latency budgets, memory, compatibility)

### 2. Gather Project Context

Read these files for conventions and architecture:
- `README.md` — project overview, feature list, architecture diagram, project structure
- `architecture.md` — pipeline stages, latency budgets, component design, concurrency model
- `vision.md` — design principles, roadmap phases, non-goals
- `pyproject.toml` — dependencies, tool configuration, Python version requirement

Understand the project structure:
```
src/linux_whisper/
    app.py              # Main orchestrator
    audio.py            # Ring buffer, VAD, capture, feedback
    cli.py              # CLI entry point
    config.py           # YAML config, validation, dataclasses
    hotkey.py           # evdev hotkey daemon
    state.py            # Async state machine
    overlay.py          # GTK4 floating pill
    tray.py             # pystray system tray
    stt/
        engine.py       # STTEngine protocol and factory
        faster_whisper.py
        moonshine.py
    polish/
        pipeline.py     # Three-stage orchestrator
        disfluency.py   # BERT / regex filler removal
        punctuation.py  # ELECTRA / rule-based punctuation
        llm.py          # Qwen3 4B via llama-cpp
    inject/
        injector.py     # Display server detection, text injection
tests/
    conftest.py
    test_audio.py
    test_cli.py
    test_config.py
    test_inject.py
    test_polish.py
    test_state.py
    test_stt.py
```

### 3. Project Conventions

Follow these Python/linux-whisper conventions:

- **Python 3.12+** — use modern syntax (type unions with `|`, `match` statements where appropriate)
- **Type hints everywhere** — all function signatures, return types, class attributes
- **Frozen dataclasses for data** — `@dataclass(frozen=True, slots=True)` for config and value objects
- **Protocol for interfaces** — `typing.Protocol` for STT engines, injectors, etc. (not ABC)
- **asyncio for coordination** — the main loop is async; CPU-bound work runs in thread pools
- **logging module** — use `logging.getLogger(__name__)`, not print statements
- **No PyTorch** — all inference via ONNX Runtime or llama.cpp
- **Ruff for linting** — target rules: E, F, I, N, W, UP, B, SIM, TCH
- **Line length 100** — per `pyproject.toml`
- **Tests with pytest** — use fixtures, parametrize, appropriate markers

### 4. Write the Plan

Structure the plan as:

```markdown
## Plan: [Issue Title]

### Summary
[1-2 sentences on the approach]

### Files to Modify
| File | Change | Risk |
|------|--------|------|
| `src/linux_whisper/<file>.py` | [what changes] | low/medium/high |
| `tests/test_<file>.py` | [what tests to add] | low |

### New Files (if any)
| File | Purpose |
|------|---------|
| `src/linux_whisper/<file>.py` | [purpose] |

### Approach

#### Step 1: [description]
[Details of what to do and why]

#### Step 2: [description]
[Details]

### Test Strategy
- [What unit tests to add]
- [What integration tests if applicable]
- [What mocks/fixtures are needed]
- [Which test markers apply: slow, integration]

### Risk Assessment
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| [risk] | low/medium/high | [impact] | [mitigation] |

### Latency/Memory Impact
[Will this affect latency budgets from architecture.md? Memory usage?
State "no impact" explicitly if none.]

### Out of Scope
[Anything related but explicitly NOT part of this implementation]
```

### 5. Validate the Plan

Before presenting:
- [ ] Every acceptance criterion from the issue is addressed
- [ ] All modified files exist in the project (or are explicitly new)
- [ ] Test strategy covers the acceptance criteria
- [ ] Risk assessment is honest (not everything is "low risk")
- [ ] No unnecessary changes beyond what the issue requires
- [ ] The plan respects latency budgets (< 5ms hotkey, < 300ms STT, < 15ms each encoder, < 350ms LLM, < 20ms injection)
- [ ] Dependencies on other issues are identified

### 6. Present and Confirm

Present the plan to the orchestrator (or user). Wait for approval before proceeding to implementation.
