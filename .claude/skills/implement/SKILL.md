---
name: implement
description: Implementation, review, and merge — full lifecycle or any subset
argument-hint: <#issue | PR-number | freeform task> [instructions]
---

# Skill: Implement (Orchestrator)

Orchestrate end-to-end implementation of a GitHub issue: plan, code, review, address, merge.

## Trigger

The user asks to implement a GitHub issue (e.g., "implement #42", "work on this issue").

## Instructions

You are the **orchestrator**. You coordinate four phases and delegate to sub-skills. You do NOT write code yourself.

### Phase 0: Load Context

1. Read the issue (use `gh issue view <number>`)
2. Read project context:
   - `README.md` — project overview, architecture, CLI
   - `architecture.md` — pipeline stages, latency budgets
   - `pyproject.toml` — dependencies, tooling
3. Verify the issue has clear acceptance criteria. If not, ask the user to clarify before proceeding.

### Phase 1: Plan

Delegate to the **implement-plan** skill.

Input: The issue number and context gathered in Phase 0.

Expected output: A concrete implementation plan with:
- Files to create or modify
- Approach for each change
- Test strategy
- Risk assessment

Review the plan. If it looks reasonable, proceed. If it has gaps, ask the planner to revise.

### Phase 2: Code

Delegate to the **implement-code** skill.

Input: The issue number and the plan from Phase 1.

Expected output:
- Implementation committed to a feature branch
- All tests pass: `python -m pytest tests/`
- Linting clean: `ruff check src/ tests/`
- PR created with a clear description

### Phase 3: Review and Address

Run three review passes on the PR. For each, analyze the diff and check for issues:

**Review: Correctness**
- Does the code do what the issue asks?
- Are there edge cases missed?
- Are error paths handled?
- Do the tests actually verify the acceptance criteria?

**Review: Testing**
- Is test coverage adequate?
- Are tests testing behavior, not implementation?
- Are there missing test cases (error paths, boundary conditions)?
- Do tests use appropriate markers (`@pytest.mark.slow`, `@pytest.mark.integration`)?

**Review: Architecture**
- Does this follow linux-whisper's patterns? (frozen dataclasses for config, Protocol for interfaces, asyncio for coordination)
- Is the code in the right module? (stt/, polish/, inject/, etc.)
- Any unnecessary coupling between pipeline stages?
- Will this impact latency budgets?

Write findings to `/tmp/linux-whisper-implement-findings-<issue>-<pass>.md`.

If findings exist, delegate to the **implement-address** skill to fix them. Then re-review. Loop until clean (max 3 iterations).

### Phase 4: Merge

Once reviews are clean:
1. Verify all tests pass one final time: `python -m pytest tests/`
2. Verify lint is clean: `ruff check src/ tests/`
3. Delegate to the **merge-pr** skill

### Escalation

Stop and ask the user before proceeding if any of these are true:
- Changes touch audio pipeline timing or latency-critical paths
- Changes modify the STT engine protocol (`STTEngine` in `stt/engine.py`)
- Changes affect hotkey handling or evdev integration
- Changes modify the state machine transitions
- The plan requires adding a new dependency to `pyproject.toml`
- The plan changes the CLI interface or config schema
- Any acceptance criterion is ambiguous or untestable
- Tests require real audio devices or model downloads (should be mocked)

### Output Format

At each phase transition, report:
```
## Phase N Complete: [Phase Name]
**Status:** [Pass/Issues Found]
**Summary:** [1-2 sentences]
**Next:** [What happens next]
```

At completion:
```
## Implementation Complete
**Issue:** #<number>
**PR:** #<pr-number>
**Changes:** [Brief summary of what was implemented]
**Tests:** [Number of new/modified tests]
**Status:** Merged to main
```
