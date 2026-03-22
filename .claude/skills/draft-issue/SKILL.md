---
name: draft-issue
description: Create well-structured GitHub issues optimized for /implement
argument-hint: <brief description of what needs to be done>
---

# Skill: Draft Issue

Draft a well-structured GitHub issue for the linux-whisper project.

## Trigger

The user asks to create, draft, or write a GitHub issue.

## Instructions

### 1. Gather Context

Before drafting, read:
- `README.md` — project overview, features, architecture, CLI reference
- `architecture.md` — pipeline stages, latency budgets, component design
- `vision.md` — design principles, roadmap, non-goals, success metrics
- `pyproject.toml` — dependencies, extras, tool configuration

Understand the 6-stage pipeline (hotkey -> audio -> STT -> polish -> inject), the hybrid polish approach, and the CPU-first architecture.

### 2. Choose Template Size

**Standard Issue** — for focused, single-concern changes (most issues):
- Touches 1-3 files
- Single pipeline stage or component
- Can be completed in one PR

**Large Issue** — for cross-cutting changes that span multiple pipeline stages:
- Touches 4+ files or multiple pipeline stages
- Needs sub-tasks or a phased approach
- May require multiple PRs

### 3. Draft the Issue

#### Standard Issue Template

```markdown
## Summary

[1-2 sentences. What is the problem or feature? Why does it matter?]

## Context

[What is the current behavior? What triggered this issue? Link to relevant
code in `src/linux_whisper/` if applicable.]

## Proposed Solution

[Concrete description of what to implement or change. Reference specific
files, classes, or functions.]

### Changes Required

- [ ] `src/linux_whisper/<file>.py` — [description of change]
- [ ] `tests/test_<file>.py` — [test coverage for the change]

## Acceptance Criteria

- [ ] [Observable, testable criterion]
- [ ] [Another criterion]
- [ ] All tests pass: `python -m pytest tests/`
- [ ] Linting clean: `ruff check src/ tests/`

## Notes

[Optional: latency impact, memory impact, alternative approaches considered,
links to relevant upstream docs.]
```

#### Large Issue Template

```markdown
## Summary

[1-2 sentences. What is the cross-cutting change?]

## Context

[Background on why this is needed. Reference architecture.md or vision.md
sections if applicable.]

## Design

[How should this be implemented? What pipeline stages are affected?
What are the key design decisions?]

### Pipeline Impact

| Stage | Component | Change | Latency Impact |
|-------|-----------|--------|----------------|
| [stage] | [component] | [what changes] | [+/- Xms or none] |

## Sub-Tasks

- [ ] **Task 1:** [description] — `src/linux_whisper/<file>.py`
- [ ] **Task 2:** [description] — `src/linux_whisper/<file>.py`
- [ ] **Task 3:** [description] — `tests/test_<file>.py`

## Acceptance Criteria

- [ ] [Observable, testable criterion]
- [ ] [Performance criterion — e.g., "p95 latency stays under 800ms"]
- [ ] All tests pass: `python -m pytest tests/`
- [ ] Linting clean: `ruff check src/ tests/`

## Notes

[Optional: phasing strategy, rollback plan, memory budget impact.]
```

### 4. Acceptance Criteria Guidance

Good acceptance criteria are:
- **Observable** — "VAD auto-stop triggers within 500ms of silence" not "VAD works better"
- **Testable** — can be verified with a pytest assertion or CLI command
- **Scoped** — tied to this issue, not aspirational improvements

Always include:
- `python -m pytest tests/` passes
- `ruff check src/ tests/` is clean
- Any latency budget impacts are documented

For performance-sensitive changes (audio pipeline, STT, polish pipeline), include specific latency or memory criteria referencing the budgets in `architecture.md`.

### 5. Validation Checklist

Before presenting the issue:

- [ ] Title is concise and starts with a verb (Add, Fix, Update, Remove, Refactor)
- [ ] Summary explains *why*, not just *what*
- [ ] Proposed solution references specific files in `src/linux_whisper/`
- [ ] Acceptance criteria are testable
- [ ] Labels are suggested (bug, enhancement, refactor, performance, etc.)
- [ ] Issue doesn't duplicate an existing open issue (check with `gh issue list`)
- [ ] Scope is appropriate — not too large for a single PR (standard) or has sub-tasks (large)

### 6. Create the Issue

Use `gh issue create` to create the issue on GitHub:

```bash
gh issue create --title "Title here" --body "$(cat <<'EOF'
[issue body]
EOF
)"
```

Add appropriate labels with `--label` if the repository has them configured.
