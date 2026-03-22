---
name: implement-address
description: Address filtered review findings for implement workflow
context: fork
agent: general-purpose
argument-hint: <pr-number> <round-number> <findings-file-path>
---

# Skill: Implement Address

Address review findings from the implement workflow.

## Trigger

Delegated from the **implement** orchestrator when review findings need to be resolved.

## Instructions

### 1. Read Findings

Read the findings file at `/tmp/linux-whisper-implement-findings-<issue>-<pass>.md`.

Each finding will be one of:
- **Must Fix** — blocks merge, must be resolved
- **Should Fix** — important improvement, fix unless there's a good reason not to
- **Nit** — minor style or preference issue, fix if easy

### 2. Triage

Categorize findings:

| Finding | Severity | Action |
|---------|----------|--------|
| [finding] | Must Fix | [how to fix] |
| [finding] | Should Fix | [how to fix] |
| [finding] | Nit | [fix or explain why not] |

### 3. Address Each Finding

For each Must Fix and Should Fix:

1. Read the relevant code
2. Make the fix
3. Run tests to verify the fix doesn't break anything:
   ```bash
   python -m pytest tests/ -v
   ruff check src/ tests/
   ```
4. Commit the fix with a clear message:
   ```bash
   git commit -m "fix: address review — <brief description>

   - [what was fixed and why]
   - [another fix if multiple]

   Refs #<issue>"
   ```

For Nits:
- Fix if the change is straightforward
- If declining a nit, note why in the commit message or report

### 4. Handle Disagreements

If you believe a finding is incorrect or the suggested fix would make things worse:

1. Do NOT silently ignore the finding
2. Write a clear explanation of why you disagree
3. Report the disagreement back to the orchestrator for resolution

Common legitimate reasons to push back:
- The suggested fix would violate a latency budget
- The finding is based on a pattern from a different project, not linux-whisper's conventions
- The test suggestion would require real hardware (audio device, display server) rather than mocks
- The finding is already handled by a different mechanism (e.g., "add error handling" when asyncio.to_thread already propagates exceptions)

### 5. Verify

After all fixes:

```bash
# Full test suite
python -m pytest tests/ -v

# Lint
ruff check src/ tests/

# Import check
python -c "import linux_whisper"
```

All must pass.

### 6. Push

```bash
git push
```

### 7. Report

Report back to the orchestrator:

```markdown
## Findings Addressed

**Findings file:** `/tmp/linux-whisper-implement-findings-<issue>-<pass>.md`

### Resolved
| Finding | Severity | Resolution |
|---------|----------|------------|
| [finding] | Must Fix | [what was done] |
| [finding] | Should Fix | [what was done] |

### Declined (with justification)
| Finding | Severity | Reason |
|---------|----------|--------|
| [finding] | Nit | [why it was declined] |

### Verification
- Tests: X passed, Y new
- Lint: clean
- Commits: N new commits pushed
```
