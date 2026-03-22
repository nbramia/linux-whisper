---
name: implement-code
description: Implement code, tests, and create PR for implement workflow
context: fork
agent: general-purpose
argument-hint: <issue-number-or-0> <task description, plan, and instructions>
---

# Skill: Implement Code

Implement the plan from a GitHub issue on a feature branch and open a PR.

## Trigger

Delegated from the **implement** orchestrator after the plan is approved.

## Instructions

### 1. Set Up Branch

```bash
git checkout main
git pull origin main
git checkout -b <branch-name>
```

Branch naming: `<type>/<issue>-<short-description>`
- `feat/42-vad-auto-stop`
- `fix/87-clipboard-restore`
- `refactor/103-stt-protocol`

### 2. Implement the Plan

Follow the plan step-by-step. For each change:

1. Read the existing file first to understand current patterns
2. Make the change following project conventions
3. Run tests after each significant change

#### Python Conventions

- **Type hints** on all functions:
  ```python
  def feed_audio(self, chunk: np.ndarray) -> list[TranscriptSegment]:
  ```
- **Frozen dataclasses** for value objects:
  ```python
  @dataclass(frozen=True, slots=True)
  class TranscriptSegment:
      text: str
      start: float
      end: float
      confidence: float
  ```
- **Protocol for interfaces** (not ABC):
  ```python
  class STTEngine(Protocol):
      def start_stream(self) -> None: ...
      def feed_audio(self, chunk: np.ndarray) -> list[TranscriptSegment]: ...
      def finalize(self) -> TranscriptResult: ...
      def reset(self) -> None: ...
  ```
- **Logging** via the standard module:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info("Model loaded: %s (%.1f MB)", model_name, size_mb)
  ```
- **Error handling** — specific exceptions, not bare `except`:
  ```python
  try:
      result = engine.finalize()
  except STTError as exc:
      logger.error("STT finalization failed: %s", exc)
      return TranscriptResult(text="", error=str(exc))
  ```
- **asyncio** for coordination:
  ```python
  async def process_utterance(self, audio: np.ndarray) -> str:
      transcript = await asyncio.to_thread(self.engine.finalize)
      polished = await self.polish_pipeline.process(transcript.text)
      return polished
  ```
- **No PyTorch imports** — use ONNX Runtime or llama-cpp-python

### 3. Write Tests

For every change, write or update tests:

```python
# tests/test_<module>.py

import pytest
from linux_whisper.<module> import <class_or_function>


class TestClassName:
    """Tests for ClassName."""

    def test_basic_behavior(self):
        """Describe what this test verifies."""
        result = function_under_test(input_data)
        assert result == expected

    @pytest.mark.parametrize("input_val,expected", [
        ("um hello", "hello"),
        ("I I think so", "I think so"),
    ])
    def test_parametrized(self, input_val, expected):
        """Test multiple input/output pairs."""
        assert clean_text(input_val) == expected

    @pytest.mark.slow
    def test_with_model(self, loaded_model):
        """Test that requires a real model (slow)."""
        result = loaded_model.transcribe(audio_fixture)
        assert result.text

    @pytest.mark.integration
    def test_full_pipeline(self):
        """Test end-to-end pipeline."""
        ...
```

Testing guidelines:
- Test behavior, not implementation details
- Mock external dependencies (audio devices, evdev, display servers, model files)
- Use `@pytest.mark.slow` for tests that load real models
- Use `@pytest.mark.integration` for full pipeline tests
- Use `conftest.py` fixtures for shared test setup
- Use `tmp_path` fixture for files, not hardcoded paths

### 4. Verify

Run the full verification suite:

```bash
# Tests pass
python -m pytest tests/ -v

# Linting clean
ruff check src/ tests/

# Import check — the package loads without errors
python -c "import linux_whisper"

# If CLI changes were made, verify they work
linux-whisper --help
linux-whisper config validate
```

All four must pass before proceeding.

### 5. Commit

Make focused commits. Each commit should be a logical unit:

```bash
git add src/linux_whisper/<file>.py tests/test_<file>.py
git commit -m "feat: add VAD auto-stop mode

Implement silence-based automatic recording stop. Silero VAD
monitors for configurable silence duration (default 500ms) and
triggers recording stop without requiring hotkey release.

Closes #42"
```

Commit message conventions:
- Prefix: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- First line: imperative mood, under 72 characters
- Body: explain *why*, not just *what*
- Reference issue with `Closes #N` or `Refs #N`

### 6. Open PR

```bash
git push -u origin <branch-name>
```

Create the PR:

```bash
gh pr create --title "<type>: <concise description>" --body "$(cat <<'EOF'
## Summary

[1-3 sentences on what this PR does and why]

Closes #<issue-number>

## Changes

- [Bullet list of key changes]
- [Another change]

## Test Evidence

```
$ python -m pytest tests/ -v
[paste relevant output or summary]

$ ruff check src/ tests/
All checks passed!
```

## Latency/Memory Impact

[State impact or "No impact on latency budgets or memory usage."]

## Checklist

- [ ] Tests pass: `python -m pytest tests/`
- [ ] Lint clean: `ruff check src/ tests/`
- [ ] Type hints on all new functions
- [ ] Logging (not print) for diagnostics
- [ ] Acceptance criteria from issue are met
EOF
)"
```

### 7. Report

Report back to the orchestrator:
- PR number and URL
- Summary of changes
- Test results (pass count, new tests added)
- Any deviations from the plan and why
