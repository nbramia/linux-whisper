---
name: mine-for-ideas
description: Analyze an open-source repo and surface ideas linux-whisper could learn from
argument-hint: <github-url> [context/guidance]
---

# Skill: Mine for Ideas

Deeply analyze a topic in the context of linux-whisper and produce a structured analysis with concrete recommendations.

## Trigger

The user asks to explore, analyze, research, or think through a technical topic related to linux-whisper (e.g., "explore whether we should switch to whisper.cpp as default", "analyze streaming vs batch STT tradeoffs", "think about how to add language support").

## Instructions

### 1. Gather Context

Before analyzing, build a thorough understanding of linux-whisper:

**Core documents:**
- `README.md` — features, architecture overview, STT model table, configuration, CLI reference
- `architecture.md` — pipeline stages, latency budgets, component design, memory budget, concurrency model, technology stack
- `vision.md` — design principles, roadmap, non-goals, success metrics

**Source code** (as relevant to the topic):
- `src/linux_whisper/app.py` — main orchestrator
- `src/linux_whisper/audio.py` — ring buffer, VAD, capture
- `src/linux_whisper/config.py` — configuration schema
- `src/linux_whisper/state.py` — state machine
- `src/linux_whisper/stt/engine.py` — STT engine protocol
- `src/linux_whisper/stt/faster_whisper.py` — CTranslate2 backend
- `src/linux_whisper/stt/moonshine.py` — ONNX backend
- `src/linux_whisper/polish/pipeline.py` — three-stage polish orchestrator
- `src/linux_whisper/polish/disfluency.py` — BERT filler removal
- `src/linux_whisper/polish/punctuation.py` — ELECTRA punctuation
- `src/linux_whisper/polish/llm.py` — Qwen3 4B via llama-cpp
- `src/linux_whisper/inject/injector.py` — text injection
- `src/linux_whisper/hotkey.py` — evdev hotkey daemon
- `src/linux_whisper/tray.py` — system tray

### 2. Frame the Analysis

Define:
- **Question:** What specific question are we trying to answer?
- **Scope:** What parts of the pipeline are affected?
- **Constraints:** What are the hard constraints? (latency budgets, CPU-first, no PyTorch, local-only, etc.)
- **Design principles to honor:** Which of linux-whisper's design principles (from vision.md) are most relevant?

### 3. Analyze

Structure the analysis around these dimensions (include all that are relevant):

#### Technical Feasibility
- Can this be done within linux-whisper's architecture?
- What pipeline stages are affected?
- Does it require new dependencies? (Remember: no PyTorch, prefer ONNX Runtime or llama.cpp)
- Does it work on both X11 and Wayland?

#### Latency Impact
Reference the specific budgets from architecture.md:
- Hotkey detection: < 5ms
- Audio capture + VAD: < 10ms
- STT: < 300ms (streaming) or < 500ms (batch)
- Polish 4a (disfluency): < 15ms
- Polish 4b (punctuation): < 15ms
- Polish 4c (LLM): < 350ms
- Text injection: < 20ms
- Total simple: < 365ms
- Total complex: < 715ms

Will this change blow any budget? Can it be made to fit?

#### Memory Impact
Reference the memory budget from architecture.md:
- Total default: ~3.5GB of 64GB
- Each component's allocation

Will this change the idle memory footprint significantly?

#### CPU vs GPU Considerations
- Does this work on CPU with AVX-512? (primary target)
- Is there a future ROCm acceleration path?
- Does this avoid CUDA dependencies?

#### Model Selection Tradeoffs (if applicable)
- Parameter count vs accuracy (WER, F1, IFEval)
- Quantization options and their quality/speed tradeoffs
- ONNX vs GGUF vs other runtime formats
- Streaming capability vs batch-only

#### User Experience Impact
- Does this change the core flow (hotkey -> speak -> text)?
- Does it affect perceived latency?
- Does it require configuration changes?
- Is it invisible to users or does it add visible complexity?

#### Audio Pipeline Design (if applicable)
- Ring buffer implications
- VAD threshold sensitivity
- Pre-roll buffer interaction
- Sample rate / bit depth considerations

### 4. Explore Alternatives

For each viable approach, provide:

```markdown
### Option A: [Name]

**Description:** [1-2 sentences]

**Pros:**
- [pro]
- [pro]

**Cons:**
- [con]
- [con]

**Latency impact:** [+/- Xms on stage Y]
**Memory impact:** [+/- XMB]
**Complexity:** low / medium / high
**Fits design principles:** [which ones it honors, which it tensions]
```

Always include at least one "do nothing" option that explains the cost of inaction.

### 5. Recommendation

```markdown
## Recommendation

**Preferred:** Option [X]

**Rationale:** [2-3 sentences explaining why, referencing specific
constraints, budgets, or design principles]

**Implementation effort:** [rough estimate — small/medium/large]

**Suggested next steps:**
1. [concrete action]
2. [concrete action]
3. [concrete action]

**Open questions:**
- [question that needs answering before committing]
- [another question]
```

### 6. Output Format

Present the full analysis as a structured document:

```markdown
# Analysis: [Topic]

## Question
[What are we trying to decide?]

## Context
[Relevant linux-whisper architecture and constraints]

## Options
[Option A, B, C analysis]

## Tradeoff Matrix

| Dimension | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| Latency impact | [value] | [value] | [value] |
| Memory impact | [value] | [value] | [value] |
| Complexity | [value] | [value] | [value] |
| UX impact | [value] | [value] | [value] |
| Fits principles | [value] | [value] | [value] |

## Recommendation
[Preferred option with rationale]

## Next Steps
[Concrete actions]
```

### Guidelines

- **Be specific.** "This adds ~200ms to stage 4c" is useful. "This might be slower" is not.
- **Reference real numbers.** Use WER percentages, latency measurements, memory sizes, parameter counts from the architecture docs.
- **Honor the constraints.** Don't recommend GPU-only solutions for a CPU-first project. Don't recommend cloud APIs for a local-only project.
- **Be honest about unknowns.** If you're estimating latency, say so. If a model's quality on this specific task is unknown, say so.
- **Consider the roadmap.** Does this align with the v0.1/v0.2/v0.3 phasing in vision.md?
