"""Polish pipeline orchestrator — chains disfluency, punctuation, and LLM stages.

The :class:`PolishPipeline` runs stages 4a → 4b → (conditional) 4c on each
transcript, respecting the user's :class:`PolishConfig` to enable or disable
individual stages.  Stage 4c (LLM) is only invoked when:

1. ``config.polish.llm`` is enabled, **and**
2. the disfluency stage flagged self-corrections, **or** ``config.polish.llm_always``
   is True.
"""

from __future__ import annotations

import logging
import time

from linux_whisper.config import PolishConfig
from linux_whisper.polish.disfluency import DisfluencyRemover, DisfluencyResult
from linux_whisper.polish.llm import LLMCorrector
from linux_whisper.polish.punctuation import PunctuationRestorer

logger = logging.getLogger(__name__)


class PolishPipeline:
    """Orchestrates the three-stage text polish pipeline.

    Stages
    ------
    1. **Disfluency removal** (4a) — BERT token classifier / regex fallback.
    2. **Punctuation + capitalisation** (4b) — ELECTRA classifier / rules.
    3. **LLM self-correction resolution** (4c) — Qwen3 4B via llama.cpp,
       conditionally invoked.

    Each stage can be individually disabled via :class:`PolishConfig`.  When
    ``config.enabled`` is False the entire pipeline is bypassed and the input
    text is returned as-is.
    """

    def __init__(self, config: PolishConfig | None = None) -> None:
        self._config = config or PolishConfig()

        # Lazy-initialise stage components only if enabled.
        self._disfluency: DisfluencyRemover | None = None
        self._punctuation: PunctuationRestorer | None = None
        self._llm: LLMCorrector | None = None

        if self._config.enabled:
            self._init_stages()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_stages(self) -> None:
        """Create stage objects respecting config flags."""
        if self._config.disfluency:
            logger.debug("Initialising disfluency remover (stage 4a)")
            self._disfluency = DisfluencyRemover()

        if self._config.punctuation:
            logger.debug("Initialising punctuation restorer (stage 4b)")
            self._punctuation = PunctuationRestorer()

        if self._config.llm:
            logger.debug("Initialising LLM corrector (stage 4c)")
            self._llm = LLMCorrector(config=self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> str:
        """Run the full polish pipeline on *text* and return cleaned output.

        The method is synchronous and intended to be called from an async
        context via ``asyncio.to_thread`` or an executor.
        """
        if not self._config.enabled:
            logger.debug("Polish pipeline disabled — returning text unchanged")
            return text

        if not text or not text.strip():
            return ""

        t0 = time.perf_counter()
        current = text
        has_self_corrections = False

        # ── Stage 4a: Disfluency removal ──────────────────────────────
        if self._disfluency is not None:
            t_stage = time.perf_counter()
            disfluency_result: DisfluencyResult = self._disfluency.process(current)
            current = disfluency_result.text
            has_self_corrections = disfluency_result.has_self_corrections
            dt = (time.perf_counter() - t_stage) * 1000
            logger.debug(
                "Stage 4a (disfluency): %.1f ms | self-corrections=%s | %r → %r",
                dt,
                has_self_corrections,
                text[:80],
                current[:80],
            )

        # ── Stage 4b: Punctuation + capitalisation ────────────────────
        if self._punctuation is not None:
            t_stage = time.perf_counter()
            current = self._punctuation.process(current)
            dt = (time.perf_counter() - t_stage) * 1000
            logger.debug("Stage 4b (punctuation): %.1f ms", dt)

        # ── Stage 4c: LLM correction (conditional) ───────────────────
        should_run_llm = (
            self._llm is not None
            and self._llm.available
            and (has_self_corrections or self._config.llm_always)
        )

        if should_run_llm:
            assert self._llm is not None  # for type narrowing
            t_stage = time.perf_counter()
            corrected = self._llm.process(current)
            dt = (time.perf_counter() - t_stage) * 1000
            logger.debug("Stage 4c (LLM): %.1f ms", dt)

            if corrected and corrected.strip():
                current = corrected
            else:
                logger.warning(
                    "LLM returned empty — keeping stage 4b output"
                )
        elif self._llm is not None and not self._llm.available:
            logger.debug("Stage 4c skipped — LLM model not loaded")
        elif not has_self_corrections and not self._config.llm_always:
            logger.debug("Stage 4c skipped — no self-corrections detected")

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Polish pipeline: %.1f ms total | %r → %r",
            total_ms,
            text[:60],
            current[:60],
        )
        return current
