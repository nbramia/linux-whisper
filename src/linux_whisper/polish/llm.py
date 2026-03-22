"""Stage 4c: LLM self-correction resolution and grammar repair (Qwen3 4B).

Only invoked when the disfluency stage (4a) flags self-correction patterns in
the transcript.  Uses llama-cpp-python to run a quantised GGUF model with a
narrowly-scoped system prompt that forbids paraphrasing.

If llama-cpp-python is not installed, the model file is missing, or inference
exceeds the timeout, the input is returned unchanged.
"""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
from threading import Thread
from typing import Any

from linux_whisper.config import MODELS_DIR, PolishConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional llama-cpp-python import
# ---------------------------------------------------------------------------
try:
    from llama_cpp import Llama

    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    Llama = None  # type: ignore[assignment,misc]
    logger.debug(
        "llama-cpp-python not installed; LLMCorrector will pass text through unchanged"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = MODELS_DIR / "llm"
_DEFAULT_MODEL_FILENAME = "Qwen3-4B-Q4_K_M.gguf"
_DEFAULT_TIMEOUT_MS = 3000  # generous for cold start; warm inference is ~300ms
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 256

# Focused system prompt — deliberately narrow.  Filler removal and punctuation
# are already handled by stages 4a / 4b; the LLM only resolves semantic
# self-corrections and grammar.
_SYSTEM_PROMPT = """\
You resolve self-corrections in dictated text. When someone changes their mind mid-sentence, keep ONLY the final version. Fix grammar. Output ONLY the result.

Examples:
"meet at 2, actually no at 4 on Friday" → "Meet at 4 on Friday."
"send to John, no wait send to Sarah instead" → "Send to Sarah."
"go with option A, actually option B is better" → "Go with option B."
"the deadline is Monday, I mean Tuesday" → "The deadline is Tuesday."
"move it to friday, actually no lets do it on monday" → "Let's do it on Monday."
/no_think"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LLMCorrector:
    """Resolve self-corrections and fix grammar via a local GGUF LLM.

    The corrector is intentionally conservative:

    * It only activates when the upstream disfluency detector flags
      self-corrections (unless ``llm_always`` is set).
    * It enforces a hard timeout (default 500 ms).  If the model doesn't
      finish in time the original text is returned.
    * Temperature is fixed at 0 for deterministic output.
    """

    def __init__(self, config: PolishConfig | None = None) -> None:
        self._config = config or PolishConfig()
        self._model: Any | None = None  # Llama instance
        self._loaded = False
        self._timeout_s = _DEFAULT_TIMEOUT_MS / 1000.0
        self._model_path: Path | None = self._resolve_model_path()

        # Don't load model at init — lazy-load on first use to save ~2.5GB RAM.
        # The model is only needed when self-corrections are detected.

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        if not _LLAMA_AVAILABLE:
            logger.info(
                "llama-cpp-python not available — LLM correction disabled"
            )
            return

        model_path = self._resolve_model_path()
        if model_path is None or not model_path.exists():
            logger.info(
                "LLM GGUF model not found at %s — LLM correction disabled",
                model_path,
            )
            return

        n_threads = self._config.llm_threads if self._config.llm_threads > 0 else max(1, os.cpu_count() or 4)

        try:
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=n_threads,
                n_threads_batch=n_threads,
                verbose=False,
            )
            self._loaded = True
            logger.info(
                "Loaded LLM model from %s (threads=%d)",
                model_path,
                n_threads,
            )
        except Exception:
            logger.exception("Failed to load LLM GGUF model")
            self._model = None

    def _resolve_model_path(self) -> Path | None:
        """Determine the GGUF model file path from config."""
        model_name = self._config.llm_model or "Qwen3-4B-Instruct-Q4_K_M"

        # If the model name looks like an absolute path, use it directly.
        if "/" in model_name or model_name.endswith(".gguf"):
            candidate = Path(model_name)
            if candidate.is_absolute():
                return candidate
            return _DEFAULT_MODEL_DIR / model_name

        # Otherwise, look for <name>.gguf in the default directory.
        return _DEFAULT_MODEL_DIR / f"{model_name}.gguf"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if the LLM can be loaded (package installed, model file exists)."""
        if not _LLAMA_AVAILABLE:
            return False
        return self._model_path is not None and self._model_path.exists()

    def _ensure_loaded(self) -> bool:
        """Lazy-load the model on first use. Returns True if ready."""
        if self._loaded and self._model is not None:
            return True
        if not self.available:
            return False
        self._try_load_model()
        return self._loaded

    def process(self, text: str, app_context: str | None = None) -> str:
        """Run LLM correction on *text*.

        Lazy-loads the model on first call (~1s). Returns the original
        text unchanged if the model can't load, times out, or hallucinates.

        Parameters
        ----------
        app_context:
            Optional context string describing the focused application,
            injected into the system prompt for tone adaptation.
        """
        if not text or not text.strip():
            return text

        if not self._ensure_loaded():
            logger.debug("LLM not available — returning text unchanged")
            return text

        result: str | None = None

        def _infer() -> None:
            nonlocal result
            try:
                result = self._run_inference(text, app_context=app_context)
            except Exception:
                logger.exception("LLM inference failed")
                result = None

        worker = Thread(target=_infer, daemon=True)
        worker.start()
        worker.join(timeout=self._timeout_s)

        if worker.is_alive():
            logger.warning(
                "LLM inference exceeded %.0f ms timeout — returning input unchanged",
                self._timeout_s * 1000,
            )
            return text

        if result is None or not result.strip():
            logger.warning("LLM returned empty output — returning input unchanged")
            return text

        # Sanity check: if the LLM output is vastly longer than the input it
        # likely hallucinated.  Reject outputs that are > 2x the input length.
        if len(result) > len(text) * 2:
            logger.warning(
                "LLM output suspiciously long (%d vs %d chars) — discarding",
                len(result),
                len(text),
            )
            return text

        return result.strip()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self, text: str, app_context: str | None = None
    ) -> str | None:
        """Build the prompt and call the model."""
        assert self._model is not None

        system_content = _SYSTEM_PROMPT
        if app_context:
            system_content = f"{_SYSTEM_PROMPT}\n\nContext: {app_context}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=_DEFAULT_MAX_TOKENS,
            temperature=_DEFAULT_TEMPERATURE,
            top_p=1.0,
            repeat_penalty=1.0,
            # Deterministic: no sampling
        )

        choices = response.get("choices", [])
        if not choices:
            return None

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str):
            return None
        # Strip Qwen3 <think>...</think> reasoning tags if present
        if "</think>" in content:
            content = content.split("</think>")[-1]
        return content.strip() or None
