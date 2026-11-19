"""Stage 4c: LLM self-correction resolution and grammar repair (Qwen3 4B).

Only invoked when the disfluency stage (4a) flags self-correction patterns in
the transcript.  Uses llama-cpp-python to run a quantised GGUF model with a
narrowly-scoped system prompt that forbids paraphrasing.

If llama-cpp-python is not installed, the model file is missing, or inference
exceeds the timeout, the input is returned unchanged.

Model lifecycle
---------------
Loading the model takes seconds, so it never happens inside a dictation if it
can be helped.  :meth:`LLMCorrector.preload` loads it on a daemon thread; the
app calls it once it is ready and again at every recording start, so the model
is warm before the first self-correction arrives.  :meth:`LLMCorrector.process`
waits for an in-progress load for at most the inference timeout and otherwise
returns the input unchanged.  After ``polish.llm_idle_unload_s`` seconds
without a call (0 = never) the model is closed to free its memory and GPU
queue; the next recording start reloads it in the background.  This mirrors
the GPU whisper worker's idle unload in ``stt/whisper_gpu.py``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import weakref
from pathlib import Path
from threading import Thread
from typing import Any

from linux_whisper.config import MODELS_DIR, PolishConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional llama-cpp-python import
# ---------------------------------------------------------------------------
try:
    from llama_cpp import Llama, llama_supports_gpu_offload

    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    Llama = None  # type: ignore[assignment,misc]
    llama_supports_gpu_offload = None  # type: ignore[assignment]
    logger.debug(
        "llama-cpp-python not installed; LLMCorrector will pass text through unchanged"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = MODELS_DIR / "llm"
_DEFAULT_MODEL_FILENAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
_DEFAULT_TIMEOUT_MS = 3000  # generous for cold start; warm inference is ~300ms
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 256

# Focused system prompt — deliberately narrow.  Filler removal and punctuation
# are already handled by stages 4a / 4b; the LLM only resolves semantic
# self-corrections and grammar.
#
# No thinking-suppression suffix is needed: the default model is an
# instruct-only checkpoint with no reasoning mode to switch off.  If you point
# ``polish.llm_model`` at a hybrid-reasoning model (the original Qwen3-4B, for
# instance) it may emit reasoning traces into the output — the benchmark
# harness counts these as ``thinking_leaks``.
#
# The "delete nothing else" rule is load-bearing.  Without it the instruct
# checkpoint compresses beyond the self-correction — dropping qualifiers like
# "just" and leading subjects like "Let's" — which reads as a summary rather
# than a transcript.
_SYSTEM_PROMPT = (
    # Split purely to satisfy the 100-column limit — the assembled string is
    # byte-identical to the single line it replaces, and is asserted as such
    # in tests. This prompt is benchmark-gated; do not reword it casually.
    "You resolve self-corrections in dictated text. When someone changes "
    "their mind mid-sentence, keep ONLY the final version. Fix grammar. "
    "Output ONLY the result.\n"
) + """\

Delete nothing except the abandoned half of a self-correction. Keep every
other word, including qualifiers and the opening subject. Never shorten,
summarise, or rephrase.

Examples:
"meet at 2, actually no at 4 on Friday" → "Meet at 4 on Friday."
"send to John, no wait send to Sarah instead" → "Send to Sarah."
"go with option A, actually option B is better" → "Go with option B."
"the deadline is Monday, I mean Tuesday" → "The deadline is Tuesday."
"move it to friday, actually no lets do it on monday" → "Let's do it on Monday."
"the deploy just needs another review" → "The deploy just needs another review."
"let's meet at two, actually make it four" → "Let's meet at four."
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


def _idle_unload_callback(corrector_ref: weakref.ref[LLMCorrector], generation: int) -> None:
    """Timer entry point that does not keep a discarded corrector alive."""
    corrector = corrector_ref()
    if corrector is not None:
        corrector._unload_if_idle(generation)


class LLMCorrector:
    """Resolve self-corrections and fix grammar via a local GGUF LLM.

    The corrector is intentionally conservative:

    * It only activates when the upstream disfluency detector flags
      self-corrections (unless ``llm_always`` is set).
    * It enforces a hard timeout (default 3 s).  If the model doesn't
      finish in time the original text is returned.
    * Temperature is fixed at 0 for deterministic output.
    """

    def __init__(self, config: PolishConfig | None = None) -> None:
        self._config = config or PolishConfig()
        self._model: Any | None = None  # Llama instance
        self._loaded = False
        self._timeout_s = _DEFAULT_TIMEOUT_MS / 1000.0
        self._model_path: Path | None = self._resolve_model_path()

        # Nothing is loaded at init: the app calls ``preload`` once it is
        # ready, on a background thread, so construction stays cheap.
        #
        # ``_lock`` guards the model reference, the load thread, the in-flight
        # count and the idle timer against the timer and load threads.  It is
        # never held across a model load or an inference, so ``preload`` on
        # the hotkey thread cannot wait behind either.
        self._lock = threading.RLock()
        self._load_thread: Thread | None = None
        self._load_done = threading.Event()
        # Inferences started and not yet finished, including ones whose
        # caller already gave up on the timeout.  Blocks idle unload.
        self._inflight = 0
        self._closed = False
        # Set when a load raised: retrying on every recording start would
        # re-run a failing multi-second GPU load each time.
        self._load_failed = False
        self._idle_unload_s: float = self._config.llm_idle_unload_s
        self._idle_timer: threading.Timer | None = None
        # Bumped on every arm/cancel so a timer that fired just before being
        # cancelled recognises itself as stale once it gets the lock.
        self._idle_generation = 0

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

        n_threads = (
            self._config.llm_threads
            if self._config.llm_threads > 0
            else max(1, os.cpu_count() or 4)
        )

        # Determine GPU offload based on config
        n_gpu_layers = 0
        device_label = "CPU"
        if self._config.llm_device == "rocm":
            if llama_supports_gpu_offload and llama_supports_gpu_offload():
                n_gpu_layers = -1  # offload all layers
                device_label = "ROCm GPU"
            else:
                logger.warning(
                    "llm_device='rocm' but GPU offload not available — falling back to CPU"
                )

        t0 = time.perf_counter()
        try:
            model = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=n_threads,
                n_threads_batch=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception:
            logger.exception("Failed to load LLM GGUF model — LLM correction disabled")
            with self._lock:
                self._load_failed = True
            return

        with self._lock:
            if self._closed:
                # ``close`` ran while the load was in progress.
                self._close_model(model)
                return
            self._model = model
            self._loaded = True
            self._arm_idle_timer()
        logger.info(
            "Loaded LLM model from %s in %.0f ms (device=%s, threads=%d, gpu_layers=%d)",
            model_path,
            (time.perf_counter() - t0) * 1000,
            device_label,
            n_threads,
            n_gpu_layers,
        )

    def _resolve_model_path(self) -> Path | None:
        """Determine the GGUF model file path from config."""
        # Falls back to the packaged default filename rather than a literal:
        # the two drifted apart previously, so an empty ``llm_model`` resolved
        # to a file that had never existed.
        model_name = self._config.llm_model or _DEFAULT_MODEL_FILENAME

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

    @property
    def loaded(self) -> bool:
        """True if the model is resident and ready for inference."""
        return self._loaded and self._model is not None

    def preload(self) -> bool:
        """Start loading the model on a daemon thread; never blocks.

        A no-op when the model is already loaded or a load is running.
        Returns True if the model is loaded or loading, False if it cannot be
        (package missing, model file absent, an earlier load failed, corrector
        closed).
        """
        with self._lock:
            if self._closed or self._load_failed:
                return False
            if self.loaded:
                return True
            if self._load_thread is not None and self._load_thread.is_alive():
                return True
            if not self.available:
                return False
            done = threading.Event()
            thread = Thread(
                target=self._background_load, args=(done,), name="llm-preload", daemon=True
            )
            self._load_done = done
            self._load_thread = thread
            thread.start()
        logger.info("Loading LLM model in the background")
        return True

    def _background_load(self, done: threading.Event) -> None:
        try:
            self._try_load_model()
        finally:
            done.set()

    def wait_until_loaded(self, timeout: float | None) -> bool:
        """Block up to *timeout* seconds (None = forever) for a running load.

        Returns True if the model is loaded afterwards.  Does not start a load.
        """
        with self._lock:
            if self.loaded:
                return True
            if self._load_thread is None:
                return False
            done = self._load_done
        done.wait(timeout)
        return self.loaded

    def _ensure_loaded(self) -> bool:
        """Make sure a load is under way and wait up to the timeout for it."""
        if self.loaded:
            return True
        if not self.preload():
            return False
        if self.wait_until_loaded(self._timeout_s):
            return True
        logger.warning(
            "LLM model still loading after %.0f ms — returning input unchanged",
            self._timeout_s * 1000,
        )
        return False

    def close(self) -> None:
        """Stop the idle timer and release the model. Safe to call more than once."""
        with self._lock:
            self._closed = True
            self._cancel_idle_timer()
            model, self._model = self._model, None
            self._loaded = False
        if model is not None:
            self._close_model(model)

    # ------------------------------------------------------------------
    # Idle unload
    # ------------------------------------------------------------------

    def _arm_idle_timer(self) -> None:
        """(Re)start the idle countdown; caller holds ``_lock``.

        A no-op while an inference is in flight: the countdown restarts when
        the last one finishes.
        """
        self._cancel_idle_timer()
        if (
            self._idle_unload_s <= 0
            or self._model is None
            or self._inflight
            or self._closed
        ):
            return
        timer = threading.Timer(
            self._idle_unload_s,
            _idle_unload_callback,
            args=(weakref.ref(self), self._idle_generation),
        )
        timer.daemon = True
        timer.name = "llm-idle-unload"
        self._idle_timer = timer
        timer.start()

    def _cancel_idle_timer(self) -> None:
        """Stop any pending idle countdown; caller holds ``_lock``."""
        self._idle_generation += 1
        timer, self._idle_timer = self._idle_timer, None
        if timer is not None:
            timer.cancel()

    def _unload_if_idle(self, generation: int) -> None:
        with self._lock:
            if generation != self._idle_generation or self._inflight or self._model is None:
                return
            model, self._model = self._model, None
            self._loaded = False
            self._cancel_idle_timer()
        logger.info("Unloading idle LLM model after %ss idle", self._idle_unload_s)
        self._close_model(model)

    @staticmethod
    def _close_model(model: Any) -> None:
        close = getattr(model, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            logger.exception("Failed to close LLM model")

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def process(self, text: str, app_context: str | None = None) -> str:
        """Run LLM correction on *text*.

        Uses the preloaded model, or waits up to the timeout for a load in
        progress (starting one if none is).  Returns the original text
        unchanged if the model isn't ready in time, inference times out, or
        the output looks hallucinated.

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

        with self._lock:
            if self._model is None:  # unloaded between the check and here
                return text
            self._inflight += 1
            self._cancel_idle_timer()

        result: str | None = None

        def _infer() -> None:
            nonlocal result
            try:
                result = self._run_inference(text, app_context=app_context)
            except Exception:
                logger.exception("LLM inference failed")
                result = None
            finally:
                with self._lock:
                    self._inflight -= 1
                    self._arm_idle_timer()

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
