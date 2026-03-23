"""Isolated GPU worker for whisper.cpp STT inference.

Runs in a separate process to avoid the pywhispercpp/onnxruntime ROCm
shared-library conflict.  The main process communicates via a pipe:

    Main process                    Worker process
    (onnxruntime OK)                (pywhispercpp OK)
    ──────────────                  ──────────────────
    send(audio_bytes)  ──────────→  recv → transcribe
                       ←──────────  send(segments, text)

This module must NOT import onnxruntime, numpy from the main package,
or anything that pulls in conflicting ROCm libraries.
"""

from __future__ import annotations

import logging
import os
import sys
from multiprocessing.connection import Connection

# Configure basic logging for the worker process
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("whisper_gpu_worker")

# Commands sent from main → worker
CMD_LOAD = "load"
CMD_TRANSCRIBE = "transcribe"
CMD_SHUTDOWN = "shutdown"


def worker_main(conn: Connection, model_path: str, n_threads: int) -> None:
    """Entry point for the GPU worker process.

    Loads the whisper.cpp model once, then loops handling transcription
    requests until a shutdown command is received.
    """
    try:
        _run(conn, model_path, n_threads)
    except Exception:
        logger.exception("Worker crashed")
    finally:
        conn.close()


def _run(conn: Connection, model_path: str, n_threads: int) -> None:
    import numpy as np
    from pywhispercpp.model import Model

    logger.info("Loading whisper.cpp model: %s (threads=%d)", model_path, n_threads)
    model = Model(model_path, n_threads=n_threads, redirect_whispercpp_logs_to=False)
    logger.info("Model loaded, GPU worker ready")

    # Signal ready
    conn.send({"status": "ready"})

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            logger.info("Connection closed, shutting down")
            break

        cmd = msg.get("cmd")

        if cmd == CMD_SHUTDOWN:
            logger.info("Shutdown requested")
            break

        if cmd == CMD_TRANSCRIBE:
            pcm_bytes = msg["audio"]
            # Convert int16 PCM to float32
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            duration = len(audio) / 16000.0

            logger.debug("Transcribing %.1fs of audio...", duration)

            try:
                segments = model.transcribe(audio)
                result_segs = []
                full_parts = []
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        result_segs.append({
                            "text": text,
                            "t0": seg.t0 / 100.0,
                            "t1": seg.t1 / 100.0,
                        })
                        full_parts.append(text)

                conn.send({
                    "status": "ok",
                    "segments": result_segs,
                    "full_text": " ".join(full_parts),
                    "duration": duration,
                })
            except Exception as e:
                logger.exception("Transcription failed")
                conn.send({
                    "status": "error",
                    "error": str(e),
                    "duration": duration,
                })
        else:
            logger.warning("Unknown command: %s", cmd)
