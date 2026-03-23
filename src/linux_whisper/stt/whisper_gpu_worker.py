#!/usr/bin/env python3
"""Isolated GPU worker for whisper.cpp STT inference.

Runs as a completely separate subprocess.  CRITICAL: pywhispercpp must
be imported BEFORE numpy to avoid a ROCm shared-library segfault.

Communication: length-prefixed JSON over stdin/stdout pipes, with raw
audio bytes sent inline after transcribe commands.
"""

# ── IMPORT ORDER MATTERS ──────────────────────────────────────────────
# pywhispercpp's ROCm/HIP C extension segfaults if numpy is loaded first.
# Import it at the top, before anything else.
from pywhispercpp.model import Model as _WhisperModel  # noqa: E402,F401

import json
import logging
import struct
import sys

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("whisper_gpu_worker")


# ── Wire protocol ────────────────────────────────────────────────────

def _recv_msg(stdin) -> dict | None:
    header = stdin.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack(">I", header)[0]
    data = stdin.read(length)
    if len(data) < length:
        return None
    return json.loads(data)


def _send_msg(stdout, msg: dict) -> None:
    data = json.dumps(msg).encode()
    stdout.write(struct.pack(">I", len(data)))
    stdout.write(data)
    stdout.flush()


# ── Main loop ────────────────────────────────────────────────────────

def main() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # First message: init with model path
    init_msg = _recv_msg(stdin)
    if init_msg is None or init_msg.get("cmd") != "init":
        logger.error("Expected init message, got: %s", init_msg)
        return

    model_path = init_msg["model_path"]
    n_threads = init_msg.get("n_threads", 4)

    logger.info("Loading model: %s (threads=%d)", model_path, n_threads)
    model = _WhisperModel(
        model_path, n_threads=n_threads, redirect_whispercpp_logs_to=False
    )
    logger.info("Model loaded — GPU worker ready")

    _send_msg(stdout, {"status": "ready"})

    while True:
        msg = _recv_msg(stdin)
        if msg is None:
            break

        cmd = msg.get("cmd")
        if cmd == "shutdown":
            logger.info("Shutdown")
            break

        if cmd == "transcribe":
            audio_len = msg["audio_length"]
            pcm_bytes = stdin.read(audio_len)
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            duration = len(audio) / 16000.0

            logger.info("Transcribing %.1fs audio...", duration)
            try:
                segs = model.transcribe(audio)
                out_segs = []
                parts = []
                for s in segs:
                    t = s.text.strip()
                    if t:
                        out_segs.append(
                            {"text": t, "t0": s.t0 / 100.0, "t1": s.t1 / 100.0}
                        )
                        parts.append(t)
                _send_msg(stdout, {
                    "status": "ok",
                    "segments": out_segs,
                    "full_text": " ".join(parts),
                    "duration": duration,
                })
            except Exception as e:
                logger.exception("Transcription failed")
                _send_msg(stdout, {
                    "status": "error",
                    "error": str(e),
                    "duration": duration,
                })


if __name__ == "__main__":
    main()
