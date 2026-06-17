"""VLMClient — talks to a local VLM through Ollama's native API.

The model is kept warm in Ollama (`keep_alive`) and called sparingly (the slow
loop). Frames are downscaled before sending. The reply is parsed as a strict JSON
plan (see `brain.prompts`); we tolerate stray code fences / surrounding text.

Why the native `/api/chat` and not the OpenAI-compatible `/v1` endpoint: only the
native API honours `options.num_ctx`. Left at the model's default (e.g. 262k for
qwen3-vl) Ollama reserves a huge KV cache — ~48 GB VRAM and ~50 s per call. Capping
the context (we send one image + a short prompt) drops that to ~7 GB and a few
seconds. The OpenAI endpoint silently ignored the cap, so we go native.

Model + host + context are configurable (see `config.py` / env vars) so the brain
can be swapped for newer models without code changes.
"""

import json

import cv2
import numpy as np
import ollama

import config
from brain.prompts import (
    DECOMPOSE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_decompose_prompt,
    build_user_prompt,
)


def _keep_alive():
    ka = config.VLM_KEEP_ALIVE
    try:
        return int(ka)          # "-1" → -1 (forever); Ollama also accepts "5m" etc.
    except (TypeError, ValueError):
        return ka


class VLMClient:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.client = ollama.Client(host=host or config.OLLAMA_HOST)
        self.model = model or config.VLM_MODEL

    def _encode(self, frame: np.ndarray) -> bytes:
        w = config.VLM_FRAME_W
        if frame.shape[1] > w:
            h = int(frame.shape[0] * w / frame.shape[1])
            frame = cv2.resize(frame, (w, h))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    def decompose(self, goal: str) -> list[str]:
        """Split a goal into ordered single-target sub-goals (text-only, run once).

        Returns at least one step; falls back to ``[goal]`` if the model returns
        nothing usable, so a simple mission behaves exactly as before.
        """
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                      {"role": "user", "content": build_decompose_prompt(goal)}],
            format="json",
            keep_alive=_keep_alive(),
            options={"temperature": 0.0, "num_ctx": config.VLM_NUM_CTX},
        )
        data = _parse_json(resp["message"]["content"] or "")
        steps = data.get("steps") or []
        steps = [str(s).strip() for s in steps if str(s).strip()]
        return steps or [goal]

    def plan(self, goal: str, frame: np.ndarray | None, detections: list[dict],
             telemetry: dict, phase: str) -> dict:
        """One planning step → parsed JSON decision (see brain.prompts)."""
        user = {"role": "user",
                "content": build_user_prompt(goal, detections, telemetry, phase)}
        if frame is not None and frame.size:
            user["images"] = [self._encode(frame)]

        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, user],
            format="json",
            keep_alive=_keep_alive(),
            options={"temperature": 0.2, "num_ctx": config.VLM_NUM_CTX},
        )
        return _parse_json(resp["message"]["content"] or "")


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):                 # strip ```json … ``` fences
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")  # last-ditch: first {...} block
        if 0 <= i < j:
            return json.loads(text[i:j + 1])
        raise
