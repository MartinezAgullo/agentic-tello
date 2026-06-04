"""VLMClient — talks to a local VLM through Ollama's OpenAI-compatible API.

The model is kept warm in Ollama (`keep_alive`) and called sparingly (the slow
loop). Frames are downscaled before sending. The reply is parsed as a strict JSON
plan (see `brain.prompts`); we tolerate stray code fences / surrounding text.

Model + host are configurable (see `config.py` / env vars) so the brain can be
swapped for newer models without code changes.
"""

import base64
import json

import cv2
import numpy as np
from openai import OpenAI

import config
from brain.prompts import SYSTEM_PROMPT, build_user_prompt


def _keep_alive():
    ka = config.VLM_KEEP_ALIVE
    try:
        return int(ka)          # "-1" → -1 (forever); Ollama also accepts "5m" etc.
    except (TypeError, ValueError):
        return ka


class VLMClient:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        base_url = (host or config.OLLAMA_HOST).rstrip("/") + "/v1"
        self.client = OpenAI(base_url=base_url, api_key="ollama")  # key unused by Ollama
        self.model = model or config.VLM_MODEL

    def _encode(self, frame: np.ndarray) -> str:
        w = config.VLM_FRAME_W
        if frame.shape[1] > w:
            h = int(frame.shape[0] * w / frame.shape[1])
            frame = cv2.resize(frame, (w, h))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode()

    def plan(self, goal: str, frame: np.ndarray | None, detections: list[dict],
             telemetry: dict, phase: str) -> dict:
        """One planning step → parsed JSON decision (see brain.prompts)."""
        content: list[dict] = [
            {"type": "text", "text": build_user_prompt(goal, detections, telemetry, phase)},
        ]
        if frame is not None and frame.size:
            url = f"data:image/jpeg;base64,{self._encode(frame)}"
            content.append({"type": "image_url", "image_url": {"url": url}})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            extra_body={"keep_alive": _keep_alive()},
        )
        return _parse_json(resp.choices[0].message.content or "")


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
