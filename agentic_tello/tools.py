"""tools.py — the single tool registry (one source of truth).

Every high-level action the agent loop takes — and, in Phase F, every action the
MCP server exposes — is defined here exactly once: a JSON schema (so it can be
advertised to a model or over MCP) bound to a handler that funnels through the
`ControlArbiter` / primitives. No actuation logic is duplicated elsewhere.

Build a registry with `build_registry(ctx)` and call `registry["set_target"].run({...})`.
Actuating tools (takeoff/land/move/rotate/emergency) must be invoked from the single
control thread; planning tools (set_target/report_done/get_*) are thread-safe.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentic_tello.tello_tools.primitives import get_telemetry as _get_telemetry
from agentic_tello.tello_tools.primitives import take_snapshot as _take_snapshot


@dataclass
class ToolContext:
    """Everything the handlers need — wired once at startup."""
    arbiter: Any          # ControlArbiter
    controller: Any       # TelloController
    worker: Any | None    # PerceptionWorker (may be None if detector failed to load)
    state: Any            # MissionState


class Tool:
    def __init__(self, name: str, description: str, parameters: dict,
                 handler: Callable[[dict], Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters          # JSON-schema object
        self._handler = handler

    def run(self, args: dict | None = None) -> Any:
        return self._handler(args or {})

    def schema(self) -> dict:
        """OpenAI / MCP-style function schema."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.parameters},
        }}


def build_registry(ctx: ToolContext) -> dict[str, Tool]:
    arb, c, st = ctx.arbiter, ctx.controller, ctx.state

    def takeoff(_):  arb.agent_takeoff();  return "took off"
    def land(_):     arb.agent_land();     return "landed"
    def move(a):     arb.agent_move(a["direction"], int(a["cm"]));  return f"moved {a['direction']} {a['cm']}cm"
    def rotate(a):   arb.agent_rotate(int(a["deg"]));               return f"rotated {a['deg']}deg"
    def emergency_stop(_): arb.emergency(); return "EMERGENCY STOP"

    def set_target(a):
        q = a.get("queries", [])
        q = q if isinstance(q, list) else [q]
        q = [s.strip() for s in q if str(s).strip()]
        st.target_queries = q
        if ctx.worker is not None:
            ctx.worker.set_queries(q)
        return f"target set to {q}"

    def take_snapshot(a):
        return _take_snapshot(c, a.get("label", "agent"))

    def get_telemetry(_):
        return _get_telemetry(c)

    def get_pose(_):
        """Dead-reckoning pose relative to takeoff (cm, heading clockwise deg)."""
        return {"x": arb.safe.x, "y": arb.safe.y, "heading": arb.safe.heading}

    def get_observation(_):
        dets = ctx.worker.detections if ctx.worker is not None else []
        return {"target": st.target_queries, "phase": st.phase, "detections": dets}

    def report_done(a):
        st.phase = "done"
        st.active = False
        st.done_reason = a.get("reason", "")
        return f"done: {st.done_reason}"

    defs = [
        Tool("takeoff", "Take off and hover.", {}, takeoff),
        Tool("land", "Land safely.", {}, land),
        Tool("move", "Move a discrete step (cm) in one direction.",
             {"direction": {"type": "string", "enum": ["forward", "back", "left", "right", "up", "down"]},
              "cm": {"type": "integer"}}, move),
        Tool("rotate", "Rotate clockwise by degrees (negative = counter-clockwise).",
             {"deg": {"type": "integer"}}, rotate),
        Tool("set_target", "Set the open-vocabulary object(s) the detector should localize.",
             {"queries": {"type": "array", "items": {"type": "string"}}}, set_target),
        Tool("take_snapshot", "Save a snapshot of the current camera frame.",
             {"label": {"type": "string"}}, take_snapshot),
        Tool("get_telemetry", "Read battery, height, attitude, etc.", {}, get_telemetry),
        Tool("get_pose", "Dead-reckoning pose (x, y, heading) relative to takeoff.", {}, get_pose),
        Tool("get_observation", "Current target, phase and detections.", {}, get_observation),
        Tool("report_done", "Declare the mission goal satisfied.",
             {"reason": {"type": "string"}}, report_done),
        Tool("emergency_stop", "Cut motors immediately (bypasses all guards).", {}, emergency_stop),
    ]
    return {t.name: t for t in defs}
