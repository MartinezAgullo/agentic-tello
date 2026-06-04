"""AgentBrain — the dual-cadence sense-plan-act loop.

Two decoupled cadences, exactly as the architecture demands:

- **Slow loop** (`_planner` thread, every `VLM_INTERVAL_S`): calls the VLM to turn
  the goal into detector targets and to judge completion. It NEVER actuates — it
  only writes `MissionState` and sets detector queries (via the tool registry), so
  it is safe to run from its own thread even while the operator is in MANUAL.

- **Fast loop** (`fast_step`, called every control tick by the single actuation
  thread): purely deterministic. Reads the latest detections + `MissionState` and
  returns ONE `Action` for the caller to execute through the arbiter. No VLM here —
  this is the Phase-D servoing reflex driving the mission's phases.

Keeping all actuation in the caller's single thread preserves the "one chokepoint"
rule (djitellopy isn't safe for concurrent sends).
"""

import threading
import time
from typing import Callable

import config
from agent import state as S
from agent.servoing import Servoer

# Actions returned by fast_step for the control thread to execute:
#   ("rc", lr, fb, ud, yaw) | ("hover",) | ("snapshot", label) | ("done",)
Action = tuple

_LOST_LIMIT = 25        # fast-ticks with no detection before approach → search
_SCAN_YAW = 0.45        # fraction of SPEED used to rotate while searching


class AgentBrain:
    def __init__(self, vlm, worker, state: S.MissionState, tools: dict,
                 get_frame: Callable, get_telemetry: Callable,
                 servoer: Servoer | None = None, log: Callable[[str], None] = print) -> None:
        self.vlm = vlm
        self.worker = worker
        self.state = state
        self.tools = tools
        self.get_frame = get_frame
        self.get_telemetry = get_telemetry
        self.servoer = servoer or Servoer()
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── mission control ─────────────────────────────────────────────────────────
    def start_mission(self, goal: str) -> None:
        self.state.reset(goal)
        self.log(f"[brain] mission armed: {goal!r} (Arm AUTO to let it fly)")
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._planner, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self.state.active = False
        self._stop.set()

    @property
    def active(self) -> bool:
        return self.state.active

    # ── slow loop: VLM planning only (no actuation) ──────────────────────────────
    def _planner(self) -> None:
        while not self._stop.is_set():
            if not self.state.active:
                time.sleep(0.2)
                continue
            try:
                dets = self.worker.detections if self.worker is not None else []
                decision = self.vlm.plan(
                    self.state.goal, self.get_frame(), dets,
                    self.get_telemetry(), self.state.phase,
                )
                self._apply(decision)
            except Exception as e:                       # Ollama down / bad JSON — keep flying
                self.log(f"[brain] plan failed: {e}")
            time.sleep(config.VLM_INTERVAL_S)

    def _apply(self, d: dict) -> None:
        target = d.get("target") or []
        target = target if isinstance(target, list) else [target]
        target = [str(t).strip() for t in target if str(t).strip()]
        if target and target != self.state.target_queries:
            self.tools["set_target"].run({"queries": target})
            self.log(f"[brain] target → {target}")
        msg = (d.get("message") or "").strip()
        if msg:
            self.state.message = msg
        if d.get("done"):
            self.tools["report_done"].run({"reason": msg or "goal reached"})
            self.log(f"[brain] mission complete: {msg}")

    # ── fast loop: deterministic phase machine (caller actuates) ──────────────────
    def fast_step(self) -> Action:
        st = self.state
        if not st.active or st.phase == S.DONE:
            return ("hover",)
        if not st.target_queries:
            return ("hover",)                            # waiting for the first VLM plan

        dets = self.worker.detections if self.worker is not None else []

        if st.phase == S.SEARCH:
            if dets:
                st.phase = S.APPROACH
                st.lost = 0
                self.log("[brain] target acquired → approach")
                return ("hover",)
            return ("rc", 0, 0, 0, int(config.SPEED * _SCAN_YAW))   # rotate to scan

        if st.phase == S.APPROACH:
            if not dets:
                st.lost += 1
                if st.lost > _LOST_LIMIT:
                    st.phase = S.SEARCH
                    st.lost = 0
                    self.log("[brain] lost target → search")
                return ("hover",)
            st.lost = 0
            frame = self.get_frame()
            if frame is None:
                return ("hover",)
            h, w = frame.shape[:2]
            lr, fb, ud, yaw, done = self.servoer.step(dets[0], w, h)
            if done:
                st.phase = S.CAPTURE
                self.log("[brain] target centered → capture")
                return ("hover",)
            return ("rc", lr, fb, ud, yaw)

        if st.phase == S.CAPTURE:
            st.phase = S.DONE
            label = st.target_queries[0] if st.target_queries else "agent"
            return ("snapshot", label)

        return ("hover",)
