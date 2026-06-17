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
from collections.abc import Callable

import config
from agent import state as S
from agent.servoing import Servoer
from brain.prompts import SEARCH_HINTS as _SEARCH_HINTS

# Actions returned by fast_step for the control thread to execute:
#   ("rc", lr, fb, ud, yaw) | ("rotate", deg) | ("move", direction, cm)
#   | ("hover",) | ("snapshot", label) | ("done",)
Action = tuple

_LOST_LIMIT = 25         # fast-ticks with no detection before approach → search

# ── in-room search pattern (within the geofence; never leaves the room) ───────
_SEARCH_YAW_STEP = 30        # degrees per discrete turn while sweeping a vantage
_SEARCH_DWELL_S = 0.7        # hold after each turn/step so the detector scans the new view
_SEARCH_STEP_CM = config.MAX_STEP_CM   # translation between vantage points
_SEARCH_MAX_VANTAGES = 4     # vantage points to try before giving up (room swept)
_SEARCH_DIRS = ("forward", "right", "back", "left")  # cycle so a blocked dir doesn't stall


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
        self._plan_failing = False        # whether the last VLM call failed (warn on edges only)

    # ── mission control ─────────────────────────────────────────────────────────
    def start_mission(self, goal: str) -> None:
        # Re-sending the same goal must not wipe an in-progress mission back to SEARCH.
        if self.state.active and goal.strip() == self.state.goal.strip():
            self.log(f"[brain] mission already running: {goal!r} (ignoring duplicate Send)")
            return
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
                if not self.state.steps:                 # one-time goal decomposition
                    steps = self.vlm.decompose(self.state.goal)
                    self.state.set_steps(steps)
                    if len(steps) > 1:
                        self.log(f"[brain] goal split into {len(steps)} steps: {steps}")
                dets = self.worker.detections if self.worker is not None else []
                decision = self.vlm.plan(
                    self.state.current_goal(), self.get_frame(), dets,
                    self.get_telemetry(), self.state.phase,
                )
                self._apply(decision)
                if self._plan_failing:                   # recovered — say so once
                    self._plan_failing = False
                    self.log("[brain] VLM reachable again — planning resumed.")
            except Exception as e:                       # Ollama down / bad JSON — keep flying
                if not self._plan_failing:               # warn on the falling edge only (no spam)
                    self._plan_failing = True
                    self.log(f"[brain] VLM unreachable, can't plan ({e}). Fast-loop search/"
                             "approach still runs on the last target; check Ollama.")
            time.sleep(config.VLM_INTERVAL_S)

    def _apply(self, d: dict) -> None:
        target = d.get("target") or []
        target = target if isinstance(target, list) else [target]
        target = [str(t).strip() for t in target if str(t).strip()]
        if target and target != self.state.target_queries:
            self.tools["set_target"].run({"queries": target})
            self.log(f"[brain] target → {target}")
        hint = str(d.get("search_hint") or "").strip().lower()
        self.state.search_hint = hint if hint in _SEARCH_HINTS else "around"
        scene = (d.get("scene") or "").strip()
        if scene and scene != self.state.scene:
            self.state.scene = scene
            self.log(f"[brain] scene: {scene}")
        msg = (d.get("message") or "").strip()
        if msg:
            self.state.message = msg
        if d.get("done"):
            if self.state.advance_step():            # more sub-goals left → pursue next
                self.log(f"[brain] step done → next: {self.state.current_goal()!r}")
            else:
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
            return self._search_step(st)

        if st.phase == S.APPROACH:
            if not dets:
                st.lost += 1
                if st.lost > _LOST_LIMIT:
                    st.phase = S.SEARCH
                    st.lost = 0
                    st.reset_search()
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
            label = st.target_queries[0] if st.target_queries else "agent"
            if st.advance_step():                        # more sub-goals → search for the next
                self.log(f"[brain] captured → next step: {st.current_goal()!r}")
            else:
                st.phase = S.DONE
                st.active = False
                st.done_reason = st.done_reason or "all steps complete"
                self.log("[brain] all steps complete")
            return ("snapshot", label)

        return ("hover",)

    def _search_step(self, st: S.MissionState) -> Action:
        """Controlled in-room search: sweep a vantage in discrete turns, then
        reposition to a new vantage (geofence refuses any step that would leave
        the room). Stops and reports once the room is swept — leaving the room is
        a future, obstacle-avoidance-gated phase, not done here.
        """
        now = time.monotonic()
        if now < st.search_dwell_until:
            return ("hover",)                         # hold still so the detector scans this view
        if st.search_swept_deg < 360:                 # keep turning in place
            st.search_swept_deg += _SEARCH_YAW_STEP
            st.search_dwell_until = now + _SEARCH_DWELL_S
            return ("rotate", _SEARCH_YAW_STEP)
        if st.search_vantages + 1 >= _SEARCH_MAX_VANTAGES:
            st.message = "target not found — room swept; hovering"
            if not st.search_exhausted:               # warn once, not every tick
                st.search_exhausted = True
                self.log("[brain] room swept, target not found — hovering. Reposition the "
                         "drone (manual), change the goal, or land.")
            return ("hover",)                         # room covered; wait for the operator
        # follow the VLM's scene-based hint if it named a translation direction,
        # otherwise cycle through directions so a geofenced one doesn't stall us
        hint = st.search_hint
        direction = hint if hint in _SEARCH_DIRS else _SEARCH_DIRS[st.search_vantages % len(_SEARCH_DIRS)]
        st.search_vantages += 1
        st.search_swept_deg = 0
        st.search_dwell_until = now + _SEARCH_DWELL_S
        self.log(f"[brain] swept, not found → reposition {direction} {_SEARCH_STEP_CM}cm")
        return ("move", direction, _SEARCH_STEP_CM)
