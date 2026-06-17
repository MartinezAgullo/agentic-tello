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

import math
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
_APPROACH_COAST_S = 0.4  # keep servoing toward the last box through brief detection dropouts

# ── in-room search pattern (within the geofence; never leaves the room) ───────
_SEARCH_YAW_STEP = 30        # degrees per discrete turn while sweeping a vantage
_SEARCH_DWELL_S = 0.7        # hold after each turn/step so the detector scans the new view
_SEARCH_STEP_CM = config.MAX_STEP_CM   # translation between vantage points
_SEARCH_MAX_VANTAGES = 4     # vantage points to try before giving up (room swept)
_SEARCH_DIRS = ("forward", "right", "back", "left")  # cycle so a blocked dir doesn't stall


def _step_desc(step: dict | None) -> str:
    """Short human label for a typed step (logs / UI)."""
    if not step:
        return "—"
    t = step.get("type", "find")
    if t == "find":
        return f"find {step.get('object', '')}".strip()
    if t == "rotate":
        return f"rotate {step.get('direction', 'left')} {step.get('degrees', 90)}°"
    if t == "move":
        return f"move {step.get('direction', 'forward')} {step.get('cm', 50)}cm"
    if t == "return":
        return "return to start"
    if t == "unsupported":
        return f"[unsupported] {step.get('text', '')}".strip()
    return t


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
        self._last_det: dict | None = None   # most recent box, for coasting through dropouts
        self._last_det_t = 0.0               # monotonic time of that box

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
                    if len(self.state.steps) > 1:
                        self.log("[brain] goal split into "
                                 f"{len(self.state.steps)} steps: "
                                 f"{[_step_desc(s) for s in self.state.steps]}")
                step = self.state.current_step()
                # Only "find" steps need the VLM (it picks the detector target). Maneuver
                # steps (rotate/move/return/unsupported) run deterministically in the fast
                # loop — planning them would only churn the detector target needlessly.
                if step is not None and step.get("type", "find") == "find":
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
        # Only (re)set the detector target while SEARCHING (or before the first target).
        # Letting the VLM churn the query every slow-tick mid-APPROACH re-embeds the
        # YOLO-World prompts and reshuffles "target #0", which destabilises detection and
        # makes the approach crawl. Once acquired, the target stays locked until lost.
        can_retarget = self.state.phase == S.SEARCH or not self.state.target_queries
        if target and target != self.state.target_queries and can_retarget:
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
            if self.state.done_reason == "" and msg:
                self.state.done_reason = msg
            if self._complete_step():                # more steps left → pursue next
                self.log(f"[brain] step done → next: {_step_desc(self.state.current_step())}")

    # ── fast loop: deterministic phase machine (caller actuates) ──────────────────
    def fast_step(self) -> Action:
        st = self.state
        if not st.active or st.phase == S.DONE:
            return ("hover",)

        step = st.current_step()
        if step is None:
            return ("hover",)
        if step.get("type", "find") != "find":           # rotate / move / return / unsupported
            return self._maneuver_step(st, step)

        if not st.target_queries:
            return ("hover",)                            # waiting for the first VLM plan

        dets = self.worker.detections if self.worker is not None else []

        if st.phase == S.SEARCH:
            if dets:
                st.phase = S.APPROACH
                st.lost = 0
                self._last_det = None             # don't coast on a previous target's box
                self.log("[brain] target acquired → approach")
                return ("hover",)
            return self._search_step(st)

        if st.phase == S.APPROACH:
            # Coast through brief detection dropouts: open-vocab boxes flicker, and hovering
            # on every missed frame is what made the approach crawl. Reuse the last box for a
            # short window so the servoer keeps closing distance instead of stalling.
            now = time.monotonic()
            if dets:
                self._last_det = dets[0]
                self._last_det_t = now
            det = dets[0] if dets else (
                self._last_det if self._last_det is not None
                and now - self._last_det_t <= _APPROACH_COAST_S else None)
            if det is None:
                st.lost += 1
                if st.lost > _LOST_LIMIT:
                    st.phase = S.SEARCH
                    st.lost = 0
                    self._last_det = None
                    st.reset_search()
                    self.log("[brain] lost target → search")
                return ("hover",)
            st.lost = 0
            frame = self.get_frame()
            if frame is None:
                return ("hover",)
            h, w = frame.shape[:2]
            lr, fb, ud, yaw, done = self.servoer.step(det, w, h)
            if done:
                st.phase = S.CAPTURE
                self.log("[brain] target centered → capture")
                return ("hover",)
            return ("rc", lr, fb, ud, yaw)

        if st.phase == S.CAPTURE:
            label = st.target_queries[0] if st.target_queries else "agent"
            if self._complete_step():                    # advance to the next step, or finish
                self.log(f"[brain] captured → next step: {_step_desc(st.current_step())}")
            return ("snapshot", label)

        return ("hover",)

    # ── shared step completion ────────────────────────────────────────────────────
    def _complete_step(self) -> bool:
        """Advance to the next step, or finish the mission if none remain. Returns True
        if a next step exists. Used by the find CAPTURE, the maneuver handlers, and the
        VLM's per-step `done`, so completion behaves identically whoever triggers it."""
        st = self.state
        if st.advance_step():
            self._last_det = None
            return True
        st.phase = S.DONE
        st.active = False
        st.done_reason = st.done_reason or "all steps complete"
        self.log("[brain] all steps complete")
        return False

    # ── fast loop: deterministic maneuver steps (no VLM, no detector) ─────────────
    def _maneuver_step(self, st: S.MissionState, step: dict) -> Action:
        t = step["type"]
        if t == "rotate":
            deg = int(step.get("degrees", 90))
            signed = deg if step.get("direction") == "right" else -deg
            self.log(f"[brain] maneuver: rotate {signed:+d}°")
            self._complete_step()                        # one discrete turn, then next step
            return ("rotate", signed)
        if t == "move":
            direction, cm = step.get("direction", "forward"), int(step.get("cm", 50))
            self.log(f"[brain] maneuver: move {direction} {cm}cm")
            self._complete_step()
            return ("move", direction, cm)
        if t == "return":
            return self._return_step(st)
        # unsupported (leave room / cross door / inter-room navigation) — skip with a note
        if not step.get("_warned"):
            step["_warned"] = True
            st.message = f"skipped (not supported yet): {step.get('text', '')}"
            self.log("[brain] unsupported step skipped — needs obstacle avoidance / room "
                     f"egress (Phase G/H, not built): {step.get('text', '')!r}")
        self._complete_step()
        return ("hover",)

    def _return_step(self, st: S.MissionState) -> Action:
        """Dead-reckon back toward the takeoff point with discrete body-frame moves.
        Straight-line, no obstacle awareness. Finishes when within one min-step of home."""
        pose = self._pose()
        if pose is None:                                 # no pose source — can't return
            self.log("[brain] return: no pose available, skipping")
            self._complete_step()
            return ("hover",)
        x, y, heading = pose["x"], pose["y"], pose["heading"]
        if math.hypot(x, y) <= config.MIN_STEP_CM:
            self.log(f"[brain] returned to start (within {config.MIN_STEP_CM}cm)")
            self._complete_step()
            return ("hover",)
        # project the home-ward vector (-x, -y) onto the body axes (heading clockwise)
        rad = math.radians(heading)
        f_comp = -x * math.sin(rad) - y * math.cos(rad)  # forward(+) / back(-)
        r_comp = -x * math.cos(rad) + y * math.sin(rad)  # right(+)  / left(-)
        if abs(f_comp) >= abs(r_comp):
            direction, mag = ("forward" if f_comp > 0 else "back"), abs(f_comp)
        else:
            direction, mag = ("right" if r_comp > 0 else "left"), abs(r_comp)
        cm = int(max(config.MIN_STEP_CM, min(config.MAX_STEP_CM, mag)))
        return ("move", direction, cm)

    def _pose(self) -> dict | None:
        tool = self.tools.get("get_pose")
        if tool is None:
            return None
        try:
            return tool.run({})
        except Exception:
            return None

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
