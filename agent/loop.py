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
import re
import threading
import time
from collections.abc import Callable

import config
from agent import state as S
from agent.servoing import Servoer
from brain.prompts import SEARCH_HINTS as _SEARCH_HINTS
from perception import markers

# Actions returned by fast_step for the control thread to execute:
#   ("rc", lr, fb, ud, yaw) | ("rotate", deg) | ("move", direction, cm)
#   | ("hover",) | ("snapshot", label) | ("done",)
Action = tuple

_LOST_LIMIT = 25         # fast-ticks with no detection before approach → search
_APPROACH_COAST_S = 0.4  # keep servoing toward the last box through brief detection dropouts
_WATCH_LEAVE_S = 3.0     # sustained absence (after a subject appeared) that ends a watch step
_CLIMB_TOL_CM = 10       # survey altitude counts as reached within this of the target height
_DEFAULT_MARKER_COUNT = 4  # how many colour markers a survey frames at once when unspecified

# ── in-room search pattern (within the geofence; never leaves the room) ───────
_SEARCH_YAW_STEP = 15        # degrees per discrete turn while sweeping a vantage; small so the
                             # narrow heading window where ALL markers co-appear isn't skipped
_SEARCH_DWELL_S = 1.2        # hold after each turn/step so the drone settles and the detector
                             # gets a stable frame before the next turn (avoid a rushed sweep)
_SEARCH_STEP_CM = config.MAX_STEP_CM   # translation between vantage points
_SEARCH_MAX_VANTAGES = 4     # vantage points to try before giving up (room swept)
_SEARCH_DIRS = ("forward", "right", "back", "left")  # cycle so a blocked dir doesn't stall


# ── deterministic step normalization (don't trust the VLM to emit the right fields) ──
# gemma3 frequently drops `count`/`approach` on a marker survey and forgets the climb, so
# the mission would approach the first marker instead of rotating, and never gain altitude.
# We re-derive these directly from the goal text, which is reliable.
def _markers_wanted(goal: str) -> int:
    """How many colour markers the goal asks to frame at once. Reads an integer that
    qualifies the marker noun ("los 4 cuadrados", "all 4 markers"); defaults to 4. The
    '\\d+ <marker-noun>' shape avoids picking up an altitude like '1.9 m'."""
    m = re.search(r"(\d+)\s+\w*\s*(cuadrad|marcador|marker|naranja|orange|square)", goal.lower())
    return int(m.group(1)) if m else _DEFAULT_MARKER_COUNT


def _target_height_cm(goal: str) -> int | None:
    """Absolute climb height (cm) parsed from the goal, or None. Only triggers when the
    goal actually asks to gain altitude (a climb verb), so a distance like 'within 1 m of
    the plant' is not mistaken for a height. Accepts '1.9 m', '1,9m', "1'9 m", '190 cm'."""
    g = goal.lower()
    if not re.search(r"sub[ei]|asciend|elev|altura|alto|rise|climb|height|up to|hover at", g):
        return None
    _m = r"m(?:et(?:er|ro)s?)?"   # m / meter(s) / metro(s)
    # Decimal metres FIRST and as a real decimal — '1,80 m' = 1.80 m = 180 cm (the prior
    # one-digit-only version fell through to the integer branch and mis-parsed '0 m' → 0).
    if m := re.search(rf"(\d+)\s*[.,']\s*(\d+)\s*{_m}\b", g):   # 1,80 m / 1.9 m / 1'9m
        h = round(float(f"{m.group(1)}.{m.group(2)}") * 100)
    elif m := re.search(r"(\d{2,3})\s*cm\b", g):               # 190 cm
        h = int(m.group(1))
    elif m := re.search(rf"\b(\d)\s*{_m}\b", g):               # 2 m (lone integer metres)
        h = int(m.group(1)) * 100
    else:
        return None
    return max(config.MIN_HEIGHT_CM, min(h, config.MAX_HEIGHT_CM))


def _normalize_steps(steps: list[dict], goal: str) -> list[dict]:
    """Make the marker-survey contract deterministic regardless of what the VLM emitted:
    colour-marker finds become fixed-vantage (approach=false) and require all N markers in
    view; and if the goal asks to gain altitude, a self-correcting `climb` to that absolute
    height is injected up front (replacing any relative 'move up' the VLM guessed)."""
    height = _target_height_cm(goal)
    n = _markers_wanted(goal)
    out: list[dict] = []
    for s in steps:
        if not isinstance(s, dict):
            out.append(s)
            continue
        t = s.get("type", "find")
        if t == "find" and markers.is_marker_query(s.get("object", "")):
            s["approach"] = False                      # frame from a vantage, never approach
            if int(s.get("count", 1) or 1) < n:
                s["count"] = n                          # wait for all markers, don't shoot on one
        # drop the VLM's altitude guesses; we inject an absolute climb below
        if height is not None and (t == "climb" or (t == "move" and s.get("direction") == "up")):
            continue
        out.append(s)
    if height is not None:
        out.insert(0, {"type": "climb", "height_cm": height})
    return out


def _is_marker_survey(step: dict) -> bool:
    """A fixed-vantage colour-marker find: the detector target is a marker query (HSV CV path)
    and we do NOT approach (explicit approach=false, or a multi-marker find which defaults to
    no-approach). Such a step is fully deterministic and needs no VLM planning."""
    if step.get("type", "find") != "find" or not markers.is_marker_query(step.get("object", "")):
        return False
    approach = step.get("approach")
    if approach is None:
        return int(step.get("count", 1) or 1) > 1     # multi-marker defaults to no-approach
    return not bool(approach)


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
    if t == "climb":
        return f"climb to {step.get('height_cm', config.MAX_HEIGHT_CM)}cm"
    if t == "return":
        return "return to start"
    if t == "watch":
        mode = "follow" if step.get("approach") else "hover"
        return (f"watch {step.get('object', '')} (every "
                f"{int(step.get('interval_s', 10))}s, {mode})").strip()
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
        self._ensure_planner()

    def start_survey(self, n: int = _DEFAULT_MARKER_COUNT, marker_query: str = "orange square",
                     height_cm: int | None = None, goal_text: str = "") -> None:
        """Arm the deterministic aerial **marker survey** directly — no VLM decomposition.

        This is the orchestrator's reliable entry point for "find the N colour markers": it
        installs the survey steps itself (optional climb to `height_cm`, then a fixed-vantage
        `find` of `marker_query` with `count=n, approach=false`), so the result never depends
        on the VLM extracting the right noun. The marker query goes verbatim to the detector
        (HSV colour CV in perception/markers.py); the planner skips VLM planning for it."""
        goal = goal_text.strip() or f"aerial marker survey: frame {n} {marker_query} marker(s)"
        if self.state.active and goal == self.state.goal.strip():
            self.log(f"[brain] survey already running: {goal!r} (ignoring duplicate)")
            return
        steps: list[dict] = []
        if height_cm:
            steps.append({"type": "climb", "height_cm": int(height_cm)})
        steps.append({"type": "find", "object": marker_query, "count": int(n), "approach": False})
        self.state.reset(goal)
        self.state.set_steps(steps)                      # planner sees steps → skips decompose
        self.log(f"[brain] marker survey armed (no VLM): {[_step_desc(s) for s in steps]}")
        self._ensure_planner()

    def _ensure_planner(self) -> None:
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
                    steps = _normalize_steps(steps, self.state.goal)  # enforce the survey contract
                    self.state.set_steps(steps)
                    if len(self.state.steps) > 1:
                        self.log("[brain] goal split into "
                                 f"{len(self.state.steps)} steps: "
                                 f"{[_step_desc(s) for s in self.state.steps]}")
                step = self.state.current_step()
                # Only "find" steps need the VLM (it picks the detector target). Maneuver
                # steps (rotate/move/return/unsupported) run deterministically in the fast
                # loop — planning them would only churn the detector target needlessly.
                # A fixed-vantage colour-marker survey is ALSO fully deterministic: the
                # detector is seeded with the marker query (HSV CV) and completion is the
                # count-based CAPTURE — so skip the VLM there too, or it could clobber the
                # pinned marker target with a different noun.
                if step is not None and step.get("type", "find") == "find" \
                        and not _is_marker_survey(step):
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
        if step.get("type") == "watch":                  # passive vigil + timed snapshots
            return self._watch_step(st, step)
        if step.get("type", "find") != "find":           # rotate / move / return / unsupported
            return self._maneuver_step(st, step)

        if not st.target_queries:
            # Seed the detector straight from the step's object so the find works even
            # before the first VLM plan (or with Ollama down) — same deterministic
            # targeting as watch/maneuver steps. The planner may still refine it while
            # SEARCHING. Critical for colour markers, whose name must reach the detector
            # verbatim to hit the HSV path.
            obj = (step.get("object") or "").strip()
            if not obj:
                return ("hover",)                        # truly ambiguous → wait for the VLM
            self.tools["set_target"].run({"queries": [obj]})
            self.log(f"[brain] find → target {obj!r} (seeded from step)")
            return ("hover",)

        dets = self.worker.detections if self.worker is not None else []

        if st.phase == S.SEARCH:
            # `count` = how many of the target must be in view at once (default 1).
            # `approach` = fly toward it before the shot; when omitted it DEFAULTS to
            # (need == 1): you cannot close in on several separate markers at once, so a
            # multi-marker find is inherently a fixed-vantage survey — frame them from
            # altitude and shoot WITHOUT approaching (the aerial-cenital case). This is
            # deterministic so the mission no longer depends on the VLM emitting
            # `approach:false`, which it frequently forgets.
            need = max(1, int(step.get("count", 1)))
            approach = bool(step["approach"]) if step.get("approach") is not None else (need == 1)
            # Best-effort: if the room was swept and N never co-appeared, shoot whatever
            # is framed anyway so the pipeline never hangs — the homography downstream is
            # the real validator (it rejects a bad frame and the orchestrator retakes).
            exhausted = (not approach) and st.search_exhausted
            if len(dets) >= need or exhausted:
                st.lost = 0
                self._last_det = None             # don't coast on a previous target's box
                if approach:
                    st.phase = S.APPROACH
                    self.log(f"[brain] target acquired ({len(dets)}/{need}) → approach")
                else:
                    st.phase = S.CAPTURE
                    self.log(f"[brain] {len(dets)} target(s) in view (need {need})"
                             + (" — room swept, best-effort" if exhausted else "")
                             + " → capture (no approach)")
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

    # ── fast loop: passive vigil with timed snapshots (no VLM needed) ─────────────
    def _watch_step(self, st: S.MissionState, step: dict) -> Action:
        """Wait in place for a subject to appear, then snapshot every `interval_s` for
        as long as it stays in view. The subject noun is explicit in the step, so the
        detector target is set deterministically — this step works even with Ollama down.
        With `approach`, the drone keeps the subject framed between shots (servoing);
        otherwise it holds a fixed hover. The step finishes once the subject, having
        appeared, stays out of view for `_WATCH_LEAVE_S`.
        """
        st.phase = S.WATCH
        obj = (step.get("object") or "person").strip()
        interval = float(step.get("interval_s", 10) or 10)
        approach = bool(step.get("approach", False))
        # Lock the detector onto the subject (no VLM round-trip — the noun is known).
        if st.target_queries != [obj]:
            self.tools["set_target"].run({"queries": [obj]})
            st.target_queries = [obj]
            self.log(f"[brain] watch → {obj} (snapshot every {interval:.0f}s, "
                     f"{'follow' if approach else 'hover'}); waiting for it to appear")

        dets = self.worker.detections if self.worker is not None else []
        now = time.monotonic()

        if not dets:
            if st.watch_seen:                            # it appeared earlier — has it left?
                if st.watch_lost_since == 0.0:
                    st.watch_lost_since = now
                elif now - st.watch_lost_since >= _WATCH_LEAVE_S:
                    self.log(f"[brain] {obj} left the view → watch step done")
                    self._complete_step()
            return ("hover",)                            # still waiting / brief dropout

        st.watch_lost_since = 0.0                         # present this tick
        if not st.watch_seen:
            st.watch_seen = True
            st.watch_next_snap = now                      # shoot immediately on first appearance
            self.log(f"[brain] {obj} appeared → snapshotting every {interval:.0f}s")

        if now >= st.watch_next_snap:
            st.watch_next_snap = now + interval           # hold still on the shot tick
            return ("snapshot", obj)
        if approach:                                      # keep it framed between shots
            frame = self.get_frame()
            if frame is not None:
                h, w = frame.shape[:2]
                lr, fb, ud, yaw, _ = self.servoer.step(dets[0], w, h)
                return ("rc", lr, fb, ud, yaw)
        return ("hover",)

    # ── fast loop: deterministic maneuver steps (no VLM, no detector) ─────────────
    def _maneuver_step(self, st: S.MissionState, step: dict) -> Action:
        t = step["type"]
        if t == "climb":
            return self._climb_step(st, step)
        if t == "rotate":
            deg = int(step.get("degrees", 90))
            signed = deg if step.get("direction") == "right" else -deg
            self.log(f"[brain] maneuver: rotate {signed:+d}°")
            self._complete_step()                        # one discrete turn, then next step
            return ("rotate", signed)
        if t == "move":
            # SafeTello clamps every move to MAX_STEP_CM, so a long move (e.g. crossing a
            # doorway, cm≈200) must be issued in chunks across ticks until it's covered.
            direction = step.get("direction", "forward")
            remaining = int(step.get("_remaining", step.get("cm", config.MAX_STEP_CM)))
            if remaining < config.MIN_STEP_CM:               # distance covered → next step
                self._complete_step()
                return ("hover",)
            chunk = min(config.MAX_STEP_CM, remaining)
            step["_remaining"] = remaining - chunk
            self.log(f"[brain] maneuver: move {direction} {chunk}cm "
                     f"({step['_remaining']}cm to go)")
            return ("move", direction, chunk)
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

    def _climb_step(self, st: S.MissionState, step: dict) -> Action:
        """Ascend to an ABSOLUTE height, self-correcting off measured telemetry each tick.
        Unlike a relative 'move up' (whose bookkeeping is lost if a chunk is refused at the
        cap), this re-reads the height every tick and keeps issuing up-steps until the target
        is reached or the cap leaves no room for a min-step — so it reliably gains altitude
        regardless of takeoff height."""
        target = min(int(step.get("height_cm", config.MAX_HEIGHT_CM)), config.MAX_HEIGHT_CM)
        h = (self.get_telemetry() or {}).get("height_cm")
        if h is None:                                    # no reading — one best-effort step, move on
            self.log("[brain] climb: no height telemetry — single best-effort ascent")
            self._complete_step()
            return ("move", "up", config.MAX_STEP_CM)
        room = config.MAX_HEIGHT_CM - h                  # headroom before the cap refuses a step
        if h >= target - _CLIMB_TOL_CM or room < config.MIN_STEP_CM:
            self.log(f"[brain] at survey altitude {h}cm (target {target}cm) → next step")
            self._complete_step()
            return ("hover",)
        chunk = min(config.MAX_STEP_CM, room, max(config.MIN_STEP_CM, target - h))
        self.log(f"[brain] climb: {h}cm → {target}cm (up {chunk}cm)")
        return ("move", "up", chunk)

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
        if st.search_swept_deg < 360:                 # keep turning in place (counter-clockwise)
            st.search_swept_deg += _SEARCH_YAW_STEP
            st.search_dwell_until = now + _SEARCH_DWELL_S
            return ("rotate", -_SEARCH_YAW_STEP)      # negative = anticlockwise (arbiter convention)
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
