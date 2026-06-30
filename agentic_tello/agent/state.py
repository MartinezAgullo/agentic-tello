"""MissionState — the small shared blackboard between the slow (VLM) and fast loops.

The VLM planner writes *what* to pursue (target object names, completion); the fast
deterministic loop reads it and decides *how* to fly (search → approach → capture).
Fields are simple scalars/lists (assignment is atomic under the GIL); `reset` takes
the lock so a mission start is consistent.
"""

import threading

# mission phases
SEARCH = "search"      # target named but not yet in view → scan
APPROACH = "approach"  # target visible → center + close in (deterministic servoing)
CAPTURE = "capture"    # centered & close → grab a snapshot
WATCH = "watch"        # passive vigil → wait for a subject, then snapshot on a timer
DONE = "done"          # goal satisfied → hover, wait for the operator


class MissionState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset("")

    def reset(self, goal: str) -> None:
        with self._lock:
            self.goal = goal
            self.steps: list[dict] = []  # ordered TYPED sub-steps (decomposed by the VLM once)
            self.step_idx = 0            # which sub-step is active
            self.phase = SEARCH
            self.target_queries: list[str] = []
            self.lost = 0          # consecutive fast-ticks the target has been out of view
            self.message = ""      # latest human-readable status from the brain
            self.done_reason = ""
            self.active = bool(goal)
            self.search_hint = "around"  # VLM advice on where to explore next (see prompts)
            self.scene = ""              # VLM's read of the space (doorways/windows/hazards)
            self.reset_search()
            self.reset_watch()

    # ── multi-step goal handling ──────────────────────────────────────────────
    def set_steps(self, steps: list[dict]) -> None:
        """Install the ordered typed sub-steps (the VLM's one-time decomposition)."""
        norm: list[dict] = []
        for s in steps:
            if isinstance(s, dict) and s.get("type"):
                norm.append(s)
            elif isinstance(s, str) and s.strip():       # tolerate the old string format
                norm.append({"type": "find", "object": s.strip(), "text": s.strip()})
        with self._lock:
            self.steps = norm
            self.step_idx = 0

    def current_step(self) -> dict | None:
        """The active typed sub-step (None if there are no steps / index out of range)."""
        if self.steps and 0 <= self.step_idx < len(self.steps):
            return self.steps[self.step_idx]
        return None

    def current_goal(self) -> str:
        """Object phrase for the active find step (used by the VLM planner / logs)."""
        st = self.current_step()
        if st is None:
            return self.goal
        if st.get("type") == "find":
            return st.get("object") or st.get("text") or self.goal
        return st.get("text") or self.goal

    def advance_step(self) -> bool:
        """Mark the current sub-goal done and move to the next. Returns True if a
        next step exists (re-armed to SEARCH with a cleared target), False if the
        whole mission is finished (caller declares DONE)."""
        with self._lock:
            if self.step_idx + 1 < len(self.steps):
                self.step_idx += 1
                self.phase = SEARCH
                self.target_queries = []
                self.lost = 0
                self.reset_search()
                self.reset_watch()
                return True
            return False

    def reset_search(self) -> None:
        """Clear the in-room search bookkeeping (call whenever (re)entering SEARCH)."""
        self.search_swept_deg = 0      # degrees yawed at the current vantage point
        self.search_vantages = 0       # vantage points visited this search
        self.search_dwell_until = 0.0  # monotonic time until the detector has scanned a view
        self.search_exhausted = False  # room fully swept, target not found (warn once)

    def reset_watch(self) -> None:
        """Clear the in-place vigil bookkeeping (call whenever (re)entering a watch step)."""
        self.watch_seen = False        # subject has appeared at least once this step
        self.watch_next_snap = 0.0     # monotonic time the next snapshot is due
        self.watch_lost_since = 0.0    # monotonic time the subject (after appearing) went absent

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "goal": self.goal,
            "phase": self.phase,
            "step": f"{self.step_idx + 1}/{len(self.steps)}" if self.steps else "",
            "target": ", ".join(self.target_queries),
            "message": self.message,
            "scene": self.scene,
        }
