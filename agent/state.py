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
DONE = "done"          # goal satisfied → hover, wait for the operator


class MissionState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset("")

    def reset(self, goal: str) -> None:
        with self._lock:
            self.goal = goal
            self.phase = SEARCH
            self.target_queries: list[str] = []
            self.lost = 0          # consecutive fast-ticks the target has been out of view
            self.message = ""      # latest human-readable status from the brain
            self.done_reason = ""
            self.active = bool(goal)

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "goal": self.goal,
            "phase": self.phase,
            "target": ", ".join(self.target_queries),
            "message": self.message,
        }
