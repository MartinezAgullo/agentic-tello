"""ControlArbiter — the one place every command funnels through.

Two modes:
  AUTO    the agent loop drives (agent_* calls execute)
  MANUAL  the operator drives (manual_* calls execute); agent_* are blocked

Operator input always wins: any manual_* call flips to MANUAL immediately, so
the agent can never fight the human for the sticks. Re-arming AUTO is explicit
(arm_auto), so the agent never silently grabs control mid-manual. Emergency
bypasses the mode gate entirely.
"""

import threading

from tello_tools.safety import SafeTello


class ArbiterBlocked(Exception):
    """Raised when an agent command is rejected because the mode is MANUAL."""


class ControlArbiter:
    AUTO = "AUTO"
    MANUAL = "MANUAL"

    def __init__(self, safe: SafeTello) -> None:
        self.safe = safe
        self.mode = self.MANUAL   # start under human control; agent must be armed
        self._lock = threading.Lock()

    # ── mode control ──────────────────────────────────────────────────────────
    def arm_auto(self) -> None:
        with self._lock:
            self.safe.hover() if self.safe.flying else None
            self.mode = self.AUTO

    def to_manual(self) -> None:
        with self._lock:
            self.mode = self.MANUAL
            if self.safe.flying:
                self.safe.hover()   # stop whatever the agent was doing

    # ── agent-facing (AUTO only) ──────────────────────────────────────────────
    def agent_takeoff(self) -> None:
        self._auto().takeoff()

    def agent_land(self) -> None:
        self._auto().land()

    def agent_move(self, direction: str, cm: int) -> None:
        self._auto().move(direction, cm)

    def agent_rotate(self, deg: int) -> None:
        self._auto().rotate(deg)

    def agent_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self._auto().rc(lr, fb, ud, yaw)

    def agent_hover(self) -> None:
        self._auto().hover()

    # ── operator-facing (preempts to MANUAL) ──────────────────────────────────
    def manual_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self._take().rc(lr, fb, ud, yaw)

    def manual_takeoff(self) -> None:
        self._take().takeoff()

    def manual_land(self) -> None:
        self._take().land()

    def manual_move(self, direction: str, cm: int) -> None:
        self._take().move(direction, cm)

    def manual_rotate(self, deg: int) -> None:
        self._take().rotate(deg)

    # ── always allowed ─────────────────────────────────────────────────────────
    def emergency(self) -> None:
        self.mode = self.MANUAL
        self.safe.emergency()

    def set_geofence(self, enabled: bool) -> None:
        """Operator override: lift/restore the dead-reckoning geofence so the agent may
        leave the room. Mode-independent (a safety control, like emergency)."""
        self.safe.geofence_enabled = bool(enabled)

    def tick(self) -> None:
        self.safe.tick()

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "flying": self.safe.flying,
            "pos": (round(self.safe.x), round(self.safe.y)),
            "heading": round(self.safe.heading),
            "geofence": self.safe.geofence_enabled,
        }

    # ── internals ──────────────────────────────────────────────────────────────
    def _auto(self) -> SafeTello:
        if self.mode != self.AUTO:
            raise ArbiterBlocked("agent command blocked — operator is in MANUAL")
        return self.safe

    def _take(self) -> SafeTello:
        """Operator input seizes control."""
        self.mode = self.MANUAL
        return self.safe
