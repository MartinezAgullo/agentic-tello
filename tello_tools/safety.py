"""SafeTello — every actuation passes through here before reaching the drone.

Enforces indoor-small-room limits:
  - discrete move size clamped to [MIN_STEP, MAX_STEP]
  - dead-reckoning geofence: refuse moves that leave a radius from takeoff
  - height caps on asc/descend and rc up-channel
  - battery floor → auto-land
  - watchdog → auto-hover if no fresh command

Pose tracking is approximate (integrates discrete moves + rotations). It is a
safety backstop, not navigation truth. Emergency stop bypasses everything.
"""

import math
import time

import config
from tello_tools.controller import TelloController


class SafetyError(Exception):
    """Raised when a command is refused by a safety cap."""


class SafeTello:
    def __init__(self, controller: TelloController) -> None:
        self.c = controller
        self.flying = False
        # pose relative to takeoff point, in cm; heading clockwise degrees
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self._last_cmd = time.monotonic()

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def takeoff(self) -> None:
        batt = self.c.get_battery()
        if batt <= config.BATTERY_FLOOR_PCT:
            raise SafetyError(f"battery {batt}% at/below floor {config.BATTERY_FLOOR_PCT}%")
        self.x = self.y = self.heading = 0.0
        self.c._takeoff()
        self.flying = True
        self._touch()

    def land(self) -> None:
        self.c._land()
        self.flying = False
        self._touch()

    # ── guarded discrete moves ────────────────────────────────────────────────
    def move(self, direction: str, cm: int) -> None:
        if not self.flying:
            raise SafetyError("not flying")
        if direction in ("up", "down"):
            self._check_vertical(direction, cm)
            cm = self._clamp_step(cm)
            self.c._move(direction, cm)
            self._touch()
            return
        cm = self._clamp_step(cm)
        nx, ny = self._predict(direction, cm)
        if math.hypot(nx, ny) > config.GEOFENCE_RADIUS_CM:
            raise SafetyError(
                f"move {direction} {cm}cm would leave geofence "
                f"(r={math.hypot(nx, ny):.0f} > {config.GEOFENCE_RADIUS_CM}cm)"
            )
        self.c._move(direction, cm)
        self.x, self.y = nx, ny
        self._touch()

    def rotate(self, deg: int) -> None:
        if not self.flying:
            raise SafetyError("not flying")
        self.c._rotate(deg)
        self.heading = (self.heading + deg) % 360
        self._touch()

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Continuous velocity control (manual sticks / servoing)."""
        s = config.SPEED
        lr, fb, ud, yaw = (max(-s, min(s, v)) for v in (lr, fb, ud, yaw))
        # block further ascent near the ceiling
        if ud > 0 and self.c.get_height() >= config.MAX_HEIGHT_CM:
            ud = 0
        self.c._rc(lr, fb, ud, yaw)
        self._touch()

    def hover(self) -> None:
        self.c._rc(0, 0, 0, 0)
        self._touch()

    def emergency(self) -> None:
        self.c.emergency()
        self.flying = False

    # ── periodic guards — call every loop tick ────────────────────────────────
    def tick(self) -> None:
        if not self.flying:
            return
        if self.c.get_battery() <= config.BATTERY_FLOOR_PCT:
            self.land()
            raise SafetyError("battery floor reached — auto-landing")
        if time.monotonic() - self._last_cmd > config.WATCHDOG_S:
            self.hover()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _touch(self) -> None:
        self._last_cmd = time.monotonic()

    def _clamp_step(self, cm: int) -> int:
        return max(config.MIN_STEP_CM, min(config.MAX_STEP_CM, int(cm)))

    def _check_vertical(self, direction: str, cm: int) -> None:
        h = self.c.get_height()
        if direction == "up" and h + cm > config.MAX_HEIGHT_CM:
            raise SafetyError(f"ascend would exceed height cap {config.MAX_HEIGHT_CM}cm")
        if direction == "down" and h - cm < config.MIN_HEIGHT_CM:
            raise SafetyError(f"descend would go below floor {config.MIN_HEIGHT_CM}cm")

    def _predict(self, direction: str, cm: int) -> tuple[float, float]:
        rad = math.radians(self.heading)
        # forward = +along heading; screen/world x = sin, y = cos
        fx, fy = math.sin(rad), math.cos(rad)
        rx, ry = math.cos(rad), -math.sin(rad)  # right = heading + 90°
        dx = dy = 0.0
        if direction == "forward":
            dx, dy = fx * cm, fy * cm
        elif direction == "back":
            dx, dy = -fx * cm, -fy * cm
        elif direction == "right":
            dx, dy = rx * cm, ry * cm
        elif direction == "left":
            dx, dy = -rx * cm, -ry * cm
        return self.x + dx, self.y + dy
