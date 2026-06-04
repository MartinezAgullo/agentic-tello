"""Deterministic visual servoing — the fast-loop reflex (Phase D).

Turns one detection box into rc velocity commands that center the target in the
frame and approach it, entirely **without** the VLM. Given the latest detection:
yaw to center horizontally, up/down to center vertically, and forward/back until
the target fills the desired fraction of the frame.

Output is normalized then scaled to `config.SPEED`; it still funnels through
`arbiter.agent_rc → SafeTello`, so the SPEED/height caps and watchdog apply.
Gains are deliberately gentle and there are deadbands so it doesn't jitter once
the target is roughly centered. Larger detections (closer objects) ⇒ bigger
`area_frac`, which is the cheap distance proxy used for the approach.
"""

import config


class Servoer:
    def __init__(
        self,
        *,
        target_area_frac: float = 0.18,  # stop when the box fills ~this much of the frame
        center_tol: float = 0.08,        # normalized half-deadband around frame center
        area_tol: float = 0.04,          # area deadband around the target size
        kp_yaw: float = 1.4,             # gains map normalized error → [-1,1] command
        kp_ud: float = 1.2,
        kp_fwd: float = 1.2,
    ) -> None:
        self.target_area_frac = target_area_frac
        self.center_tol = center_tol
        self.area_tol = area_tol
        self.kp_yaw = kp_yaw
        self.kp_ud = kp_ud
        self.kp_fwd = kp_fwd

    def step(self, det: dict, frame_w: int, frame_h: int) -> tuple[int, int, int, int, bool]:
        """Map one detection to (lr, fb, ud, yaw, done).

        Errors are normalized to [-1, 1]; gains turn them into rc velocities
        (SafeTello clamps to ±config.SPEED). `done` is True once the target is
        centered and at the desired size. lr stays 0 — we center with yaw.
        """
        cx, cy = det["center"]
        ex = (cx - frame_w / 2) / (frame_w / 2)        # +right of center
        ey = (cy - frame_h / 2) / (frame_h / 2)        # +below center
        ea = self.target_area_frac - det["area_frac"]  # +too far (need to approach)

        yaw = ud = fb = 0.0
        if abs(ex) > self.center_tol:
            yaw = self.kp_yaw * ex                     # turn toward the target
        if abs(ey) > self.center_tol:
            ud = -self.kp_ud * ey                      # target high in frame ⇒ climb
        centered = abs(ex) <= self.center_tol and abs(ey) <= self.center_tol
        if centered and abs(ea) > self.area_tol:
            fb = self.kp_fwd * (ea / self.target_area_frac)  # +fwd to approach, -back if too close
        done = centered and abs(ea) <= self.area_tol

        s = config.SPEED
        def scale(v):
            return int(max(-s, min(s, v * s)))
        return 0, scale(fb), scale(ud), scale(yaw), done
