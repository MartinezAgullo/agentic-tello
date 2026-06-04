"""Non-actuating primitives: capture a frame to disk, read telemetry.

These only read from the controller, so they need no safety gate.
"""

import os
import time

import cv2

import config
from tello_tools.controller import TelloController


def take_snapshot(controller: TelloController, label: str = "snap") -> str | None:
    frame = controller.get_frame()
    if frame is None or frame.size == 0:
        return None
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    fname = os.path.join(
        config.SNAPSHOT_DIR, f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    cv2.imwrite(fname, frame)
    return fname


def get_telemetry(controller: TelloController) -> dict:
    state = controller.get_state()
    return {
        "battery": controller.get_battery(),
        "height_cm": controller.get_height(),
        "temp_c": state.get("templ"),
        "flight_time_s": state.get("time"),
        "stream_fps": round(controller.frame_read.decode_fps, 1)
        if controller.frame_read else 0.0,
    }
