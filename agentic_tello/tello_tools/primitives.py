"""Non-actuating primitives: capture a frame to disk, read telemetry.

These only read from the controller, so they need no safety gate.

Every snapshot is written with a sidecar `<stem>.json` carrying the drone's
telemetry at capture time (height, IMU attitude, battery…). That metadata is
what lets the BEV / cenital tooling reconstruct an accurate ground projection
offline — height and pitch come from the drone, not guessed.
"""

import json
import os
import time

import cv2

from agentic_tello import config
from agentic_tello.tello_tools.controller import TelloController


def snapshot_metadata(controller: TelloController, label: str = "snap") -> dict:
    """Telemetry to persist alongside a snapshot — the BEV pipeline reads this."""
    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "telemetry": get_telemetry(controller),
        # raw djitellopy state: pitch/roll/yaw (deg), h (cm), tof (cm), baro, ...
        "state": controller.get_state(),
    }


def take_snapshot(
    controller: TelloController, label: str = "snap", dest_dir: str | None = None
) -> str | None:
    """Capture the current frame to ``dest_dir`` (default ``config.SNAPSHOT_DIR``).

    Writes ``<label>_<timestamp>.jpg`` plus a ``.json`` telemetry sidecar. Pass
    ``dest_dir=config.PENDING_SNAPSHOT_DIR`` to queue a frame for 3D reconstruction.
    """
    frame = controller.get_frame()
    if frame is None or frame.size == 0:
        return None
    out_dir = dest_dir or config.SNAPSHOT_DIR
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{label}_{time.strftime('%Y%m%d_%H%M%S')}")
    fname = f"{stem}.jpg"
    cv2.imwrite(fname, frame)
    try:
        with open(f"{stem}.json", "w") as f:
            json.dump(snapshot_metadata(controller, label), f, indent=2)
    except Exception:
        pass  # never let a metadata hiccup lose the image
    return fname


def get_telemetry(controller: TelloController) -> dict:
    state = controller.get_state()
    return {
        "battery": controller.get_battery(),
        "height_cm": controller.get_height(),
        "temp_c": state.get("templ"),
        "flight_time_s": state.get("time"),
        "stream_fps": round(controller.frame_read.decode_fps, 1) if controller.frame_read else 0.0,
    }
