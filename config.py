"""Central configuration: every safety cap, cadence, and model name lives here.

Indoor-small-room defaults — conservative on purpose. Tune the geofence/height
to your actual room before flying.

Model names and the Ollama host are **env-overridable** so you can swap in newer
models without editing code — e.g.::

    VLM_MODEL=qwen3-vl:8b DETECTOR_MODEL=yolov8x-worldv2.pt uv run python main.py

The defaults below track what's currently pulled locally; bump them (or set the
env var) as you update the system's brain.
"""

import math
import os

# ── video stream ────────────────────────────────────────────────────────────
TELLO_STREAM_PORT = 11111
TELLO_STREAM_URL = f"udp://0.0.0.0:{TELLO_STREAM_PORT}"
LIVE_GAP_S = 0.025  # decoded-frame gap below this ⇒ burning backlog, skip it

# ── control ─────────────────────────────────────────────────────────────────
SPEED = 40  # cm/s for rc control (manual + servoing); keep low indoors
KEEPALIVE_S = 5  # send a zero-rc heartbeat at least this often

# ── safety caps (SafeTello enforces these) ──────────────────────────────────
MAX_HEIGHT_CM = 180  # refuse ascend commands above this
MIN_HEIGHT_CM = 30  # don't descend below this while flying
GEOFENCE_RADIUS_CM = 200  # dead-reckoning box radius from takeoff point
MAX_STEP_CM = 50  # largest single discrete move
MIN_STEP_CM = 20  # djitellopy move_* floor
BATTERY_FLOOR_PCT = 15  # auto-land at/below this
WATCHDOG_S = 2.0  # no fresh command within this ⇒ auto-hover

# ── perception ──────────────────────────────────────────────────────────────
# open-vocab detector (ultralytics YOLO-World); override with DETECTOR_MODEL
DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", "yolov8s-worldv2.pt")
DETECT_CONF = float(os.getenv("DETECT_CONF", "0.25"))
DETECT_EVERY = 1  # run detector every Nth new frame (1 = every new frame)
# Cap detector rate so its (GIL-holding) inference can't starve the video decode
# thread. ~15 fps is plenty for servoing and leaves the decoder its GIL slices.
# Raise once the detector is in its own process (see perception/README.md).
DETECT_MAX_FPS = float(os.getenv("DETECT_MAX_FPS", "15"))

# ── brain (VLM via Ollama) ──────────────────────────────────────────────────
# `ollama list` shows what's pulled; default to the local vision model. Swap via
# VLM_MODEL/OLLAMA_HOST env vars as newer models land — no code change needed.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VLM_MODEL = os.getenv("VLM_MODEL", "gemma3:12b")  # non-thinking VLM: low, predictable latency
VLM_INTERVAL_S = float(
    os.getenv("VLM_INTERVAL_S", "3.0")
)  # min seconds between VLM calls (slow loop)
VLM_FRAME_W = int(os.getenv("VLM_FRAME_W", "640"))  # downscale width sent to the VLM
VLM_KEEP_ALIVE = os.getenv("VLM_KEEP_ALIVE", "-1")  # keep model warm in Ollama (-1 = forever)
# context window: the model's native default (e.g. 262k) reserves a huge KV cache and
# makes each call ~15x slower. We only send one image + a short prompt, so cap it small.
VLM_NUM_CTX = int(os.getenv("VLM_NUM_CTX", "4096"))

# ── camera intrinsics (still photo) ─────────────────────────────────────────
# DJI RoboMaster TT / Tello stills — approximate, uncalibrated. The *honest* way
# to get these is a one-off checkerboard calibration (cv2.calibrateCamera) on this
# unit, which yields fx/fy directly (FOV = 2*atan(W/2fx)). Until then we use the
# published FOV and derive VFOV from the aspect ratio. Env-overridable.
#
# DJI publishes "FOV 82.6°" without saying horizontal vs diagonal; we treat it as
# the *diagonal* FOV. For the 4:3 still that gives HFOV 70.3°, VFOV 55.6°
# (HFOV = 2*atan((W/d)*tan(DFOV/2)), d = sqrt(W^2+H^2)). Still a guess, not a
# measurement — calibrate or do a known-floor-ruler check to confirm.
CAM_PHOTO_W = int(os.getenv("CAM_PHOTO_W", "2592"))
CAM_PHOTO_H = int(os.getenv("CAM_PHOTO_H", "1936"))
CAM_DFOV_DEG = float(os.getenv("CAM_DFOV_DEG", "82.6"))  # diagonal FOV (published spec)
CAM_HFOV_DEG = float(
    os.getenv("CAM_HFOV_DEG", "")
    or math.degrees(
        2.0
        * math.atan(
            (CAM_PHOTO_W / math.hypot(CAM_PHOTO_W, CAM_PHOTO_H))
            * math.tan(math.radians(CAM_DFOV_DEG) / 2.0)
        )
    )
)
# VFOV: leave unset to derive from HFOV + aspect (square-pixel assumption).
CAM_VFOV_DEG = float(os.getenv("CAM_VFOV_DEG", "0")) or None

# ── web ui ──────────────────────────────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000

# ── output ──────────────────────────────────────────────────────────────────
SNAPSHOT_DIR = "snapshots"
# Standard (operator) snapshots from the Snapshot button / P key land here.
CAPTURES_DIR = os.path.join(SNAPSHOT_DIR, "captures")
# Bird's-eye-view / pseudo-orthophoto outputs land in a sub-folder of snapshots.
BEV_DIR = os.path.join(SNAPSHOT_DIR, "cenital_view")

# ── 3D photogrammetry (PyODM → local OpenDroneMap node) ──────────────────────
# Offline reconstruction storage, kept under the snapshots/ tree. Snapshots queued
# for the next "craft 3D model" run land in PENDING; once consumed they move to
# PROCESSED, and the textured model assets are downloaded into a timestamped
# sub-folder of MODELS_3D_DIR.
STORAGE_DIR = os.path.join(SNAPSHOT_DIR, "storage_3D")
PENDING_SNAPSHOT_DIR = os.path.join(STORAGE_DIR, "pending_snapshots")
PROCESSED_DIR = os.path.join(STORAGE_DIR, "processed")
MODELS_3D_DIR = os.path.join(STORAGE_DIR, "3D_models")

# OpenDroneMap processing node (NodeODM, typically a CUDA-enabled Docker container
# on this same host). Point these elsewhere to offload processing to another box
# without touching code — only the endpoint changes.
ODM_HOST = os.getenv("ODM_HOST", "localhost")
ODM_PORT = int(os.getenv("ODM_PORT", "3000"))
ODM_TOKEN = os.getenv("ODM_TOKEN", "")  # empty unless the node enforces auth
ODM_POLL_INTERVAL_S = float(os.getenv("ODM_POLL_INTERVAL_S", "3.0"))

# ── 3D reconstruction quality (ODM task options) ─────────────────────────────
# Edit these to tune the reconstruction. The active value on each line is the
# balanced default; the commented line right below it is the maximum-quality
# value (best result, regardless of runtime / memory). Swap them as you wish.
#

# feature-quality: resolution at which image features are extracted.
ODM_FEATURE_QUALITY = "high"  # ultra | high | medium | low | lowest
#ODM_FEATURE_QUALITY = "ultra"          # max quality

# min-num-features: minimum features extracted per image (more → easier matching).
ODM_MIN_NUM_FEATURES = 10000
#ODM_MIN_NUM_FEATURES = 50000           # max quality

# pc-geometric: extra geometric filtering of the dense point cloud (drops the
# spurious points that often abort OpenMVS densification, e.g. error 134).
ODM_PC_GEOMETRIC = False
#ODM_PC_GEOMETRIC = True                 # max quality

# pc-quality: density of the final point cloud.
ODM_PC_QUALITY = "medium"  # ultra | high | medium | low | lowest
#ODM_PC_QUALITY = "ultra"               # max quality
