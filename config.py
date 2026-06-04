"""Central configuration: every safety cap, cadence, and model name lives here.

Indoor-small-room defaults — conservative on purpose. Tune the geofence/height
to your actual room before flying.

Model names and the Ollama host are **env-overridable** so you can swap in newer
models without editing code — e.g.::

    VLM_MODEL=qwen3-vl:8b DETECTOR_MODEL=yolov8x-worldv2.pt uv run python main.py

The defaults below track what's currently pulled locally; bump them (or set the
env var) as you update the system's brain.
"""

import os


# ── video stream ────────────────────────────────────────────────────────────
TELLO_STREAM_PORT = 11111
TELLO_STREAM_URL  = f"udp://0.0.0.0:{TELLO_STREAM_PORT}"
LIVE_GAP_S        = 0.025   # decoded-frame gap below this ⇒ burning backlog, skip it

# ── control ─────────────────────────────────────────────────────────────────
SPEED          = 40    # cm/s for rc control (manual + servoing); keep low indoors
KEEPALIVE_S    = 5     # send a zero-rc heartbeat at least this often

# ── safety caps (SafeTello enforces these) ──────────────────────────────────
MAX_HEIGHT_CM     = 180   # refuse ascend commands above this
MIN_HEIGHT_CM     = 30    # don't descend below this while flying
GEOFENCE_RADIUS_CM = 200  # dead-reckoning box radius from takeoff point
MAX_STEP_CM       = 50    # largest single discrete move
MIN_STEP_CM       = 20    # djitellopy move_* floor
BATTERY_FLOOR_PCT = 15    # auto-land at/below this
WATCHDOG_S        = 2.0   # no fresh command within this ⇒ auto-hover

# ── perception ──────────────────────────────────────────────────────────────
# open-vocab detector (ultralytics YOLO-World); override with DETECTOR_MODEL
DETECTOR_MODEL   = os.getenv("DETECTOR_MODEL", "yolov8s-worldv2.pt")
DETECT_CONF      = float(os.getenv("DETECT_CONF", "0.25"))
DETECT_EVERY     = 1      # run detector every Nth new frame (1 = every new frame)

# ── brain (VLM via Ollama) ──────────────────────────────────────────────────
# `ollama list` shows what's pulled; default to the local vision model. Swap via
# VLM_MODEL/OLLAMA_HOST env vars as newer models land — no code change needed.
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VLM_MODEL      = os.getenv("VLM_MODEL", "qwen3-vl:8b")
VLM_INTERVAL_S = float(os.getenv("VLM_INTERVAL_S", "3.0"))  # min seconds between VLM calls (slow loop)
VLM_FRAME_W    = int(os.getenv("VLM_FRAME_W", "640"))       # downscale width sent to the VLM
VLM_KEEP_ALIVE = os.getenv("VLM_KEEP_ALIVE", "-1")          # keep model warm in Ollama (-1 = forever)

# ── web ui ──────────────────────────────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000

# ── output ──────────────────────────────────────────────────────────────────
SNAPSHOT_DIR = "snapshots"
