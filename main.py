"""Single entrypoint — the full agentic Tello system.

Wires drone + perception + agent brain + web UI and serves the dashboard. The web
server's control thread owns all actuation; the detector and the VLM planner load
off-thread so startup never blocks. Open http://<host>:8000 after launch.

    uv run python main.py

Models are env-configurable (see config.py): e.g.
    VLM_MODEL=qwen3-vl:8b DETECTOR_MODEL=yolov8x-worldv2.pt uv run python main.py
"""

import uvicorn

import config
from web.server import app

if __name__ == "__main__":
    print(f"Agentic Tello → http://{config.WEB_HOST}:{config.WEB_PORT}  "
          f"(VLM={config.VLM_MODEL}, detector={config.DETECTOR_MODEL})", flush=True)
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)
