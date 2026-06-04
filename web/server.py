"""Phase C — web dashboard + manual flight control.

Run from the project root:  uv run python -m web.server
Then open http://<host>:8000  (e.g. http://localhost:8000).

Design: a single dedicated **control thread** owns every drone actuation (djitellopy
is not safe for concurrent sends). WebSocket handlers never touch the drone directly —
they enqueue discrete commands or set the manual stick vector. Any nonzero manual input
preempts to MANUAL automatically (operator always wins). MJPEG carries the video so a
plain <img> shows it; a WebSocket carries telemetry + the decision log both ways.

A `render_frame()` hook returns the frame to show — raw now, detection-overlaid once
Phase B lands. The AUTO mode is a no-op until the Phase E agent loop drives it.
"""

import asyncio
import collections
import errno
import queue
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from agent.loop import AgentBrain
from agent.servoing import Servoer
from agent.state import MissionState
from brain.vlm_client import VLMClient
from perception.detector import COCO_CLASSES, Detector
from perception.worker import PerceptionWorker
from tello_tools.arbiter import ArbiterBlocked, ControlArbiter
from tello_tools.controller import TelloController
from tello_tools.primitives import get_telemetry, take_snapshot
from tello_tools.safety import SafeTello, SafetyError
from tools import ToolContext, build_registry

# ── shared state (written by the control thread, read by handlers) ────────────
_cmd_q: "queue.Queue[tuple]" = queue.Queue()
_manual_vec = (0, 0, 0, 0)        # (lr, fb, ud, yaw) in {-1,0,1}; reassigned atomically
_status: dict = {"ready": False}
_log: collections.deque = collections.deque(maxlen=300)
_goal: str | None = None
_stop = threading.Event()

_sys: dict = {"controller": None, "safe": None, "arb": None}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _log.append(line)
    print(line, flush=True)


# ── control thread: the only place the drone is actuated ──────────────────────
def _control_loop() -> None:
    try:
        log("Connecting to Tello…")
        c = TelloController()          # binds UDP 8889 here — can fail if port busy
        c.connect()
        c.start_stream()
        log("Connected, stream started.")
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            _status["error"] = "Tello control port 8889 busy — stale process still running"
            log("Tello control port 8889 is ALREADY IN USE — a previous run (web.server "
                "or bench_test) is probably still alive. Free it and restart:")
            log("    pgrep -af 'web.server|uvicorn'   # find stale processes")
            log("    ss -lunap | grep 8889            # (or: lsof -iUDP:8889) who holds it")
            log("    pkill -f web.server              # kill them, then re-run the server")
        else:
            _status["error"] = f"Tello connect failed: {e}"
            log(f"Tello connect failed: {e} — check the drone WiFi. UI runs, commands off.")
        return
    except Exception as e:
        _status["error"] = f"Tello connect failed: {e}"
        log(f"Tello connect failed: {e} — check the drone WiFi. UI runs, commands off.")
        return

    safe = SafeTello(c)
    arb = ControlArbiter(safe)
    servoer = Servoer()
    _sys.update(controller=c, safe=safe, arb=arb, servoer=servoer)
    _status["ready"] = True

    # load the detector + agent brain off-thread so model init never blocks flight
    threading.Thread(target=_init_perception, args=(c, arb), daemon=True).start()

    last_tel = 0.0
    low_batt_warned = False
    while not _stop.is_set():
        # 1) discrete commands
        while True:
            try:
                name, args = _cmd_q.get_nowait()
            except queue.Empty:
                break
            _handle_cmd(arb, c, name, args)

        # 2) drive: operator sticks always win; otherwise AUTO servoes (Phase D)
        if safe.flying:
            try:
                if _manual_vec != (0, 0, 0, 0):
                    if arb.mode == ControlArbiter.AUTO:   # the human just grabbed the sticks
                        log("operator input → MANUAL (AUTO preempted). Re-arm AUTO to resume.")
                    lr, fb, ud, yaw = (v * config.SPEED for v in _manual_vec)
                    arb.manual_rc(lr, fb, ud, yaw)        # nonzero input seizes MANUAL
                elif arb.mode == ControlArbiter.AUTO:
                    brain = _sys.get("brain")
                    if brain is not None and brain.active:
                        _run_agent(arb, c, brain)         # Phase E: VLM-planned mission
                    else:
                        _auto_servo(arb, c, servoer)      # Phase D: track top detection, no VLM
                else:
                    arb.manual_rc(0, 0, 0, 0)             # MANUAL idle — hold hover
            except (SafetyError, ArbiterBlocked) as e:
                log(f"safety: {e}")

        # 3) watchdog + battery floor
        try:
            arb.tick()
        except SafetyError as e:
            log(f"safety: {e}")

        # 4) telemetry snapshot ~3 Hz
        now = time.monotonic()
        if now - last_tel > 0.3:
            last_tel = now
            tel = get_telemetry(c)
            w = _sys.get("perception")
            det = ({"det_n": len(w.detections), "det_fps": round(w.det_fps, 1),
                    "queries": ", ".join(w.queries)} if w is not None else {})
            brain = _sys.get("brain")
            mission = brain.state.snapshot() if brain is not None else {}
            _status.update(ready=True, goal=_goal, mission=mission,
                           **det, **arb.status(), **tel)

            # pre-warn before the auto-land floor (hysteresis so it fires once)
            batt = tel.get("battery") or 100
            if not low_batt_warned and batt <= config.BATTERY_FLOOR_PCT + 5:
                low_batt_warned = True
                log(f"battery {batt}% — low. Auto-land floor is {config.BATTERY_FLOOR_PCT}%; "
                    "land soon (manually) to avoid a forced landing.")
            elif low_batt_warned and batt > config.BATTERY_FLOOR_PCT + 10:
                low_batt_warned = False

        time.sleep(0.05)

    try:
        if safe.flying:
            log("Auto-landing before exit…")
            arb.manual_land()
    except Exception as e:
        log(f"landing on exit failed: {e}")
    c.shutdown()
    log("Drone disconnected, UDP ports released.")


def _handle_cmd(arb: ControlArbiter, c: TelloController, name: str, args: tuple) -> None:
    try:
        if name == "takeoff":
            arb.manual_takeoff(); log("Takeoff")
        elif name == "land":
            arb.manual_land(); log("Land")
        elif name == "mode":
            (arb.arm_auto if args[0] == "AUTO" else arb.to_manual)()
            log(f"Mode → {args[0]}")
            if args[0] == "AUTO":
                brain = _sys.get("brain")
                w = _sys.get("perception")
                has_goal = brain is not None and brain.active
                has_target = w is not None and bool(w.queries)
                if not has_goal and not has_target:
                    log("AUTO armed but there's no Goal and no Detect target — the drone "
                        "will just hover. Send a Goal (then it searches/approaches) or type "
                        "a Detect query (then it servoes toward it).")
                elif not arb.safe.flying:
                    log("AUTO armed with a mission but the drone isn't airborne — press "
                        "Takeoff to start flying it.")
        elif name == "emergency":
            arb.emergency(); log("EMERGENCY STOP")
        elif name == "snapshot":
            fn = take_snapshot(c, "manual"); log(f"Snapshot: {fn}")
        elif name == "goal":
            global _goal
            _goal = args[0]
            brain = _sys.get("brain")
            if brain is not None:
                brain.start_mission(args[0])             # arms the mission; Arm AUTO to fly it
            else:
                log(f"Goal stored: {args[0]!r} (brain still loading)")
    except (SafetyError, ArbiterBlocked) as e:
        log(f"refused: {e}")
    except Exception as e:
        log(f"command error: {e}")


def _init_perception(c: TelloController, arb: ControlArbiter) -> None:
    worker = None
    try:
        log("Loading detector (YOLO-World)… first run downloads weights.")
        det = Detector()
        worker = PerceptionWorker(c.get_frame, det).start()
        _sys["perception"] = worker
        log(f"Detector ready on {det.device}. Type queries in the Detect box.")
    except Exception as e:
        log(f"Detector load failed: {e} — flight still works, no detection.")

    # build the agent brain on top: VLM planner + the single tool registry
    try:
        state = MissionState()
        ctx = ToolContext(arbiter=arb, controller=c, worker=worker, state=state)
        tools = build_registry(ctx)
        brain = AgentBrain(VLMClient(), worker, state, tools,
                           c.get_frame, lambda: get_telemetry(c), log=log)
        _sys.update(state=state, tools=tools, brain=brain)
        log(f"Brain ready (VLM {config.VLM_MODEL} via {config.OLLAMA_HOST}). "
            "Set a Goal, then Arm AUTO.")
    except Exception as e:
        log(f"Brain init failed: {e} — manual + Phase-D AUTO still work.")


def _run_agent(arb: ControlArbiter, c: TelloController, brain: AgentBrain) -> None:
    """Phase E: execute one fast-loop Action from the brain (the actuation chokepoint)."""
    act = brain.fast_step()
    kind = act[0]
    if kind == "rc":
        arb.agent_rc(act[1], act[2], act[3], act[4])
    elif kind == "rotate":                  # discrete search turn (blocks briefly)
        arb.agent_rotate(act[1])
    elif kind == "move":                    # reposition to a new vantage (geofence-guarded)
        arb.agent_move(act[1], act[2])
    elif kind == "snapshot":
        fn = brain.tools["take_snapshot"].run({"label": act[1]})
        log(f"[brain] snapshot saved: {fn}")
    else:                                   # "hover" / "done"
        arb.agent_hover()


def _auto_servo(arb: ControlArbiter, c: TelloController, servoer: Servoer) -> None:
    """Phase D: deterministic servoing toward the top detection (no VLM).

    Nothing to track ⇒ hover (the Phase-E agent loop adds search/replanning).
    """
    w = _sys.get("perception")
    if w is None or not w.detections:
        arb.agent_hover()
        return
    frame = c.get_frame()
    if frame is None:
        arb.agent_hover()
        return
    h, fw = frame.shape[:2]
    lr, fb, ud, yaw, done = servoer.step(w.detections[0], fw, h)
    arb.agent_hover() if done else arb.agent_rc(lr, fb, ud, yaw)


def render_frame() -> np.ndarray | None:
    """Latest frame, with detection boxes drawn on it when the worker is active."""
    c = _sys["controller"]
    if c is None:
        return None
    frame = c.get_frame()
    if frame is None:
        return None
    w = _sys.get("perception")
    if w is not None and w.queries and w.detections:
        return Detector.annotate(frame, w.detections)
    return frame


# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_control_loop, daemon=True)
    t.start()
    yield
    print(flush=True)   # break the line after the ^C echo
    log("Ctrl+C — landing if airborne and releasing the drone…")
    _stop.set()
    t.join(timeout=10)  # let it finish landing + closing sockets before we exit
    log("Shutdown complete. Fly safe 👋")


app = FastAPI(title="Agentic Tello", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open("web/static/index.html") as f:
        return f.read()


@app.get("/video")
def video() -> StreamingResponse:
    boundary = "frame"

    def gen():
        placeholder = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(placeholder, "waiting for stream...", (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)
        while not _stop.is_set():
            try:
                frame = render_frame()
                if frame is None:
                    frame = placeholder
                if frame.shape[1] > 640:
                    h = int(frame.shape[0] * 640 / frame.shape[1])
                    frame = cv2.resize(frame, (640, h))
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield (b"--" + boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            except Exception as e:                 # one bad frame must not kill the stream
                print(f"[video] frame error (stream continues): {e}", flush=True)
            time.sleep(0.033)

    return StreamingResponse(
        gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}"
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    sent_log = 0

    async def push():
        nonlocal sent_log
        while True:
            payload = {"type": "status", "status": _status}
            new = list(_log)[sent_log:]
            if new:
                sent_log = len(_log)
                payload["log"] = new
            await websocket.send_json(payload)
            await asyncio.sleep(0.25)

    pusher = asyncio.create_task(push())
    try:
        while True:
            msg = await websocket.receive_json()
            _handle_client(msg)
    except WebSocketDisconnect:
        pass
    finally:
        pusher.cancel()


def _handle_client(msg: dict) -> None:
    global _manual_vec
    t = msg.get("type")
    if t == "rc":
        _manual_vec = (msg.get("lr", 0), msg.get("fb", 0),
                       msg.get("ud", 0), msg.get("yaw", 0))
    elif t == "cmd":
        _cmd_q.put((msg["name"], tuple(msg.get("args", []))))
    elif t == "mode":
        _cmd_q.put(("mode", (msg["value"],)))
    elif t == "goal":
        _cmd_q.put(("goal", (msg["text"],)))
    elif t == "queries":
        w = _sys.get("perception")
        if w is not None:
            w.set_queries(msg.get("text", "").split(","))
            log(f"Detect queries: {msg.get('text', '')!r}")
    elif t == "queries_all":
        w = _sys.get("perception")
        if w is not None:
            w.set_queries(list(COCO_CLASSES))
            log(f"Detect queries: ALL ({len(COCO_CLASSES)} COCO classes)")
    elif t == "emergency":
        _cmd_q.put(("emergency", ()))


# mount static after routes so "/" stays our handler
app.mount("/static", StaticFiles(directory="web/static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)
