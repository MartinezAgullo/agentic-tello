"""Phase C — web dashboard + manual flight control.

Run from the project root:  uv run python -m agentic_tello.web.server
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
import concurrent.futures
import errno
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agentic_tello import config
from agentic_tello.agent.loop import AgentBrain
from agentic_tello.agent.servoing import Servoer
from agentic_tello.agent.state import MissionState
from agentic_tello.brain.vlm_client import VLMClient
from agentic_tello.perception.bev import generate_bev_panel
from agentic_tello.perception.detector import COCO_CLASSES, Detector
from agentic_tello.perception.worker import PerceptionWorker
from agentic_tello.photogrammetry import (
    PhotogrammetryError,
    craft_3d_model,
    list_models,
    list_pending_images,
)
from agentic_tello.tello_tools.arbiter import ArbiterBlocked, ControlArbiter
from agentic_tello.tello_tools.controller import TelloController
from agentic_tello.tello_tools.primitives import get_telemetry, take_snapshot
from agentic_tello.tello_tools.safety import SafeTello, SafetyError
from agentic_tello.tools import ToolContext, build_registry

_HERE = Path(__file__).resolve().parent

# ── shared state (written by the control thread, read by handlers) ────────────
# Each queue item is (name, args, future): the future is None for fire-and-forget
# callers (the WebSocket UI) and a concurrent.futures.Future for REST callers that
# need the command's result/error back. The control thread resolves it.
_cmd_q: "queue.Queue[tuple]" = queue.Queue()
_manual_vec = (0, 0, 0, 0)        # (lr, fb, ud, yaw) in {-1,0,1}; reassigned atomically
_status: dict = {"ready": False}
_log: collections.deque = collections.deque(maxlen=300)
_goal: str | None = None
_mission: dict = {}               # latest mission triggered via POST /mission (id, goal, ts)
_stop = threading.Event()
_craft_busy = threading.Lock()  # guards against overlapping "craft 3D model" runs

_sys: dict = {"controller": None, "safe": None, "arb": None}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _log.append(line)
    print(line, flush=True)


def _enqueue(name: str, args: tuple = ()) -> None:
    """Fire-and-forget: drop a command on the control thread, don't wait (WS path)."""
    _cmd_q.put((name, tuple(args), None))


def _call(name: str, args: tuple = (), timeout: float = 10.0):
    """Enqueue a command and block until the control thread returns its result.

    Raises whatever the handler raised (SafetyError / ArbiterBlocked / …) or
    concurrent.futures.TimeoutError if the control thread never serviced it.
    """
    fut: concurrent.futures.Future = concurrent.futures.Future()
    _cmd_q.put((name, tuple(args), fut))
    return fut.result(timeout=timeout)


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
        # 1) discrete commands — resolve the caller's future (REST) or log (WS)
        while True:
            try:
                name, args, fut = _cmd_q.get_nowait()
            except queue.Empty:
                break
            try:
                res = _handle_cmd(arb, c, name, args)
                if fut is not None:
                    fut.set_result(res)
            except (SafetyError, ArbiterBlocked) as e:
                if fut is not None:
                    fut.set_exception(e)
                else:
                    log(f"refused: {e}")
            except Exception as e:
                if fut is not None:
                    fut.set_exception(e)
                else:
                    log(f"command error: {e}")

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


def _handle_cmd(arb: ControlArbiter, c: TelloController, name: str, args: tuple):
    """Execute one discrete command on the control thread, returning a result string.

    Does NOT catch SafetyError / ArbiterBlocked — the drain loop in `_control_loop`
    handles them (logging for WS callers, surfacing them on the future for REST).
    """
    if name == "takeoff":
        arb.manual_takeoff(); log("Takeoff"); return "took off"
    elif name == "land":
        arb.manual_land(); log("Land"); return "landed"
    elif name == "move":
        arb.agent_move(args[0], int(args[1]))
        log(f"Move {args[0]} {args[1]}cm"); return f"moved {args[0]} {args[1]}cm"
    elif name == "rotate":
        arb.agent_rotate(int(args[0]))
        log(f"Rotate {args[0]}deg"); return f"rotated {args[0]}deg"
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
        return f"mode → {args[0]}"
    elif name == "emergency":
        arb.emergency(); log("EMERGENCY STOP"); return "EMERGENCY STOP"
    elif name == "geofence":
        on = bool(args[0])
        arb.set_geofence(on)
        if on:
            log(f"Geofence RE-ARMED (radius {config.GEOFENCE_RADIUS_CM}cm from takeoff).")
        else:
            log("⚠ Geofence DISABLED — the agent may now leave the room / cross doorways. "
                "No obstacle avoidance exists; fly low/slow with E-STOP in reach.")
        return f"geofence {'on' if on else 'off'}"
    elif name == "snapshot":
        label = args[0] if args else "manual"
        fn = take_snapshot(c, label, dest_dir=config.CAPTURES_DIR)
        log(f"Snapshot: {fn}"); return fn
    elif name == "cenital":
        fn = take_snapshot(c, "cenital")
        if fn is None:
            log("Cenital: no frame yet — is the stream up?")
        else:
            floor = bool(args[0]) if args else False
            log(f"Cenital: snapshot {fn} — generating BEV (floor-seg={floor})…")
            # BEV is CPU work (warp + seg); run off the control thread so it
            # never starves flight. It only reads the saved file, no drone I/O.
            threading.Thread(target=_make_cenital, args=(fn, floor), daemon=True).start()
        return fn
    elif name == "snapshot_3d":
        fn = take_snapshot(c, "snap3d", dest_dir=config.PENDING_SNAPSHOT_DIR)
        if fn is None:
            log("3D-snapshot: no frame yet — is the stream up?")
        else:
            n = len(list_pending_images())
            log(f"3D-snapshot: {fn} ({n} image(s) pending reconstruction)")
            _status["model3d"] = {"pending": n, "ts": int(time.time() * 1000)}
        return fn
    elif name == "goal":
        global _goal
        _goal = args[0]
        brain = _sys.get("brain")
        if brain is not None:
            brain.start_mission(args[0])             # arms the mission; Arm AUTO to fly it
        else:
            log(f"Goal stored: {args[0]!r} (brain still loading)")
        return f"goal set: {args[0]!r}"
    elif name == "survey":
        # Deterministic colour-marker survey (no VLM): the brain installs the steps itself.
        n, marker_query, height_cm, goal_text = args
        _goal = goal_text or f"marker survey: {n}x {marker_query}"
        brain = _sys.get("brain")
        if brain is not None:
            brain.start_survey(n=int(n), marker_query=marker_query,
                               height_cm=height_cm, goal_text=goal_text)
        else:
            log(f"Survey requested ({n}x {marker_query}) but brain still loading")
        return f"survey armed: {n}x {marker_query}"
    else:
        raise ValueError(f"unknown command {name!r}")


def _make_cenital(image_path: str, floor_seg: bool) -> None:
    """Render the cenital BEV panel off-thread; publish its path for the UI."""
    try:
        res = generate_bev_panel(image_path, floor_seg=floor_seg)
        _status["cenital"] = {"path": res["panel_path"], "ts": int(time.time() * 1000)}
        log(f"[cenital] H={res['height']:.2f}m → {os.path.basename(res['panel_path'])}")
    except Exception as e:
        log(f"Cenital failed: {e}")


def _run_craft() -> None:
    """Drive one 3D reconstruction off-thread; publish progress to the UI."""
    if not _craft_busy.acquire(blocking=False):
        return  # another run slipped in; the handler already warned
    try:
        def on_progress(p: dict) -> None:
            _status["model3d"] = {**p, "busy": True, "ts": int(time.time() * 1000)}

        log("[3d] craft 3D model: starting reconstruction…")
        res = craft_3d_model(log=log, on_progress=on_progress)
        _status["model3d"] = {
            "stage": "done",
            "progress": 100.0,
            "status": "COMPLETED",
            "busy": False,
            "name": res.name,
            "models": list_models(),
            "ts": int(time.time() * 1000),
        }
        log(f"[3d] model ready: {res.name} — open the 3D Models tab to view it.")
    except PhotogrammetryError as e:
        _status["model3d"] = {"stage": "error", "busy": False, "error": str(e),
                              "ts": int(time.time() * 1000)}
        log(f"[3d] reconstruction failed: {e}")
    except Exception as e:
        _status["model3d"] = {"stage": "error", "busy": False, "error": str(e),
                              "ts": int(time.time() * 1000)}
        log(f"[3d] unexpected error: {e}")
    finally:
        _craft_busy.release()


def _start_craft() -> None:
    """Kick off a reconstruction unless one is already running.

    3D reconstruction is fully drone-independent (it only reads saved snapshots
    and talks to the ODM node), so it is dispatched directly here rather than
    through the drone command queue — that way it still works when the Tello
    isn't connected and the control thread has exited.
    """
    if _craft_busy.locked():
        log("craft 3D model: a reconstruction is already running — please wait.")
    else:
        threading.Thread(target=_run_craft, daemon=True).start()


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
        if fn:                                  # expose it over GET /mission/photo for the Mac
            _status["mission_photo"] = {"path": fn, "ts": int(time.time() * 1000),
                                        "mission_id": _mission.get("id"), "label": act[1]}
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
    with open(_HERE / "static" / "index.html") as f:
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


@app.get("/cenital/panel")
def cenital_panel() -> Response:
    """Serve the most recently generated cenital panel image (304/404 until one exists)."""
    cv = _status.get("cenital") or {}
    path = cv.get("path")
    if not path or not os.path.exists(path):
        return Response(status_code=404)
    return FileResponse(path)


@app.get("/api/models")
def api_models() -> list[dict]:
    """List crafted 3D models for the viewer tab (newest first)."""
    return list_models()


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
        _enqueue(msg["name"], tuple(msg.get("args", [])))
    elif t == "mode":
        _enqueue("mode", (msg["value"],))
    elif t == "goal":
        _enqueue("goal", (msg["text"],))
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
        _enqueue("emergency", ())
    elif t == "snapshot_3d":
        _enqueue("snapshot_3d", ())  # needs the drone (captures a frame)
    elif t == "craft_3d":
        _start_craft()  # drone-independent — dispatch directly, not via the control thread


# ── REST control surface ──────────────────────────────────────────────────────
# A thin HTTP API so an external orchestrator (the Mac agent, or the tello_mcp
# proxy) can send goals and drive the drone without a WebSocket. Handlers only
# enqueue onto the control thread (`_call`/`_enqueue`) or read `_status` — they
# never touch djitellopy directly, so they can't starve the video decoder.
def _run_cmd(name: str, args: tuple = (), timeout: float = 10.0):
    """Run a control-thread command for a REST handler, mapping failures to HTTP."""
    if _sys.get("arb") is None:
        raise HTTPException(503, _status.get("error") or "drone not ready")
    try:
        return _call(name, args, timeout)
    except (SafetyError, ArbiterBlocked) as e:
        raise HTTPException(409, str(e)) from e         # refused by a safety guard / mode gate
    except concurrent.futures.TimeoutError as e:
        raise HTTPException(504, "command timed out — drone busy or unreachable") from e
    except HTTPException:
        raise
    except Exception as e:                              # noqa: BLE001 — surface the message
        raise HTTPException(500, str(e)) from e


# --- mission lifecycle (the orchestrator's main path) ------------------------
@app.post("/mission", status_code=202)
def post_mission(body: dict) -> dict:
    """Start an autonomous mission: (goal | marker survey) → arm AUTO → takeoff. Returns
    immediately; the mission runs for seconds on the control thread (poll GET /mission/status).

    Two modes:
    - NL goal:        body = {"goal": "..."} → the VLM decomposes it.
    - Marker survey:  body = {"markers": N, "marker_query"?: "orange square",
                              "survey_height_cm"?: 160, "goal"?: "..."} → the drone runs the
      deterministic colour-marker survey (climb + fixed-vantage find, no approach), bypassing
      the VLM. This is the reliable "find the N markers" capability."""
    goal = (body.get("goal") or "").strip()
    markers = body.get("markers")
    if not goal and markers is None:
        raise HTTPException(422, "missing 'goal' (or 'markers' for a marker survey)")
    if _sys.get("arb") is None:
        raise HTTPException(503, _status.get("error") or "drone not ready")
    global _mission
    mid = f"m_{int(time.time() * 1000)}"
    label = goal or f"marker survey ({markers})"
    _mission = {"id": mid, "goal": label, "ts": int(time.time() * 1000)}
    _status.pop("mission_photo", None)                 # clear the previous mission's photo
    if markers is not None:                            # deterministic marker survey
        _enqueue("survey", (int(markers), body.get("marker_query") or "orange square",
                            body.get("survey_height_cm"), goal))
    else:
        _enqueue("goal", (goal,))                      # async: same sequence as the WS UI
    _enqueue("mode", ("AUTO",))
    # Only take off if grounded. On a re-shoot the drone is already hovering at survey
    # altitude — re-issuing takeoff there just errors ('takeoff unsuccessful') and the
    # survey re-runs from where it is, so the drone holds position instead of fighting.
    arb = _sys.get("arb")
    if arb is None or not arb.safe.flying:
        _enqueue("takeoff", ())
    else:
        log(f"[rest] mission {mid}: drone already airborne — skipping takeoff (re-shoot in place)")
    log(f"[rest] mission {mid} started: {label!r}")
    return {"mission_id": mid, "goal": label, "ts": _mission["ts"]}


@app.get("/mission/status")
def mission_status() -> dict:
    """Mission blackboard for polling — phase == 'done' means the goal is satisfied."""
    mission = _status.get("mission", {}) or {}
    photo = _status.get("mission_photo") or {}
    has_photo = bool(photo.get("path") and os.path.exists(photo["path"]))
    return {
        "mission_id": _mission.get("id"),
        "ready": _status.get("ready", False),
        "error": _status.get("error"),
        "goal": _status.get("goal"),
        "phase": mission.get("phase"),
        "mission": mission,
        "photo_available": has_photo,
        "photo_ts": photo.get("ts") if has_photo else None,
    }


@app.get("/mission/photo")
def mission_photo() -> Response:
    """Download the latest snapshot the mission captured (404 until one exists)."""
    photo = _status.get("mission_photo") or {}
    path = photo.get("path")
    if not path or not os.path.exists(path):
        return Response(status_code=404)
    return FileResponse(path, media_type="image/jpeg",
                        filename=os.path.basename(path))


# --- fine-grained control (so the MCP proxy keeps full control) --------------
@app.post("/control/takeoff")
def control_takeoff() -> dict:
    return {"result": _run_cmd("takeoff")}


@app.post("/control/land")
def control_land() -> dict:
    return {"result": _run_cmd("land")}


@app.post("/control/mode")
def control_mode(body: dict) -> dict:
    mode = str(body.get("mode", "")).upper()
    if mode not in ("AUTO", "MANUAL"):
        raise HTTPException(422, "mode must be 'AUTO' or 'MANUAL'")
    return {"result": _run_cmd("mode", (mode,))}


@app.post("/control/move")
def control_move(body: dict) -> dict:
    direction = body.get("direction")
    cm = body.get("cm")
    if direction not in ("forward", "back", "left", "right", "up", "down") or cm is None:
        raise HTTPException(422, "need direction (forward/back/left/right/up/down) and cm")
    return {"result": _run_cmd("move", (direction, int(cm)))}


@app.post("/control/rotate")
def control_rotate(body: dict) -> dict:
    if body.get("deg") is None:
        raise HTTPException(422, "need 'deg' (negative = counter-clockwise)")
    return {"result": _run_cmd("rotate", (int(body["deg"]),))}


@app.post("/control/emergency")
def control_emergency() -> dict:
    return {"result": _run_cmd("emergency")}


@app.post("/control/geofence")
def control_geofence(body: dict) -> dict:
    return {"result": _run_cmd("geofence", (bool(body.get("on", True)),))}


@app.post("/control/snapshot")
def control_snapshot(body: dict | None = None) -> dict:
    label = (body or {}).get("label", "manual")
    fn = _run_cmd("snapshot", (label,))
    if fn is None:
        raise HTTPException(503, "no frame yet — is the stream up?")
    return {"result": fn}


@app.post("/control/target")
def control_target(body: dict) -> dict:
    """Set the open-vocabulary detector queries (deterministic servoing target)."""
    tools = _sys.get("tools")
    if tools is None:
        raise HTTPException(503, "brain/detector not ready")
    queries = body.get("queries", [])
    return {"result": tools["set_target"].run({"queries": queries})}


@app.post("/mission/done")
def mission_done(body: dict | None = None) -> dict:
    """Declare the current goal satisfied (sets phase=done, disarms the mission)."""
    tools = _sys.get("tools")
    if tools is None:
        raise HTTPException(503, "brain not ready")
    return {"result": tools["report_done"].run({"reason": (body or {}).get("reason", "")})}


# --- read-only telemetry / state (thread-safe; no control thread needed) -----
@app.get("/telemetry")
def get_telemetry_route() -> dict:
    c = _sys.get("controller")
    if c is None:
        raise HTTPException(503, _status.get("error") or "drone not ready")
    return get_telemetry(c)


@app.get("/pose")
def get_pose_route() -> dict:
    arb = _sys.get("arb")
    if arb is None:
        raise HTTPException(503, "drone not ready")
    return {"x": arb.safe.x, "y": arb.safe.y, "heading": arb.safe.heading}


@app.get("/observation")
def get_observation_route() -> dict:
    tools = _sys.get("tools")
    if tools is None:
        raise HTTPException(503, "brain/detector not ready")
    return tools["get_observation"].run({})


@app.get("/status")
def get_status_route() -> dict:
    """Full HUD snapshot (mode, flying, battery, detections, mission, …)."""
    return _status


# mount static after routes so "/" stays our handler
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")

# serve crafted 3D model assets (OBJ/MTL/textures) for the Three.js viewer
os.makedirs(config.MODELS_3D_DIR, exist_ok=True)
app.mount("/models", StaticFiles(directory=config.MODELS_3D_DIR), name="models")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)
