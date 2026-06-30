# `web` — Dashboard & Manual Flight Control

Browser UI for the agentic Tello: live video, telemetry, event log, a goal box, and
**manual flight**. It sits on top of the core control library (`tello_tools`) and is the
place where manual flying is validated before any autonomy is added.

## What's in here

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app + the single drone **control thread** |
| `static/index.html` | self-contained dashboard (HTML/CSS/JS, no build step) |

## How it's wired (important for safety)

```
browser  ──WebSocket──▶  command queue / stick vector
   ▲                          │
   │ MJPEG video + WS status   ▼
   └──────────────  control thread (the ONLY thing that touches the drone)
                              │
                              ▼
                ControlArbiter → SafeTello → TelloController → Tello
```

- **One control thread owns all actuation** — djitellopy is not safe for concurrent
  sends, so WebSocket handlers never command the drone directly; they only enqueue
  discrete commands or set the manual stick vector.
- **Operator always wins:** any nonzero manual input preempts to **MANUAL**.
- **Watchdog + battery floor** run every loop tick (auto-hover on silence, auto-land at
  the floor) via `arbiter.tick()`.
- **AUTO** mode is a no-op until the Phase E agent loop exists — toggling to AUTO just
  hovers for now.

## Run it

From the **project root** (not from `web/`), on the drone's WiFi (`TELLO-XXXXXX`):

```bash
cd ~/Desktop/agentic-tello
uv run python -m web.server
```

Then open **http://localhost:8000** (or `http://<spark-ip>:8000` from another machine —
the server binds `0.0.0.0`). Port/host are set in the root `config.py`.

> The server starts even without the drone (degraded mode: video shows a placeholder,
> commands are disabled until the control thread connects). Check the terminal log for
> `Connected, stream started.` vs `Tello connect failed`.

## REST API (for an external orchestrator)

Besides the browser WebSocket, the server exposes a plain HTTP surface so another agent
(e.g. one running on the Mac, reaching the Spark over the forwarded port) can send goals,
drive the drone, and download photos. The `tello_mcp` proxy forwards to exactly these.
Handlers only enqueue onto the control thread or read shared status — they never touch
djitellopy directly, so they can't starve the video decoder. Refused commands return
**409** (safety guard / mode gate), not-ready returns **503**, bad input **422**.

| Method & path | Body | Purpose |
|---------------|------|---------|
| `POST /mission` | `{"goal": "<NL>"}` | Autonomous mission: goal → arm AUTO → takeoff. Returns `202 {mission_id, goal, ts}` (async — poll status). |
| `GET /mission/status` | — | Blackboard for polling; `phase == "done"` ⇒ goal satisfied; `photo_available` flags a capture. |
| `GET /mission/photo` | — | The latest snapshot the mission captured (`image/jpeg`; **404** until one exists). |
| `POST /mission/done` | `{"reason"?}` | Declare the current goal satisfied. |
| `POST /control/mode` | `{"mode": "AUTO"\|"MANUAL"}` | Arm AUTO / return to MANUAL. |
| `POST /control/takeoff` · `/land` · `/emergency` | — | Discrete actuation. |
| `POST /control/move` | `{"direction", "cm"}` | Step in a direction. |
| `POST /control/rotate` | `{"deg"}` | Yaw (negative = CCW). |
| `POST /control/geofence` | `{"on": bool}` | Arm/disarm the geofence. |
| `POST /control/snapshot` | `{"label"?}` | Capture a frame to disk; returns its path. |
| `POST /control/target` | `{"queries": [...]}` | Set open-vocab detector queries. |
| `GET /telemetry` · `/pose` · `/observation` · `/status` | — | Read-only context. |

```bash
# fire an autonomous mission and pull the photo when it's done
curl -XPOST localhost:8000/mission -H 'content-type: application/json' \
     -d '{"goal":"go to the plant and take a picture"}'
curl localhost:8000/mission/status            # poll until "phase":"done"
curl -o photo.jpg localhost:8000/mission/photo
```

> No auth: a `POST /mission` makes the drone take off, and the server binds `0.0.0.0`.
> Keep it on a trusted link (the Spark↔Mac port-forward); don't expose port 8000 publicly.

## Controls

**Click the video pane first** so the page captures your keystrokes.

| Key | Action | | Key | Action |
|-----|--------|-|-----|--------|
| `W` / `S` | forward / back | | `↑` / `↓` | ascend / descend |
| `A` / `D` | strafe left / right | | `←` / `→` | yaw left / right |
| `T` | takeoff | | `L` | land |
| `Esc` | **EMERGENCY STOP** | | | |

Buttons mirror these, plus **Snapshot** (saves to `snapshots/`), the **AUTO/MANUAL**
toggle, and a **goal box** (stored now; the agent consumes it in Phase E).
Releasing all movement keys holds a hover (watchdog). Losing window focus releases the
sticks automatically.

## First-flight checklist (props ON — low and slow)

1. Keep the physical space clear and **E-STOP within reach** (`Esc` or the red button).
2. Takeoff → gentle nudges on each axis → confirm directions match expectations.
3. Test **E-STOP** — motors must cut instantly.
4. Toggle **AUTO** then back to **MANUAL**; confirm a key press preempts AUTO instantly.
5. Release keys → confirm a stable **hover**, not drift.
6. Tune `config.py` (`GEOFENCE_RADIUS_CM`, `MAX_HEIGHT_CM`, `MIN/MAX_STEP_CM`, `SPEED`)
   to your room.

## WebSocket protocol (for reference / future clients)

Client → server (JSON):
```jsonc
{"type":"rc","lr":0,"fb":1,"ud":0,"yaw":0}   // stick units in {-1,0,1}; scaled by config.SPEED
{"type":"cmd","name":"takeoff"}               // takeoff | land | snapshot
{"type":"mode","value":"AUTO"}                // AUTO | MANUAL
{"type":"goal","text":"go to the plant ..."}
{"type":"emergency"}
```
Server → client (JSON): `{"type":"status","status":{...telemetry+mode...},"log":[...]}`.

## Troubleshooting

**UI loads but commands do nothing / log says `port 8889 ... ALREADY IN USE`.**
djitellopy binds UDP **8889** (control) and **11111** (video). If a previous run didn't
exit cleanly, those stay held and the control thread can't connect — the dashboard works
but the drone is disabled (header shows the error in red). Free the port and restart:

```bash
pgrep -af 'web.server|uvicorn'    # find stale processes
ss -lunap | grep 8889             # or: lsof -iUDP:8889  — see who holds it
pkill -f web.server               # kill them (or: kill <PID>)
```

Then re-run the server. (Common cause: starting the server in the background and killing
only the `uv` wrapper, which leaves the child Python holding the socket. Prefer running
it in the foreground and stopping with Ctrl+C.)

**`Tello connect failed` in the log.** Not a port issue — check you're on the drone's
WiFi (`TELLO-XXXXXX`) and it's powered on.

## Notes & next steps

- Detection overlay plugs into `render_frame()` in `server.py` (Phase B).
- The goal box and AUTO mode become live once the Phase E agent loop drives the arbiter.
- Startup/shutdown use the FastAPI **lifespan** API (`lifespan=` on the app).
