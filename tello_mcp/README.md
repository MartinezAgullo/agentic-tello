# `tello_mcp` — MCP server over the shared tool registry

> ⚠️ **The in-process agent does NOT use this MCP server.** The agent brain
> (`agent/loop.py`) calls the tool registry **directly in-process** — e.g.
> `tools["set_target"].run(...)` — because it lives in the *same* process as the tools.
> MCP exists only for an **external** client (a separate Claude/LLM process) to reach the
> same registry across a process boundary. Routing the in-process agent through MCP would
> add a stdio serialization round-trip (ms + async overhead) for zero benefit, wreck the
> fast-loop cadence, and force serializing shared state (`MissionState`, detections) that
> the agent reads live by reference. Same source of truth (`tools.py`) for both paths;
> MCP is the boundary surface, not the agent's call path. See *Who calls what* below.

A thin [Model Context Protocol](https://modelcontextprotocol.io) server (stdio) that
exposes the project's high-level tools so an **external** agent (Claude, another LLM, any
MCP client) can drive the drone or read its state. It is a **pure HTTP proxy**: every
tool and resource is a request to the web server's REST surface (`web/server.py`). The
web server is the single process that owns the drone; this just forwards.

> **It no longer opens its own drone connection.** It used to wire
> `TelloController → SafeTello → ControlArbiter` and bind UDP 8889, which meant it could
> not run alongside `web.server`. Now it binds no drone port, so it **runs concurrently
> with the web server** — which is the intended topology: the web server owns the Tello
> on the Spark; the Mac-side agent reaches it over the forwarded port through this proxy.

Point it at the web server with `TELLO_WEB_URL` (default `http://localhost:8000`).

## What it does (and doesn't)

- ✅ Request/response **planning + high-level acts**: `start_mission` (NL goal → AUTO →
  takeoff), arm/manual, takeoff/land, move/rotate by step, set detector target, snapshot,
  telemetry, declare done, e-stop — plus `get_mission_photo` to pull the latest capture.
- ✅ **Safety preserved, server-side.** Actuating REST endpoints pass through the same
  arbiter / `SafeTello` guards (mode gate, geofence, height/battery caps, watchdog) and
  the single drone control thread. A 409 from the server surfaces as `BLOCKED:`.
- ✅ **Concurrent with the web server.** No djitellopy access here → no UDP-port conflict.
  It *requires* the web server to be running; if it's down, tools return an error.
- ❌ **Not the fast loop.** Continuous velocity control (`rc`, servoing) is deliberately
  not a tool — a request per control tick would wreck the cadence.
- ❌ **Adds no guards of its own.** All safety is enforced by the web server it proxies.

## Who calls what

The agent never goes through MCP; the proxy and the agent both end at the same REST/tool
surface, but the proxy crosses a process (and possibly host) boundary over HTTP:

```
agent/loop.py (in-process, on the Spark)  ──direct function call──► tools.py ──► ControlArbiter → SafeTello → TelloController
                                                                       ▲
external MCP client ──MCP stdio──► tello_mcp.server ──HTTP──► web.server REST ──┘
   (e.g. the Mac-side agent)                          :8000
```

- **In-process agent path** (normal operation): `agent/loop.py` → `tools[...].run(...)`.
  No serialization, no MCP, no extra process.
- **MCP proxy path** (optional, external): an MCP client → `tello_mcp.server` → HTTP →
  the web server's REST endpoints. Used to drive missions from outside the Spark process.

If you're wondering "does the running system need the MCP server?" — **no**. `main.py`
runs the full agent + web UI without it.

## Tools

Each tool is one HTTP call to a web-server REST endpoint. Mission-level and fine-grained
control both surface, so an MCP client keeps the full control it had when this server drove
the drone directly:

| Tool | REST endpoint | Does |
|------|---------------|------|
| `start_mission` | `POST /mission` | NL goal → arm AUTO → takeoff; returns a `mission_id` |
| `mission_status` | `GET /mission/status` | Poll the blackboard (`phase=='done'` ⇒ goal met) |
| `get_mission_photo` | `GET /mission/photo` | Download the latest mission capture as an image |
| `report_done` | `POST /mission/done` | Declare the goal satisfied |
| `arm_auto` / `to_manual` | `POST /control/mode` | Mode gate — **arm AUTO before takeoff/move/rotate** |
| `takeoff` / `land` | `POST /control/{takeoff,land}` | Take off / land |
| `move` / `rotate` | `POST /control/{move,rotate}` | Discrete step (cm) / yaw (deg) |
| `set_target` | `POST /control/target` | Set open-vocab detector queries |
| `take_snapshot` | `POST /control/snapshot` | Capture a frame to disk on the Spark |
| `emergency_stop` | `POST /control/emergency` | Cut motors (bypasses guards) |
| `get_telemetry` / `get_pose` / `get_observation` / `get_status` | `GET /telemetry`,`/pose`,`/observation`,`/status` | Read-only context |

A typical autonomous session is one call: `start_mission` → poll `mission_status` until
`phase == "done"` → `get_mission_photo`. Or drive manually: `arm_auto` → `takeoff` →
`set_target` → `move`/`rotate` → `take_snapshot` → `land`. Refusals (geofence, MANUAL,
caps) come back as `BLOCKED: …`; other failures as `ERROR: …` — the server never crashes
on a bad call.

## Resources (read-only context)

MCP separates **Tools** (actions the model invokes) from **Resources** (read-only state
the client/model *pulls*). The drone's live state is exposed as JSON resources so a client
can read context without spending a tool call:

| URI | Contents |
|-----|----------|
| `tello://telemetry` | battery, height, attitude, temperature, stream fps |
| `tello://observation` | target queries, mission phase, live detections |
| `tello://status` | control mode (AUTO/MANUAL), flying, position, heading |

Each resource read is a `GET` against the web server. The `get_telemetry` /
`get_observation` / `get_status` **tools** are kept too, so tool-only clients (e.g. Claude
Code) can still read state without resource support.

## Layout

A thin adaptation of the conventional MCP-server scaffolding — the parts that already
exist in this repo are reused, not duplicated (single source of truth):

```
tello_mcp/
├── server.py     # MCP server (low-level Server) + the tool table → HTTP forwarder
├── resources.py  # read-only Resources (telemetry/observation/status) over HTTP
└── README.md
#  REST target → web/server.py   (the process that owns the drone)
#  TELLO_WEB_URL  selects it      (default http://localhost:8000)
```

## Run

Start the **web server first** (it owns the drone), then this proxy. From the project root
(root-relative imports — don't run from inside `tello_mcp/`):

```bash
uv run python -m web.server                              # terminal 1: owns the Tello
uv run python -m tello_mcp.server                        # terminal 2: the MCP proxy
TELLO_WEB_URL=http://spark:8000 uv run python -m tello_mcp.server   # or point at a remote host
```

Logs go to **stderr** (stdout is the MCP transport). On startup it probes the web server
and warns if it's unreachable; tools error until the server is up. No UDP-port conflict —
it can run at the same time as `web.server`.

### Register with an MCP client (e.g. Claude Code)

```bash
claude mcp add tello -- uv run --directory ~/Desktop/agentic-tello python -m tello_mcp.server
```

Or add it to your client's MCP config manually:

```json
{
  "mcpServers": {
    "tello": {
      "command": "uv",
      "args": ["run", "--directory", "/home/agullo/Desktop/agentic-tello",
               "python", "-m", "tello_mcp.server"]
    }
  }
}
```

> First flights low and slow, props-on only after `bench_test.py`, E-STOP in reach.
> An external agent driving over MCP is still bounded by the same `config.py` caps.
