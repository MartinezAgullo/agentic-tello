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
MCP client) can drive the drone or read its state. It wires the same stack the web UI uses
— `TelloController → SafeTello → ControlArbiter` — and reuses `build_registry(ctx)` from
the root `tools.py` **verbatim**: no actuation logic is duplicated here.

> **Why low-level `Server`, not FastMCP?** FastMCP infers each tool's schema from a typed
> Python signature, so using it would mean re-declaring all 13 tools here and duplicating
> the JSON schemas that already live in `tools.py`. The low-level server registers that
> existing registry as-is. FastMCP is the right default for a *greenfield* server that
> writes its tools from scratch; here the registry is the single source of truth.

## What it does (and doesn't)

- ✅ Request/response **planning + high-level acts**: arm, takeoff, move/rotate by step,
  set detector target, snapshot, telemetry, declare done, e-stop.
- ✅ **One chokepoint.** Every actuation — tool calls *and* the periodic `arb.tick()`
  watchdog — runs on a single control thread (a 1-worker executor), so djitellopy never
  gets concurrent sends. The async MCP handlers only enqueue onto it.
- ✅ **Safety preserved.** Actuating tools still pass through the arbiter / `SafeTello`
  guards (mode gate, geofence, height/battery caps, watchdog). `emergency_stop` is the
  single guard-bypassing escape hatch.
- ❌ **Not the fast loop.** Continuous velocity control (`rc`, servoing) is deliberately
  not a tool — an MCP round-trip per control tick would wreck the cadence.
- ❌ **Not concurrent with the web server.** It opens the single djitellopy connection
  (binds UDP 8889). Run `tello_mcp` *or* `main.py`, not both.

## Who calls what

Two independent callers reach the **same** registry; the agent never goes through MCP:

```
agent/loop.py (in-process)  ──direct function call──┐
                                                    ▼
external client ──MCP stdio──► tello_mcp.server ──► tools.py  (build_registry)
   (separate Claude/LLM)                              │
                                                      ▼
                                  ControlArbiter → SafeTello → TelloController
```

- **In-process agent path** (the one actually used in normal operation): `agent/loop.py`
  → `tools[...].run(...)`. No serialization, no MCP, no extra process.
- **MCP path** (optional, external only): a separate client → `tello_mcp.server` → the
  same `tools.py`. Only used when you deliberately drive the drone from outside the system.

If you're wondering "does the running system need the MCP server?" — **no**. `main.py`
runs the full agent + web UI without it.

## Tools

The 10 registry tools (see `tello_tools/README.md` → *Tool registry*) plus 3 mode-control
tools added here, because the registry's actuating tools call `arb.agent_*` and require
**AUTO** — without these they'd be unreachable:

| Tool | Does |
|------|------|
| `arm_auto` | Mode → AUTO so actuating tools execute. **Call before takeoff/move/rotate.** |
| `to_manual` | Mode → MANUAL; actuating tools blocked, drone holds hover. |
| `get_status` | Read mode (AUTO/MANUAL), flying, position, heading. |

A typical session: `arm_auto` → `takeoff` → `set_target` → `move`/`rotate` →
`take_snapshot` → `land`. Tool refusals (geofence, MANUAL, caps) come back as
`BLOCKED: …`; other failures as `ERROR: …` — the server never crashes on a bad call.

## Resources (read-only context)

MCP separates **Tools** (actions the model invokes) from **Resources** (read-only state
the client/model *pulls*). The drone's live state is exposed as JSON resources so a client
can read context without spending a tool call:

| URI | Contents |
|-----|----------|
| `tello://telemetry` | battery, height, attitude, temperature, stream fps |
| `tello://observation` | target queries, mission phase, live detections |
| `tello://status` | control mode (AUTO/MANUAL), flying, position, heading |

The `get_telemetry` / `get_observation` / `get_status` **tools** are kept too, so
tool-only clients (e.g. Claude Code) can still read state without resource support.

## Layout

A thin adaptation of the conventional MCP-server scaffolding — the parts that already
exist in this repo are reused, not duplicated (single source of truth):

```
tello_mcp/
├── server.py     # MCP server init (low-level Server) + lifespan + single control thread
├── resources.py  # read-only Resources (telemetry/observation/status)
└── README.md
#  tools  →  root tools.py      (the shared registry — "actions")
#  sdk    →  tello_tools/       (the real drone/safety stack — "drone_sdk")
#  config →  root config.py     (env-overridable caps; no separate .env)
```

## Run

From the project root (root-relative imports — don't run from inside `tello_mcp/`):

```bash
uv run python -m tello_mcp.server
```

Logs go to **stderr** (stdout is the MCP transport). It connects to the drone first; if
UDP 8889 is busy, free the stale process (`pkill -f 'web.server|tello_mcp'`) and retry.

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
