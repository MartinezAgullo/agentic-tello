# `tello_mcp` — MCP server over the shared tool registry

A thin [Model Context Protocol](https://modelcontextprotocol.io) server (stdio) that
exposes the project's high-level tools so an **external** agent (Claude, another LLM, any
MCP client) can drive the drone or read its state. It wires the same stack the web UI uses
— `TelloController → SafeTello → ControlArbiter` — and reuses `build_registry(ctx)` from
the root `tools.py` **verbatim**: no actuation logic is duplicated here.

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
