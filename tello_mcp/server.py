"""tello_mcp.server — a thin MCP server over the same tool registry (`tools.py`).

This advertises the project's high-level tools over MCP (stdio) so an external agent
(Claude, another LLM, an MCP client) can drive the drone or read its state. It wires
the **same** stack the web UI uses — `TelloController → SafeTello → ControlArbiter` —
and reuses `build_registry(ctx)` verbatim: no actuation logic is duplicated here.

Design rules it honours (see project `CLAUDE.md`):

- **One chokepoint.** djitellopy is not safe for concurrent sends, so *every* actuation
  — tool calls and the periodic `arb.tick()` watchdog — runs on a single control thread
  (a 1-worker executor). The async MCP handlers only enqueue onto it.
- **Not the fast loop.** MCP is request/response for planning + manual high-level acts;
  continuous servoing (`rc`) is deliberately NOT a tool. Don't drive a per-tick reflex
  over MCP — the round-trip would wreck the cadence.
- **Safety is not bypassed.** Actuating tools still pass through the arbiter/SafeTello
  guards (mode gate, geofence, height/battery caps, watchdog). `emergency_stop` is the
  one guard-bypassing escape hatch, exactly as elsewhere.

Run it from the project root (root-relative imports):

    uv run python -m tello_mcp.server

Because it opens the single djitellopy connection (binds UDP 8889), it cannot run at the
same time as `main.py` / the web server — use one or the other.
"""

import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from agent.state import MissionState
from tello_tools.arbiter import ArbiterBlocked, ControlArbiter
from tello_tools.controller import TelloController
from tello_tools.safety import SafeTello, SafetyError
from tools import ToolContext, build_registry

# Detector is optional: if it fails to load (or no CUDA), set_target/get_observation
# still work, they just won't have detections. Import lazily so a missing model never
# stops the control tools from being served.
try:
    from perception.detector import Detector
    from perception.worker import PerceptionWorker
except Exception:                       # noqa: BLE001 — perception is best-effort here
    Detector = None
    PerceptionWorker = None


def _log(msg: str) -> None:
    """Logs go to stderr — stdout is the MCP transport and must stay clean."""
    print(f"[tello-mcp] {msg}", file=sys.stderr, flush=True)


# ── single control thread: the only place the drone is actuated ───────────────
_ctl = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tello-ctl")


def _on_ctl(fn, *args):
    """Run a blocking actuation on the one control thread and wait for it."""
    return _ctl.submit(fn, *args).result()


async def _await_ctl(fn, *args):
    """Async wrapper: schedule on the control thread without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ctl, lambda: fn(*args))


def _connect() -> tuple[TelloController, ControlArbiter, MissionState, object | None, dict]:
    """Wire the full stack exactly like the web server, on the control thread."""
    _log("connecting to Tello…")
    c = TelloController()               # binds UDP 8889 — fails if a prior run holds it
    c.connect()
    c.start_stream()
    _log("connected, stream started.")

    safe = SafeTello(c)
    arb = ControlArbiter(safe)          # starts in MANUAL; client must arm_auto to fly
    state = MissionState()

    worker = None
    if Detector is not None and PerceptionWorker is not None:
        try:
            det = Detector()
            worker = PerceptionWorker(c.get_frame, det).start()
            _log("detector loaded.")
        except Exception as e:          # noqa: BLE001 — degrade, don't die
            _log(f"detector unavailable ({e}); set_target/get_observation run without detections.")

    ctx = ToolContext(arbiter=arb, controller=c, worker=worker, state=state)
    tools = build_registry(ctx)
    return c, arb, state, worker, tools


def _watchdog(arb: ControlArbiter, stop: threading.Event) -> None:
    """Periodic safety tick (auto-hover on command gap, auto-land at battery floor),
    serialized onto the same single control thread as every actuation."""
    while not stop.is_set():
        try:
            _on_ctl(arb.tick)
        except SafetyError as e:
            _log(f"safety: {e}")
        except Exception:               # noqa: BLE001 — executor shutting down
            return
        time.sleep(0.1)


def _build_server(arb: ControlArbiter, tools: dict) -> Server:
    server = Server("tello")

    # ── server-level tools layered on the shared registry ────────────────────
    # The registry's actuating tools call arb.agent_*, which require AUTO. These
    # three let an MCP client manage the mode gate and read the HUD — without them
    # the registry tools would be unreachable (ArbiterBlocked) or blind.
    def _arm_auto(_):  arb.arm_auto();  return "mode → AUTO (agent tools now execute)"
    def _to_manual(_): arb.to_manual(); return "mode → MANUAL (agent tools blocked; drone hovers)"
    def _status(_):    return arb.status()

    server_tools = {
        "arm_auto": ("Hand control to the agent so actuating tools execute (mode → AUTO). "
                     "Required before takeoff/move/rotate/land.", {}, _arm_auto),
        "to_manual": ("Return control to the operator (mode → MANUAL); actuating tools are "
                      "then blocked and the drone holds hover.", {}, _to_manual),
        "get_status": ("Read the control state: mode (AUTO/MANUAL), flying, position, heading.",
                       {}, _status),
    }

    def _mcp_tools() -> list[types.Tool]:
        out: list[types.Tool] = []
        for name, (desc, props, _fn) in server_tools.items():
            out.append(types.Tool(
                name=name, description=desc,
                inputSchema={"type": "object", "properties": props},
            ))
        for t in tools.values():
            out.append(types.Tool(
                name=t.name, description=t.description,
                inputSchema={"type": "object", "properties": t.parameters},
            ))
        return out

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:        # noqa: D401
        return _mcp_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        args = arguments or {}
        try:
            if name in server_tools:
                fn = server_tools[name][2]
                result = await _await_ctl(fn, args)
            elif name in tools:
                # Every registry tool runs on the single control thread, serialized
                # against the watchdog → no concurrent djitellopy sends.
                result = await _await_ctl(tools[name].run, args)
            else:
                return [types.TextContent(type="text", text=f"ERROR: unknown tool {name!r}")]
            return [types.TextContent(type="text", text=str(result))]
        except (SafetyError, ArbiterBlocked) as e:
            # Expected refusals (geofence, MANUAL, caps) — report, don't crash.
            return [types.TextContent(type="text", text=f"BLOCKED: {e}")]
        except Exception as e:                          # noqa: BLE001
            return [types.TextContent(type="text", text=f"ERROR: {e}")]

    return server


async def _serve() -> None:
    try:
        c, arb, _state, worker, tools = _on_ctl(_connect)
    except OSError as e:
        import errno
        if e.errno == errno.EADDRINUSE:
            _log("control port 8889 is ALREADY IN USE — a previous run (web.server / "
                 "bench_test / another tello_mcp) still holds it. Free it and retry:")
            _log("    pkill -f 'web.server|tello_mcp'   # then re-run")
        else:
            _log(f"Tello connect failed: {e} — check the drone WiFi.")
        raise SystemExit(1) from e
    except Exception as e:                              # noqa: BLE001
        _log(f"Tello connect failed: {e} — check the drone WiFi.")
        raise SystemExit(1) from e

    stop = threading.Event()
    wd = threading.Thread(target=_watchdog, args=(arb, stop), daemon=True)
    wd.start()

    server = _build_server(arb, tools)
    _log("serving on stdio. Tools ready — arm_auto then takeoff to fly.")
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        stop.set()
        try:
            if arb.safe.flying:
                _log("auto-landing before exit…")
                _on_ctl(arb.manual_land)
        except Exception as e:                          # noqa: BLE001
            _log(f"landing on exit failed: {e}")
        if worker is not None:
            try: worker.stop()
            except Exception: pass                      # noqa: BLE001,E722
        _on_ctl(c.shutdown)
        _ctl.shutdown(wait=False)
        _log("drone disconnected, UDP ports released.")


def main() -> None:
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
