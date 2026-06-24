"""tello_mcp.server — a thin HTTP proxy bridging MCP ⇆ the web server's REST API.

This used to open its **own** djitellopy connection (UDP 8889) and wire the full
control stack, which meant it could NOT run at the same time as `web.server`. It is
now a pure proxy: every tool and resource is an HTTP call to the running web server's
REST surface (see `web/server.py`). The web server is the single process that owns the
drone; this just forwards. Consequences:

- It binds no drone port → it CAN run alongside `web.server` (the intended topology:
  the web server owns the Tello on the Spark; an external agent on the Mac reaches it
  over the forwarded port via this proxy or directly over REST).
- It REQUIRES the web server to be up. Point it with `TELLO_WEB_URL`
  (default `http://localhost:8000`). If the server is down, tools return an error —
  they do not fly anything.
- Safety stays server-side: the arbiter / SafeTello caps and the AUTO/MANUAL mode gate
  are enforced by the web server. `arm_auto` is still required before actuating tools.
  This proxy adds no guards of its own.

Run it from the project root:

    uv run python -m tello_mcp.server                 # talks to http://localhost:8000
    TELLO_WEB_URL=http://spark:8000 uv run python -m tello_mcp.server
"""

import asyncio
import base64
import json
import os
import sys

import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from tello_mcp.resources import register_resources

WEB_URL = os.environ.get("TELLO_WEB_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT = 15.0


def _log(msg: str) -> None:
    """Logs go to stderr — stdout is the MCP transport and must stay clean."""
    print(f"[tello-mcp] {msg}", file=sys.stderr, flush=True)


# ── tool table ────────────────────────────────────────────────────────────────
# name -> (description, json-schema properties, HTTP method, path, body-builder)
# The body-builder maps the MCP arguments to the JSON body (None ⇒ no body, e.g. GETs
# and parameterless POSTs). Mirrors the root tool registry so an MCP client keeps the
# same fine-grained control it had when this server drove the drone directly.
_Tool = tuple[str, dict, str, str, object]
_TOOLS: dict[str, _Tool] = {
    # mission lifecycle (the orchestrator's main path)
    "start_mission": (
        "Start an autonomous mission from a natural-language goal (e.g. 'go to the "
        "plant and take a picture'). The drone arms AUTO, takes off, then searches / "
        "approaches / captures on its own. Returns a mission_id; poll mission_status.",
        {"goal": {"type": "string"}},
        "POST", "/mission", lambda a: {"goal": a.get("goal", "")}),
    "mission_status": (
        "Poll the mission blackboard. phase=='done' means the goal is satisfied; "
        "photo_available flags a capture you can fetch with get_mission_photo.",
        {}, "GET", "/mission/status", None),
    "report_done": (
        "Declare the current mission goal satisfied (phase → done, disarms it).",
        {"reason": {"type": "string"}},
        "POST", "/mission/done", lambda a: {"reason": a.get("reason", "")}),
    # mode gate
    "arm_auto": (
        "Hand control to the agent (mode → AUTO). Required before takeoff/move/rotate.",
        {}, "POST", "/control/mode", lambda a: {"mode": "AUTO"}),
    "to_manual": (
        "Return control to the operator (mode → MANUAL); the drone holds hover and "
        "actuating tools are then blocked.",
        {}, "POST", "/control/mode", lambda a: {"mode": "MANUAL"}),
    # actuation
    "takeoff": ("Take off and hover.", {}, "POST", "/control/takeoff", None),
    "land": ("Land safely.", {}, "POST", "/control/land", None),
    "move": (
        "Move a discrete step (cm) in one direction.",
        {"direction": {"type": "string",
                       "enum": ["forward", "back", "left", "right", "up", "down"]},
         "cm": {"type": "integer"}},
        "POST", "/control/move",
        lambda a: {"direction": a.get("direction"), "cm": a.get("cm")}),
    "rotate": (
        "Rotate clockwise by degrees (negative = counter-clockwise).",
        {"deg": {"type": "integer"}},
        "POST", "/control/rotate", lambda a: {"deg": a.get("deg")}),
    "set_target": (
        "Set the open-vocabulary object(s) the detector should localize / servo toward.",
        {"queries": {"type": "array", "items": {"type": "string"}}},
        "POST", "/control/target", lambda a: {"queries": a.get("queries", [])}),
    "take_snapshot": (
        "Capture the current camera frame to disk on the Spark; returns its path. "
        "Use get_mission_photo to download the latest mission capture as an image.",
        {"label": {"type": "string"}},
        "POST", "/control/snapshot", lambda a: {"label": a.get("label", "manual")}),
    "emergency_stop": (
        "Cut motors immediately (bypasses all guards).",
        {}, "POST", "/control/emergency", None),
    # read-only context (also available as MCP resources)
    "get_telemetry": (
        "Read battery, height, attitude, temperature, stream fps.",
        {}, "GET", "/telemetry", None),
    "get_pose": (
        "Dead-reckoning pose (x, y, heading) relative to takeoff.",
        {}, "GET", "/pose", None),
    "get_observation": (
        "Current target queries, mission phase and live detections.",
        {}, "GET", "/observation", None),
    "get_status": (
        "Read the control HUD: mode (AUTO/MANUAL), flying, battery, position, heading.",
        {}, "GET", "/status", None),
}

# served alongside _TOOLS but handled specially (returns an image, not JSON text)
_PHOTO_TOOL = (
    "get_mission_photo",
    "Download the latest snapshot the mission captured, as an image. Returns text "
    "'no mission photo yet' until a capture exists.",
)


def _format(r: httpx.Response) -> types.TextContent:
    """Render a JSON REST response as MCP text, mapping HTTP errors to BLOCKED/ERROR."""
    try:
        data = r.json()
        text = json.dumps(data)
    except ValueError:
        data, text = None, r.text
    detail = data.get("detail") if isinstance(data, dict) else None
    if r.status_code == 409:                # refused by a safety guard / mode gate
        return types.TextContent(type="text", text=f"BLOCKED: {detail or text}")
    if r.status_code >= 400:
        return types.TextContent(type="text", text=f"ERROR: {detail or text}")
    return types.TextContent(type="text", text=text)


def _build_server(client: httpx.AsyncClient) -> Server:
    server = Server("tello")

    def _mcp_tools() -> list[types.Tool]:
        out = [types.Tool(name=name, description=desc,
                          inputSchema={"type": "object", "properties": props})
               for name, (desc, props, *_rest) in _TOOLS.items()]
        out.append(types.Tool(name=_PHOTO_TOOL[0], description=_PHOTO_TOOL[1],
                              inputSchema={"type": "object", "properties": {}}))
        return out

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:        # noqa: D401
        return _mcp_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None):
        args = arguments or {}
        try:
            if name == _PHOTO_TOOL[0]:
                return await _get_photo(client)
            spec = _TOOLS.get(name)
            if spec is None:
                return [types.TextContent(type="text", text=f"ERROR: unknown tool {name!r}")]
            _desc, _props, method, path, build = spec
            kwargs = {}
            if build is not None and method != "GET":
                kwargs["json"] = build(args)
            r = await client.request(method, path, **kwargs)
            return [_format(r)]
        except httpx.HTTPError as e:
            return [types.TextContent(
                type="text", text=f"ERROR: web server unreachable at {WEB_URL} ({e})")]

    register_resources(server, client)
    return server


async def _get_photo(client: httpx.AsyncClient) -> list[types.ImageContent | types.TextContent]:
    r = await client.get("/mission/photo")
    if r.status_code == 404:
        return [types.TextContent(type="text", text="no mission photo yet")]
    if r.status_code >= 400:
        return [types.TextContent(type="text", text=f"ERROR: {r.text}")]
    return [types.ImageContent(
        type="image",
        data=base64.b64encode(r.content).decode(),
        mimeType=r.headers.get("content-type", "image/jpeg"),
    )]


async def _serve() -> None:
    async with httpx.AsyncClient(base_url=WEB_URL, timeout=_TIMEOUT) as client:
        # Best-effort reachability probe so the operator gets a clear message early.
        try:
            await client.get("/status")
            _log(f"connected to web server at {WEB_URL}.")
        except httpx.HTTPError as e:
            _log(f"WARNING: web server not reachable at {WEB_URL} ({e}). "
                 "Start it (uv run python -m web.server) — tools will error until it is up.")

        server = _build_server(client)
        _log("serving on stdio. Tools ready — arm_auto then takeoff, or start_mission.")
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


def main() -> None:
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
