"""tello_mcp.resources — read-only MCP Resources (the drone's live context).

MCP distinguishes **Tools** (actions the model invokes — takeoff, move, …) from
**Resources** (read-only context the client/model *pulls* — battery, pose, what the
detector sees). The actions are advertised by `tello_mcp.server`; this module adds the
resource half so a client can read state without burning a tool call.

Like the tools, these are now thin HTTP proxies: each resource read is a GET against
the web server's REST surface (the single process that owns the drone). No djitellopy
access happens here.

URIs:
    tello://telemetry    battery, height, attitude, temp, stream fps
    tello://observation  current target queries, mission phase, live detections
    tello://status       full control HUD (mode, flying, position, heading, …)
"""

import json

import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents

# resource name -> (REST path, description)
_RESOURCES = {
    "telemetry": ("/telemetry", "Battery, height, attitude, temperature, stream fps."),
    "observation": ("/observation", "Target queries, mission phase, live detections."),
    "status": ("/status", "Full control HUD (mode AUTO/MANUAL, flying, position, heading)."),
}


def register_resources(server: Server, client: httpx.AsyncClient) -> None:
    """Wire the read-only resources onto an existing low-level `Server`, served over HTTP."""

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:        # noqa: D401
        return [
            types.Resource(uri=f"tello://{name}", name=name, description=desc,
                           mimeType="application/json")
            for name, (_path, desc) in _RESOURCES.items()
        ]

    @server.read_resource()
    async def read_resource(uri: types.AnyUrl) -> list[ReadResourceContents]:
        name = str(uri).removeprefix("tello://").strip("/")
        entry = _RESOURCES.get(name)
        try:
            if entry is None:
                data = {"error": f"unknown resource {uri!s}"}
            else:
                r = await client.get(entry[0])
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            data = {"error": f"web server unreachable ({e})"}
        return [ReadResourceContents(
            content=json.dumps(data, default=str), mime_type="application/json",
        )]
