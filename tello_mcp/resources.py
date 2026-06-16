"""tello_mcp.resources — read-only MCP Resources (the drone's live context).

MCP distinguishes **Tools** (actions the model invokes — takeoff, move, …) from
**Resources** (read-only context the client/model *pulls* — battery, pose, what the
detector sees). The actions already live in the root `tools.py`; this module adds the
resource half so a client can subscribe to state without burning a tool call.

Resource reads still funnel through the single control thread (`run_on_ctl`): the
underlying telemetry getters query djitellopy, so they must not race an actuation send.

URIs:
    tello://telemetry    battery, height, attitude, temp, stream fps
    tello://observation  current target queries, mission phase, live detections
    tello://status       control mode (AUTO/MANUAL), flying, position, heading
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents

_RESOURCES = [
    ("tello://telemetry", "telemetry", "Battery, height, attitude, temperature, stream fps."),
    ("tello://observation", "observation", "Target queries, mission phase, live detections."),
    ("tello://status", "status", "Control mode (AUTO/MANUAL), flying, position, heading."),
]


def register_resources(
    server: Server,
    arb: Any,
    tools: dict,
    run_on_ctl: Callable[..., Awaitable[Any]],
) -> None:
    """Wire the three read-only resources onto an existing low-level `Server`."""

    async def _read(name: str) -> dict:
        if name == "telemetry":
            return await run_on_ctl(tools["get_telemetry"].run, {})
        if name == "observation":
            return await run_on_ctl(tools["get_observation"].run, {})
        if name == "status":
            return await run_on_ctl(arb.status)
        raise KeyError(name)

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:        # noqa: D401
        return [
            types.Resource(uri=uri, name=name, description=desc, mimeType="application/json")
            for uri, name, desc in _RESOURCES
        ]

    @server.read_resource()
    async def read_resource(uri: types.AnyUrl) -> list[ReadResourceContents]:
        name = str(uri).removeprefix("tello://").strip("/")
        try:
            data = await _read(name)
        except KeyError:
            data = {"error": f"unknown resource {uri!s}"}
        return [ReadResourceContents(
            content=json.dumps(data, default=str), mime_type="application/json",
        )]
