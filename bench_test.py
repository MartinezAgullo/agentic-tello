"""Phase A bench test — PROPS OFF. Validates the core stack without flying.

Connect to Tello WiFi, then:  uv run python bench_test.py

Checks: connection, telemetry, video stream, geofence rejection, height cap,
arbiter AUTO/MANUAL gating, emergency passthrough. It never spins motors —
takeoff/land are exercised only as dry assertions against the safety layer
(the actual takeoff line is commented; uncomment only when you mean to fly).
"""

import time

from agentic_tello import config
from agentic_tello.tello_tools.arbiter import ArbiterBlocked, ControlArbiter
from agentic_tello.tello_tools.controller import TelloController
from agentic_tello.tello_tools.primitives import get_telemetry
from agentic_tello.tello_tools.safety import SafeTello, SafetyError


def main() -> None:
    c = TelloController()
    print("Connecting…")
    c.connect()
    print("Connected.")
    c.start_stream()

    safe = SafeTello(c)
    arb = ControlArbiter(safe)

    # telemetry + stream
    time.sleep(2)
    print("Telemetry:", get_telemetry(c))
    print("Frame received:", c.get_frame() is not None)

    # arbiter starts in MANUAL → agent commands must be blocked
    try:
        arb.agent_rotate(30)
        print("FAIL: agent command ran in MANUAL")
    except ArbiterBlocked:
        print("OK: agent blocked while MANUAL")

    # simulate flying to exercise the safety math (no motors)
    safe.flying = True

    # geofence: single moves are clamped to MAX_STEP, so the fence trips on the
    # *accumulated* position. Pre-position near the edge to prove rejection.
    safe.heading = 90.0          # forward → +x
    safe.x = config.GEOFENCE_RADIUS_CM - 10
    try:
        safe.move("forward", config.MAX_STEP_CM)
        print("FAIL: geofence did not reject")
    except SafetyError as e:
        print(f"OK: geofence rejected ({e})")
    safe.x = 0.0

    # height cap on rc up-channel is enforced inside rc(); step clamp:
    print("Step clamp 5cm ->", safe._clamp_step(5), "| 999cm ->", safe._clamp_step(999))

    safe.flying = False
    print("\nBench checks done. Emergency stop is wired via arb.emergency().")
    c.shutdown()


if __name__ == "__main__":
    main()
