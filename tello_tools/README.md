# `tello_tools` — Core Control Library

The hardware layer for the agentic Tello controller: drone connection, low-latency
video, and **every safety guard**. Higher layers (perception, brain, web UI) build on
this; they should never talk to djitellopy directly.

## Layering

Every actuation funnels through one chokepoint — top to bottom:

```
your code / agent loop / web UI
        │
        ▼
ControlArbiter   ── AUTO vs MANUAL; operator input always preempts the agent
        │
        ▼
SafeTello        ── clamps steps, geofence, height/battery caps, watchdog
        │
        ▼
TelloController  ── raw djitellopy + LowLatencyFrameRead video stream
        │
        ▼
      Tello
```

Rule of thumb: **agents/UI call `ControlArbiter`**, never `SafeTello` or
`TelloController` directly — except `arbiter.emergency()`, which bypasses every guard.

## Components

| File | Class / fn | Purpose |
|------|-----------|---------|
| `controller.py` | `TelloController` | connection, video, **raw** actuation (`_takeoff`, `_rc`, …) |
| `controller.py` | `LowLatencyFrameRead` | UDP-drain + continuous backlog-skip frame reader |
| `safety.py` | `SafeTello` | guarded actuation; raises `SafetyError` when a cap is hit |
| `arbiter.py` | `ControlArbiter` | AUTO/MANUAL mode gate; raises `ArbiterBlocked` |
| `primitives.py` | `take_snapshot`, `get_telemetry` | non-actuating reads |

All limits live in the project-root `config.py` (geofence radius, height cap, step
size, battery floor, watchdog, stream params).

## Prerequisites

1. Dependencies installed (from the project root):
   ```bash
   cd ~/Desktop/agentic-tello
   uv sync
   ```
2. Connected to the drone's WiFi (`TELLO-XXXXXX`).
3. Run everything with `uv run` so the project venv + root imports resolve.

> Imports are root-relative (`import config`, `from tello_tools... import ...`), so run
> scripts **from the project root**, not from inside `tello_tools/`.

## Quick start

```python
import time
from tello_tools.controller import TelloController
from tello_tools.safety import SafeTello
from tello_tools.arbiter import ControlArbiter
from tello_tools.primitives import get_telemetry, take_snapshot

# 1. connect + start video
c = TelloController()
c.connect()
c.start_stream()
time.sleep(2)                      # let the stream warm up

# 2. wrap in the safety + arbiter layers
safe = SafeTello(c)
arb  = ControlArbiter(safe)        # starts in MANUAL

# 3. reads need no guard
print(get_telemetry(c))            # battery, height, temp, stream fps
take_snapshot(c, label="test")     # → snapshots/test_YYYYmmdd_HHMMSS.jpg

# 4a. MANUAL flight (operator) — any manual_* call seizes control
arb.manual_takeoff()
arb.manual_rc(lr=0, fb=20, ud=0, yaw=0)   # ease forward (clamped to config.SPEED)
time.sleep(1)
arb.manual_land()

# 4b. AUTO flight (agent) — must arm first, else ArbiterBlocked
arb.arm_auto()
arb.agent_takeoff()
arb.agent_move("forward", 30)      # cm; clamped to [MIN_STEP, MAX_STEP], geofenced
arb.agent_rotate(45)               # degrees, clockwise
arb.agent_land()

# 5. shut down cleanly
c.shutdown()
```

### Keeping the drone safe in a loop

Call `arb.tick()` every iteration of your control loop. It auto-hovers if no command
arrived within `WATCHDOG_S` and auto-lands at the battery floor (raises `SafetyError`
when it does, so catch it):

```python
from tello_tools.safety import SafetyError
while running:
    try:
        arb.tick()
    except SafetyError as e:
        print("safety:", e)        # e.g. battery floor → auto-landed
    ...
```

### Modes & preemption

- `arb.arm_auto()` — hand control to the agent (`agent_*` calls execute).
- `arb.to_manual()` — return control to the operator; agent is frozen (hovers).
- Any `manual_*` call **flips to MANUAL automatically** — the human never fights the
  agent for the sticks. Re-arming AUTO is always explicit.
- `arb.emergency()` — cut motors **now**, bypassing all guards.
- `arb.status()` — `{mode, flying, pos, heading}` for a HUD.

## Bench test (props OFF — do this first)

A no-flight validation of the whole stack lives at the project root:

```bash
cd ~/Desktop/agentic-tello
uv run python bench_test.py
```

It checks connection, telemetry, a live frame, that the agent is blocked while MANUAL,
and that the geofence rejects an out-of-bounds move — **without spinning the motors**.
Run this before any flight, then do first flights low, slow, with E-STOP in reach.

## Notes

- `TelloController` exposes raw `_`-prefixed actuation by design — go through
  `SafeTello`/`ControlArbiter` instead, except for `emergency()`.
- `LowLatencyFrameRead` continuously skips stale backlog frames so latency can't
  accumulate, and the decode thread must not be GIL-starved — see the project
  `CLAUDE.md` ("Streaming + concurrency lessons") before adding worker threads.
