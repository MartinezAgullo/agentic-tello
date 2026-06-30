# `agent` — Dual-Cadence Sense-Plan-Act Loop

Wires the slow VLM planner (`brain/`) and the fast deterministic controller into a single
mission loop. Owns the phase machine, the mission state blackboard, and the visual servoing
reflex.

## Contents

| File | Purpose |
|------|---------|
| `loop.py` | `AgentBrain` — slow-loop planner thread + `fast_step()` called every control tick |
| `state.py` | `MissionState` — shared blackboard between slow and fast loops (phases, targets, steps) |
| `servoing.py` | `Servoer` — maps one detection box to `rc` velocity commands (center + approach) |

## Phase machine

Each sub-step of a mission advances through:

```
SEARCH → APPROACH → CAPTURE → (next step or DONE)
```

- **SEARCH** — target named but not in view; the fast loop runs a scan pattern (rotate +
  translate between vantage points).
- **APPROACH** — target visible; `Servoer` centers it and closes in via `rc` commands.
- **CAPTURE** — centered and close enough; snapshot is taken.
- **WATCH** — passive variant: wait for a subject to appear, then snapshot on a timer.
- **DONE** — goal satisfied; hover and wait for the operator.

The fast loop drives these transitions deterministically; the slow loop (VLM) only sets
*what* to look for and judges *whether* it's done.

## Multi-step goals

The VLM decomposes a goal into ordered typed steps (`find`, `move`, `rotate`, `climb`, …)
once at mission start. The phase machine runs once per step. For the deterministic marker
survey mode, `start_survey()` installs the steps directly, bypassing the VLM entirely.
