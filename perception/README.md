# `perception` — Open-Vocabulary Detection

The **fast** perception layer. A YOLO-World model localizes arbitrary, free-text
targets ("potted plant", "red backpack on the chair") so the agent can servo toward
them — without invoking the slow VLM. The VLM only reasons about these boxes occasionally.

## Contents

| File | Purpose |
|------|---------|
| `detector.py` | `Detector` — YOLO-World wrapper: `set_queries()`, `detect()`, `annotate()` |
| `worker.py` | `PerceptionWorker` — runs the detector on the live feed in its own thread |

## How it works

- **`Detector.set_queries(["potted plant", "chair"])`** sets the open-vocab classes
  (encoded once via CLIP). **`detect(frame)`** returns, per hit:
  `{label, score, box, center, area_frac}`. `area_frac` (box area ÷ frame area) is a
  cheap distance proxy — bigger = closer — used by the Phase D approach controller.
- **`PerceptionWorker`** reads the latest frame, gated on frame **identity** + a short
  sleep so it never busy-loops the GIL and starves the video decoder (the cardinal rule
  in the project `CLAUDE.md`). Query changes are applied inside the worker loop, so the
  model is never reconfigured mid-inference.

## Dependencies

`ultralytics` (pulls `torch`/`torchvision`, CUDA build on the Spark) and `clip-anytorch`
(provides the `clip` text encoder YOLO-World needs for open-vocab queries). Already added
to the project. First `Detector()` downloads the weights; first `set_queries` caches the
CLIP text features.

## Verify it (in the web UI — easiest)

It's wired into the dashboard. On the drone's WiFi:

```bash
cd ~/Desktop/agentic-tello
uv run python -m web.server     # open http://localhost:8000
```

Wait for `Detector ready on cuda.` in the terminal, then type comma-separated targets in
the **Detect** box (e.g. `person, potted plant, chair`) and hit Detect. Green boxes should
appear on the live feed, and the telemetry panel shows **Detections** (count) and
**Det fps**. Clear the box + Detect again to stop.

> Watch `Stream fps` vs `Det fps`. If running detection makes `Stream fps` sag well below
> ~24–30, the detector is starving the decode thread — per `CLAUDE.md`, move the worker to
> a separate process. On the GB10 (CUDA releases the GIL) it should hold.

## In-thread vs. its own process — the trade-off

The detector runs **in its own thread** today, not its own process. That's deliberate, and
the choice is a trade-off, not a free lunch:

- **What a separate process *buys* you:** it removes **GIL** contention. In a separate
  process the video-decode thread never waits on the GIL while Python orchestrates
  inference, so the stream holds its frame rate no matter what the detector does. This is
  the fix to reach for *if* `Stream fps` collapses when detection is on.
- **What it *costs* you (why it isn't the default):**
  1. **IPC / frame copies.** Every frame (~2–3 MB raw at 720p) must cross the process
     boundary, and detections must come back. A naive `Queue`/`pickle` copy per frame can
     *eat the very gain you were chasing*; doing it right means `multiprocessing.shared_memory`
     — more code and more ways to get it wrong.
  2. **Duplicate CUDA context + model.** Each process loads its own CUDA context and its own
     copy of the weights → hundreds of MB of extra VRAM and a slower cold start. Nothing is
     shared in memory.
  3. **Still one GPU.** A separate process removes the *GIL* conflict, **not** the *GPU
     compute* conflict — both still time-slice the same physical GPU. The win is purely on
     the CPU/Python side.
  4. **Operational complexity.** Process lifecycle, clean shutdown, cross-process error
     handling, and debugging all get harder.
- **Why in-thread is the right default:** simpler, lower latency (zero IPC copies), one
  shared model/CUDA context, easy to debug — and **CUDA calls release the GIL**, so on a
  capable GPU the detector already spends most of its time GIL-free. The starvation only
  bites if a *CPU-bound* Python section in the worker hogs the GIL. Principle: don't pay the
  IPC/complexity tax until the `Stream fps` readout proves you need it (matches `CLAUDE.md`:
  *"promote the detector to its own process **if** it starves the decoder"* — conditional).

## Use from code

```python
from perception.detector import Detector
from perception.worker import PerceptionWorker

det = Detector()                              # device auto: cuda / mps / cpu
worker = PerceptionWorker(controller.get_frame, det).start()
worker.set_queries(["potted plant"])
...
worker.detections        # latest list of detections (read anytime)
worker.det_fps           # detector throughput
```
