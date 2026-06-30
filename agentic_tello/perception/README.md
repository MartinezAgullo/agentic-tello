# `perception` — Open-Vocabulary Detection

The **fast** perception layer. A YOLO-World model localizes arbitrary, free-text targets so
the agent can servo toward them — without invoking the slow VLM.

## Contents

| File | Purpose |
|------|---------|
| `detector.py` | `Detector` — YOLO-World wrapper: `set_queries()`, `detect()`, `annotate()` |
| `worker.py` | `PerceptionWorker` — runs the detector on the live feed in its own thread |
| `markers.py` | HSV colour-marker detection — classical CV for fiducial-like floor markers |

## How it works

- **`Detector.set_queries(["potted plant", "chair"])`** sets the open-vocab classes (encoded
  once via CLIP). **`detect(frame)`** returns per hit: `{label, score, box, center, area_frac}`.
- **`PerceptionWorker`** reads the latest frame, gated on frame identity + a short sleep so
  it never busy-loops the GIL and starves the video decoder.

## GIL rule

The detector runs **in-thread** (not its own process). CUDA calls release the GIL, so on a
capable GPU the detector spends most of its time GIL-free. If `Stream fps` sags with detection
active, the detector is starving the decode thread — promote it to its own process (IPC via
shared memory). Don't pay the IPC/complexity tax until the readout proves you need it.

## Colour-marker detection (`markers.py`)

YOLO-World is unreliable on the orange square markers used for aerial-survey alignment.
`markers.py` uses **deterministic HSV segmentation + contour filtering** instead — cheaper
(CPU-only) and far more reliable for this specific task. Returns the same dict shape as YOLO,
so the rest of the system consumes it identically.

**Activation:** query-driven. When the detection query contains `orange` (or `naranja`),
`markers.is_marker_query()` returns `True` and the worker routes through HSV instead of YOLO.

This is used by the **marker survey mode** (`POST /mission {"markers": N}`), which bypasses
the VLM entirely — see the main README's [Marker survey mode](../../README.md#marker-survey-mode).
