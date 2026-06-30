# `brain` — VLM Planning (Slow Loop)

The **slow** half of the dual-cadence architecture. A local VLM (Gemma 3 / Qwen via Ollama)
is called every few seconds to turn a natural-language goal into detector targets and to judge
whether the goal is satisfied. It **never actuates** — it only writes to `MissionState`.

## Contents

| File | Purpose |
|------|---------|
| `vlm_client.py` | Ollama client (native `/api/chat`): frame encoding, JSON parsing, keep-alive |
| `prompts.py` | System/user prompts for planning (`SYSTEM_PROMPT`) and goal decomposition (`DECOMPOSE_SYSTEM_PROMPT`) |

## Two decoupled cadences

The system splits perception and action into two loops that run independently:

| Loop | Cadence | Thread | Job |
|------|---------|--------|-----|
| **Slow** (this package) | every `VLM_INTERVAL_S` (~3 s) | `_planner` in `agent/loop.py` | decide *what* to pursue — set detector queries, judge completion |
| **Fast** (`agent/`) | every control tick | single actuation thread | decide *how* to fly — deterministic servoing, phase transitions |

The slow loop **only writes shared state** (`MissionState` + detector queries); the fast loop
**only reads** it and emits one `Action` per tick. This decoupling is critical: the VLM takes
seconds per call, but the drone needs sub-second reflexes to servo toward a target.

### Why native Ollama API

The OpenAI-compatible `/v1` endpoint silently ignores `num_ctx`. Left at the model default
(e.g. 262k for qwen3-vl), Ollama reserves a huge KV cache (~48 GB VRAM, ~50 s/call). The
native `/api/chat` with a capped context drops that to ~7 GB and a few seconds.

### GIL considerations

<!-- TODO: document the specific GIL starvation incident if it recurs.
     The concern: the VLM client runs in a background thread; if its Python-side
     processing (JSON parsing, frame encoding, prompt assembly) holds the GIL for
     too long, it can starve the video-decode thread and tank Stream fps — the same
     class of problem the perception worker faces (see perception/README.md).
     In practice the VLM call itself is I/O-bound (HTTP to Ollama, releases the GIL),
     so the window is narrow, but under heavy load or with a slower model it could
     surface. If it does: move the planner to its own process or offload the
     encode/parse to a thread pool. -->

The VLM call is I/O-bound (HTTP request to Ollama, which releases the GIL), so GIL contention
with the video-decode thread has not been a problem in practice. The risk is the same as with
the perception worker (see `perception/README.md`): if CPU-bound Python work in this thread
grows, it could starve the decoder. The mitigation would be the same — promote to a separate
process.
