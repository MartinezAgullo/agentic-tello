"""PerceptionWorker — runs the detector on the live feed in its own thread.

Gated on frame **identity** + a short sleep so it never busy-loops the GIL and
starves the video decode thread (the cardinal rule from CLAUDE.md). Publishes the
latest detections for the web overlay / agent loop to read.

Query changes are applied at the top of the loop (not from the caller's thread), so
the model is never reconfigured mid-inference.

If detector inference ever starves the decoder despite the gate, promote this to a
separate process (the model releases the GIL during CUDA calls, so on the GB10 it
should be fine in-thread).
"""

import threading
import time
from typing import Callable

import numpy as np

from perception.detector import Detector


class PerceptionWorker:
    def __init__(self, get_frame: Callable[[], np.ndarray | None],
                 detector: Detector) -> None:
        self._get_frame = get_frame
        self.detector = detector
        self.detections: list[dict] = []
        self.det_fps = 0.0
        self._pending: list[str] | None = None   # queries waiting to be applied
        self._last = None                          # identity of last frame processed
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "PerceptionWorker":
        self._thread.start()
        return self

    def set_queries(self, queries: list[str]) -> None:
        self._pending = queries          # applied in the worker loop (atomic reassign)

    @property
    def queries(self) -> list[str]:
        return self.detector.queries

    def _run(self) -> None:
        win_start = time.monotonic()
        win_count = 0
        while not self._stopped:
            if self._pending is not None:
                q, self._pending = self._pending, None
                self.detector.set_queries(q)
                if not q:
                    self.detections = []

            frame = self._get_frame()
            if frame is None or frame is self._last or not self.detector.queries:
                time.sleep(0.01)         # nothing new (or nothing to look for) — yield
                continue
            self._last = frame
            try:
                self.detections = self.detector.detect(frame)
            except Exception as e:
                print(f"[perception] detect error: {e}", flush=True)
                time.sleep(0.05)
                continue

            win_count += 1
            elapsed = time.monotonic() - win_start
            if elapsed >= 1.0:
                self.det_fps = win_count / elapsed
                win_start = time.monotonic()
                win_count = 0

    def stop(self) -> None:
        self._stopped = True
