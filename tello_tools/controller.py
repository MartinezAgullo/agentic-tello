"""TelloController — owns the drone connection and the low-latency video stream.

This is the raw hardware layer: it talks to djitellopy directly and exposes
unguarded actuation (`_takeoff`, `_rc`, …). Callers should go through
SafeTello / ControlArbiter, not here, except for emergency stop.

The video path (LowLatencyFrameRead) is ported from the previous controller.
Two hard-won lessons are baked in (see the project CLAUDE.md):
  1. Drain the OS UDP buffer before PyAV opens the port.
  2. Continuously skip stale backlog frames so latency can't accumulate.
"""

import logging
import socket
import threading
import time

import av
import numpy as np
from djitellopy import Tello

import config

# djitellopy adds its own noisy StreamHandler on import — silence it.
_dji_logger = logging.getLogger("djitellopy")
_dji_logger.handlers.clear()
_dji_logger.setLevel(logging.CRITICAL)


def _drain_udp_buffer(port: int) -> int:
    """Discard everything already queued in the OS UDP socket buffer.

    Opens the port non-blocking, reads until EAGAIN, closes. PyAV opens the
    same port right after and starts from the current frame.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", port))
    s.setblocking(False)
    count = 0
    try:
        while True:
            s.recv(65536)
            count += 1
    except (BlockingIOError, OSError):
        pass
    finally:
        s.close()
    return count


class LowLatencyFrameRead:
    """Minimal-latency PyAV reader that never lets backlog accumulate.

    Publishes the most recent decoded frame to `self.frame`. When PyAV delivers
    frames faster than real time (a backlog), it skips the BGR conversion and
    doesn't publish — freeing CPU to clear the backlog and never showing a stale
    frame. `self.draining` reflects that state for the UI.
    """

    def __init__(self, url: str, port: int) -> None:
        self.frame: np.ndarray | None = None
        self.draining = True
        self.decode_fps = 0.0
        self.skip_fps = 0.0
        self.error: str | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, args=(url, port), daemon=True)
        self._thread.start()

    def _run(self, url: str, port: int) -> None:
        _drain_udp_buffer(port)
        # Aggressive low-latency probing can make av.open() fail to detect the
        # stream if the first UDP packets are slow/partial. Retry, and on later
        # attempts relax probesize/analyzeduration so we still get video (a little
        # extra startup latency beats a black screen).
        base = {
            "fflags": "nobuffer",
            "flags": "low_delay",
            "max_delay": "0",
            "reorder_queue_size": "0",
        }
        container = None
        for attempt in range(6):
            if self._stopped:
                return
            lenient = attempt >= 2
            opts = {**base,
                    "probesize": "5000000" if lenient else "32",
                    "analyzeduration": "1000000" if lenient else "0"}
            try:
                container = av.open(url, options=opts)
                self.error = None
                print(f"[stream] video opened (attempt {attempt + 1}, "
                      f"{'lenient' if lenient else 'low-latency'} probe)", flush=True)
                break
            except Exception as e:
                self.error = f"av.open failed (attempt {attempt + 1}/6): {e}"
                print(f"[stream] {self.error}", flush=True)
                time.sleep(0.6)
        if container is None:
            print("[stream] gave up opening video — check streamon / port 11111", flush=True)
            return

        try:
            ctx = container.streams.video[0].codec_context
            ctx.thread_count = 1
            ctx.flags |= 0x00080000   # AV_CODEC_FLAG_LOW_DELAY
            ctx.flags2 |= 0x00000001  # AV_CODEC_FLAG2_FAST
        except Exception:
            pass

        last_decode = time.monotonic()
        win_start = time.monotonic()
        win_decoded = win_skipped = 0
        first = True
        try:
            for packet in container.demux(container.streams.video[0]):
                if self._stopped:
                    break
                for frame in packet.decode():
                    if self._stopped:
                        break
                    now = time.monotonic()
                    gap = now - last_decode
                    last_decode = now
                    win_decoded += 1
                    if gap < config.LIVE_GAP_S:
                        self.draining = True
                        win_skipped += 1
                        continue
                    self.draining = False
                    self.frame = frame.to_ndarray(format="bgr24")
                    if first:
                        first = False
                        print("[stream] first frame decoded — live", flush=True)
                    win_elapsed = now - win_start
                    if win_elapsed >= 1.0:
                        self.decode_fps = win_decoded / win_elapsed
                        self.skip_fps = win_skipped / win_elapsed
                        win_start = now
                        win_decoded = win_skipped = 0
        except Exception as e:
            if not self._stopped:
                self.error = f"stream decode error: {e}"
                print(f"[stream] {self.error}", flush=True)
        finally:
            try:
                container.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stopped = True


class TelloController:
    """Raw drone + stream owner. Unguarded actuation; wrap with SafeTello."""

    def __init__(self) -> None:
        self.drone = Tello()
        self.frame_read: LowLatencyFrameRead | None = None
        self._connected = False

    def connect(self) -> None:
        self.drone.connect()
        self._connected = True

    def start_stream(self) -> None:
        try:
            self.drone.streamoff()
        except Exception:
            pass
        time.sleep(0.3)
        self.drone.streamon()
        self.frame_read = LowLatencyFrameRead(
            config.TELLO_STREAM_URL, config.TELLO_STREAM_PORT
        )

    # ── reads ────────────────────────────────────────────────────────────────
    def get_frame(self) -> np.ndarray | None:
        return self.frame_read.frame if self.frame_read else None

    def get_battery(self) -> int:
        return self.drone.get_battery()

    def get_state(self) -> dict:
        try:
            return self.drone.get_current_state()
        except Exception:
            return {}

    def get_height(self) -> int:
        try:
            return self.drone.get_height()
        except Exception:
            return 0

    # ── raw actuation (call via SafeTello) ────────────────────────────────────
    def _takeoff(self) -> None:
        self.drone.takeoff()

    def _land(self) -> None:
        self.drone.land()

    def _rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self.drone.send_rc_control(lr, fb, ud, yaw)

    def _move(self, direction: str, cm: int) -> None:
        getattr(self.drone, f"move_{direction}")(cm)

    def _rotate(self, deg: int) -> None:
        if deg >= 0:
            self.drone.rotate_clockwise(deg)
        else:
            self.drone.rotate_counter_clockwise(-deg)

    def emergency(self) -> None:
        """Cut motors immediately. Bypasses every guard."""
        self.drone.emergency()

    def shutdown(self) -> None:
        if self.frame_read:
            self.frame_read.stop()
        try:
            self.drone.end()
        except Exception:
            pass
