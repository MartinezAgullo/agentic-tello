"""Offline 3D reconstruction — turn drone snapshots into a textured 3D model.

This module drives a **local OpenDroneMap (ODM) processing node** through
**PyODM**. The node is expected to run as a CUDA-enabled Docker container on the
same host (the NVIDIA DGX Spark); only its HTTP endpoint is referenced here, so
the processing node can later be moved to a separate machine purely by changing
``ODM_HOST`` / ``ODM_PORT`` in :mod:`config` — no code change required.

Workflow (the "craft 3D model" button):
    1. Collect every image waiting in ``snapshots/storage_3D/pending_snapshots/``.
    2. Submit them to the ODM node as a single reconstruction task.
    3. Poll the task asynchronously, reporting progress through a callback.
    4. On success, download the assets (textured ``.obj`` + ``.mtl`` + textures
       and the ``.ply`` point cloud) into a fresh, timestamped sub-folder of
       ``snapshots/storage_3D/3D_models/``.
    5. Move the consumed source images out of ``pending_snapshots/`` into
       ``snapshots/storage_3D/processed/`` so they are not reprocessed.

The heavy lifting runs on the ODM node, not in this process: this module only
submits work, polls, and shuffles files, so it is safe to call from a background
thread without starving the video decode loop.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pyodm import Node
from pyodm.exceptions import NodeConnectionError, NodeResponseError, TaskFailedError

import config

# Source images we are willing to feed into the reconstruction.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ODM task options tuned for small indoor image sets from a single drone camera.
# Override per call via the ``options`` argument; see the OpenDroneMap docs for
# the full list of switches.
DEFAULT_OPTIONS: dict = {
    "use-3dmesh": True,  # full 3D mesh (orbit/indoor capture), not a 2.5D surface
    "pc-quality": "medium",  # point-cloud density vs. runtime trade-off
    "mesh-octree-depth": 11,
    "texturing-single-material": True,  # one material → simpler OBJ for web viewers
    "dsm": False,  # no digital surface model needed for an indoor object scan
    "fast-orthophoto": False,
}

ProgressCallback = Callable[[dict], None]
LogCallback = Callable[[str], None]


class PhotogrammetryError(RuntimeError):
    """Raised when a reconstruction cannot start or complete."""


@dataclass
class CraftResult:
    """Outcome of a successful :func:`craft_3d_model` run."""

    name: str  # model folder name, e.g. "model_20260623_143000"
    model_dir: str  # absolute/relative path to the downloaded asset tree
    obj_path: str | None  # textured OBJ, if one was produced
    ply_path: str | None  # point cloud, if one was produced
    n_images: int  # number of source images consumed
    processed: list[str] = field(default_factory=list)  # where the images moved to


# ── file discovery ────────────────────────────────────────────────────────────
def list_pending_images(pending_dir: str | None = None) -> list[str]:
    """Return the sorted list of images queued for reconstruction."""
    pending_dir = pending_dir or config.PENDING_SNAPSHOT_DIR
    if not os.path.isdir(pending_dir):
        return []
    images = [
        os.path.join(pending_dir, name)
        for name in os.listdir(pending_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(images)


def _find_first(root: str, *suffixes: str) -> str | None:
    """Depth-first search for the first file ending in any of ``suffixes``."""
    matches: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(suffixes):
                matches.append(os.path.join(dirpath, name))
    # Prefer the shallowest, shortest path (ODM nests the canonical asset directly).
    matches.sort(key=lambda p: (p.count(os.sep), len(p)))
    return matches[0] if matches else None


def list_models(models_dir: str | None = None) -> list[dict]:
    """Describe every crafted model for the web viewer.

    Each entry exposes the OBJ/MTL filenames and the directory that holds them,
    relative to ``models_dir`` so the web layer can turn them into static URLs.
    """
    models_dir = models_dir or config.MODELS_3D_DIR
    if not os.path.isdir(models_dir):
        return []

    models: list[dict] = []
    for name in sorted(os.listdir(models_dir), reverse=True):  # newest first
        model_dir = os.path.join(models_dir, name)
        if not os.path.isdir(model_dir):
            continue
        obj_path = _find_first(model_dir, ".obj")
        if obj_path is None:
            continue  # still processing or a failed/partial download
        obj_dir = os.path.dirname(obj_path)
        # Prefer the MTL whose stem matches the OBJ (e.g. odm_textured_model_geo.obj
        # → odm_textured_model_geo.mtl, the one the OBJ's `mtllib` line references);
        # fall back to any MTL in the same directory.
        obj_stem = os.path.splitext(os.path.basename(obj_path))[0]
        matched_mtl = os.path.join(obj_dir, f"{obj_stem}.mtl")
        mtl_path = matched_mtl if os.path.exists(matched_mtl) else _find_first(obj_dir, ".mtl")
        models.append(
            {
                "name": name,
                # Path of the asset directory relative to the models root, used by
                # the web layer to build the static URL the 3D viewer fetches from.
                "rel_dir": os.path.relpath(obj_dir, models_dir),
                "obj": os.path.basename(obj_path),
                "mtl": os.path.basename(mtl_path) if mtl_path else None,
                "created": os.path.getmtime(model_dir),
            }
        )
    return models


# ── the reconstruction pipeline ────────────────────────────────────────────────
def _move_into(dest_dir: str, src: str) -> str:
    """Move ``src`` into ``dest_dir``, avoiding name clashes; return the new path."""
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(src)
    dest = os.path.join(dest_dir, base)
    if os.path.exists(dest):
        stem, ext = os.path.splitext(base)
        dest = os.path.join(dest_dir, f"{stem}_{int(time.time() * 1000)}{ext}")
    shutil.move(src, dest)
    return dest


def craft_3d_model(
    *,
    log: LogCallback = print,
    on_progress: ProgressCallback | None = None,
    node: Node | None = None,
    options: dict | None = None,
) -> CraftResult:
    """Run one full reconstruction over the pending snapshots.

    Blocking — intended to be called from a background thread. Progress is
    reported through ``on_progress`` as ``{"stage", "progress", "status"}`` dicts.

    Raises:
        PhotogrammetryError: no pending images, or the ODM node is unreachable /
            the task failed.
    """
    images = list_pending_images()
    if not images:
        raise PhotogrammetryError(
            "No pending snapshots to process — capture a few 3D snapshots first."
        )

    if node is None:
        node = Node(config.ODM_HOST, config.ODM_PORT, token=config.ODM_TOKEN)

    opts = {**DEFAULT_OPTIONS, **(options or {})}

    def report(stage: str, progress: float, status: str) -> None:
        if on_progress is not None:
            on_progress({"stage": stage, "progress": round(progress, 1), "status": status})

    log(f"[3d] submitting {len(images)} image(s) to ODM at {config.ODM_HOST}:{config.ODM_PORT}…")
    report("submitting", 0.0, "UPLOADING")

    try:
        task = node.create_task(images, opts)
    except (NodeConnectionError, NodeResponseError) as exc:
        raise PhotogrammetryError(
            f"Cannot reach the OpenDroneMap node at {config.ODM_HOST}:{config.ODM_PORT} "
            f"({exc}). Is the NodeODM container running?"
        ) from exc

    log(f"[3d] task {task.uuid} created; processing…")

    def status_cb(info) -> None:
        # `info` is a pyodm TaskInfo; status is an enum, progress is 0–100.
        status_name = getattr(getattr(info, "status", None), "name", "RUNNING")
        report("processing", float(getattr(info, "progress", 0.0) or 0.0), status_name)

    try:
        task.wait_for_completion(status_callback=status_cb, interval=config.ODM_POLL_INTERVAL_S)
    except TaskFailedError as exc:
        # Surface the last lines of the node's console output to aid debugging.
        try:
            tail = "\n".join(task.output()[-15:])
        except Exception:
            tail = "(console output unavailable)"
        raise PhotogrammetryError(f"ODM task {task.uuid} failed:\n{tail}") from exc
    except (NodeConnectionError, NodeResponseError) as exc:
        raise PhotogrammetryError(f"Lost contact with the ODM node: {exc}") from exc

    # ── download assets into a fresh, timestamped model folder ──────────────────
    name = f"model_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(config.MODELS_3D_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    log(f"[3d] downloading assets → {model_dir}")
    report("downloading", 100.0, "DOWNLOADING")
    try:
        task.download_assets(model_dir)
    except (NodeConnectionError, NodeResponseError) as exc:
        raise PhotogrammetryError(f"Asset download failed: {exc}") from exc

    obj_path = _find_first(model_dir, ".obj")
    ply_path = _find_first(model_dir, ".ply")

    # ── retire the consumed source images (images + telemetry sidecars) ─────────
    moved: list[str] = []
    for img in images:
        moved.append(_move_into(config.PROCESSED_DIR, img))
        sidecar = os.path.splitext(img)[0] + ".json"
        if os.path.exists(sidecar):
            _move_into(config.PROCESSED_DIR, sidecar)

    log(
        f"[3d] done: {name} (obj={'yes' if obj_path else 'no'}, "
        f"ply={'yes' if ply_path else 'no'}); {len(moved)} image(s) archived."
    )
    report("done", 100.0, "COMPLETED")

    return CraftResult(
        name=name,
        model_dir=model_dir,
        obj_path=obj_path,
        ply_path=ply_path,
        n_images=len(images),
        processed=moved,
    )
