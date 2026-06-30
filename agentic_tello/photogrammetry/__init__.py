"""Offline 3D photogrammetry — drone snapshots → textured 3D model.

Thin wrapper around a local OpenDroneMap node, driven via PyODM. See
:mod:`photogrammetry.odm` for the processing pipeline.
"""

from agentic_tello.photogrammetry.odm import (
    PhotogrammetryError,
    craft_3d_model,
    list_models,
    list_pending_images,
)

__all__ = [
    "PhotogrammetryError",
    "craft_3d_model",
    "list_models",
    "list_pending_images",
]
