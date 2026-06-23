# photogrammetry/ — offline 3D reconstruction

Turns a set of drone snapshots into a textured 3D model, fully offline, using
[OpenDroneMap](https://www.opendronemap.org/) driven through
[PyODM](https://github.com/OpenDroneMap/PyODM).

## Pipeline

All paths below live under `snapshots/storage_3D/`.

```
pending_snapshots/   →  ODM node  →  3D_models/<model>/   (assets)
   (3D-snapshot)         (craft)      processed/          (consumed images)
```

1. **3D-snapshot** (web UI) saves the live frame + telemetry sidecar into
   `snapshots/storage_3D/pending_snapshots/`.
2. **Craft 3D model** (web UI) submits every pending image to the ODM node,
   polls it asynchronously, downloads the textured `.obj`/`.mtl`/textures and the
   `.ply` point cloud into a timestamped folder under
   `snapshots/storage_3D/3D_models/`, then moves the source images to
   `snapshots/storage_3D/processed/`.
3. The **3D Models** tab renders the result with a Three.js OBJ viewer.

## Running the processing node

`craft_3d_model` talks to a NodeODM HTTP endpoint (default `localhost:3000`).
Start a CUDA-enabled node in Docker on the DGX Spark:

```bash
docker run -d -p 3000:3000 --gpus all opendronemap/nodeodm:gpu
```

(omit `--gpus all` / use `opendronemap/nodeodm` for the CPU image on the MacBook).

Point the controller elsewhere without code changes via env vars:

```bash
ODM_HOST=192.168.1.50 ODM_PORT=3000 uv run python -m web.server
```

## Notes

- Reconstruction needs **several overlapping views** — orbit the object and take
  many 3D-snapshots before crafting.
- Processing runs on the ODM node (minutes), so the web layer dispatches it on a
  background thread; flight and telemetry never block on it.
- The Three.js viewer loads from a CDN by default. For a fully-offline DGX
  deployment, vendor the modules under `web/static/vendor/` and update the import
  map in `web/static/index.html`.
