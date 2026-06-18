"""Bird's-Eye-View (BEV) from a single drone frame — Inverse Perspective Mapping.

Given one photo from a camera at a known height H, looking roughly horizontally
(pitch ~ 0, roll ~ 0), this reprojects the *ground* part of the image onto a
metric top-down grid (a pseudo-orthophoto). No neural nets — pure projective
geometry: pinhole model + ray/ground-plane intersection.

Key facts this module is built on
----------------------------------
* The map from a single world plane (the ground, Z=0) to the image is a
  **homography**. So "analytic IPM" and "cv2.warpPerspective" are the *same*
  transform — we expose both and check they agree (see `compare_methods`).
* The linear-angle formula `phi = (u-cx)/(W/2) * HFOV/2` is an *approximation*.
  A real (rectilinear) lens obeys `phi = atan((u-cx)/fx)`. We use the exact
  pinhole projection via the intrinsic matrix K everywhere.

Coordinate conventions
-----------------------
* Camera frame (OpenCV): x right, y down, z forward (optical axis).
* World frame: X right, Y forward, Z up. Camera sits at (0, 0, H).
* BEV image: +X (right) maps to columns, +Y (forward, away from drone) maps to
  rows growing *upward*, so the drone is at the bottom-centre.

Run as a script
---------------
    uv run python -m perception.bev --image shot.jpg --height 1.5 --hfov 82.6

Adds a metric grid, a valid-ground mask, and (optionally) auto horizon / pitch
estimation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

import config

# DJI RoboMaster TT / Tello still-photo defaults live in config.py (env-overridable).
DEFAULT_WIDTH = config.CAM_PHOTO_W
DEFAULT_HEIGHT = config.CAM_PHOTO_H
DEFAULT_HFOV_DEG = config.CAM_HFOV_DEG


def load_snapshot_metadata(image_path: str) -> dict | None:
    """Read the `<stem>.json` sidecar written next to a drone snapshot, if any.

    Returns the parsed dict, or None when there's no sidecar (e.g. an external
    image). The drone's height and IMU attitude live under `telemetry`/`state`.
    """
    stem, _ = os.path.splitext(image_path)
    path = f"{stem}.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _height_m_from_meta(meta: dict) -> float | None:
    """Best ground height (metres) from snapshot metadata. Prefers ToF (downward
    distance sensor) when present and plausible, else barometric height."""
    state = meta.get("state") or {}
    tof = state.get("tof")  # cm, downward time-of-flight
    if isinstance(tof, (int | float)) and 10 <= tof <= 800:
        return tof / 100.0
    h = (meta.get("telemetry") or {}).get("height_cm") or state.get("h")
    return h / 100.0 if isinstance(h, (int | float)) and h > 0 else None


# ── camera model ──────────────────────────────────────────────────────────────
@dataclass
class CameraModel:
    """Pinhole intrinsics. Principal point assumed centred (uncalibrated)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
        vfov_deg: float | None = None,
    ) -> CameraModel:
        """Build K from FOV. If `vfov_deg` is None it's derived assuming square
        pixels (fx == fy), which is the right thing to do when you only trust the
        horizontal FOV spec — see `vfov_from_hfov`.
        """
        cx, cy = width / 2.0, height / 2.0
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        if vfov_deg is None:
            fy = fx  # square pixels: consistent with vfov_from_hfov
        else:
            fy = (height / 2.0) / np.tan(np.radians(vfov_deg) / 2.0)
        return cls(width, height, fx, fy, cx, cy)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.fy))))


def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    """Vertical FOV implied by the horizontal FOV and the aspect ratio, assuming
    square pixels: VFOV = 2*atan(tan(HFOV/2) * H / W)."""
    t = np.tan(np.radians(hfov_deg) / 2.0) * (height / width)
    return float(np.degrees(2.0 * np.arctan(t)))


def hfov_from_dfov(dfov_deg: float, width: int, height: int) -> float:
    """Horizontal FOV from a *diagonal* FOV spec (square pixels).

    DJI publishes the Tello camera as "FOV 82.6°" without saying horizontal vs
    diagonal. If it's the diagonal, the real HFOV is narrower:
        HFOV = 2*atan( (W/d) * tan(DFOV/2) ),  d = sqrt(W^2 + H^2)
    For the 4:3 still (2592x1936): 82.6° diagonal -> ~70.3° horizontal.
    """
    d = float(np.hypot(width, height))
    t = np.tan(np.radians(dfov_deg) / 2.0) * (width / d)
    return float(np.degrees(2.0 * np.arctan(t)))


# ── extrinsics: camera orientation ─────────────────────────────────────────────
def rotation_cam_to_world(pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Rotation R such that  P_world_dir = R @ P_cam_dir.

    pitch_deg > 0  => nose down (camera looks toward the ground).
    roll_deg  > 0  => roll to the right.
    World = X right, Y forward, Z up; camera = x right, y down, z forward.
    """
    # Base: camera optical axis (z) -> world forward (Y); camera down (y) -> world -Z.
    r0 = np.array(
        [
            [1.0, 0.0, 0.0],  # world X (right)  <- cam x
            [0.0, 0.0, 1.0],  # world Y (fwd)    <- cam z
            [0.0, -1.0, 0.0],  # world Z (up)     <- -cam y
        ]
    )
    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    # Pitch about world X (right axis); positive = nose down (forward dips to -Z).
    cp, sp = np.cos(p), np.sin(p)
    rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    # Roll about world Y (forward axis).
    cr, sr = np.cos(r), np.sin(r)
    ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    return ry @ rx @ r0


# ── core geometry ───────────────────────────────────────────────────────────────
def ground_homography(
    cam: CameraModel,
    height_m: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """3x3 homography mapping ground metric coords (X, Y, 1) -> image pixel (u, v, 1).

    Derivation: a ground point is P=(X,Y,0); camera at C=(0,0,H). In camera frame
    P_cam = R_cw (P - C) with R_cw = R^T (world->camera). Expanding,
        s[u,v,1]^T = K [ c1 | c2 | -H*c3 ] [X,Y,1]^T
    where c1,c2,c3 are the columns of R_cw. This *is* the IPM transform.
    """
    r_cw = rotation_cam_to_world(pitch_deg, roll_deg).T  # world -> camera
    c1, c2, c3 = r_cw[:, 0], r_cw[:, 1], r_cw[:, 2]
    m = np.column_stack([c1, c2, -height_m * c3])
    h = cam.K @ m
    # NB: don't normalise by h[2,2] — it's 0 at pitch=0. Homography scale is
    # arbitrary (cancels in the projective divide); keep the raw matrix.
    return h


def pixels_to_ground(
    cam: CameraModel,
    uv: np.ndarray,
    height_m: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward IPM: image pixels -> ground metric coords by ray/plane intersection.

    `uv` is (N,2). Returns (XY (N,2) in metres, valid (N,) bool). Invalid = pixel's
    ray is above the horizon or points away from the ground.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    # Ray direction in camera frame, then rotate to world.
    dirs_cam = np.column_stack(
        [(uv[:, 0] - cam.cx) / cam.fx, (uv[:, 1] - cam.cy) / cam.fy, np.ones(len(uv))]
    )
    r = rotation_cam_to_world(pitch_deg, roll_deg)
    d = dirs_cam @ r.T  # world direction, columns X,Y,Z
    # Camera at height H; hit ground when H + t*d_z = 0 -> t = -H/d_z, need d_z<0.
    dz = d[:, 2]
    valid = dz < -1e-9
    t = np.where(valid, -height_m / np.where(valid, dz, -1.0), np.nan)
    xy = np.column_stack([t * d[:, 0], t * d[:, 1]])  # X right, Y forward
    return xy, valid


class BEVProjector:
    """Precomputed inverse-perspective remap onto a metric top-down grid.

    Backward mapping (BEV cell -> source pixel) avoids holes. Build once, then
    `warp` any frame of the same camera/height cheaply with cv2.remap.
    """

    def __init__(
        self,
        cam: CameraModel,
        height_m: float,
        x_range: tuple[float, float] = (-3.0, 3.0),
        y_range: tuple[float, float] = (0.5, 8.0),
        m_per_px: float = 0.02,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> None:
        self.cam = cam
        self.height_m = height_m
        self.x_range = x_range
        self.y_range = y_range
        self.m_per_px = m_per_px
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg

        self.out_w = int(round((x_range[1] - x_range[0]) / m_per_px))
        self.out_h = int(round((y_range[1] - y_range[0]) / m_per_px))

        # Metric coordinate of each output cell centre.
        xs = x_range[0] + (np.arange(self.out_w) + 0.5) * m_per_px
        # Row 0 is the farthest (top of image); drone (small Y) at the bottom.
        ys = y_range[1] - (np.arange(self.out_h) + 0.5) * m_per_px
        gx, gy = np.meshgrid(xs, ys)  # (out_h, out_w)

        h_g2i = ground_homography(cam, height_m, pitch_deg, roll_deg)
        pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=0)  # 3xN
        proj = h_g2i @ pts
        u = proj[0] / proj[2]
        v = proj[1] / proj[2]
        # Mark cells whose source pixel falls outside the image as invalid.
        in_img = (proj[2] > 0) & (u >= 0) & (u <= cam.width - 1) & (v >= 0) & (v <= cam.height - 1)
        u = np.where(in_img, u, -1.0)
        v = np.where(in_img, v, -1.0)
        self.map_x = u.reshape(self.out_h, self.out_w).astype(np.float32)
        self.map_y = v.reshape(self.out_h, self.out_w).astype(np.float32)
        self.h_ground_to_image = h_g2i

    def warp(self, image: np.ndarray) -> np.ndarray:
        """Analytic IPM: remap the source frame onto the metric BEV grid."""
        return cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def homography_image_to_bev(self) -> np.ndarray:
        """3x3 homography image -> BEV pixels, for cv2.warpPerspective comparison."""
        # Affine metric(X,Y) -> bev pixel (col, row).
        x0, _ = self.x_range
        _, y1 = self.y_range
        a = np.array(
            [
                [1.0 / self.m_per_px, 0.0, -x0 / self.m_per_px],
                [0.0, -1.0 / self.m_per_px, y1 / self.m_per_px],
                [0.0, 0.0, 1.0],
            ]
        )
        return a @ np.linalg.inv(self.h_ground_to_image)

    def warp_opencv(self, image: np.ndarray) -> np.ndarray:
        """Same BEV via cv2.warpPerspective — should match `warp` to sub-pixel."""
        return cv2.warpPerspective(image, self.homography_image_to_bev(), (self.out_w, self.out_h))

    def ground_mask(self) -> np.ndarray:
        """Source-image mask: which input pixels project onto the in-range ground.

        Returns a (H, W) uint8 mask (255 = valid ground used by the BEV).
        """
        ys, xs = np.mgrid[0 : self.cam.height, 0 : self.cam.width]
        uv = np.column_stack([xs.ravel(), ys.ravel()])
        xy, valid = pixels_to_ground(self.cam, uv, self.height_m, self.pitch_deg, self.roll_deg)
        in_range = (
            valid
            & (xy[:, 0] >= self.x_range[0])
            & (xy[:, 0] <= self.x_range[1])
            & (xy[:, 1] >= self.y_range[0])
            & (xy[:, 1] <= self.y_range[1])
        )
        return (in_range.reshape(self.cam.height, self.cam.width) * 255).astype(np.uint8)


# ── optional extras ─────────────────────────────────────────────────────────────
def detect_horizon(image: np.ndarray, cam: CameraModel) -> tuple[float | None, float | None]:
    """Estimate the horizon row and the implied pitch error.

    Crude but useful indoors-free: the strongest near-horizontal line (Hough) in
    the central band is taken as the horizon. With pitch=0 the horizon sits at
    v = cy, so a measured row v_h implies  pitch_err = atan((cy - v_h) / fy).
    Returns (v_horizon, pitch_error_deg) or (None, None) if nothing convincing.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=image.shape[1] // 3,
        maxLineGap=40,
    )
    if lines is None:
        return None, None
    best_v, best_len = None, 0.0
    for x1, y1, x2, y2 in lines[:, 0, :]:
        if abs(y2 - y1) > 0.05 * abs(x2 - x1 + 1e-6):  # not horizontal enough
            continue
        length = abs(x2 - x1)
        if length > best_len:
            best_len, best_v = length, (y1 + y2) / 2.0
    if best_v is None:
        return None, None
    pitch_err = float(np.degrees(np.arctan((cam.cy - best_v) / cam.fy)))
    return best_v, pitch_err


def compare_methods(proj: BEVProjector, image: np.ndarray) -> dict:
    """Confirm analytic remap == warpPerspective (they share one homography)."""
    a = proj.warp(image).astype(np.float32)
    b = proj.warp_opencv(image).astype(np.float32)
    overlap = (a.sum(axis=2) > 0) & (b.sum(axis=2) > 0)
    diff = np.abs(a - b).mean(axis=2)
    return {
        "mean_abs_diff": float(diff[overlap].mean()) if overlap.any() else 0.0,
        "max_abs_diff": float(diff[overlap].max()) if overlap.any() else 0.0,
    }


def draw_metric_grid(bev: np.ndarray, proj: BEVProjector, step_m: float = 1.0) -> np.ndarray:
    """Overlay a metric grid (lines every `step_m`) with axis labels."""
    out = bev.copy()
    x0, x1 = proj.x_range
    y0, y1 = proj.y_range
    for xm in np.arange(np.ceil(x0 / step_m) * step_m, x1 + 1e-6, step_m):
        col = int((xm - x0) / proj.m_per_px)
        cv2.line(out, (col, 0), (col, proj.out_h - 1), (60, 60, 60), 1)
        cv2.putText(
            out, f"{xm:+.0f}m", (col + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
        )
    for ym in np.arange(np.ceil(y0 / step_m) * step_m, y1 + 1e-6, step_m):
        row = int((y1 - ym) / proj.m_per_px)
        cv2.line(out, (0, row), (proj.out_w - 1, row), (60, 60, 60), 1)
        cv2.putText(
            out, f"{ym:.0f}m", (4, row - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
        )
    return out


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 200, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Paint `mask` (255=ground) translucently over `image` for visual inspection.

    This shows *what the geometry assumes is the ground plane* on the real photo —
    so you can see directly where it overshoots (walls, furniture, columns that
    fall inside the trapezoid but aren't on z=0).
    """
    out = image.copy()
    sel = mask > 0
    tint = np.zeros_like(image)
    tint[:] = color
    out[sel] = (alpha * tint[sel] + (1 - alpha) * image[sel]).astype(np.uint8)
    # outline the mask boundary so the extent is crisp
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


# ── CLI demo ────────────────────────────────────────────────────────────────────
def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-image BEV / IPM for a Tello frame")
    ap.add_argument("--image", required=True)
    ap.add_argument(
        "--height",
        type=float,
        default=None,
        help="drone height (m). Default: read from the snapshot's .json sidecar.",
    )
    ap.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG)
    ap.add_argument(
        "--dfov",
        type=float,
        default=None,
        help="diagonal FOV; if set, HFOV is derived from it (Tello's 82.6° may be diagonal)",
    )
    ap.add_argument(
        "--vfov", type=float, default=config.CAM_VFOV_DEG, help="default: derived from HFOV"
    )
    ap.add_argument("--pitch", type=float, default=0.0, help="nose-down deg (>0)")
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--mpp", type=float, default=0.02, help="metres per BEV pixel")
    ap.add_argument("--xrange", type=float, nargs=2, default=(-3.0, 3.0))
    ap.add_argument("--yrange", type=float, nargs=2, default=(0.5, 8.0))
    ap.add_argument(
        "--auto-pitch", action="store_true", help="detect horizon and use it to correct pitch"
    )
    ap.add_argument(
        "--use-meta-attitude",
        action="store_true",
        help="also take pitch/roll from the snapshot's IMU metadata (validate signs on a real flight first)",
    )
    ap.add_argument("--show", action="store_true", help="also open a preview window")
    ap.add_argument("--out", default=None, help="override the auto output path")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read {args.image}")
    h_img, w_img = image.shape[:2]

    hfov = args.hfov if args.dfov is None else hfov_from_dfov(args.dfov, w_img, h_img)
    if args.dfov is not None:
        print(f"DFOV {args.dfov:.1f} -> HFOV {hfov:.1f} (assuming diagonal spec)")
    cam = CameraModel.from_fov(w_img, h_img, hfov, args.vfov)
    print(
        f"camera  {w_img}x{h_img}  fx={cam.fx:.1f} fy={cam.fy:.1f}  "
        f"HFOV={cam.hfov_deg:.1f} VFOV={cam.vfov_deg:.1f}"
    )

    # Height + attitude: CLI flags win; otherwise fall back to the drone metadata.
    meta = load_snapshot_metadata(args.image)
    height = args.height
    pitch, roll = args.pitch, args.roll
    if meta is not None:
        if height is None:
            height = _height_m_from_meta(meta)
            if height is not None:
                print(f"height from metadata: {height:.2f} m")
        if args.use_meta_attitude:
            st = meta.get("state") or {}
            pitch = float(st.get("pitch", pitch))
            roll = float(st.get("roll", roll))
            print(f"attitude from metadata: pitch={pitch:+.1f} roll={roll:+.1f} deg")
    if height is None:
        height = 1.5
        print(f"no --height and no metadata; assuming {height:.2f} m")

    v_h, pitch_err = detect_horizon(image, cam)
    if v_h is not None:
        print(f"horizon row ~ {v_h:.0f}px (cy={cam.cy:.0f}) -> pitch err {pitch_err:+.2f}deg")
        if args.auto_pitch:
            pitch = pitch_err
            print(f"using detected pitch = {pitch:+.2f}deg")
    else:
        print("horizon: not detected")

    proj = BEVProjector(
        cam,
        height,
        tuple(args.xrange),
        tuple(args.yrange),
        args.mpp,
        pitch,
        roll,
    )
    bev = draw_metric_grid(proj.warp(image), proj)
    mask = proj.ground_mask()

    cmp = compare_methods(proj, image)
    print(
        f"analytic vs warpPerspective: mean|diff|={cmp['mean_abs_diff']:.3f} "
        f"max={cmp['max_abs_diff']:.1f} (≈0 confirms IPM is a homography)"
    )

    panel_h = 600
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_on_photo = overlay_mask(image, mask)
    strip = np.hstack(
        [
            _resize_to_height(image, panel_h),
            _resize_to_height(mask_on_photo, panel_h),
            _resize_to_height(mask_bgr, panel_h),
            _resize_to_height(bev, panel_h),
        ]
    )

    # Auto-save into snapshots/cenital_view/: the BEV alone + the 3-up panel.
    os.makedirs(config.BEV_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    bev_path = args.out or os.path.join(config.BEV_DIR, f"{stem}_bev.jpg")
    panel_path = os.path.join(config.BEV_DIR, f"{stem}_panel.jpg")
    cv2.imwrite(bev_path, bev)
    cv2.imwrite(panel_path, strip)
    print(f"saved BEV   -> {bev_path}")
    print(f"saved panel -> {panel_path}")

    if args.show:
        cv2.imshow("original | ground mask | BEV", strip)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
