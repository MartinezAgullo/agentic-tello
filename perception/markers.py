"""Color-marker detection — classical CV for fiducial-like floor markers.

YOLO-World matches the visual appearance of concrete *nouns* (a chair, a plant).
It is unreliable on abstract geometric/colour targets like the orange square
markers used for aerial-survey alignment: they are small, viewed at an angle and
split by black bands. For those, deterministic HSV colour segmentation + contour
filtering is both far more reliable and far cheaper (CPU, every frame, no GIL
contention with the video decoder).

`detect()` returns the SAME dict shape as `perception.detector.Detector.detect`
({label, score, box, center, area_frac}) so the agent loop / web overlay consume
it identically — the fast loop just counts boxes, it doesn't care how they were
found.
"""

import cv2
import numpy as np

# ── marker colour specs ───────────────────────────────────────────────────────
# Each spec: label + one or more HSV ranges (OpenCV hue is 0-179). Multiple ranges
# let a colour wrap the hue origin (e.g. red) or cover a light/dark spread. Tune the
# saturation/value floors to your room: a white floor is low-saturation, so a high
# S floor (>~80) rejects it while keeping the saturated marker. Black bands are low
# V, so they drop out and the morphology close re-merges the orange fragments.
#
# Measured marker colour: HEX #e25f00 → RGB(226,95,0) → standard HSV (25°,100%,89%).
# In OpenCV units (H 0-179, S/V 0-255) that is H≈13, S≈255, V≈227. The range below is
# centred there with margin: the hue band stays in orange, while the S/V FLOORS are kept
# low enough to keep the marker under shade/glare yet high enough that the (low-saturation)
# white floor is rejected. Saturation is the real discriminator against the floor — raise
# its floor if the floor leaks in, lower the V floor if shadowed markers drop out.
_SPECS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "orange": [((5, 80, 70), (22, 255, 255))],
}

# query keywords (English + Spanish) that select a colour spec instead of YOLO
_KEYWORDS: dict[str, str] = {
    "orange": "orange",
    "naranja": "orange",
}

# blob acceptance, as fractions of the frame area (resolution-independent)
_MIN_AREA_FRAC = 3e-4    # reject specks / sensor noise
_MAX_AREA_FRAC = 0.15    # reject a large orange object filling the view (not a marker)
_MIN_ASPECT = 0.35       # bbox w/h must be roughly square-ish (angled view tolerated)
_MAX_ASPECT = 2.8
_MIN_FILL = 0.30         # mask pixels / bbox area — squares are solid; rejects thin streaks


def is_marker_query(query: str) -> bool:
    """True if this query should be served by colour CV rather than YOLO."""
    q = query.lower()
    return any(k in q for k in _KEYWORDS)


def specs_for_queries(queries: list[str]) -> list[tuple[str, list]]:
    """Map marker queries to (label, hsv_ranges). Non-marker queries are ignored
    here (they stay on YOLO). The label echoes the operator's query so the overlay
    and snapshot filename read naturally."""
    out: list[tuple[str, list]] = []
    for q in queries:
        ql = q.lower()
        for kw, color in _KEYWORDS.items():
            if kw in ql and color in _SPECS:
                out.append((q.strip(), _SPECS[color]))
                break
    return out


def detect(frame: np.ndarray, specs: list[tuple[str, list]]) -> list[dict]:
    """Find colour-marker blobs for each spec. Returns detector-shaped dicts,
    sorted by score (descending), same as the YOLO path."""
    if frame is None or frame.size == 0 or not specs:
        return []
    h, w = frame.shape[:2]
    frame_area = float(w * h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    out: list[dict] = []
    for label, ranges in specs:
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        # close gaps where black bands split the orange into fragments, then open
        # to drop isolated speckle
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area_frac = (bw * bh) / frame_area
            if area_frac < _MIN_AREA_FRAC or area_frac > _MAX_AREA_FRAC:
                continue
            aspect = bw / float(bh) if bh else 0.0
            if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
                continue
            fill = cv2.contourArea(c) / float(bw * bh) if bw and bh else 0.0
            if fill < _MIN_FILL:
                continue
            x1, y1, x2, y2 = float(x), float(y), float(x + bw), float(y + bh)
            out.append({
                "label": label,
                "score": float(min(1.0, fill)),      # solid square ⇒ high score
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "area_frac": area_frac,
            })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out
