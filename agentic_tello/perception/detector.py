"""Open-vocabulary object detection — the fast perception layer.

Wraps an ultralytics YOLO-World model. Unlike fixed-class YOLO, you give it free
text queries ("potted plant", "red backpack") and it localizes them. The agent's
fast loop uses these boxes to servo toward a target; the VLM only reasons about
them occasionally.

First use downloads the weights (`config.DETECTOR_MODEL`).
"""


import cv2
import numpy as np

import config
from perception import markers

# Default broad vocabulary for the "detect all" button — the 80 COCO classes the
# YOLO-World backbone was distilled on (its sweet spot). Open-vocab still lets you
# type anything; this is just a convenient catch-all preset.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _auto_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        from ultralytics import YOLO
        self.device = device or _auto_device()
        self.model = YOLO(model or config.DETECTOR_MODEL)
        self.model.to(self.device)
        self.queries: list[str] = []
        self._yolo_queries: list[str] = []               # subset served by YOLO (label lookup)
        self._marker_specs: list[tuple[str, list]] = []  # colour-CV targets (see markers.py)

    def set_queries(self, queries: list[str]) -> None:
        """Set the things to look for (open-vocabulary class names).

        Queries naming a colour marker (e.g. "orange square") are routed to the
        deterministic HSV colour detector instead of YOLO-World, which is unreliable
        on abstract geometric/colour fiducials. The rest stay on YOLO.
        """
        queries = [q.strip() for q in queries if q.strip()]
        self.queries = queries
        self._marker_specs = markers.specs_for_queries(queries)
        self._yolo_queries = [q for q in queries if not markers.is_marker_query(q)]
        if self._yolo_queries:
            self.model.set_classes(self._yolo_queries)

    def detect(self, frame: np.ndarray, conf: float | None = None) -> list[dict]:
        """Return detections for the current queries on `frame`.

        Each: {label, score, box:(x1,y1,x2,y2), center:(cx,cy), area_frac}.
        area_frac (box area / frame area) is a cheap distance proxy for approach.
        """
        if not self.queries or frame is None or frame.size == 0:
            return []
        # Colour markers go through deterministic HSV CV (markers.py); YOLO handles the rest.
        out: list[dict] = markers.detect(frame, self._marker_specs) if self._marker_specs else []
        if self._yolo_queries:
            res = self.model.predict(
                frame, conf=conf or config.DETECT_CONF,
                device=self.device, verbose=False,
            )[0]
            h, w = frame.shape[:2]
            for b in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                cls = int(b.cls[0])
                out.append({
                    "label": self._yolo_queries[cls] if cls < len(self._yolo_queries) else str(cls),
                    "score": float(b.conf[0]),
                    "box": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                    "area_frac": ((x2 - x1) * (y2 - y1)) / float(w * h),
                })
        out.sort(key=lambda d: d["score"], reverse=True)
        return out

    @staticmethod
    def annotate(frame: np.ndarray, dets: list[dict]) -> np.ndarray:
        """Draw boxes + labels onto a copy of the frame (for the web overlay)."""
        img = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            tag = f"{d['label']} {d['score']:.2f}"
            cv2.rectangle(img, (x1, y1 - 18), (x1 + 9 * len(tag), y1), (0, 255, 0), -1)
            cv2.putText(img, tag, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img
