"""
Phone-in-hand detection (distraction) – both models are trained on the COCO dataset,
which has a "cell phone" class.

Backends (picked automatically):
  yolo       Ultralytics YOLOv8n  – used if you `pip install ultralytics` (downloads yolov8n.pt once)
  mediapipe  MediaPipe EfficientDet-Lite0 – no extra install; the 4 MB model is downloaded
             once into models/efficientdet_lite0.tflite

The detector runs every few frames to keep the app fast.
"""
from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

EFFICIENTDET_URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/"
                    "efficientdet_lite0/int8/1/efficientdet_lite0.tflite")
COCO_PHONE = 67  # "cell phone" in YOLO's COCO class list


class PhoneDetector:
    def __init__(self, model_dir="models", backend="auto", min_score=0.35):
        self.backend, self.error, self._model = None, None, None
        self.min_score = min_score
        if backend in ("auto", "yolo"):
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(Path(model_dir) / "yolov8n.pt") if (Path(model_dir) / "yolov8n.pt").exists()
                                   else "yolov8n.pt")
                self.backend = "YOLOv8n"
                return
            except Exception as exc:  # not installed or no download
                self.error = f"YOLO unavailable ({exc.__class__.__name__})"
                if backend == "yolo":
                    return
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            path = Path(model_dir) / "efficientdet_lite0.tflite"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[phone] downloading EfficientDet-Lite0 -> {path}")
                tmp = path.with_suffix(".part")
                urllib.request.urlretrieve(EFFICIENTDET_URL, tmp)   # no half-written model on failure
                tmp.replace(path)
            opts = vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE, score_threshold=min_score,
                category_allowlist=["cell phone"], max_results=3)
            self._model = vision.ObjectDetector.create_from_options(opts)
            self.backend = "EfficientDet-Lite0"
            self.error = None
        except Exception as exc:
            self.error = f"phone detection disabled: {exc}"
            self._model = None

    @property
    def ok(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray):
        """-> list of (x0, y0, x1, y1, score) in pixels for detected phones."""
        if self._model is None:
            return []
        try:
            if self.backend == "YOLOv8n":
                r = self._model(frame_bgr, classes=[COCO_PHONE], conf=self.min_score, verbose=False)[0]
                return [(*map(float, b), float(c)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
            import mediapipe as mp

            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            res = self._model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            out = []
            for d in res.detections:
                bb = d.bounding_box
                out.append((bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height,
                            d.categories[0].score))
            return out
        except Exception as exc:
            self.error = str(exc)
            return []


@lru_cache(maxsize=2)
def get_phone_detector(model_dir="models", backend="auto"):
    det = PhoneDetector(model_dir, backend)
    print(f"[phone] backend: {det.backend or 'none'}" + (f" ({det.error})" if det.error else ""))
    return det


def near_face(box, face_box) -> bool:
    """Phone held at the ear or in front of the face/chest (face box enlarged)."""
    if face_box is None:
        return True   # face hidden by the phone counts too
    fx0, fy0, fx1, fy1 = face_box
    w, h = fx1 - fx0, fy1 - fy0
    ex0, ey0, ex1, ey1 = fx0 - w, fy0 - 0.3 * h, fx1 + w, fy1 + 1.5 * h
    x0, y0, x1, y1 = box[:4]
    return not (x1 < ex0 or x0 > ex1 or y1 < ey0 or y0 > ey1)
