"""
Core drowsiness and distraction detection logic.

Signals per frame (MediaPipe Face Mesh landmarks):
  - EAR  (Eye Aspect Ratio)    -> eye closure, blinks, PERCLOS
  - MAR  (Mouth Aspect Ratio)  -> yawning
  - Head pose (solvePnP)       -> head nodding (pitch) and looking away (yaw)
  - Optional CNNs (train_model.py) -> eye-closed and yawn probabilities,
    fused with the geometric signals.

Extra features
  - Distraction warning (inspired by the EU ADDW requirement): head turned
    away from the road, or face lost, for longer than `distraction_seconds`.
  - Night mode: dark frames are enhanced with CLAHE before landmark detection.
  - Estimated KSS (Karolinska Sleepiness Scale 1-9) derived from the fatigue
    score. This is an approximation for display, not a validated measurement.
  - eye_mode = "fusion" | "ear" | "cnn" for the ablation study (evaluate.py).

A short calibration (default 3 s, eyes open, looking ahead) personalises the
EAR threshold and the neutral head pitch / yaw.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from phone_detector import get_phone_detector, near_face
from temporal import FPS as SEQ_FPS, WINDOW as SEQ_WINDOW, frame_vector, load_temporal, predict_window

# MediaPipe Face Mesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]   # p1..p6 (corner, top, top, corner, bottom, bottom)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = [(81, 178), (13, 14), (311, 402)]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]

# Head pose: nose tip, chin, eye corners (image-left, image-right), mouth corners
POSE_IDX = [1, 152, 33, 263, 57, 287]
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

KSS_LABELS = {
    1: "Extremely alert", 2: "Very alert", 3: "Alert", 4: "Rather alert",
    5: "Neither alert nor sleepy", 6: "Some signs of sleepiness",
    7: "Sleepy, no effort to stay awake", 8: "Sleepy, some effort to stay awake",
    9: "Very sleepy, fighting sleep",
}


@dataclass
class DetectorConfig:
    ear_threshold: float = 0.21        # used if calibration is off
    mar_threshold: float = 0.60
    eye_closed_seconds: float = 1.5    # continuous closure -> alarm
    yawn_seconds: float = 1.0          # mouth open this long = one yawn
    head_tilt_degrees: float = 18.0    # pitch change from neutral
    head_down_seconds: float = 2.0     # sustained tilt -> alarm
    distraction_degrees: float = 30.0  # yaw change from neutral = looking away
    distraction_seconds: float = 2.5   # looking away / face lost this long -> alarm
    perclos_window: float = 60.0       # seconds
    alarm_score: float = 70.0          # fatigue score that triggers the alarm
    low_light_enhance: bool = True     # CLAHE night mode
    low_light_threshold: float = 70.0  # mean brightness (0-255) below which frames are enhanced
    dark_threshold: float = 25.0       # below this and no face found -> camera is "blind" (too dark)
    dark_seconds: float = 2.0          # how long before "too dark" is reported
    night_eye_model: bool = True       # use the infrared-trained eye CNN (eye_night.keras) for IR / dark frames
    calibrate: bool = True
    calibration_seconds: float = 3.0
    use_cnn: bool = True
    eye_mode: str = "fusion"           # "fusion" (CNN+EAR) | "ear" | "cnn" | "classical" (best classical ML + EAR)
    fusion_weight: float = 0.6         # fusion rule: w*P(closed) + (1-w)*[EAR < thr] >= fusion_threshold
    fusion_threshold: float = 0.5      # (tuned on your own clips with tune_fusion.py)
    # --- real-world robustness ---
    circadian_risk: bool = True        # more sensitive at 1-5 AM, 2-4 PM and 6 PM-1 AM
    time_on_task_min: float = 90.0     # continuous driving after which fatigue is raised (0 = off)
    speed_gate_kmh: float = 0.0        # no alarms below this GPS speed, e.g. parked (0 = off)
    eye_occlusion_check: bool = True   # sunglasses / hidden eyes -> rely on head + mouth
    predict_window_min: float = 10.0   # window for the fatigue trend prediction
    use_temporal: bool = True          # LSTM/GRU sequence model (train_temporal.py), if trained
    phone_detection: bool = True       # phone-in-hand detection (phone_detector.py)
    phone_seconds: float = 2.0         # phone visible near the face this long -> alarm


# ------------------------------------------------------------ helpers ----
def eye_aspect_ratio(p: np.ndarray) -> float:
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2.0 * c + 1e-6))


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    vertical = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in MOUTH_VERTICAL])
    horizontal = np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]])
    return float(vertical / (horizontal + 1e-6))


def circadian_period(t: float):
    """Time-of-day risk factor. Alertness dips at 1-5 AM and 2-4 PM (IIT Bombay study),
    and most sleep-related crashes in the AIIMS Rishikesh study happened 6 PM - midnight."""
    h = time.localtime(t).tm_hour
    if 1 <= h < 5:
        return 1.25, "Night low (1–5 AM)"
    if 14 <= h < 16:
        return 1.10, "Afternoon dip (2–4 PM)"
    if h >= 18 or h < 1:
        return 1.10, "Evening (6 PM–1 AM)"
    return 1.0, "Daytime"


def eyes_occluded(gray: np.ndarray, pts: np.ndarray) -> bool:
    """Sunglasses check: the eye region is much darker and flatter than the rest of the face."""
    def region(idx, pad):
        p = pts[idx]
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        w, h = x1 - x0, y1 - y0
        a, b = int(max(x0 - pad * w, 0)), int(max(y0 - pad * h, 0))
        c, d = int(min(x1 + pad * w, gray.shape[1])), int(min(y1 + pad * h, gray.shape[0]))
        return gray[b:d, a:c]
    eyes = np.concatenate([region(LEFT_EYE, 0.3).ravel(), region(RIGHT_EYE, 0.3).ravel()])
    face = region(FACE_OVAL, 0.0)
    if eyes.size < 20 or face.size < 100:
        return False
    ratio = eyes.mean() / (face.mean() + 1e-6)
    return bool(ratio < 0.5 and eyes.std() < 25)


def estimate_kss(score: float) -> int:
    """Rough mapping of the 0-100 fatigue score onto the 1-9 KSS scale."""
    return int(np.clip(round(1 + score / 100 * 8), 1, 9))


def is_infrared(bgr: np.ndarray) -> bool:
    """IR / night-vision cameras give (almost) grey images: the colour channels are nearly equal."""
    small = bgr[::8, ::8].astype(np.int16)
    return float(np.abs(small[:, :, 0] - small[:, :, 1]).mean() + np.abs(small[:, :, 1] - small[:, :, 2]).mean()) < 6.0


def enhance_low_light(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB + mild gamma lift (night mode)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    lut = np.array([((i / 255.0) ** 0.7) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


def _square_crop(img, x0, y0, x1, y1, scale, size):
    h, w = img.shape[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * scale / 2
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, w)), int(min(cy + half, h))
    if c - a < 4 or d - b < 4:
        return None
    return cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=4)
def load_keras_model(path: str):
    """Load a model once and share it between detector instances."""
    if not path or not Path(path).exists():
        return None
    try:
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras

        model = keras.models.load_model(path)
        print(f"[detector] loaded {path}")
        return model
    except Exception as exc:  # TensorFlow missing or incompatible file
        print(f"[detector] could not load {path}: {exc}")
        return None


@lru_cache(maxsize=4)
def load_classical_model(path: str):
    """Best classical model saved by train_classical.py (Random Forest, SVM, ...)."""
    if not path or not Path(path).exists():
        return None
    try:
        import joblib

        bundle = joblib.load(path)
        print(f"[detector] loaded {bundle['name']} from {path}")
        return bundle
    except Exception as exc:
        print(f"[detector] could not load {path}: {exc}")
        return None


# ----------------------------------------------------------- detector ----
class DrowsinessDetector:
    def __init__(self, config: DetectorConfig | None = None,
                 eye_model_path: str | None = "models/eye_model.keras",
                 yawn_model_path: str | None = "models/yawn_model.keras"):
        self.cfg = config or DetectorConfig()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.eye_model = load_keras_model(eye_model_path) if self.cfg.use_cnn else None
        self.yawn_model = load_keras_model(yawn_model_path) if self.cfg.use_cnn else None
        base = Path(eye_model_path).parent if eye_model_path else Path("models")
        night_path = base / "eye_night.keras"
        self.night_model = (load_keras_model(str(night_path))
                            if self.cfg.use_cnn and self.cfg.night_eye_model and night_path.exists() else None)
        self.ir_votes = deque(maxlen=30)
        self.ir_mode = False
        self.using_night = False
        self.dark_since = None
        self.classical_eye = load_classical_model(str(base / "eye_classical.joblib"))
        self.temporal = load_temporal(str(base))
        self.phone = get_phone_detector(str(base)) if self.cfg.phone_detection else None
        self.frame_i = 0
        self.last_pts = None
        self.speed_kmh = None      # set from the phone/laptop GPS by the server
        self.ear_scale = 1.0       # learned from "false alarm" feedback (per driver)
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self):
        self.start = None
        self.calibrated = not self.cfg.calibrate
        self._calib_ear, self._calib_pitch, self._calib_yaw = [], [], []
        self.ear_thr = self.cfg.ear_threshold * self.ear_scale
        self.base_pitch = None
        self.base_yaw = None
        self.closed_since = None
        self.mouth_open_since = None
        self.yawn_counted = False
        self.tilt_since = None
        self.nod_counted = False
        self.away_since = None
        self.away_counted = False
        self.face_lost_since = None
        self.closed_hist = deque()   # (t, closed)
        self.blinks = deque()
        self.yawns = deque()
        self.nods = deque()
        self.distractions = deque()
        self.score_samples = deque()        # (t, score) every 5 s for the trend prediction
        self.seq = deque(maxlen=SEQ_WINDOW)  # per-frame vectors for the temporal model
        self.seq_t = self.pred_t = None
        self.temporal_prob = None
        self.phone_boxes = []
        self.phone_since = None
        self.phone_counted = False
        self.phone_events = deque()
        self.occl_votes = deque(maxlen=15)  # sunglasses check, majority vote
        self.eyes_visible = True
        self.alarm_events = 0
        self.alarm_times = deque()
        self.alarm_since = None
        self._last_alarm = False

    def recalibrate(self):
        self.reset()

    # -- adaptive personalisation ----------------------------------------
    PROFILE_KEYS = ("eye_closed_seconds", "head_tilt_degrees", "distraction_seconds",
                    "distraction_degrees", "alarm_score")

    def apply_feedback(self, reason: str) -> dict:
        """Driver pressed "False alarm": make that alarm a little harder to trigger."""
        c = self.cfg
        if reason == "eyes_closed":
            self.ear_scale = max(0.80, self.ear_scale * 0.95)
            self.ear_thr *= 0.95
            c.eye_closed_seconds = min(c.eye_closed_seconds + 0.25, 3.0)
        elif reason == "head_down":
            c.head_tilt_degrees = min(c.head_tilt_degrees + 2, 30)
        elif reason == "distracted":
            c.distraction_seconds = min(c.distraction_seconds + 0.5, 6.0)
            c.distraction_degrees = min(c.distraction_degrees + 3, 50)
        elif reason == "high_fatigue":
            c.alarm_score = min(c.alarm_score + 3, 90)
        return self.profile()

    def profile(self) -> dict:
        return {"ear_scale": round(self.ear_scale, 3),
                **{k: getattr(self.cfg, k) for k in self.PROFILE_KEYS}}

    def load_profile(self, p: dict):
        self.ear_scale = float(p.get("ear_scale", 1.0))
        for k in self.PROFILE_KEYS:
            if k in p:
                setattr(self.cfg, k, p[k])

    def close(self):
        self.face_mesh.close()

    @property
    def cnn_loaded(self) -> dict:
        return {"eye": self.eye_model is not None, "yawn": self.yawn_model is not None,
                "night": self.night_model is not None,
                "classical": None if self.classical_eye is None else self.classical_eye["name"]}

    def _eye_model_name(self) -> str:
        mode = self.cfg.eye_mode
        if self.using_night and mode in ("fusion", "cnn"):
            return "Night IR CNN" + (" + EAR" if mode == "fusion" else " only")
        if mode == "classical" and self.classical_eye is not None:
            return f"{self.classical_eye['name']} + EAR"
        if mode == "fusion" and self.eye_model is not None:
            return "MobileNetV2 CNN + EAR"
        if mode == "cnn" and self.eye_model is not None:
            return "MobileNetV2 CNN only"
        return "EAR only"

    # -- models -----------------------------------------------------------
    def _model_size(self, model):
        return int(model.input_shape[1])

    def _eye_closed_prob(self, rgb, pts, model=None, gray=False):
        model = model or self.eye_model
        size = self._model_size(model)
        crops = []
        for idx in (LEFT_EYE, RIGHT_EYE):
            e = pts[idx]
            (x0, y0), (x1, y1) = e.min(0), e.max(0)
            crop = _square_crop(rgb, x0, y0, x1, y1, 1.8, size)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return None
        batch = np.stack(crops).astype(np.float32)
        if gray:   # the night model was trained on grey infrared images
            batch = np.repeat(batch.mean(axis=3, keepdims=True), 3, axis=3)
        return float(model(batch, training=False).numpy().mean())

    def _eye_closed_prob_classical(self, rgb, pts):
        from features import batch_features

        crops = []
        for idx in (LEFT_EYE, RIGHT_EYE):
            e = pts[idx]
            (x0, y0), (x1, y1) = e.min(0), e.max(0)
            crop = _square_crop(rgb, x0, y0, x1, y1, 1.8, 64)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return None
        return float(self.classical_eye["model"].predict_proba(batch_features(crops))[:, 1].mean())

    def _yawn_prob(self, rgb, pts):
        size = self._model_size(self.yawn_model)
        face = pts[FACE_OVAL]
        (x0, y0), (x1, y1) = face.min(0), face.max(0)
        crop = _square_crop(rgb, x0, y0, x1, y1, 1.15, size)
        if crop is None:
            return None
        return float(self.yawn_model(crop[None].astype(np.float32), training=False).numpy()[0, 0])

    @staticmethod
    def _head_pose(pts, w, h):
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts[POSE_IDX].astype(np.float64), cam,
                                   np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = cv2.RQDecomp3x3(rmat)[0]
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return float(pitch), float(yaw), float(roll)

    def _eyes_closed(self, ear, eye_prob):
        ear_closed = ear < self.ear_thr
        mode = self.cfg.eye_mode
        if mode == "ear" or eye_prob is None:
            return ear_closed
        if mode == "cnn":
            return eye_prob >= 0.5
        # "fusion" (CNN + EAR) and "classical" (classical ML + EAR) use the same rule
        w = self.cfg.fusion_weight
        return (w * eye_prob + (1 - w) * float(ear_closed)) >= self.cfg.fusion_threshold

    # -- main entry -------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> dict:
        t = time.time() if timestamp is None else timestamp
        if self.start is None:
            self.start = t
        h, w = frame_bgr.shape[:2]

        brightness = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())
        enhanced = self.cfg.low_light_enhance and brightness < self.cfg.low_light_threshold
        work = enhance_low_light(frame_bgr) if enhanced else frame_bgr
        self.ir_votes.append(brightness > 8 and is_infrared(frame_bgr))   # black frames are grey too
        self.ir_mode = sum(self.ir_votes) > len(self.ir_votes) // 2
        light = dict(brightness=brightness, enhanced=enhanced)

        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        self.frame_i += 1
        if self.phone and self.phone.ok and self.cfg.phone_detection and self.frame_i % 3 == 0:
            self.phone_boxes = self.phone.detect(work)
        face_box = None
        if res.multi_face_landmarks:
            fo = np.array([(res.multi_face_landmarks[0].landmark[i].x * w, res.multi_face_landmarks[0].landmark[i].y * h)
                           for i in FACE_OVAL])
            face_box = (*fo.min(0), *fo.max(0))
        phone_now = self.calibrated and any(near_face(b, face_box) for b in self.phone_boxes)
        if phone_now:
            if self.phone_since is None:
                self.phone_since, self.phone_counted = t, False
            if t - self.phone_since >= self.cfg.phone_seconds and not self.phone_counted:
                self.phone_events.append(t)
                self.phone_counted = True
        else:
            self.phone_since = None
        self._phone_dur = t - self.phone_since if self.phone_since is not None else 0.0

        if not res.multi_face_landmarks:
            self.last_pts = None
            self.closed_since = self.mouth_open_since = self.tilt_since = self.away_since = None
            lost = 0.0
            # Too dark to see anything: this is a camera problem, not the driver looking away.
            # Report it instead of raising a false "eyes on road" alarm.
            if brightness < self.cfg.dark_threshold:
                self.dark_since = self.dark_since or t
                self.face_lost_since = None
                return self._result(t, face=False, **light)
            self.dark_since = None
            if self.calibrated:
                self.face_lost_since = self.face_lost_since or t
                lost = t - self.face_lost_since
                self._feed_temporal(t, frame_vector(face=False))
            return self._result(t, face=False, away_dur=lost, **light)
        self.face_lost_since = None
        self.dark_since = None

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        self.last_pts = pts

        ear = (eye_aspect_ratio(pts[LEFT_EYE]) + eye_aspect_ratio(pts[RIGHT_EYE])) / 2
        mar = mouth_aspect_ratio(pts)
        if self.cfg.eye_occlusion_check:   # sunglasses -> eye signals unreliable
            self.occl_votes.append(eyes_occluded(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), pts))
            self.eyes_visible = sum(self.occl_votes) <= len(self.occl_votes) // 2
        else:
            self.eyes_visible = True
        pitch, yaw, roll = self._head_pose(pts, w, h)
        mode = self.cfg.eye_mode
        if mode == "classical" and self.classical_eye is not None:
            eye_prob = self._eye_closed_prob_classical(rgb, pts)
        elif mode in ("fusion", "cnn") and self.night_model is not None and (self.ir_mode or enhanced):
            eye_prob = self._eye_closed_prob(rgb, pts, self.night_model, gray=True)
        elif mode in ("fusion", "cnn") and self.eye_model is not None:
            eye_prob = self._eye_closed_prob(rgb, pts)
        else:
            eye_prob = None
        self.using_night = eye_prob is not None and self.night_model is not None and (self.ir_mode or enhanced)
        yawn_prob = self._yawn_prob(rgb, pts) if self.yawn_model is not None else None
        signals = dict(ear=ear, mar=mar, pitch=pitch, yaw=yaw, roll=roll,
                       eye_prob=eye_prob, yawn_prob=yawn_prob, **light)

        # Calibration: learn personal open-eye EAR and neutral head pose
        if not self.calibrated:
            self._calib_ear.append(ear)
            self._calib_pitch.append(pitch)
            self._calib_yaw.append(yaw)
            if t - self.start >= self.cfg.calibration_seconds and len(self._calib_ear) >= 10:
                self.ear_thr = float(np.clip(0.75 * np.median(self._calib_ear) * self.ear_scale, 0.12, 0.30))
                self.base_pitch = float(np.median(self._calib_pitch))
                self.base_yaw = float(np.median(self._calib_yaw))
                self.calibrated = True
                self.start = t  # session starts after calibration
            return self._result(t, face=True, calibrating=True, **signals)
        if self.base_pitch is None:
            self.base_pitch, self.base_yaw = pitch, yaw

        # Eyes (ignored when hidden, e.g. sunglasses: head pose + mouth still work)
        eyes_closed = self.eyes_visible and self._eyes_closed(ear, eye_prob)
        if eyes_closed:
            self.closed_since = self.closed_since or t
            closed_dur = t - self.closed_since
        else:
            if self.closed_since is not None:
                d = t - self.closed_since
                if 0.05 <= d < self.cfg.eye_closed_seconds:
                    self.blinks.append(t)
            self.closed_since = None
            closed_dur = 0.0

        # Mouth: yawn when open wide long enough
        mouth_open = mar > self.cfg.mar_threshold or (yawn_prob is not None and yawn_prob > 0.85)
        if mouth_open:
            if self.mouth_open_since is None:
                self.mouth_open_since, self.yawn_counted = t, False
            yawn_dur = t - self.mouth_open_since
            if yawn_dur >= self.cfg.yawn_seconds and not self.yawn_counted:
                self.yawns.append(t)
                self.yawn_counted = True
        else:
            self.mouth_open_since = None
            yawn_dur = 0.0
        yawning = yawn_dur >= self.cfg.yawn_seconds

        # Head pitch: sustained change from neutral = nod / head drop
        tilt = abs(pitch - self.base_pitch) > self.cfg.head_tilt_degrees
        if tilt:
            if self.tilt_since is None:
                self.tilt_since, self.nod_counted = t, False
            tilt_dur = t - self.tilt_since
            if tilt_dur >= 0.5 and not self.nod_counted:
                self.nods.append(t)
                self.nod_counted = True
        else:
            self.tilt_since = None
            tilt_dur = 0.0

        # Head yaw: looking away from the road (distraction)
        away = abs(yaw - self.base_yaw) > self.cfg.distraction_degrees
        if away:
            if self.away_since is None:
                self.away_since, self.away_counted = t, False
            away_dur = t - self.away_since
            if away_dur >= self.cfg.distraction_seconds and not self.away_counted:
                self.distractions.append(t)
                self.away_counted = True
        else:
            self.away_since = None
            away_dur = 0.0

        if self.eyes_visible:
            self.closed_hist.append((t, eyes_closed))
        self._feed_temporal(t, frame_vector(ear, self.ear_thr, mar, pitch - self.base_pitch, yaw - self.base_yaw,
                                            roll, eyes_closed, yawning, True))
        return self._result(t, face=True, eyes_closed=eyes_closed, closed_dur=closed_dur,
                            yawning=yawning, tilt=tilt, tilt_dur=tilt_dur, away=away,
                            away_dur=away_dur, **signals)

    # -- temporal model ----------------------------------------------------
    def _feed_temporal(self, t, vec):
        """Sample at SEQ_FPS; once the window is full, predict P(drowsy) once per second."""
        if not (self.cfg.use_temporal and self.temporal):
            self.temporal_prob = None
            return
        if self.seq_t is None or t - self.seq_t >= 0.9 / SEQ_FPS:
            self.seq.append(vec)
            self.seq_t = t
        if len(self.seq) == SEQ_WINDOW and (self.pred_t is None or t - self.pred_t >= 1.0):
            self.temporal_prob = predict_window(self.temporal, np.stack(self.seq))
            self.pred_t = t

    # -- aggregation ------------------------------------------------------
    def _prune(self, t):
        while self.closed_hist and t - self.closed_hist[0][0] > self.cfg.perclos_window:
            self.closed_hist.popleft()
        for dq, win in ((self.blinks, 60.0), (self.yawns, 300.0), (self.nods, 300.0),
                        (self.distractions, 300.0), (self.phone_events, 300.0)):
            while dq and t - dq[0] > win:
                dq.popleft()

    def _result(self, t, face, calibrating=False, eyes_closed=False, closed_dur=0.0,
                yawning=False, tilt=False, tilt_dur=0.0, away=False, away_dur=0.0,
                ear=None, mar=None, pitch=None, yaw=None, roll=None, eye_prob=None,
                yawn_prob=None, brightness=None, enhanced=False):
        self._prune(t)
        elapsed = max(t - self.start, 1e-6)
        perclos = (float(np.mean([c for _, c in self.closed_hist]))
                   if self.closed_hist else 0.0)
        blink_rate = len(self.blinks) * 60.0 / min(max(elapsed, 1.0), 60.0)
        yawns, nods = len(self.yawns), len(self.nods)

        # fatigue score = signals + time-on-task, scaled by the time-of-day risk
        p_perclos = min(perclos / 0.30, 1.0) * 50
        p_yawn = min(yawns / 3, 1.0) * 25
        p_nod = min(nods / 3, 1.0) * 25
        drive_min = 0.0 if calibrating else elapsed / 60
        task = self.cfg.time_on_task_min
        p_task = min((drive_min - task) / 30 * 10, 20.0) if task and drive_min > task else 0.0
        circ, period = circadian_period(t) if self.cfg.circadian_risk else (1.0, "Off")
        base = p_perclos + p_yawn + p_nod + p_task
        score = min(base * circ, 100.0)
        # sequence model (LSTM/GRU) opinion: 30% of the score once it has 3 s of data
        p_temporal = None
        if self.temporal_prob is not None and face and not calibrating:
            p_temporal = self.temporal_prob * 100
            score = min(0.7 * score + 0.3 * p_temporal, 100.0)
        trend_score = score
        if closed_dur > 0:
            score = max(score, min(closed_dur / self.cfg.eye_closed_seconds, 1.0) * 100)

        # fatigue trend: linear fit over the last N minutes -> minutes until score reaches 60
        if face and not calibrating and (not self.score_samples or t - self.score_samples[-1][0] >= 5):
            self.score_samples.append((t, trend_score))
        while self.score_samples and t - self.score_samples[0][0] > self.cfg.predict_window_min * 60:
            self.score_samples.popleft()
        slope = eta = None
        if len(self.score_samples) >= 6:
            ts = np.array([s[0] for s in self.score_samples])
            ys = np.array([s[1] for s in self.score_samples])
            if ts[-1] - ts[0] >= 30:
                slope = float(np.polyfit((ts - ts[0]) / 60, ys, 1)[0])  # points per minute
                if slope > 0.5 and trend_score < 60:
                    eta = (60 - trend_score) / slope

        dark_dur = t - self.dark_since if (self.dark_since is not None and not face) else 0.0
        blind = dark_dur >= self.cfg.dark_seconds
        distracted = self.calibrated and not calibrating and away_dur >= self.cfg.distraction_seconds
        phone_dur = getattr(self, "_phone_dur", 0.0)
        reason = None
        if not calibrating:
            if face and closed_dur >= self.cfg.eye_closed_seconds:
                reason = "eyes_closed"
            elif face and tilt_dur >= self.cfg.head_down_seconds:
                reason = "head_down"
            elif phone_dur >= self.cfg.phone_seconds:
                reason = "phone_use"
            elif distracted:
                reason = "distracted"
            elif face and score >= self.cfg.alarm_score:
                reason = "high_fatigue"
        # speed gating: no alarms while the vehicle is stopped (GPS speed below the gate)
        gated = bool(self.cfg.speed_gate_kmh and self.speed_kmh is not None
                     and self.speed_kmh < self.cfg.speed_gate_kmh)
        if gated:
            reason = None
        alarm = reason is not None
        if alarm and not self._last_alarm:
            self.alarm_events += 1
            self.alarm_times.append(t)
        self._last_alarm = alarm
        self.alarm_since = (self.alarm_since or t) if alarm else None
        while self.alarm_times and t - self.alarm_times[0] > 600:
            self.alarm_times.popleft()

        # Escalating alarm level (0-3), loosely following the EU DDAW idea of an
        # early warning (KSS 7) and a mandatory warning (KSS 8+):
        #   1 = early warning   : fatigue rising, yawning or looking away
        #   2 = alarm           : microsleep, head drop, distraction or high score
        #   3 = critical alarm  : alarm lasting >= 3 s, or 3+ alarms in 10 minutes
        alarm_seconds = t - self.alarm_since if alarm else 0.0
        if alarm and (alarm_seconds >= 3.0 or len(self.alarm_times) >= 3):
            level = 3
        elif alarm:
            level = 2
        elif not calibrating and not gated and (phone_dur > 0 or blind):
            level = 1
        elif face and not calibrating and not gated and (score >= 35 or yawning or away
                                                        or (eta is not None and eta <= 10)):
            level = 1
        else:
            level = 0

        if calibrating:
            status = "calibrating"
        elif reason in ("distracted", "phone_use"):
            status = "distracted"
        elif alarm or (face and score >= 60):
            status = "drowsy"
        elif not face and blind:
            status = "too_dark"
        elif not face:
            status = "no_face"
        elif score >= 35 or yawning or away:
            status = "warning"
        else:
            status = "alert"

        kss = estimate_kss(score)
        rnd = lambda v, n=3: None if v is None else round(float(v), n)  # noqa: E731
        return {
            "status": status,
            "alarm": alarm,
            "alarm_reason": reason,
            "alarm_level": level,
            "alarm_seconds": round(alarm_seconds, 2),
            "alarms_last_10min": len(self.alarm_times),
            "face_detected": face,
            "calibrating": calibrating,
            "fatigue_score": round(score, 1),
            "est_kss": kss,
            "est_kss_label": KSS_LABELS[kss],
            "ear": rnd(ear),
            "ear_threshold": rnd(self.ear_thr),
            "mar": rnd(mar),
            "eye_closed_prob": rnd(eye_prob),
            "yawn_prob": rnd(yawn_prob),
            "eyes_closed": bool(eyes_closed),
            "eyes_closed_seconds": round(closed_dur, 2),
            "yawning": bool(yawning),
            "yawn_count": yawns,
            "blink_rate": round(blink_rate, 1),
            "perclos": round(perclos, 3),
            "head_pitch": rnd(None if pitch is None or self.base_pitch is None
                              else pitch - self.base_pitch, 1),
            "head_yaw": rnd(None if yaw is None or self.base_yaw is None
                            else yaw - self.base_yaw, 1),
            "head_roll": rnd(roll, 1),
            "head_tilted": bool(tilt),
            "nod_count": nods,
            "looking_away": bool(away or (not face and away_dur > 0)),
            "looking_away_seconds": round(away_dur, 2),
            "distraction_count": len(self.distractions),
            "brightness": rnd(brightness, 1),
            "low_light": bool(brightness is not None and brightness < self.cfg.low_light_threshold),
            "night_mode": bool(enhanced),
            "camera_blind": blind,
            "dark_seconds": round(dark_dur, 1),
            "ir_camera": bool(self.ir_mode),
            "night_model": self.night_model is not None,
            "night_model_active": bool(self.using_night),
            "eyes_visible": bool(self.eyes_visible) if face else None,
            "risk_period": period,
            "circadian_factor": circ,
            "drive_minutes": round(drive_min, 1),
            "score_parts": {"perclos": round(p_perclos, 1), "yawn": round(p_yawn, 1),
                            "nod": round(p_nod, 1), "time_on_task": round(p_task, 1),
                            "temporal": None if p_temporal is None else round(p_temporal, 1)},
            "temporal_prob": None if self.temporal_prob is None else round(self.temporal_prob, 3),
            "phone_detected": phone_dur > 0,
            "phone_seconds": round(phone_dur, 2),
            "phone_backend": self.phone.backend if (self.phone and self.phone.ok) else None,
            "temporal_model": self.temporal["meta"]["best_model"] if (self.temporal and self.cfg.use_temporal) else None,
            "fatigue_trend": None if slope is None else round(slope, 2),
            "predicted_drowsy_min": None if eta is None else round(eta, 1),
            "speed_kmh": None if self.speed_kmh is None else round(self.speed_kmh, 1),
            "speed_gated": gated,
            "alarm_events": self.alarm_events,
            "session_seconds": round(elapsed, 1) if not calibrating else 0.0,
            "cnn": self.cnn_loaded,
            "eye_mode": self.cfg.eye_mode,
            "eye_model_used": self._eye_model_name(),
            "timestamp": t,
        }

    # -- drawing ----------------------------------------------------------
    def annotate(self, frame, r: dict):
        colors = {"alert": (80, 200, 80), "warning": (0, 190, 255),
                  "drowsy": (40, 40, 230), "distracted": (220, 80, 200), "calibrating": (255, 180, 0),
                  "no_face": (160, 160, 160), "too_dark": (60, 60, 200)}
        col = colors[r["status"]]
        if self.last_pts is not None:
            for idx in (LEFT_EYE, RIGHT_EYE):
                cv2.polylines(frame, [self.last_pts[idx].astype(np.int32)], True, col, 1)
            mouth = [61, 81, 13, 311, 291, 402, 14, 178]
            cv2.polylines(frame, [self.last_pts[mouth].astype(np.int32)], True, col, 1)

        for b in self.phone_boxes:
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (200, 80, 220), 2)
            cv2.putText(frame, f"phone {b[4]:.2f}", (int(b[0]), int(b[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 80, 220), 1)
        cv2.rectangle(frame, (0, 0), (290, 197), (20, 20, 20), -1)
        lines = [
            f"STATUS: {r['status'].upper()}",
            f"Fatigue score: {r['fatigue_score']}  (est. KSS {r['est_kss']})",
            f"EAR: {r['ear']} (thr {r['ear_threshold']})",
            f"MAR: {r['mar']}",
            f"PERCLOS: {r['perclos'] * 100:.0f}%",
            f"Blinks/min: {r['blink_rate']}  Yawns: {r['yawn_count']}",
            f"Pitch: {r['head_pitch']}  Yaw: {r['head_yaw']}",
            f"Night: {'IR cam ' if r['ir_camera'] else ''}{'CLAHE ON' if r['night_mode'] else 'off'}"
            f"{' TOO DARK' if r['camera_blind'] else ''}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 if i else 0.65, col if i == 0 else (235, 235, 235), 1 + (i == 0))
        if r["alarm"]:
            h, w = frame.shape[:2]
            msg = ("PUT THE PHONE DOWN!" if r["alarm_reason"] == "phone_use" else
                   "EYES ON ROAD!" if r["alarm_reason"] == "distracted" else "WAKE UP!")
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            cv2.putText(frame, msg, (w // 2 - 160, h - 40), cv2.FONT_HERSHEY_DUPLEX,
                        1.4, (0, 0, 255), 3)
        return frame


def config_dict(cfg: DetectorConfig) -> dict:
    return asdict(cfg)
