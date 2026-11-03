"""
Shared helpers for the temporal (sequence) models: LSTM, GRU, BiLSTM, 1D-CNN.

Instead of one image, these models look at the last WINDOW frames (3 s at 10 fps)
of 8 per-frame signals, so they can learn *patterns over time* – slow blinks,
long closures, drooping head, yawns – that a single frame cannot show.

Per-frame vector (same in training and in the live detector):
  ear_ratio   EAR / personal threshold   (< 1 means eyes closing)
  mar         mouth aspect ratio
  pitch       head pitch vs neutral (degrees / 30)
  yaw         head yaw vs neutral   (degrees / 45)
  roll        |head roll| / 45
  closed      1 if eyes judged closed
  yawning     1 if yawning
  face        1 if a face was found
"""
from functools import lru_cache
from pathlib import Path
import json

import numpy as np

FEATURES = ["ear_ratio", "mar", "pitch", "yaw", "roll", "closed", "yawning", "face"]
WINDOW = 30          # frames per sequence
FPS = 10             # sampling rate of the sequences


def frame_vector(ear=None, ear_thr=None, mar=None, pitch=None, yaw=None, roll=None,
                 closed=False, yawning=False, face=True) -> np.ndarray:
    if not face or ear is None:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0], np.float32)
    return np.array([
        ear / max(ear_thr or 0.21, 1e-3),
        mar or 0.0,
        (pitch or 0.0) / 30.0,
        (yaw or 0.0) / 45.0,
        abs(roll or 0.0) / 45.0,
        float(closed),
        float(yawning),
        1.0,
    ], np.float32)


def vector_from_result(r: dict) -> np.ndarray:
    """Per-frame vector from a DrowsinessDetector result dict."""
    return frame_vector(r.get("ear"), r.get("ear_threshold"), r.get("mar"), r.get("head_pitch"),
                        r.get("head_yaw"), r.get("head_roll"), r.get("eyes_closed"), r.get("yawning"),
                        r.get("face_detected") and not r.get("calibrating"))


def window_stats(x: np.ndarray) -> np.ndarray:
    """Summary features of a window, for the classical baseline (Random Forest)."""
    return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0)])


@lru_cache(maxsize=2)
def load_temporal(model_dir: str = "models"):
    """Best sequence model saved by train_temporal.py, or None."""
    d = Path(model_dir)
    model_path, meta_path = d / "temporal_model.keras", d / "temporal_meta.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    try:
        from tensorflow import keras

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("source", "")).startswith("SYNTHETIC"):
            print("[temporal] model was trained on synthetic test data – not used live. "
                  "Train on UTA-RLDD or your own clips to enable it.")
            return None
        model = keras.models.load_model(model_path)
        print(f"[temporal] loaded {meta.get('best_model')} ({meta.get('window')} frames)")
        return {"model": model, "meta": meta,
                "mean": np.array(meta["mean"], np.float32), "std": np.array(meta["std"], np.float32)}
    except Exception as exc:
        print(f"[temporal] could not load: {exc}")
        return None


def predict_window(bundle, frames: np.ndarray) -> float:
    x = (frames - bundle["mean"]) / bundle["std"]
    return float(bundle["model"](x[None].astype(np.float32), training=False).numpy()[0, 0])
