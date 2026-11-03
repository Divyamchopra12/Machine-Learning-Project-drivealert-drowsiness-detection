"""
Hand-crafted image features for the classical ML models (Random Forest, SVM, ...).

Classical models cannot learn features from raw pixels the way a CNN does, so each
image is turned into a fixed-length feature vector first:

  HOG (Histogram of Oriented Gradients) on a 64x64 grayscale image
      -> describes edges and shapes (eyelid line, open pupil, open mouth)
  + intensity histogram (16 bins)
      -> describes overall brightness / dark pupil area

Used by train_classical.py (training) and detector.py (live prediction), so the
features are always computed the same way.
"""
import cv2
import numpy as np

FEAT_SIZE = 64
_HOG = cv2.HOGDescriptor((FEAT_SIZE, FEAT_SIZE), (16, 16), (8, 8), (8, 8), 9)
HOG_LEN = int(_HOG.getDescriptorSize())  # 1764
HIST_BINS = 16
FEATURE_LEN = HOG_LEN + HIST_BINS


def image_features(img: np.ndarray) -> np.ndarray:
    """img: HxW (gray) or HxWx3 (RGB/BGR, uint8 or float 0-255) -> 1-D float32 vector."""
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(img.astype(np.uint8), (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)  # less sensitive to lighting
    hog = _HOG.compute(gray).ravel()
    hist = cv2.calcHist([gray], [0], None, [HIST_BINS], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([hog, hist]).astype(np.float32)


def batch_features(imgs) -> np.ndarray:
    return np.stack([image_features(i) for i in imgs])
