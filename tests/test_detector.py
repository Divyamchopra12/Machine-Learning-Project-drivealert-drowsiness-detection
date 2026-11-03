"""Detector behaviour on synthetic frames (no camera, no trained models needed)."""
import numpy as np

from detector import DetectorConfig, DrowsinessDetector, is_infrared


def make_detector(**kw):
    cfg = DetectorConfig(use_cnn=False, calibrate=False, phone_detection=False, use_temporal=False,
                         circadian_risk=False, **kw)
    return DrowsinessDetector(cfg, None, None)


def test_total_darkness_reports_too_dark_not_distraction():
    det = make_detector()
    black = np.full((240, 320, 3), 3, np.uint8)
    r = None
    for i in range(40):                      # 4 s at 10 fps
        r = det.process(black, timestamp=100 + i * 0.1)
    det.close()
    assert r["status"] == "too_dark"
    assert r["camera_blind"] is True
    assert r["alarm"] is False               # no false "eyes on road" alarm


def test_no_face_in_light_becomes_distraction_alarm():
    det = make_detector()
    grey = np.full((240, 320, 3), 128, np.uint8)
    r = None
    for i in range(40):
        r = det.process(grey, timestamp=100 + i * 0.1)
    det.close()
    assert r["camera_blind"] is False
    assert r["alarm_reason"] == "distracted"


def test_infrared_frame_detection():
    g = np.random.default_rng(0).integers(40, 200, (120, 160), dtype=np.uint8)
    assert is_infrared(np.dstack([g, g, g]))
    colour = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert not is_infrared(colour)


def test_fusion_rule_uses_config_weights():
    det = make_detector(eye_mode="fusion", fusion_weight=0.6, fusion_threshold=0.5)
    det.ear_thr = 0.2
    assert det._eyes_closed(0.15, 0.9)          # both say closed
    assert not det._eyes_closed(0.30, 0.2)      # both say open
    assert det._eyes_closed(0.30, 0.9)          # 0.6*0.9 = 0.54 >= 0.5 -> CNN wins
    det.cfg.fusion_weight = 0.4
    assert not det._eyes_closed(0.30, 0.9)      # 0.4*0.9 = 0.36 < 0.5
    det.close()
