"""Eye/mouth geometry and scoring helpers – the maths behind every alarm."""
import time

import numpy as np

from detector import circadian_period, estimate_kss, eye_aspect_ratio


def eye(open_px):
    # p1..p6: corner, top, top, corner, bottom, bottom (width 30 px)
    return np.array([[0, 0], [10, -open_px], [20, -open_px], [30, 0], [20, open_px], [10, open_px]], float)


def test_ear_open_eye_is_larger_than_closed():
    assert eye_aspect_ratio(eye(6)) > 0.3
    assert eye_aspect_ratio(eye(1)) < 0.1


def test_ear_is_scale_invariant():
    assert abs(eye_aspect_ratio(eye(6)) - eye_aspect_ratio(eye(6) * 3)) < 1e-6


def test_kss_estimate_range():
    assert estimate_kss(0) == 1
    assert estimate_kss(100) == 9
    assert all(1 <= estimate_kss(s) <= 9 for s in range(0, 101, 5))


def at_hour(h):
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, 30, 0, 0, 0, -1))


def test_circadian_risk_periods():
    assert circadian_period(at_hour(3))[0] == 1.25      # night low 1-5 AM
    assert circadian_period(at_hour(15))[0] == 1.10     # afternoon dip
    assert circadian_period(at_hour(10))[0] == 1.0      # daytime
