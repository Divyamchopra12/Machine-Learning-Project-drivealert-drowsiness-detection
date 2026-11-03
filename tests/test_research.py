"""Statistics used in the report."""
import numpy as np

from significance import bootstrap, f1_acc, mcnemar
from tune_fusion import alarms_from, fused


def test_f1_and_accuracy():
    y = np.array([1, 1, 0, 0])
    assert f1_acc(y, np.array([1, 1, 0, 0])) == (1.0, 1.0)
    acc, f1 = f1_acc(y, np.array([1, 0, 0, 0]))
    assert acc == 0.75 and abs(f1 - 2 / 3) < 1e-9


def test_mcnemar_identical_models_not_significant():
    y = np.array([1, 0] * 50)
    b, c, p = mcnemar(y, y.copy(), y.copy())
    assert (b, c, p) == (0, 0, 1.0)


def test_mcnemar_clearly_better_model_is_significant():
    y = np.ones(100, int)
    ref = np.zeros(100, int)
    ref[:50] = 1                 # reference right on 50
    other = np.ones(100, int)    # other right on all 100
    b, c, p = mcnemar(y, ref, other)
    assert b == 0 and c == 50 and p < 0.001


def test_bootstrap_interval_contains_estimate():
    y = np.array([1, 0] * 100)
    pred = y.copy()
    pred[:20] = 1 - pred[:20]
    (lo, hi), _ = bootstrap(y, pred, n=300)
    assert lo <= f1_acc(y, pred)[0] <= hi


def test_alarm_timing():
    t = np.arange(0, 5, 0.1)
    closed = (t >= 1) & (t < 3)                  # 2 s closure
    assert alarms_from(t, closed, 1.5) == [t[t >= 2.5][0]]
    assert alarms_from(t, closed, 2.5) == []     # too short for a 2.5 s rule


def test_fused_rule_without_cnn_falls_back_to_ear():
    rows = np.array([(0.0, 0.15, 0.21, None, True), (0.1, 0.30, 0.21, None, False)], dtype=object)
    assert list(fused(rows, 0.6, 0.5, 1.0)) == [True, False]
