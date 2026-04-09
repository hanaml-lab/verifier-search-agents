import numpy as np

from ccv import PlattCalibrator, expected_calibration_error


def test_platt_recovers_logistic_map():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 2, 20000)
    y = (rng.random(v.size) < 1 / (1 + np.exp(-(0.5 + 1.5 * v)))).astype(int)
    c = PlattCalibrator().fit(v, y)
    assert abs(c.w[0] - 0.5) < 0.1 and abs(c.w[1] - 1.5) < 0.1


def test_calibration_reduces_ece_of_overconfident_scores():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 5000)
    y = (rng.random(v.size) < 1 / (1 + np.exp(-v))).astype(int)
    raw = 1 / (1 + np.exp(-3 * v))  # overconfident reading
    c = PlattCalibrator().fit(v, y)
    assert expected_calibration_error(c.predict(v), y) < expected_calibration_error(raw, y) / 2


def test_separable_data_stays_finite():
    c = PlattCalibrator().fit([-2, -1, 1, 2], [0, 0, 1, 1])
    assert np.all(np.isfinite(c.w))
