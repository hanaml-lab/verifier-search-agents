import numpy as np
import pytest

from ccv import crc_threshold, false_finish_rate


def test_threshold_controls_risk_on_fresh_data():
    rng = np.random.default_rng(0)
    def draw(n):
        q = rng.beta(2, 2, n)
        return q, (rng.random(n) < q).astype(int)
    q_cal, y_cal = draw(4000)
    risks = []
    for _ in range(30):
        t = crc_threshold(q_cal, y_cal, 0.1)
        q, y = draw(4000)
        risks.append(false_finish_rate(q, y, t)[0])
        q_cal, y_cal = draw(4000)
    assert np.mean(risks) <= 0.11


def test_no_feasible_threshold_returns_one():
    assert crc_threshold(np.full(50, 0.5), np.zeros(50), 0.1) == 1.0


def test_bad_alpha():
    with pytest.raises(ValueError):
        crc_threshold([0.5], [1], 0.0)


def test_false_finish_rate_counts():
    ffr, acc = false_finish_rate([0.9, 0.8, 0.2], [1, 0, 0], 0.5)
    assert ffr == 0.5 and abs(acc - 2 / 3) < 1e-12
