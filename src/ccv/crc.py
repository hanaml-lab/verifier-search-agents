"""False-finish control by conformal risk control (Section 3.4 of the paper)."""
from __future__ import annotations

import numpy as np


def false_finish_rate(probs, labels, threshold: float) -> tuple[float, float]:
    """Return (false-finish rate among accepted, acceptance rate) for a submit threshold."""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels).ravel()
    acc = p >= threshold
    if not acc.any():
        return 0.0, 0.0
    return float(((y == 0) & acc).sum() / acc.sum()), float(acc.mean())


def crc_threshold(probs, labels, alpha: float, n_min: int = 20) -> float:
    """Smallest threshold t whose corrected risk (fails+1)/(accepted+1) is <= alpha
    with at least n_min accepted calibration points. Returns 1.0 (accept nothing
    except certainties) when no threshold qualifies."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels).ravel()
    order = np.argsort(-p)
    ps, fails = p[order], (y[order] == 0).astype(int)
    cum_fail = np.cumsum(fails)
    n_acc = np.arange(1, len(ps) + 1)
    risk = (cum_fail + 1) / (n_acc + 1)
    # the accepted set for threshold ps[i] includes every tie with ps[i]
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]
    ok = (risk <= alpha) & (n_acc >= n_min) & last_of_tie
    if not ok.any():
        return 1.0
    return float(ps[np.where(ok)[0].max()])
