"""Post-hoc calibration of verifier scores."""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


class PlattCalibrator:
    """Maps raw verifier scores v to q = sigmoid(w0 + w1 * v) by maximum likelihood.

    A small L2 penalty keeps the fit finite when the calibration set is separable.
    """

    def __init__(self, l2: float = 1e-4, max_iter: int = 100):
        self.l2 = l2
        self.max_iter = max_iter
        self.w = np.array([0.0, 1.0])

    def fit(self, scores, labels) -> "PlattCalibrator":
        v = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(labels, dtype=float).ravel()
        if v.shape != y.shape or v.size == 0:
            raise ValueError("scores and labels must be non-empty and the same shape")
        X = np.stack([np.ones_like(v), v], axis=1)
        w = np.array([0.0, 1.0])
        for _ in range(self.max_iter):
            p = _sigmoid(X @ w)
            g = X.T @ (y - p) - self.l2 * w
            H = -(X * (p * (1 - p))[:, None]).T @ X - self.l2 * np.eye(2)
            step = np.linalg.solve(H, g)
            w = w - step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.w = w
        return self

    def predict(self, scores):
        return _sigmoid(self.w[0] + self.w[1] * np.asarray(scores, dtype=float))


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """Equal-width binned ECE."""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
