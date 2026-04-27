"""Verifier models.

`LogisticVerifier` is the CPU reference used in the toy setting and tests: a
progress-weighted logistic model with a pairwise same-task ranking term, i.e. the
objective of Eq. (3) applied to fixed feature vectors. `HFVerifier` wraps a
causal LM with a scalar head for real trajectories (requires torch/transformers).
"""
from __future__ import annotations

import numpy as np


class LogisticVerifier:
    def __init__(self, dim: int, lr: float = 0.1, epochs: int = 200, pairwise_weight: float = 0.3, l2: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, dim)
        self.b = 0.0
        self.lr, self.epochs, self.lam, self.l2 = lr, epochs, pairwise_weight, l2

    def score(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b

    def fit(self, X, y, progress=None, task_ids=None):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        wts = 0.5 + (np.ones(len(y)) if progress is None else np.asarray(progress, dtype=float))
        pairs = self._pairs(y, task_ids) if (task_ids is not None and self.lam > 0) else None
        for _ in range(self.epochs):
            s = self.score(X)
            p = 1 / (1 + np.exp(-s))
            g = wts * (p - y)
            gw = X.T @ g / len(y) + self.l2 * self.w
            gb = g.mean()
            if pairs is not None and len(pairs[0]):
                i, j = pairs
                d = s[i] - s[j]
                c = -1 / (1 + np.exp(d))  # d/dd log(1+e^{-d})
                gw += self.lam * (X[i] - X[j]).T @ c / len(i)
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    @staticmethod
    def _pairs(y, task_ids, max_pairs=20000, seed=0):
        rng = np.random.default_rng(seed)
        task_ids = np.asarray(task_ids)
        I, J = [], []
        for t in np.unique(task_ids):
            idx = np.where(task_ids == t)[0]
            pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
            for a in pos:
                for b in neg:
                    I.append(a); J.append(b)
        I, J = np.array(I, dtype=int), np.array(J, dtype=int)
        if len(I) > max_pairs:
            sel = rng.choice(len(I), max_pairs, replace=False)
            I, J = I[sel], J[sel]
        return I, J


class HFVerifier:  # pragma: no cover - needs GPU weights
    """Scalar-head verifier over rendered trajectory prefixes (Qwen3-8B in the paper)."""

    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 24576):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=torch.bfloat16).to(device).eval()
        self.device, self.max_length = device, max_length

    def score(self, rendered_prefixes: list[str]):
        self.tok.truncation_side = "left"
        enc = self.tok(rendered_prefixes, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
        with self.torch.no_grad():
            return self.model(**enc).logits.squeeze(-1).float().cpu().numpy()
