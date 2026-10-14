"""A CPU toy setting that exercises the full CCV pipeline in seconds.

Each task has a latent difficulty; each rollout has a hidden outcome and emits a
noisy feature vector whose informativeness grows with progress, mimicking how
execution evidence accumulates along a real agent trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIM = 6


def _sig(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class ToyTask:
    tid: int
    difficulty: float


class ToyRollout:
    def __init__(self, task: ToyTask, rng: np.random.Generator, ability: float = 0.0, mean_tokens: float = 0.6):
        self.task, self.rng = task, rng
        self.success = int(rng.random() < _sig(ability - task.difficulty))
        self.total_tokens = float(rng.lognormal(np.log(mean_tokens) - 0.18, 0.6))
        self._latent = rng.normal(0, 1, DIM)
        self._dir = np.array([1.0, 0.8, 0.6, 0.0, 0.0, -0.4])
        self.tokens = 0.0

    def features(self, progress: float):
        signal = (2 * self.success - 1) * self._dir * (0.25 + 0.75 * progress ** 0.8)
        x = signal + self._latent * 0.9
        return np.r_[x[:-1], progress]

    def run_to(self, progress: float):
        self.tokens = max(self.tokens, progress * self.total_tokens)
        return self.features(progress)

    def finish(self):
        self.tokens = self.total_tokens
        return {"x": self.features(1.0), "y": self.success, "tid": self.task.tid}


def make_tasks(n: int, seed: int, sd: float = 3.0):
    rng = np.random.default_rng(seed)
    return [ToyTask(i, float(d)) for i, d in enumerate(rng.normal(0, sd, n))]


def training_set(tasks, rollouts_per_task: int, seed: int, ability: float = 0.0):
    rng = np.random.default_rng(seed)
    X, y, prog, tid = [], [], [], []
    for t in tasks:
        for _ in range(rollouts_per_task):
            r = ToyRollout(t, rng, ability)
            for p in (rng.beta(1.2, 1.0), 1.0):
                X.append(r.features(p))
                y.append(r.success)
                prog.append(p)
                tid.append(t.tid)
    return np.array(X), np.array(y), np.array(prog), np.array(tid)
