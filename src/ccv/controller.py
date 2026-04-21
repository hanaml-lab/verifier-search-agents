"""Budgeted sequential search controlled by a calibrated completion verifier (Algorithm 1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class Rollout(Protocol):
    """A rollout that can be advanced to a prefix checkpoint and then to completion."""

    def run_to(self, progress: float) -> Any: ...  # returns a prefix observation
    def finish(self) -> Any: ...                   # returns the completed trajectory
    @property
    def tokens(self) -> float: ...                 # tokens consumed so far


@dataclass
class SearchConfig:
    tau_stop: float = 0.88
    tau_prune: float = 0.25
    prune_progress: float = 0.30
    k_max: int = 16
    submit_threshold: float | None = None  # from crc_threshold; None disables the flag
    prune: bool = True


@dataclass
class SearchResult:
    best: Any
    best_q: float
    accepted: bool
    rollouts_started: int
    rollouts_pruned: int
    tokens: float
    history: list[dict] = field(default_factory=list)


def adaptive_search(new_rollout: Callable[[], Rollout],
                    q_prefix: Callable[[Any], float],
                    q_final: Callable[[Any], float],
                    cfg: SearchConfig) -> SearchResult:
    best, best_q, pruned, tokens, hist = None, -1.0, 0, 0.0, []
    n = 0
    for n in range(1, cfg.k_max + 1):
        r = new_rollout()
        if cfg.prune:
            prefix = r.run_to(cfg.prune_progress)
            qp = float(q_prefix(prefix))
            if qp < cfg.tau_prune:
                pruned += 1
                tokens += r.tokens
                hist.append({"n": n, "q_prefix": qp, "pruned": True})
                continue
        traj = r.finish()
        tokens += r.tokens
        q = float(q_final(traj))
        hist.append({"n": n, "q_final": q, "pruned": False})
        if q > best_q:
            best, best_q = traj, q
        if best_q >= cfg.tau_stop:
            break
    accepted = cfg.submit_threshold is not None and best_q >= cfg.submit_threshold
    return SearchResult(best, best_q, accepted, n, pruned, tokens, hist)


def best_of_k(new_rollout: Callable[[], Rollout], score: Callable[[Any], float], k: int):
    """Fixed-budget baseline: run k rollouts to completion and return the argmax."""
    trajs, tokens = [], 0.0
    for _ in range(k):
        r = new_rollout()
        t = r.finish()
        tokens += r.tokens
        trajs.append(t)
    scores = [score(t) for t in trajs]
    i = max(range(k), key=lambda j: scores[j])
    return trajs[i], scores[i], tokens
