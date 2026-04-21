import numpy as np

from ccv import SearchConfig, adaptive_search, best_of_k


class FakeRollout:
    def __init__(self, q, y, tokens=1.0):
        self.q, self.y, self.total, self.tokens = q, y, tokens, 0.0

    def run_to(self, p):
        self.tokens = p * self.total
        return self.q

    def finish(self):
        self.tokens = self.total
        return {"q": self.q, "y": self.y}


def seq(items):
    it = iter(items)
    return lambda: FakeRollout(*next(it))


def test_stops_at_first_confident_candidate():
    res = adaptive_search(seq([(0.5, 0), (0.95, 1), (0.99, 1)]), lambda q: q, lambda t: t["q"], SearchConfig(tau_stop=0.9, prune=False))
    assert res.rollouts_started == 2 and res.best["y"] == 1 and res.tokens == 2.0


def test_prunes_low_prefix_and_charges_partial_tokens():
    res = adaptive_search(seq([(0.1, 0), (0.95, 1)]), lambda q: q, lambda t: t["q"], SearchConfig(tau_stop=0.9, tau_prune=0.25))
    assert res.rollouts_pruned == 1 and abs(res.tokens - 1.3) < 1e-9


def test_respects_cap_and_flags_acceptance():
    items = [(0.6, 0)] * 20
    res = adaptive_search(seq(items), lambda q: q, lambda t: t["q"], SearchConfig(tau_stop=0.9, k_max=5, submit_threshold=0.7))
    assert res.rollouts_started == 5 and not res.accepted


def test_best_of_k_argmax():
    traj, s, tok = best_of_k(seq([(0.2, 0), (0.7, 1), (0.5, 0)]), lambda t: t["q"], 3)
    assert traj["y"] == 1 and s == 0.7 and tok == 3.0
