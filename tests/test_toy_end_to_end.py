import time
from pathlib import Path

import yaml

from ccv.demo import alpha_sweep, run

ROOT = Path(__file__).resolve().parents[1]


def test_toy_pipeline_runs_fast_and_beats_pass1():
    cfg = yaml.safe_load((ROOT / "configs" / "toy.yaml").read_text())
    t0 = time.time()
    res = run(cfg)
    assert time.time() - t0 < 60
    assert res["adaptive"] > res["pass@1"] + 0.1
    assert res["adaptive_tokens"] < res["bok_tokens"]
    assert res["ece_platt"] <= res["ece_raw"] + 1e-9
    assert res["false_finish_rate"] < 0.2


def test_false_finish_rate_tracks_alpha_across_seeds():
    cfg = yaml.safe_load((ROOT / "configs" / "toy.yaml").read_text())
    cfg["sweep"] = {"alphas": [0.05, 0.2], "seeds": [0, 1, 2]}
    out = alpha_sweep(cfg)
    rates = {a: v["fails"] / v["accepted"] for a, v in out.items()}
    assert rates[0.05] < rates[0.2]
    assert all(abs(r - a) < 0.05 for a, r in rates.items())
    assert out[0.05]["accepted"] < out[0.2]["accepted"]
