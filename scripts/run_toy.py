#!/usr/bin/env python
"""End-to-end CCV on the CPU toy setting: train -> calibrate -> CRC -> search -> report."""
import argparse
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccv import (PlattCalibrator, SearchConfig, adaptive_search, best_of_k, crc_threshold,  # noqa: E402
                 expected_calibration_error, false_finish_rate)
from ccv.toy import ToyRollout, make_tasks, training_set  # noqa: E402
from ccv.verifier import LogisticVerifier  # noqa: E402


def run(cfg: dict) -> dict:
    s = cfg["seed"]
    train, cal, test = (make_tasks(cfg[k], s + i) for i, k in enumerate(["n_train_tasks", "n_cal_tasks", "n_test_tasks"]))
    X, y, prog, tid = training_set(train, cfg["rollouts_per_train_task"], s + 10, cfg["ability"])
    v = LogisticVerifier(X.shape[1], **cfg["verifier"]).fit(X, y, prog, tid)

    # calibration on disjoint tasks: final states and prefixes at the pruning checkpoint
    rng = np.random.default_rng(s + 20)
    pp = cfg["search"]["prune_progress"]
    fin, pre, lab = [], [], []
    for t in cal:
        for _ in range(cfg["rollouts_per_cal_task"]):
            r = ToyRollout(t, rng, cfg["ability"])
            pre.append(v.score(r.features(pp))); fin.append(v.score(r.features(1.0))); lab.append(r.success)
    fin, pre, lab = map(np.array, (fin, pre, lab))
    cal_f = PlattCalibrator().fit(fin, lab)
    cal_p = PlattCalibrator().fit(pre, lab)
    t_alpha = crc_threshold(cal_f.predict(fin), lab, cfg["crc"]["alpha"], cfg["crc"]["n_min"])

    sc = cfg["search"]
    scfg = SearchConfig(tau_stop=sc["tau_stop"], tau_prune=sc["tau_prune"], prune_progress=pp, k_max=sc["k_max"], submit_threshold=t_alpha)
    q_pre = lambda x: cal_p.predict(v.score(x))
    q_fin = lambda traj: cal_f.predict(v.score(traj["x"]))

    out = {"pass1": [], "bok": [], "bok_tok": [], "ad": [], "ad_tok": [], "ad_roll": [], "acc_flag": [], "acc_y": [], "q_test": [], "y_test": []}
    trng = np.random.default_rng(s + 30)
    for t in test:
        mk = lambda: ToyRollout(t, trng, cfg["ability"])
        r1 = mk(); out["pass1"].append(r1.finish()["y"])
        traj, _, tok = best_of_k(mk, q_fin, sc["best_of_k"])
        out["bok"].append(traj["y"]); out["bok_tok"].append(tok)
        res = adaptive_search(mk, q_pre, q_fin, scfg)
        y_best = res.best["y"] if res.best is not None else 0
        out["ad"].append(y_best); out["ad_tok"].append(res.tokens); out["ad_roll"].append(res.rollouts_started)
        out["acc_flag"].append(res.accepted); out["acc_y"].append(y_best)
        r2 = mk(); tr = r2.finish(); out["q_test"].append(q_fin(tr)); out["y_test"].append(tr["y"])
    q, yt = np.array(out["q_test"]), np.array(out["y_test"])
    ffr, accr = false_finish_rate(q, yt, t_alpha)
    return {
        "pass@1": float(np.mean(out["pass1"])),
        f"best-of-{sc['best_of_k']}": float(np.mean(out["bok"])), "bok_tokens": float(np.mean(out["bok_tok"])),
        "adaptive": float(np.mean(out["ad"])), "adaptive_tokens": float(np.mean(out["ad_tok"])),
        "adaptive_rollouts": float(np.mean(out["ad_roll"])),
        "ece_raw_on_cal": expected_calibration_error(1 / (1 + np.exp(-fin)), lab),
        "ece_calibrated_on_cal": expected_calibration_error(cal_f.predict(fin), lab),
        "submit_threshold": t_alpha, "false_finish_rate": ffr, "acceptance": accr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "toy.toml"))
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    t0 = time.time()
    res = run(cfg)
    w = max(len(k) for k in res)
    for k, val in res.items():
        print(f"{k:<{w}}  {val:.4f}")
    print(f"{'elapsed_s':<{w}}  {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
