"""End-to-end CCV on the CPU toy setting: train the verifier, calibrate it on disjoint tasks, set the CRC submit
threshold, search, and report.

    python -m ccv.demo [--config configs/toy.yaml]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from .calibration import PlattCalibrator, expected_calibration_error
from .controller import SearchConfig, adaptive_search, best_of_k
from .crc import crc_threshold, false_finish_rate
from .toy import ToyRollout, make_tasks, training_set
from .verifier import LogisticVerifier


def calibrated_verifier(cfg: dict, seed: int):
    """Train the verifier, then Platt-calibrate it on disjoint tasks at the final state and at the pruning point."""
    train, cal = make_tasks(cfg["n_train_tasks"], seed), make_tasks(cfg["n_cal_tasks"], seed + 1)
    X, y, prog, tid = training_set(train, cfg["rollouts_per_train_task"], seed + 10, cfg["ability"])
    v = LogisticVerifier(X.shape[1], **cfg["verifier"]).fit(X, y, prog, tid)
    rng = np.random.default_rng(seed + 20)
    pp = cfg["search"]["prune_progress"]
    fin, pre, lab = [], [], []
    for t in cal:
        for _ in range(cfg["rollouts_per_cal_task"]):
            r = ToyRollout(t, rng, cfg["ability"])
            pre.append(v.score(r.features(pp)))
            fin.append(v.score(r.features(1.0)))
            lab.append(r.success)
    fin, pre, lab = map(np.array, (fin, pre, lab))
    return v, PlattCalibrator().fit(fin, lab), PlattCalibrator().fit(pre, lab), fin, lab


def run(cfg: dict) -> dict:
    """pass@1, best-of-k and adaptive search on the test tasks of seed cfg["seed"]."""
    s, sc = cfg["seed"], cfg["search"]
    v, cal_f, cal_p, fin, lab = calibrated_verifier(cfg, s)
    test = make_tasks(cfg["n_test_tasks"], s + 2)
    t_alpha = crc_threshold(cal_f.predict(fin), lab, cfg["crc"]["alpha"], cfg["crc"]["n_min"])
    scfg = SearchConfig(tau_stop=sc["tau_stop"], tau_prune=sc["tau_prune"], prune_progress=sc["prune_progress"],
                        k_max=sc["k_max"], submit_threshold=t_alpha)

    def q_pre(x):
        return cal_p.predict(v.score(x))

    def q_fin(traj):
        return cal_f.predict(v.score(traj["x"]))

    rows = {k: [] for k in ("pass1", "pass1_tok", "bok", "bok_tok", "ad", "ad_tok", "ad_roll", "q_test", "y_test")}
    trng = np.random.default_rng(s + 30)
    for t in test:
        def new_rollout(t=t):
            return ToyRollout(t, trng, cfg["ability"])

        r1 = new_rollout()
        rows["pass1"].append(r1.finish()["y"])
        rows["pass1_tok"].append(r1.tokens)
        traj, _, tok = best_of_k(new_rollout, q_fin, sc["best_of_k"])
        rows["bok"].append(traj["y"])
        rows["bok_tok"].append(tok)
        res = adaptive_search(new_rollout, q_pre, q_fin, scfg)
        rows["ad"].append(res.best["y"] if res.best is not None else 0)
        rows["ad_tok"].append(res.tokens)
        rows["ad_roll"].append(res.rollouts_started)
        tr = new_rollout().finish()
        rows["q_test"].append(q_fin(tr))
        rows["y_test"].append(tr["y"])
    ffr, accepted = false_finish_rate(np.array(rows["q_test"]), np.array(rows["y_test"]), t_alpha)
    mean = {k: float(np.mean(x)) for k, x in rows.items()}
    return {
        "selectors": {
            "pass@1": (mean["pass1"], mean["pass1_tok"], 1.0),
            f"best-of-{sc['best_of_k']}": (mean["bok"], mean["bok_tok"], float(sc["best_of_k"])),
            "CCV adaptive search": (mean["ad"], mean["ad_tok"], mean["ad_roll"]),
        },
        "ece_raw": expected_calibration_error(1 / (1 + np.exp(-fin)), lab),
        "ece_platt": expected_calibration_error(cal_f.predict(fin), lab),
        "threshold": t_alpha, "false_finish_rate": ffr, "accepted": accepted,
        "pass@1": mean["pass1"], "adaptive": mean["ad"], "adaptive_tokens": mean["ad_tok"], "bok_tokens": mean["bok_tok"],
    }


def alpha_sweep(cfg: dict) -> dict:
    """Realised false-finish rate on fresh test completions, pooled over seeds, for each target alpha."""
    out = {a: {"threshold": [], "accepted": 0, "fails": 0, "n": 0} for a in cfg["sweep"]["alphas"]}
    for s in cfg["sweep"]["seeds"]:
        v, cal_f, _, fin, lab = calibrated_verifier(cfg, s)
        rng = np.random.default_rng(s + 40)
        finals = [ToyRollout(t, rng, cfg["ability"]).finish() for t in make_tasks(cfg["n_test_tasks"], s + 2)]
        q = cal_f.predict(v.score(np.array([f["x"] for f in finals])))
        y = np.array([f["y"] for f in finals])
        for a, acc in out.items():
            t = crc_threshold(cal_f.predict(fin), lab, a, cfg["crc"]["n_min"])
            ok = q >= t
            acc["threshold"].append(t)
            acc["accepted"] += int(ok.sum())
            acc["fails"] += int((ok & (y == 0)).sum())
            acc["n"] += len(y)
    return out


def config_path(name: str) -> Path:
    """``name`` from the working directory, else from the checkout this package is installed from."""
    path = Path(name)
    if not path.exists() and (Path(__file__).resolve().parents[2] / name).exists():
        return Path(__file__).resolve().parents[2] / name
    return path


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="CCV on the CPU toy setting.")
    ap.add_argument("--config", default="configs/toy.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(config_path(args.config).read_text())
    t0 = time.time()
    res = run(cfg)
    print(f"seed {cfg['seed']}: {cfg['n_train_tasks']} training, {cfg['n_cal_tasks']} calibration and "
          f"{cfg['n_test_tasks']} test tasks")
    print(f"{'selector':<22}{'solved':>8}{'tokens':>9}{'rollouts':>10}")
    for name, (acc, tok, roll) in res["selectors"].items():
        print(f"{name:<22}{acc:>8.3f}{tok:>9.2f}{roll:>10.2f}")
    print(f"ECE on calibration tasks: {res['ece_raw']:.3f} raw, {res['ece_platt']:.3f} after Platt scaling")
    print(f"submit threshold at alpha = {cfg['crc']['alpha']:.2f}: {res['threshold']:.3f}; it accepts "
          f"{100 * res['accepted']:.1f}% of test completions, {100 * res['false_finish_rate']:.1f}% of them false finishes")
    sweep = alpha_sweep(cfg)
    seeds = cfg["sweep"]["seeds"]
    print(f"\nfalse-finish control, seeds {seeds[0]}-{seeds[-1]} pooled ({sweep[cfg['sweep']['alphas'][0]]['n']} test completions)")
    print(f"{'alpha':>6}{'accepted':>10}{'false finishes':>16}")
    for a, acc in sweep.items():
        print(f"{a:>6.2f}{100 * acc['accepted'] / acc['n']:>9.1f}%{100 * acc['fails'] / max(acc['accepted'], 1):>15.1f}%")
    print(f"done in {time.time() - t0:.1f}s")
    res["sweep"] = sweep
    return res


if __name__ == "__main__":
    main()
