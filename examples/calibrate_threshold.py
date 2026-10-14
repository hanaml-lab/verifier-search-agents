"""Calibrate your own verifier's scores and set CCV's submit threshold.

    python examples/calibrate_threshold.py examples/data/verifier_scores.csv --alpha 0.10

The CSV has one row per finished completion: ``task_id``, ``score`` (the verifier's raw score, a logit) and
``passed`` (1 or 0 when the hidden tests have been run, empty for a new completion). Rows with an outcome must come
from tasks that the verifier was not trained on. They fit the Platt calibration and the conformal risk control
threshold that bounds, at level alpha, the share of accepted completions that fail the hidden tests. Rows without an
outcome are then scored and either submitted or held back.

The bundled sample holds raw scores of the toy setting's verifier (``ccv.toy``, seed 0): 150 held-out tasks with
four completions each, and eight new completions.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from ccv import PlattCalibrator, crc_threshold, expected_calibration_error, false_finish_rate


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--alpha", type=float, default=0.10, help="bound on the share of accepted completions that fail")
    ap.add_argument("--n-min", type=int, default=20, help="fewest accepted calibration completions behind a threshold")
    args = ap.parse_args(argv)

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    labelled = [r for r in rows if r["passed"].strip() != ""]
    new = [r for r in rows if r["passed"].strip() == ""]
    s = np.array([float(r["score"]) for r in labelled])
    y = np.array([int(r["passed"]) for r in labelled])
    cal = PlattCalibrator().fit(s, y)
    q = cal.predict(s)
    t = crc_threshold(q, y, args.alpha, args.n_min)
    ffr, accepted = false_finish_rate(q, y, t)
    tasks = len({r["task_id"] for r in labelled})
    print(f"calibration: {len(labelled)} completions on {tasks} tasks, {100 * y.mean():.1f}% pass")
    print(f"ECE {expected_calibration_error(1 / (1 + np.exp(-s)), y):.3f} raw, {expected_calibration_error(q, y):.3f} after Platt scaling")
    print(f"submit threshold at alpha = {args.alpha:.2f}: q >= {t:.3f} (on the calibration set it accepts "
          f"{100 * accepted:.1f}%, {100 * ffr:.1f}% of them failing)")
    decisions = []
    if new:
        print(f"\n{'task':<10}{'score':>8}{'q':>8}  decision")
        for r in new:
            qi = float(cal.predict(np.array([float(r["score"])]))[0])
            decision = "submit" if qi >= t else "hold back"
            decisions.append({"task_id": r["task_id"], "q": qi, "decision": decision})
            print(f"{r['task_id']:<10}{float(r['score']):>8.2f}{qi:>8.3f}  {decision}")
    return {"threshold": t, "decisions": decisions}


if __name__ == "__main__":
    main()
