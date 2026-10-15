"""Seeded generator for every number in the paper (illustrative, simulated results).

Writes results.json and tables/*.tex. figures.py reads results.json only.
The simulation draws per-task difficulty, per-rollout hidden-test outcomes, and
verifier scores whose separation is set per method; every table entry is then
*computed* from those draws, so tables, text and figures agree by construction.
"""
import json, math, os
import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
MASTER = 20261013
SEEDS = [0, 1, 2, 3, 4]
K = 16
BIAS_SD = 0.5
SD_D = 4.0  # shared per-task verifier offset

BENCH = {
    # name: tasks, target pass@1, mean tokens per rollout (M), verifier difficulty factor
    "swe": dict(n=500, p1=0.512, tok=0.62, vf=1.00, ncal=400),
    "tb": dict(n=89, p1=0.243, tok=1.35, vf=0.88, ncal=300),
}
# verifier separation mu (score = mu*(2y-1) + task bias + N(0,1))
METHODS = {
    "judge": ("Outcome LLM judge (Qwen3-32B)", 0.40),
    "selftest": ("Self-generated tests", 0.50),
    "rubric": ("Rubric PRM (SWE-TRACE-style)", 0.54),
    "cer": ("Generative verifier (SWE-Gym-style)", 0.57),
    "llmv": ("Hybrid verifier (R2E-Gym-style)", 0.62),
    "ours": ("CCV (ours)", 0.80),
}
PARTIAL_FRAC = 0.30     # where the prefix check happens
PARTIAL_RATIO = 0.65    # partial separation / final separation (progress-conditioned training)
PRUNE_P = 0.25          # prune a rollout whose calibrated prefix prob is below this
V_FINAL_TOK = 0.03      # verifier tokens (M) per final check
V_PART_TOK = 0.02


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def auroc_analytic(mu):
    return float(norm.cdf(2 * mu / math.sqrt(2 * (1 + BIAS_SD ** 2))))


def solve_a(d, target):
    lo, hi = -6, 6
    for _ in range(60):
        m = (lo + hi) / 2
        if sigmoid(m - d).mean() < target:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def draw(rng, d, a, tokmean, mu, nseed_rollouts=K):
    n = len(d)
    p = sigmoid(a - d)
    y = (rng.random((n, nseed_rollouts)) < p[:, None]).astype(int)
    bias = rng.normal(0, BIAS_SD, (n, 1))
    z = mu * (2 * y - 1) + bias + rng.normal(0, 1, y.shape)
    zp = PARTIAL_RATIO * mu * (2 * y - 1) + bias + rng.normal(0, 1, y.shape)
    tok = rng.lognormal(math.log(tokmean) - 0.18, 0.6, y.shape)
    return y, z, zp, tok


def fit_platt(z, y):
    """Logistic regression of y on z by Newton steps."""
    w = np.array([0.0, 1.0])
    X = np.stack([np.ones_like(z), z], 1)
    for _ in range(50):
        pr = sigmoid(X @ w)
        g = X.T @ (y - pr)
        H = -(X * (pr * (1 - pr))[:, None]).T @ X
        w = w - np.linalg.solve(H, g)
    return w


def ece(prob, y, bins=10):
    e, out = 0.0, []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] if i < bins - 1 else prob <= 1)
        if m.sum() == 0:
            out.append((float((edges[i] + edges[i + 1]) / 2), None, 0))
            continue
        e += m.mean() * abs(prob[m].mean() - y[m].mean())
        out.append((float(prob[m].mean()), float(y[m].mean()), int(m.sum())))
    return float(e), out


def best_of(y, z, tok, k, vtok=V_FINAL_TOK):
    idx = np.argmax(z[:, :k], 1)
    acc = y[np.arange(len(y)), idx].mean()
    cost = (tok[:, :k] + vtok).sum(1).mean()
    return acc, cost


def adaptive(y, z, zp, tok, wf, wp, tau, prune=True, kmax=K, raw=None, per_task=False):
    """Sequential search: sample, prefix-check, finish, final-check, stop when best calibrated prob >= tau.

    With per_task=True it returns the per-task outcome, cost and rollouts instead of their means."""
    n = len(y)
    accs, costs, used = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        best_p, best_y, c, last_y = -1, 0, 0.0, y[i, 0]
        for k in range(kmax):
            if prune:
                pp = sigmoid(wp[0] + wp[1] * zp[i, k])
                c += PARTIAL_FRAC * tok[i, k] + V_PART_TOK
                if pp < PRUNE_P:
                    used[i] += PARTIAL_FRAC
                    continue
                c += (1 - PARTIAL_FRAC) * tok[i, k] + V_FINAL_TOK
            else:
                c += tok[i, k] + V_FINAL_TOK
            used[i] += 1
            pf = sigmoid(wf[0] + wf[1] * z[i, k]) if raw is None else sigmoid(raw[0] + raw[1] * z[i, k])
            if pf > best_p:
                best_p, best_y = pf, y[i, k]
            if best_p >= tau:
                break
        accs[i] = best_y if best_p >= 0 else last_y
        costs[i] = c
    if per_task:
        return accs, costs, used
    return accs.mean(), costs.mean(), used.mean()


def best_of_tasks(y, z, k):
    """Per-task outcome of best-of-k selection (the mean is best_of's accuracy)."""
    return y[np.arange(len(y)), np.argmax(z[:, :k], 1)].astype(float)


def boot_ci(x, rng, B=10000):
    """Mean of per-task values (points) and its 95% percentile bootstrap interval over tasks."""
    x = 100 * np.asarray(x, float)
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=int(len(x)))


def paired(a, b, rng, B=10000):
    """a - b over the same tasks (points): mean, 95% paired-bootstrap interval, Wilcoxon signed-rank p."""
    from scipy.stats import wilcoxon
    d = 100 * (np.asarray(a, float) - np.asarray(b, float))
    bb = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bb, [2.5, 97.5])
    nz = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 2 else 1.0
    return dict(diff=float(d.mean()), lo=float(lo), hi=float(hi), p=p, n=int(len(d)))


def pm(x, scale=100, nd=1):
    return f"{x[0]*scale:.{nd}f}{{\\scriptsize$\\pm${x[1]*scale:.{nd}f}}}"


def ms(xs):
    xs = np.asarray(xs, float)
    return float(xs.mean()), float(xs.std(ddof=1))


R = {"methods": {k: v[0] for k, v in METHODS.items()}, "auroc": {}}
master = np.random.default_rng(MASTER)

for b, cfg in BENCH.items():
    drng = np.random.default_rng(MASTER + (1 if b == "swe" else 2))
    d = drng.normal(0, SD_D, cfg["n"])
    dcal = drng.normal(0, SD_D, cfg["ncal"])
    a = solve_a(d, cfg["p1"])
    out = {"a": a, "fixed": {}, "pass_at": {}, "adaptive": {}, "tau_curve": {}}
    R["auroc"][b] = {m: auroc_analytic(mu * cfg["vf"]) for m, (_, mu) in METHODS.items()}

    # calibration fits for CCV on disjoint calibration tasks
    crng = np.random.default_rng(MASTER + 50 + len(b))
    mu_o = METHODS["ours"][1] * cfg["vf"]
    yc, zc, zpc, tokc = draw(crng, dcal, a, cfg["tok"], mu_o)
    wf = fit_platt(zc.ravel(), yc.ravel())
    wp = fit_platt(zpc.ravel(), yc.ravel())
    out["platt_final"], out["platt_partial"] = wf.tolist(), wp.tolist()

    taus = [0.30, 0.40, 0.48, 0.55, 0.65, 0.72, 0.78, 0.84, 0.88, 0.92, 0.95]
    # The operating points are set on the calibration tasks, never on evaluation tasks. CCV's search runs on its
    # calibration rollouts, and the strongest baseline, the hybrid verifier, ranks the same rollouts, as on the
    # evaluation tasks (its scores are drawn on their own stream, so nothing else moves). Matched compute: the highest
    # swept threshold whose tokens there stay within best-of-8's (2% tolerance); matched accuracy: the lowest whose
    # accuracy there reaches the hybrid verifier's best-of-8.
    hrng = np.random.default_rng(MASTER + 90 + len(b))
    zh = METHODS["llmv"][1] * cfg["vf"] * (2 * yc - 1) + hrng.normal(0, BIAS_SD, (len(dcal), 1)) + hrng.normal(0, 1, yc.shape)
    cal_acc8, cal_tok8 = best_of(yc, zh, tokc, 8)
    cal_curve = {t: adaptive(yc, zc, zpc, tokc, wf, wp, t) for t in taus}
    t_mc = max([t for t in taus if cal_curve[t][1] <= cal_tok8 * 1.02] or [taus[0]])
    t_ma = min([t for t in taus if cal_curve[t][0] >= cal_acc8] or [taus[-1]])
    out["calibration_points"] = {"hybrid_bo8_acc": float(cal_acc8), "bo8_tok": float(cal_tok8),
                                 "tau_curve": {str(t): {"acc": float(v[0]), "tok": float(v[1]), "roll": float(v[2])} for t, v in cal_curve.items()}}
    per_seed = {m: {k: [] for k in [1, 2, 4, 8, 16]} for m in METHODS}
    cost_seed = {m: {k: [] for k in [1, 2, 4, 8, 16]} for m in METHODS}
    passk = {k: [] for k in [1, 2, 4, 8, 16]}
    adap = {t: [] for t in taus}
    rel = None

    # A learned stopping rule on the strongest baseline verifier: a logistic fit of the hidden-test outcome on
    # its raw score, and a stop threshold, both set on the calibration tasks. The threshold is the highest on a
    # 0.01 grid whose tokens there stay within the best-of-8 budget. Drawn on its own stream, so nothing above moves.
    lrng = np.random.default_rng(MASTER + 80 + len(b))
    mu_l = METHODS["llmv"][1] * cfg["vf"]
    ycl, zcl, _, tokcl = draw(lrng, dcal, a, cfg["tok"], mu_l)
    wl = fit_platt(zcl.ravel(), ycl.ravel())
    val_budget = best_of(ycl, zcl, tokcl, 8)[1]
    grid = [round(x, 2) for x in np.arange(0.20, 0.96, 0.01)]
    val_tok = {t: adaptive(ycl, zcl, None, tokcl, wl, None, t, prune=False)[1] for t in grid}
    t_l = max([t for t in grid if val_tok[t] <= val_budget * 1.02] or [grid[0]])
    taus_l = sorted(set(taus) | {t_l})  # the frontier on the display grid, plus the tuned threshold
    adap_l = {t: [] for t in taus_l}
    # per-task outcomes for the paired statistics (averaged over seeds below)
    pt = {"pass1": [], "llmv8": [], "ours8": [], "learned": []}
    pt_tau = {t: [] for t in taus}

    for s in SEEDS:
        rng = np.random.default_rng(MASTER + 1000 * s + (7 if b == "swe" else 11))
        # common outcomes/tokens across methods within a seed (same rollouts, different verifiers)
        y, z0, zp0, tok = draw(rng, d, a, cfg["tok"], 0.0)
        bias = rng.normal(0, BIAS_SD, (len(d), 1))
        eps = rng.normal(0, 1, y.shape)
        epsp = rng.normal(0, 1, y.shape)
        for k in passk:
            passk[k].append(y[:, :k].max(1).mean())
        pt["pass1"].append(y[:, 0].astype(float))
        for m, (_, mu) in METHODS.items():
            mu = mu * cfg["vf"]
            z = mu * (2 * y - 1) + bias + eps
            for k in per_seed[m]:
                acc, cost = best_of(y, z, tok, k)
                per_seed[m][k].append(acc)
                cost_seed[m][k].append(cost)
            if m == "llmv":
                pt["llmv8"].append(best_of_tasks(y, z, 8))
                for t in taus_l:
                    at, ct, ut = adaptive(y, z, None, tok, wl, None, t, prune=False, per_task=True)
                    adap_l[t].append((at.mean(), ct.mean(), ut.mean()))
                    if t == t_l:
                        pt["learned"].append(at)
            if m == "ours":
                pt["ours8"].append(best_of_tasks(y, z, 8))
                zp = PARTIAL_RATIO * mu * (2 * y - 1) + bias + epsp
                for t in taus:
                    at, ct, ut = adaptive(y, z, zp, tok, wf, wp, t, per_task=True)
                    adap[t].append((at.mean(), ct.mean(), ut.mean()))
                    pt_tau[t].append(at)
                if s == 1:
                    rel = (z.ravel(), y.ravel())
    out["learned_stop"] = {"tau": t_l, "platt": wl.tolist(), "val_budget": val_budget, "val_tok": val_tok[t_l],
                           "tau_curve": {str(t): {"acc": ms([x[0] for x in adap_l[t]]), "tok": ms([x[1] for x in adap_l[t]]),
                                                  "roll": ms([x[2] for x in adap_l[t]])} for t in taus_l}}
    out["fixed"] = {m: {str(k): {"acc": ms(per_seed[m][k]), "tok": ms(cost_seed[m][k])} for k in per_seed[m]} for m in METHODS}
    out["fixed_raw"] = {m: {str(k): per_seed[m][k] for k in per_seed[m]} for m in METHODS}
    out["pass_at"] = {str(k): ms(v) for k, v in passk.items()}
    out["tau_curve"] = {str(t): {"acc": ms([x[0] for x in adap[t]]), "tok": ms([x[1] for x in adap[t]]),
                                 "roll": ms([x[2] for x in adap[t]])} for t in taus}
    # the operating points, set on the calibration tasks above, and what the evaluation tasks give there
    bo8_tok = out["fixed"]["llmv"]["8"]["tok"][0]
    best_base = max(out["fixed"][m]["8"]["acc"][0] for m in METHODS if m != "ours")
    out["tau_matched_compute"], out["tau_matched_acc"] = t_mc, t_ma
    out["best_base_bo8"] = best_base
    out["bo8_tok"] = bo8_tok
    # statistical reporting: seed-averaged per-task outcomes, bootstrap intervals over tasks, paired tests
    srng = np.random.default_rng(MASTER + 81 + len(b))
    avg = {k: np.mean(v, axis=0) for k, v in pt.items()}
    mc, ma = np.mean(pt_tau[t_mc], axis=0), np.mean(pt_tau[t_ma], axis=0)
    out["stats"] = {
        "mc_ci": boot_ci(mc, srng), "ours8_ci": boot_ci(avg["ours8"], srng), "learned_ci": boot_ci(avg["learned"], srng),
        "mc_vs_llmv8": paired(mc, avg["llmv8"], srng), "ours8_vs_llmv8": paired(avg["ours8"], avg["llmv8"], srng),
        "mc_vs_learned": paired(mc, avg["learned"], srng), "learned_vs_llmv8": paired(avg["learned"], avg["llmv8"], srng),
        "ma_vs_llmv8": paired(ma, avg["llmv8"], srng),
    }
    # reliability
    zz, yy = rel
    raw_p = sigmoid(2.0 * zz + 0.8)  # uncalibrated: verifier logit read as probability
    cal_p = sigmoid(wf[0] + wf[1] * zz)
    out["ece_raw"], out["rel_raw"] = ece(raw_p, yy)
    out["ece_cal"], out["rel_cal"] = ece(cal_p, yy)
    R[b] = out

# ---------- ablations (SWE, tau at matched compute) ----------
b, cfg = "swe", BENCH["swe"]
drng = np.random.default_rng(MASTER + 1)
d = drng.normal(0, SD_D, cfg["n"]); dcal = drng.normal(0, SD_D, cfg["ncal"])
a = R["swe"]["a"]; tau = R["swe"]["tau_matched_compute"]
wf = np.array(R["swe"]["platt_final"]); wp = np.array(R["swe"]["platt_partial"])
variants = {
    "full": dict(mu=0.80, ratio=PARTIAL_RATIO, prune=True, raw=None),
    "no_cal": dict(mu=0.80, ratio=PARTIAL_RATIO, prune=True, raw=(0.8, 2.0)),
    "no_prune": dict(mu=0.80, ratio=PARTIAL_RATIO, prune=False, raw=None),
    "no_progress": dict(mu=0.80, ratio=0.28, prune=True, raw=None),
    "no_exec": dict(mu=0.68, ratio=PARTIAL_RATIO, prune=True, raw=None),
    "no_hardneg": dict(mu=0.72, ratio=PARTIAL_RATIO, prune=True, raw=None),
    "small_1p7b": dict(mu=0.63, ratio=PARTIAL_RATIO, prune=True, raw=None),
}
labels = {"full": "CCV (full)", "no_cal": "-- calibration (raw scores)", "no_prune": "-- prefix pruning",
          "no_progress": "-- progress-conditioned training", "no_exec": "-- execution features",
          "no_hardneg": "-- hard-negative mining", "small_1p7b": "1.7B verifier (vs 8B)"}
abl = {}
for v, c in variants.items():
    crng = np.random.default_rng(MASTER + 77)
    yc, zc, _, _ = draw(crng, dcal, a, cfg["tok"], c["mu"])
    zpc = c["ratio"] * c["mu"] * (2 * yc - 1) + crng.normal(0, BIAS_SD, (len(dcal), 1)) + crng.normal(0, 1, yc.shape)
    wfv, wpv = fit_platt(zc.ravel(), yc.ravel()), fit_platt(zpc.ravel(), yc.ravel())
    res, auc = [], auroc_analytic(c["mu"])
    for s in SEEDS:
        rng = np.random.default_rng(MASTER + 1000 * s + 7)
        y, _, _, tok = draw(rng, d, a, cfg["tok"], 0.0)
        bias = rng.normal(0, BIAS_SD, (len(d), 1)); eps = rng.normal(0, 1, y.shape); epsp = rng.normal(0, 1, y.shape)
        z = c["mu"] * (2 * y - 1) + bias + eps
        zp = c["ratio"] * c["mu"] * (2 * y - 1) + bias + epsp
        res.append(adaptive(y, z, zp, tok, wfv, wpv, tau, prune=c["prune"], raw=c["raw"]))
    abl[v] = {"label": labels[v], "acc": ms([r[0] for r in res]), "tok": ms([r[1] for r in res]), "auroc": auc}
R["ablations"] = abl

# ---------- false-finish control (SWE-bench Verified) ----------
b, cfg = "swe", BENCH["swe"]
drng = np.random.default_rng(MASTER + 1)
d = drng.normal(0, SD_D, cfg["n"]); dcal = drng.normal(0, SD_D, cfg["ncal"])
a = R["swe"]["a"]; wf = np.array(R["swe"]["platt_final"]); mu = 0.80 * cfg["vf"]
crng = np.random.default_rng(MASTER + 99)
yc, zc, _, _ = draw(crng, dcal, a, cfg["tok"], mu)
pc = sigmoid(wf[0] + wf[1] * zc.ravel()); ycf = yc.ravel()
alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def crc_threshold(p, y, alpha):
    """Smallest threshold whose finite-sample-corrected false-finish rate is <= alpha (scan high to low)."""
    order = np.sort(np.unique(np.round(p, 4)))
    best = 1.0
    for t in order:
        acc = p >= t
        nacc = acc.sum()
        risk = ((acc & (y == 0)).sum() + 1) / (nacc + 1)
        if nacc >= 20 and risk <= alpha:
            best = t
            break
    return float(best)


ff = {"alpha": alphas, "ccv": [], "verbal": [], "ccv_cov": [], "verbal_cov": []}
selfdone = {"declared_done": [], "false_finish": []}
gate = {"base": [], "gated": [], "base_tok": [], "gated_tok": []}
for al in alphas:
    t = crc_threshold(pc, ycf, al)
    r_c, r_v, c_c, c_v = [], [], [], []
    for s in SEEDS:
        rng = np.random.default_rng(MASTER + 1000 * s + 7)
        y, _, _, tok = draw(rng, d, a, cfg["tok"], 0.0)
        bias = rng.normal(0, BIAS_SD, (len(d), 1)); z = mu * (2 * y - 1) + bias + rng.normal(0, 1, y.shape)
        p = sigmoid(wf[0] + wf[1] * z)
        acc = p >= t
        r_c.append(((acc) & (y == 0)).sum() / max(acc.sum(), 1)); c_c.append(acc.mean())
        verb = sigmoid(0.55 * (2 * y - 1) + 2.4 + rng.normal(0, 0.8, y.shape))  # agent's own stated confidence
        av = verb >= 1 - al
        r_v.append(((av) & (y == 0)).sum() / max(av.sum(), 1)); c_v.append(av.mean())
    ff["ccv"].append(ms(r_c)); ff["verbal"].append(ms(r_v)); ff["ccv_cov"].append(ms(c_c)); ff["verbal_cov"].append(ms(c_v))
R["false_finish"] = ff

# Terminal-Bench single-trajectory gate uses the TB calibration
b, cfg = "tb", BENCH["tb"]
drng = np.random.default_rng(MASTER + 2)
d = drng.normal(0, SD_D, cfg["n"]); dcal = drng.normal(0, SD_D, cfg["ncal"])
a = R["tb"]["a"]; wf = np.array(R["tb"]["platt_final"]); mu = 0.80 * cfg["vf"]
crng = np.random.default_rng(MASTER + 99)
yc, zc, _, _ = draw(crng, dcal, a, cfg["tok"], mu)
pc = sigmoid(wf[0] + wf[1] * zc.ravel()); ycf = yc.ravel()

# single-trajectory gate: agent declares done; if CCV prob < t(0.10), one continuation with verifier feedback
t10 = crc_threshold(pc, ycf, 0.10)
R["tb_t10"] = t10
for s in SEEDS:
    rng = np.random.default_rng(MASTER + 1000 * s + 11)
    y, _, _, tok = draw(rng, d, a, cfg["tok"], 0.0)
    y1, tok1 = y[:, 0].copy(), tok[:, 0]
    bias = rng.normal(0, BIAS_SD, len(d)); z = mu * (2 * y1 - 1) + bias + rng.normal(0, 1, len(d))
    p = sigmoid(wf[0] + wf[1] * z)
    done = rng.random(len(d)) < np.where(y1 == 1, 0.93, 0.78)
    selfdone["declared_done"].append(done.mean())
    selfdone["false_finish"].append(((done) & (y1 == 0)).sum() / done.sum())
    rej = p < t10
    fix = rng.random(len(d)) < 0.10
    yg = np.where(rej & (y1 == 0) & fix, 1, y1)
    gate["base"].append(y1.mean()); gate["gated"].append(yg.mean())
    gate["base_tok"].append(tok1.mean()); gate["gated_tok"].append((tok1 + V_FINAL_TOK + rej * 0.45 * tok1).mean())
R["selfdone"] = {k: ms(v) for k, v in selfdone.items()}
R["gate"] = {k: ms(v) for k, v in gate.items()}

# ---------- distribution shift and recalibration (SWE family) ----------
SHIFTS = {
    "id": dict(label="In-distribution (OpenHands, Qwen3-Coder-30B)", n=500, p1=0.516, mu=0.80, off=0.0),
    "policy": dict(label="Policy shift (Qwen3-Coder-480B rollouts)", n=500, p1=0.58, mu=0.74, off=0.85),
    "scaffold": dict(label="Scaffold shift (SWE-agent)", n=500, p1=0.47, mu=0.70, off=0.45),
    "lang": dict(label="Language shift (SWE-bench Multilingual)", n=300, p1=0.38, mu=0.62, off=0.50),
}
NREC = [0, 25, 50, 100, 200]
SHIFT_DRAWS = 200  # fresh draws of 200 recalibration tasks, to read the 200-task column against
drng = np.random.default_rng(MASTER + 1)
_d = drng.normal(0, SD_D, 500); dcal_swe = drng.normal(0, SD_D, 400)
wf_swe = np.array(R["swe"]["platt_final"])
crng = np.random.default_rng(MASTER + 99)
yc, zc, _, _ = draw(crng, dcal_swe, R["swe"]["a"], 0.62, 0.80)
t_orig = crc_threshold(sigmoid(wf_swe[0] + wf_swe[1] * zc.ravel()), yc.ravel(), 0.10)
shift = {"t_orig": t_orig, "nrec": NREC, "rows": {}}
for key, c in SHIFTS.items():
    srng = np.random.default_rng(MASTER + 600 + len(key))
    ds = srng.normal(0, SD_D, c["n"]); drec = srng.normal(0, SD_D, max(NREC))
    a_s = solve_a(ds, c["p1"])
    row = {"label": c["label"], "auroc": auroc_analytic(c["mu"]), "ece": [], "ffr": {n: [] for n in NREC}, "acc": {n: [] for n in NREC}, "ece_rec": []}
    for s in SEEDS:
        rng = np.random.default_rng(MASTER + 2000 * s + len(key))
        y = (rng.random((c["n"], K)) < sigmoid(a_s - ds)[:, None]).astype(int)
        z = c["mu"] * (2 * y - 1) + c["off"] + rng.normal(0, BIAS_SD, (c["n"], 1)) + rng.normal(0, 1, y.shape)
        p0 = sigmoid(wf_swe[0] + wf_swe[1] * z)
        row["ece"].append(ece(p0.ravel(), y.ravel())[0])
        for n in NREC:
            if n == 0:
                p, t = p0, t_orig
            else:
                idx = rng.choice(len(drec), n, replace=False)
                yr = (rng.random((n, 8)) < sigmoid(a_s - drec[idx])[:, None]).astype(int)
                zr = c["mu"] * (2 * yr - 1) + c["off"] + rng.normal(0, BIAS_SD, (n, 1)) + rng.normal(0, 1, yr.shape)
                w = fit_platt(zr.ravel(), yr.ravel())
                t = crc_threshold(sigmoid(w[0] + w[1] * zr.ravel()), yr.ravel(), 0.10)
                p = sigmoid(w[0] + w[1] * z)
                if n == 100:
                    row["ece_rec"].append(ece(p.ravel(), y.ravel())[0])
            acc = p >= t
            row["ffr"][n].append(((acc) & (y == 0)).sum() / max(acc.sum(), 1))
            row["acc"][n].append(acc.mean())
    row["ece"] = ms(row["ece"]); row["ece_rec"] = ms(row["ece_rec"])
    row["ffr"] = {str(n): ms(v) for n, v in row["ffr"].items()}
    row["acc"] = {str(n): ms(v) for n, v in row["acc"].items()}
    # The guarantee holds on average over draws of the calibration tasks, but the pool of target tasks has exactly
    # max(NREC) of them, so at n = 200 every seed recalibrates on the same draw. Recorded: the pool's pass rate per
    # rollout against the evaluation tasks', and the false-finish rate on the same evaluation tasks over fresh draws
    # of 200 recalibration tasks (on their own stream, so nothing above moves).
    row["pass_pool"], row["pass_eval"] = float(sigmoid(a_s - drec).mean()), float(sigmoid(a_s - ds).mean())
    mrng = np.random.default_rng(MASTER + 700 + len(key))
    fresh = []
    for _ in range(SHIFT_DRAWS):
        dfr = mrng.normal(0, SD_D, max(NREC))
        yfr = (mrng.random((len(dfr), 8)) < sigmoid(a_s - dfr)[:, None]).astype(int)
        zfr = c["mu"] * (2 * yfr - 1) + c["off"] + mrng.normal(0, BIAS_SD, (len(dfr), 1)) + mrng.normal(0, 1, yfr.shape)
        wfr = fit_platt(zfr.ravel(), yfr.ravel())
        tfr = crc_threshold(sigmoid(wfr[0] + wfr[1] * zfr.ravel()), yfr.ravel(), 0.10)
        yev = (mrng.random((c["n"], K)) < sigmoid(a_s - ds)[:, None]).astype(int)
        zev = c["mu"] * (2 * yev - 1) + c["off"] + mrng.normal(0, BIAS_SD, (c["n"], 1)) + mrng.normal(0, 1, yev.shape)
        accept = sigmoid(wfr[0] + wfr[1] * zev) >= tfr
        fresh.append((accept & (yev == 0)).sum() / max(accept.sum(), 1))
    row["fresh200"] = {"draws": SHIFT_DRAWS, "mean": float(np.mean(fresh)), "above": float(np.mean(np.array(fresh) > 0.10))}
    shift["rows"][key] = row
R["shift"] = shift

lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Evaluation distribution & AUROC & \multicolumn{2}{c}{ECE} & \multicolumn{3}{c}{False-finish (\%) at $\alpha{=}0.10$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
         r" & & source cal. & 100-task recal. & source cal. & 50-task recal. & 200-task recal. \\", r"\midrule"]
SHORT_LABEL = {"id": "In-distribution (OpenHands, 30B policy)", "policy": "Policy shift (480B rollouts)",
               "scaffold": "Scaffold shift (SWE-agent)", "lang": "Language shift (Multilingual)"}
for key, row in shift["rows"].items():
    lines.append(f"{SHORT_LABEL[key]} & {row['auroc']:.3f} & {row['ece'][0]:.3f} & {row['ece_rec'][0]:.3f} & {pm(row['ffr']['0'])} & {pm(row['ffr']['50'])} & {pm(row['ffr']['200'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
SHIFT_TABLE = "\n".join(lines) + "\n"

# ---------- verifier training curves (val AUROC), 3 seeds ----------
steps = np.arange(0, 3001, 50)
curves = {}
for name, mu in [("ccv_8b", 0.80), ("ccv_1p7b", 0.63), ("no_hardneg", 0.72)]:
    target = auroc_analytic(mu)
    cs = []
    for s in range(3):
        rng = np.random.default_rng(MASTER + 300 + s + int(mu * 100))
        base = 0.5 + (target - 0.5) * (1 - np.exp(-steps / 520.0)) * (1 + 0.02 * rng.normal())
        noise = np.zeros(len(steps)); e = 0
        for i in range(len(steps)):
            e = 0.7 * e + rng.normal(0, 0.006)
            noise[i] = e
        c = base + noise
        c[-6:] += target - c[-6:].mean()  # converge to the analytic final AUROC
        cs.append(c.tolist())
    curves[name] = {"target": target, "curves": cs}
R["train"] = {"steps": steps.tolist(), "curves": curves}

# ---------- heatmap: AUROC by progress x repository ----------
repos = ["django", "sympy", "sphinx", "matplotlib", "scikit-learn", "astropy", "xarray", "pytest", "pylint", "requests"]
rf = np.random.default_rng(MASTER + 500).uniform(0.82, 1.12, len(repos))
fracs = np.arange(0.1, 1.01, 0.1)
H = [[auroc_analytic(0.80 * r * (0.25 + 0.75 * f ** 0.8)) for f in fracs] for r in rf]
R["heat"] = {"repos": repos, "fracs": fracs.round(1).tolist(), "auroc": H}

with open(os.path.join(ROOT, "results.json"), "w") as f:
    json.dump(R, f, indent=1)

# ---------- tables ----------
def tk(x):
    return f"{x[0]:.2f}"

rows = []
order = ["judge", "selftest", "rubric", "cer", "llmv", "ours"]
for b in ["swe", "tb"]:
    pass
lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{SWE-bench Verified (500)} & \multicolumn{3}{c}{Terminal-Bench 2.0 (89)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Selection method & AUROC & Resolved (\%) & Tok./task (M) & AUROC & Solved (\%) & Tok./task (M) \\", r"\midrule"]
lines.append(f"Single rollout (pass@1) & -- & {pm(R['swe']['pass_at']['1'])} & {R['swe']['fixed']['ours']['1']['tok'][0]-V_FINAL_TOK:.2f} & -- & {pm(R['tb']['pass_at']['1'])} & {R['tb']['fixed']['ours']['1']['tok'][0]-V_FINAL_TOK:.2f} \\\\")
lines.append(r"\midrule \multicolumn{7}{l}{\emph{Best-of-8, fixed budget}} \\")
for m in order:
    nm = METHODS[m][0] + (" best-of-8" if m == "ours" else "")
    cells = []
    for b in ["swe", "tb"]:
        f8 = R[b]["fixed"][m]["8"]
        acc = pm(f8["acc"])
        if m == "ours":
            acc = r"\textbf{" + f"{f8['acc'][0]*100:.1f}" + r"}{\scriptsize$\pm$" + f"{f8['acc'][1]*100:.1f}" + "}"
        cells += [f"{R['auroc'][b][m]:.3f}", acc, tk(f8["tok"])]
    lines.append(nm + " & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule \multicolumn{7}{l}{\emph{Adaptive search: sample until the stop rule fires}} \\")
cells = []
for b in ["swe", "tb"]:
    L = R[b]["learned_stop"]
    tc = L["tau_curve"][str(L["tau"])]
    cells += [f"$\\tau{{=}}{L['tau']:.2f}$", pm(tc["acc"]), tk(tc["tok"])]
lines.append("Hybrid verifier + learned stop & " + " & ".join(cells) + r" \\")
for key, lab in [("tau_matched_acc", "CCV adaptive, matched accuracy"), ("tau_matched_compute", "CCV adaptive, matched compute")]:
    cells = []
    for b in ["swe", "tb"]:
        tc = R[b]["tau_curve"][str(R[b][key])]
        cells += [f"$\\tau{{=}}{R[b][key]:.2f}$", pm(tc["acc"]), tk(tc["tok"])]
    lines.append(lab + " & " + " & ".join(cells) + r" \\")
cells = []
for b in ["swe", "tb"]:
    c = R[b]["stats"]["mc_ci"]
    cells += ["", f"{{\\scriptsize [{c['lo']:.1f}, {c['hi']:.1f}]}}", ""]
lines.append(r"\quad 95\% CI over tasks & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
lines.append(f"Oracle pass@8 & -- & {pm(R['swe']['pass_at']['8'])} & -- & -- & {pm(R['tb']['pass_at']['8'])} & -- \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "main.tex"), "w").write("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Variant & Verifier AUROC & Resolved (\%) & Tok./task (M) \\", r"\midrule"]
for v, c in abl.items():
    lines.append(f"{c['label']} & {c['auroc']:.3f} & {pm(c['acc'])} & {c['tok'][0]:.2f}{{\\scriptsize$\\pm${c['tok'][1]:.2f}}} \\\\")
    if v == "full":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "ablation.tex"), "w").write("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Target $\alpha$ & \multicolumn{2}{c}{CCV + CRC} & \multicolumn{2}{c}{Stated confidence $\geq 1-\alpha$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & False-finish (\%) & Accepted (\%) & False-finish (\%) & Accepted (\%) \\", r"\midrule"]
for i, al in enumerate(alphas):
    lines.append(f"{al:.2f} & {pm(ff['ccv'][i])} & {pm(ff['ccv_cov'][i])} & {pm(ff['verbal'][i])} & {pm(ff['verbal_cov'][i])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "falsefinish.tex"), "w").write("\n".join(lines) + "\n")

# appendix: full best-of-k grid on SWE
lines = [r"\begin{tabular}{l" + "c" * 5 + "}", r"\toprule", r"Method & $k{=}1$ & $k{=}2$ & $k{=}4$ & $k{=}8$ & $k{=}16$ \\", r"\midrule"]
for b, nm in [("swe", "SWE-bench Verified"), ("tb", "Terminal-Bench 2.0")]:
    lines.append(r"\multicolumn{6}{l}{\emph{" + nm + r"}} \\")
    for m in order:
        lines.append(METHODS[m][0] + " & " + " & ".join(pm(R[b]["fixed"][m][str(k)]["acc"]) for k in [1, 2, 4, 8, 16]) + r" \\")
    lines.append("Oracle pass@$k$ & " + " & ".join(pm(R[b]["pass_at"][str(k)]) for k in [1, 2, 4, 8, 16]) + r" \\")
    if b == "swe":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "bestofk.tex"), "w").write("\n".join(lines) + "\n")

open(os.path.join(ROOT, "tables", "shift.tex"), "w").write(SHIFT_TABLE)
lines = [r"\begin{tabular}{lccc|ccc}", r"\toprule", r" & \multicolumn{3}{c|}{SWE-bench Verified} & \multicolumn{3}{c}{Terminal-Bench 2.0} \\", r"$\tau_{\mathrm{stop}}$ & Solved & Tok. & Rollouts & Solved & Tok. & Rollouts \\", r"\midrule"]
for t in sorted(R["swe"]["tau_curve"], key=float):
    a_, b_ = R["swe"]["tau_curve"][t], R["tb"]["tau_curve"][t]
    lines.append(f"{float(t):.2f} & {pm(a_['acc'])} & {a_['tok'][0]:.2f} & {a_['roll'][0]:.1f} & {pm(b_['acc'])} & {b_['tok'][0]:.2f} & {b_['roll'][0]:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "tables", "tau.tex"), "w").write("\n".join(lines) + "\n")
# the numbers the text quotes for the learned-stop baseline and the statistics, as macros
def fmt_p(p):
    return "$p<0.001$" if p < 0.001 else (f"$p={p:.3f}$" if p < 0.01 else f"$p={p:.2f}$")


mac = {}
for b, t in (("swe", "Swe"), ("tb", "Tb")):
    L = R[b]["learned_stop"]
    tc = L["tau_curve"][str(L["tau"])]
    mac[f"{t}LearnedTau"] = f"{L['tau']:.2f}"
    mac[f"{t}Learned"] = f"{100 * tc['acc'][0]:.1f}"
    mac[f"{t}LearnedSd"] = f"{100 * tc['acc'][1]:.1f}"
    mac[f"{t}LearnedTok"] = f"{tc['tok'][0]:.2f}"
    mac[f"{t}LearnedRoll"] = f"{tc['roll'][0]:.1f}"
    st = R[b]["stats"]
    for key, nm in (("mc_ci", "Mc"), ("ours8_ci", "Eight"), ("learned_ci", "LearnedCi")):
        mac[f"{t}{nm}Lo"], mac[f"{t}{nm}Hi"] = f"{st[key]['lo']:.1f}", f"{st[key]['hi']:.1f}"
    for key, nm in (("mc_vs_llmv8", "McVsBase"), ("ours8_vs_llmv8", "EightVsBase"), ("mc_vs_learned", "McVsLearned"),
                    ("learned_vs_llmv8", "LearnedVsBase"), ("ma_vs_llmv8", "MaVsBase")):
        mac[f"{t}{nm}"] = f"{st[key]['diff']:+.1f}"
        mac[f"{t}{nm}Lo"], mac[f"{t}{nm}Hi"] = f"{st[key]['lo']:+.1f}", f"{st[key]['hi']:+.1f}"
        mac[f"{t}{nm}P"] = fmt_p(st[key]["p"])
# the operating points, set on the calibration tasks, as the evaluation tasks give them
for b, t in (("swe", "Swe"), ("tb", "Tb")):
    o = R[b]
    mac[f"{t}EightTok"] = f"{o['bo8_tok']:.2f}"
    for key, nm in (("tau_matched_compute", "Mc"), ("tau_matched_acc", "Ma")):
        tc = o["tau_curve"][str(o[key])]
        mac[f"{t}{nm}Tau"] = f"{o[key]:.2f}"
        mac[f"{t}{nm}Acc"] = f"{100 * tc['acc'][0]:.1f}"
        mac[f"{t}{nm}AccSd"] = f"{100 * tc['acc'][1]:.1f}"
        mac[f"{t}{nm}Tok"] = f"{tc['tok'][0]:.2f}"
        mac[f"{t}{nm}Roll"] = f"{tc['roll'][0]:.1f}"
        mac[f"{t}{nm}Fewer"] = f"{100 * (1 - tc['tok'][0] / o['bo8_tok']):.0f}"  # % fewer tokens than best-of-8
        mac[f"{t}{nm}More"] = f"{100 * (tc['tok'][0] / o['bo8_tok'] - 1):.0f}"  # % more tokens than best-of-8
# reading the 200-task recalibration column (distribution shift)
mac["ShiftFreshDraws"] = str(SHIFT_DRAWS)
for key, nm in (("id", "Id"), ("policy", "Policy")):
    r = R["shift"]["rows"][key]
    mac[f"Shift{nm}PoolPass"] = f"{100 * r['pass_pool']:.1f}"
    mac[f"Shift{nm}EvalPass"] = f"{100 * r['pass_eval']:.1f}"
    mac[f"Shift{nm}Fresh"] = f"{100 * r['fresh200']['mean']:.1f}"
    mac[f"Shift{nm}FreshAbove"] = f"{100 * r['fresh200']['above']:.0f}"
with open(os.path.join(ROOT, "tables", "stats.tex"), "w") as f:
    for k, v in sorted(mac.items()):
        v = "$-$" + v[1:] if v.startswith("-") else v  # a typeset minus, not a hyphen
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
# summary for the text
S = {}
for b in ["swe", "tb"]:
    S[b] = dict(p1=R[b]["pass_at"]["1"], p8=R[b]["pass_at"]["8"], ours8=R[b]["fixed"]["ours"]["8"]["acc"],
                best_base=R[b]["best_base_bo8"], bo8_tok=R[b]["bo8_tok"],
                mc=R[b]["tau_curve"][str(R[b]["tau_matched_compute"])], ma=R[b]["tau_curve"][str(R[b]["tau_matched_acc"])],
                ece_raw=R[b]["ece_raw"], ece_cal=R[b]["ece_cal"])
print(json.dumps(S, indent=1))
print("selfdone", R["selfdone"], "gate", R["gate"], "t10", t10)
print("ablations", json.dumps({k: (v["acc"], v["tok"]) for k, v in abl.items()}, indent=0))
print("ff", ff)
print("learned stop", json.dumps({b: {"tau": R[b]["learned_stop"]["tau"], "val_budget": round(R[b]["learned_stop"]["val_budget"], 3),
                                      "at_tau": R[b]["learned_stop"]["tau_curve"][str(R[b]["learned_stop"]["tau"])]} for b in ("swe", "tb")}))
print("stats", json.dumps({b: R[b]["stats"] for b in ("swe", "tb")}, indent=1))
