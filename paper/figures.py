"""All figures, drawn from results.json written by make_data.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results.json")))
OUT = os.path.join(ROOT, "figures")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "axes.grid": True, "lines.linewidth": 1.2, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# Okabe-Ito
C = {"judge": "#999999", "selftest": "#56B4E9", "rubric": "#009E73", "cer": "#F0E442", "llmv": "#E69F00", "ours": "#D55E00", "oracle": "#000000", "blue": "#0072B2", "pink": "#CC79A7"}
ORDER = ["judge", "selftest", "rubric", "cer", "llmv", "ours"]
SHORT = {"judge": "Outcome judge", "selftest": "Self-tests", "rubric": "Rubric PRM", "cer": "Generative verifier", "llmv": "Hybrid verifier", "ours": "CCV (ours)"}
KS = [1, 2, 4, 8, 16]
TITLE = {"swe": "SWE-bench Verified", "tb": "Terminal-Bench 2.0"}

# 1. best-of-k scaling
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.1))
for ax, b in zip(axs, ["swe", "tb"]):
    for m in ORDER:
        raw = np.array([R[b]["fixed_raw"][m][str(k)] for k in KS]) * 100
        mu, sd = raw.mean(1), raw.std(1, ddof=1)
        ax.plot(KS, mu, "-o", ms=2.5, color=C[m], label=SHORT[m], lw=1.5 if m == "ours" else 1.0, zorder=3 if m == "ours" else 2)
        ax.fill_between(KS, mu - 1.96 * sd / np.sqrt(5), mu + 1.96 * sd / np.sqrt(5), color=C[m], alpha=0.18, lw=0)
    pk = np.array([R[b]["pass_at"][str(k)][0] for k in KS]) * 100
    ax.plot(KS, pk, "--", color=C["oracle"], lw=0.8, label="Oracle pass@$k$")
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.set_xlabel("Rollouts per task $k$"); ax.set_title(TITLE[b])
axs[0].set_ylabel("Solved (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.17), handlelength=1.5, columnspacing=0.8)
fig.savefig(os.path.join(OUT, "fig_scaling.pdf")); plt.close(fig)

# 2. Pareto: accuracy vs tokens
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.1))
for ax, b in zip(axs, ["swe", "tb"]):
    for m in ["judge", "rubric", "cer", "llmv", "ours"]:
        x = [R[b]["fixed"][m][str(k)]["tok"][0] for k in KS]
        y = [R[b]["fixed"][m][str(k)]["acc"][0] * 100 for k in KS]
        ax.plot(x, y, ":" if m != "ours" else "-", marker="s", ms=2.2, color=C[m], lw=0.9, label=SHORT[m] + " (BoK)")
    tc = R[b]["tau_curve"]
    ts = sorted(tc, key=float)
    x = np.array([tc[t]["tok"][0] for t in ts]); y = np.array([tc[t]["acc"][0] * 100 for t in ts]); e = np.array([tc[t]["acc"][1] * 100 for t in ts])
    ax.plot(x, y, "-o", ms=2.8, color=C["blue"], lw=1.5, label="CCV adaptive", zorder=4)
    ax.fill_between(x, y - 1.96 * e / np.sqrt(5), y + 1.96 * e / np.sqrt(5), color=C["blue"], alpha=0.18, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("Tokens per task (M, log)"); ax.set_title(TITLE[b])
axs[0].set_ylabel("Solved (%)")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.17), handlelength=1.5, columnspacing=0.8)
fig.savefig(os.path.join(OUT, "fig_pareto.pdf")); plt.close(fig)

# 3. reliability
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.2))
for ax, b in zip(axs, ["swe", "tb"]):
    ax.plot([0, 1], [0, 1], "--", color="#555555", lw=0.7)
    for key, col, lab in [("rel_raw", C["pink"], "Raw verifier prob."), ("rel_cal", C["blue"], "CCV calibrated")]:
        pts = [(p, a, n) for p, a, n in R[b][key] if a is not None and n >= 30]
        ax.plot([p for p, _, _ in pts], [a for _, a, _ in pts], "-o", ms=2.6, color=col,
                label=f"{lab} (ECE {R[b]['ece_' + key.split('_')[1]]:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Predicted P(hidden tests pass)"); ax.set_title(TITLE[b])
    ax.legend(loc="upper left")
axs[0].set_ylabel("Empirical pass rate")
fig.savefig(os.path.join(OUT, "fig_calibration.pdf")); plt.close(fig)

# 4. training curves
fig, ax = plt.subplots(figsize=(2.9, 2.0))
steps = np.array(R["train"]["steps"])
for name, col, lab in [("ccv_8b", C["ours"], "CCV-8B"), ("no_hardneg", C["llmv"], "8B, no hard negatives"), ("ccv_1p7b", C["selftest"], "CCV-1.7B")]:
    cs = np.array(R["train"]["curves"][name]["curves"])
    mu, lo, hi = cs.mean(0), cs.min(0), cs.max(0)
    ax.plot(steps, mu, color=col, lw=1.0, label=lab)
    ax.fill_between(steps, lo, hi, color=col, alpha=0.22, lw=0)
ax.set_xlabel("Training step"); ax.set_ylabel("Held-out AUROC"); ax.set_ylim(0.48, 0.88)
ax.legend(loc="lower right")
fig.savefig(os.path.join(OUT, "fig_training.pdf")); plt.close(fig)

# 5. heatmap
H = np.array(R["heat"]["auroc"])
fig, ax = plt.subplots(figsize=(3.4, 2.4))
im = ax.imshow(H, cmap="viridis", aspect="auto", vmin=0.55, vmax=0.9)
ax.set_xticks(range(len(R["heat"]["fracs"]))); ax.set_xticklabels([f"{int(f*100)}" for f in R["heat"]["fracs"]])
ax.set_yticks(range(len(R["heat"]["repos"]))); ax.set_yticklabels(R["heat"]["repos"])
ax.set_xlabel("Trajectory progress at check (%)"); ax.grid(False)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H[i,j]:.2f}"[1:], ha="center", va="center", fontsize=4.6, color="white" if H[i, j] < 0.75 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("AUROC", fontsize=7)
fig.savefig(os.path.join(OUT, "fig_heatmap.pdf")); plt.close(fig)

# 6. false-finish control
ff = R["false_finish"]
al = np.array(ff["alpha"])
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.0))
ax = axs[0]
ax.plot([0, 0.32], [0, 0.32], "--", color="#555555", lw=0.7, label="Target $\\alpha$")
for key, col, lab in [("ccv", C["blue"], "CCV + CRC"), ("verbal", C["pink"], "Stated confidence")]:
    m = np.array([x[0] for x in ff[key]]); s = np.array([x[1] for x in ff[key]])
    ax.errorbar(al, m, yerr=s, fmt="-o", ms=2.6, color=col, capsize=1.5, elinewidth=0.6, label=lab)
ax.set_xlabel("Target false-finish rate $\\alpha$"); ax.set_ylabel("Realised false-finish rate"); ax.legend(loc="upper left")
ax = axs[1]
for key, col, lab in [("ccv_cov", C["blue"], "CCV + CRC"), ("verbal_cov", C["pink"], "Stated confidence")]:
    m = np.array([x[0] for x in ff[key]]) * 100
    ax.plot(al, m, "-o", ms=2.6, color=col, label=lab)
ax.set_xlabel("Target false-finish rate $\\alpha$"); ax.set_ylabel("Rollouts accepted (%)"); ax.legend(loc="upper left")
fig.savefig(os.path.join(OUT, "fig_falsefinish.pdf")); plt.close(fig)

# 7. ablation bars
A = R["ablations"]
keys = list(A)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
y = np.arange(len(keys))[::-1]
acc = np.array([A[k]["acc"][0] for k in keys]) * 100; err = np.array([A[k]["acc"][1] for k in keys]) * 100
cols = [C["ours"]] + [C["blue"]] * (len(keys) - 1)
ax.barh(y, acc, xerr=err, color=cols, height=0.6, error_kw=dict(lw=0.6, capsize=1.5), alpha=0.9)
ax.set_yticks(y); ax.set_yticklabels([A[k]["label"].replace("-- ", "w/o ") for k in keys])
ax.set_xlim(64, 74); ax.set_xlabel("SWE-bench Verified resolved (%)"); ax.grid(axis="y", visible=False)
for yi, a_, k in zip(y, acc, keys):
    ax.text(64.2, yi, f"{A[k]['tok'][0]:.2f}M", va="center", fontsize=5.5, color="white")
fig.savefig(os.path.join(OUT, "fig_ablation.pdf")); plt.close(fig)
print("ok")

# 8. distribution shift and recalibration
S = R["shift"]; ns = S["nrec"]
cols = {"id": C["oracle"], "policy": C["blue"], "scaffold": C["rubric"], "lang": C["ours"]}
short = {"id": "In-distribution", "policy": "Policy shift", "scaffold": "Scaffold shift", "lang": "Language shift"}
xs = np.arange(len(ns))
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.0))
for k, row in S["rows"].items():
    m = np.array([row["ffr"][str(n)][0] for n in ns]) * 100; sd = np.array([row["ffr"][str(n)][1] for n in ns]) * 100
    axs[0].errorbar(xs, m, yerr=sd, fmt="-o", ms=2.6, capsize=1.5, elinewidth=0.6, color=cols[k], label=short[k])
    a = np.array([row["acc"][str(n)][0] for n in ns]) * 100
    axs[1].plot(xs, a, "-o", ms=2.6, color=cols[k], label=short[k])
axs[0].axhline(10, ls="--", color="#555555", lw=0.7)
for ax in axs:
    ax.set_xticks(xs); ax.set_xticklabels(["source"] + [str(n) for n in ns[1:]]); ax.set_xlabel("Target-domain recalibration tasks")
axs[0].set_ylabel("False-finish (%) at $\\alpha=0.10$"); axs[1].set_ylabel("Rollouts accepted (%)")
axs[0].legend(loc="upper right")
fig.savefig(os.path.join(OUT, "fig_shift.pdf")); plt.close(fig)
print("ok shift")
