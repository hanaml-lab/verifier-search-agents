# ccv: Calibrated Completion Verifiers

Reference implementation for **"Knowing When to Submit: Calibrated Completion Verifiers for Budgeted Search over Coding and Terminal Agent Trajectories"** (hanaML technical report, 2026).

CCV treats the verifier as a calibrated estimator of $\Pr(\text{hidden tests pass} \mid \text{trajectory prefix})$ instead of a ranker, and uses it to control test-time search:

- **Calibration**: Platt scaling of verifier scores on tasks disjoint from training and evaluation (`ccv.calibration`).
- **Budgeted sequential search** (Algorithm 1): prune unpromising prefixes at progress ρ≈0.3, stop as soon as the best calibrated score passes τ_stop, cap at K_max rollouts (`ccv.controller`).
- **False-finish control**: a submit threshold from conformal risk control that bounds the fraction of accepted completions that fail the hidden tests at level α (`ccv.crc`).
- **Verifier models**: a CPU logistic reference with the paper's progress-weighted and pairwise objective (`ccv.verifier.LogisticVerifier`), and a scalar-head LM wrapper for real trajectories (`ccv.verifier.HFVerifier`). `ccv.render` turns OpenHands-style step logs into verifier input with execution features.

## Install

```bash
pip install -e ".[dev]"          # core + tests (numpy only)
pip install -e ".[paper]"        # + scipy, matplotlib to regenerate the report's tables and figures
pip install -e ".[hf]"           # + torch, transformers for the LM verifier
```

## Quickstart (CPU, under 5 seconds)

```bash
python scripts/run_toy.py        # configs/toy.toml
pytest -q                        # 13 tests
```

The toy setting draws tasks with latent difficulty and rollouts whose features grow more informative with progress. The script trains the verifier, calibrates it on disjoint tasks, sets the CRC submit threshold, and compares pass@1, best-of-8 and adaptive search:

```
pass@1                 0.5000
best-of-8              0.7867
bok_tokens             4.7681
adaptive               0.8067
adaptive_tokens        1.8286
false_finish_rate      0.1198   # target alpha = 0.10
```

### Using the controller with your own agent

```python
from ccv import SearchConfig, adaptive_search, crc_threshold, PlattCalibrator

cal = PlattCalibrator().fit(cal_scores, cal_labels)              # held-out tasks
t_alpha = crc_threshold(cal.predict(cal_scores), cal_labels, alpha=0.10)
cfg = SearchConfig(tau_stop=0.88, tau_prune=0.25, k_max=16, submit_threshold=t_alpha)
result = adaptive_search(new_rollout, q_prefix, q_final, cfg)    # new_rollout() -> object with run_to/finish/tokens
```

## Results (from the report)

| Selector | SWE-bench Verified | Tok./task (M) | Terminal-Bench 2.0 | Tok./task (M) |
|---|---|---|---|---|
| Single rollout (pass@1) | 51.6 ± 1.2 | 0.61 | 23.6 ± 2.6 | 1.36 |
| Hybrid verifier (R2E-Gym-style), best-of-8 | 64.0 ± 0.8 | 5.21 | 36.2 ± 3.0 | 10.91 |
| CCV, best-of-8 | 67.0 ± 0.9 | 5.21 | 37.8 ± 2.2 | 10.91 |
| CCV adaptive, matched accuracy | 65.1 ± 1.3 | 2.01 | 40.2 ± 1.5 | 6.06 |
| CCV adaptive, matched compute | 71.5 ± 1.5 | 5.30 | 43.8 ± 2.5 | 10.95 |
| Oracle pass@8 | 73.8 ± 0.9 | — | 43.8 ± 1.1 | — |

The report's numbers come from a seeded simulation study calibrated to public baselines (see its Limitations section). `scripts/reproduce_paper.sh` regenerates every table and figure from `paper/make_data.py` and `paper/figures.py`.

## Layout

```
src/ccv/        calibration.py  crc.py  controller.py  verifier.py  render.py  toy.py
configs/        toy.toml  swe_verified.toml  terminal_bench.toml
scripts/        run_toy.py  reproduce_paper.sh
paper/          make_data.py  figures.py
tests/          pytest suite
```

## Full-scale runs

`configs/swe_verified.toml` and `configs/terminal_bench.toml` record the paper settings (Qwen3-Coder policies in OpenHands and Terminus 2; Qwen3-8B verifier). Rollout collection uses the official SWE-bench and Terminal-Bench harnesses. This repository holds the verifier, calibration, search and risk-control logic. It does not include agent scaffolds or model weights.

## Citation

```bibtex
@techreport{agarwal2026ccv,
  title       = {Knowing When to Submit: Calibrated Completion Verifiers for Budgeted Search over Coding and Terminal Agent Trajectories},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2026}
}
```

## License

Apache-2.0. See `LICENSE`.
