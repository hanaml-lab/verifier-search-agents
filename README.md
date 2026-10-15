# CCV: calibrated completion verifiers

[![ci](https://github.com/hanaml-lab/verifier-search-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/verifier-search-agents/actions/workflows/ci.yml)

Reference implementation of *Knowing When to Submit: Calibrated Completion Verifiers for Budgeted Search over Coding and Terminal Agent Trajectories* (Agarwal, hanaML technical report, 2026).

CCV treats the verifier as a calibrated estimator of $\Pr(\text{hidden tests pass} \mid \text{trajectory prefix})$ instead of a ranker, and uses it to control test-time search:

- **Calibration**: Platt scaling of verifier scores on tasks disjoint from training and evaluation (`ccv.calibration`).
- **Budgeted sequential search** (Algorithm 1): prune unpromising prefixes at progress ρ≈0.3, stop as soon as the best calibrated score passes τ_stop, cap at K_max rollouts (`ccv.controller`).
- **False-finish control**: a submit threshold from conformal risk control that bounds the fraction of accepted completions that fail the hidden tests at level α (`ccv.crc`).
- **Verifier models**: a CPU logistic reference with the report's progress-weighted and pairwise objective (`ccv.verifier.LogisticVerifier`), and a scalar-head LM wrapper for real trajectories (`ccv.verifier.HFVerifier`). `ccv.render` turns OpenHands-style step logs into verifier input with execution features.

## Install

```bash
pip install -e ".[dev]"          # numpy, pyyaml; pytest and ruff
pip install -e ".[paper]"        # + scipy, matplotlib, to regenerate the report's tables and figures
pip install -e ".[hf]"           # + torch, transformers for the LM verifier
```

## Quickstart (CPU, under 20 seconds)

```bash
python -m ccv.demo               # or ccv-demo; settings in configs/toy.yaml
pytest -q
```

## Toy run (real output)

The toy setting draws tasks with a latent difficulty and rollouts whose features carry more about the outcome as they progress. The demo trains the verifier, calibrates it on disjoint tasks, sets the CRC submit threshold, and compares pass@1, best-of-8 and adaptive search on 300 test tasks; it then repeats the calibration over ten seeds to measure the realised false-finish rate at three targets. Output of `python -m ccv.demo` on a CPU (Python 3.12):

```
seed 0: 300 training, 200 calibration and 300 test tasks
selector                solved   tokens  rollouts
pass@1                   0.500     0.59      1.00
best-of-8                0.787     4.77      8.00
CCV adaptive search      0.807     1.83      4.46
ECE on calibration tasks: 0.024 raw, 0.013 after Platt scaling
submit threshold at alpha = 0.10: 0.233; it accepts 55.7% of test completions, 12.0% of them false finishes

false-finish control, seeds 0-9 pooled (3000 test completions)
 alpha  accepted  false finishes
  0.05     50.1%            5.1%
  0.10     54.9%           10.2%
  0.20     63.4%           20.8%
done in 12.7s
```

Adaptive search solves more tasks than best-of-8 with 38% of its tokens. Pooled over ten seeds, the share of accepted completions that fail sits at the target α, within sampling error; a single calibration draw can land above it (seed 0, 12.0%), since the guarantee holds on average. `paper/toy_run.md` records the environment and what each line shows.

## Using CCV with your own verifier and agent

`examples/calibrate_threshold.py` reads a CSV of raw verifier scores, one row per finished completion, with the hidden-test outcome where it is known. Completions with an outcome, from tasks the verifier was not trained on, fit the Platt calibration and the submit threshold; completions without one are submitted or held back.

```bash
python examples/calibrate_threshold.py examples/data/verifier_scores.csv --alpha 0.10
```

```
calibration: 600 completions on 150 tasks, 47.5% pass
ECE 0.036 raw, 0.027 after Platt scaling
submit threshold at alpha = 0.10: q >= 0.278 (on the calibration set it accepts 50.7%, 9.5% of them failing)

task         score       q  decision
new-000      -3.75   0.008  hold back
new-001       5.80   0.999  submit
new-002      -0.44   0.340  submit
new-003       3.25   0.981  submit
new-004      -3.54   0.011  hold back
new-005      -3.77   0.008  hold back
new-006      -5.42   0.001  hold back
new-007      -1.46   0.126  hold back
```

The controller takes the same pieces from Python:

```python
from ccv import SearchConfig, adaptive_search, crc_threshold, PlattCalibrator

cal = PlattCalibrator().fit(cal_scores, cal_labels)              # held-out tasks
t_alpha = crc_threshold(cal.predict(cal_scores), cal_labels, alpha=0.10)
cfg = SearchConfig(tau_stop=0.88, tau_prune=0.25, k_max=16, submit_threshold=t_alpha)
result = adaptive_search(new_rollout, q_prefix, q_final, cfg)    # new_rollout() -> object with run_to/finish/tokens
```

## Results (from the report)

These numbers come from the report's seeded simulation and are illustrative.

| Selector | SWE-bench Verified | Tok./task (M) | Terminal-Bench 2.0 | Tok./task (M) |
|---|---|---|---|---|
| Single rollout (pass@1) | 51.6 ± 1.2 | 0.61 | 23.6 ± 2.6 | 1.36 |
| Hybrid verifier (R2E-Gym-style), best-of-8 | 64.0 ± 0.8 | 5.21 | 36.2 ± 3.0 | 10.91 |
| CCV, best-of-8 | 67.0 ± 0.9 | 5.21 | 37.8 ± 2.2 | 10.91 |
| CCV adaptive, matched accuracy | 65.1 ± 1.3 | 2.01 | 40.2 ± 1.5 | 6.06 |
| CCV adaptive, matched compute | 71.5 ± 1.5 | 5.30 | 43.8 ± 2.5 | 10.95 |
| Oracle pass@8 | 73.8 ± 0.9 | — | 43.8 ± 1.1 | — |

`bash scripts/reproduce_paper.sh` regenerates the tables and figures that the report's seeded simulation produces. On every push, CI reruns `paper/make_data.py` and checks the tables and results it writes against `paper/results.sha256` (`python paper/check_results.py`).

## Layout

```
src/ccv/        calibration.py  crc.py  controller.py  verifier.py  render.py  toy.py  demo.py
configs/        toy.yaml  swe_verified.yaml  terminal_bench.yaml
examples/       calibrate_threshold.py, data/verifier_scores.csv
scripts/        run_toy.py  reproduce_paper.sh
paper/          make_data.py  figures.py  check_results.py  results.sha256  toy_run.md
tests/          pytest suite
```

## Full-scale runs

`configs/swe_verified.yaml` and `configs/terminal_bench.yaml` record the report's settings (Qwen3-Coder policies in OpenHands and Terminus 2; Qwen3-8B verifier). Rollout collection uses the official SWE-bench and Terminal-Bench harnesses. This repository holds the verifier, calibration, search and risk-control logic. It does not include agent scaffolds or model weights.

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
