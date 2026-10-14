# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the report's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that the method works end to end on a problem with a known answer; it does not reproduce the report's
experiments.

- Command: `python -m ccv.demo` (settings in `configs/toy.yaml`)
- Code: commit `eb363f7`
- Environment: Python 3.12.13, numpy 2.5.4, PyYAML 6.0.3; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 14 s

## The toy setting

Tasks have a latent difficulty d ~ N(0, 3^2), and a rollout passes its hidden tests with probability
sigmoid(-d). Each rollout emits a feature vector at every point of its progress; the features carry more about the
outcome as the rollout advances, as execution evidence does in a real trajectory (`src/ccv/toy.py`). The logistic
reference verifier is trained with the report's progress-weighted and within-task pairwise objective on 300 tasks
(8 rollouts each). It is Platt-calibrated on 200 disjoint tasks, at the final state and at the pruning point
(progress 0.3), and the conformal risk control threshold is set there at alpha = 0.10. Search uses tau_stop = 0.85,
tau_prune = 0.25 and at most 16 rollouts, against pass@1 and best-of-8 on 300 test tasks. A second table repeats
the calibration for ten seeds and measures the realised false-finish rate on 3,000 fresh test completions for
alpha = 0.05, 0.10 and 0.20.

## Output

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

## What it shows

- Adaptive search solves more test tasks than best-of-8 (0.807 against 0.787) with 38% of its tokens (1.83 against
  4.77 per task, in the toy's token units), because it stops once a calibrated score passes tau_stop and prunes weak prefixes:
  it starts 4.46 rollouts per task on average instead of 8.
- Platt scaling roughly halves the verifier's calibration error on held-out tasks (ECE 0.024 to 0.013).
- Pooled over ten seeds, the share of accepted completions that fail the hidden tests sits at the target: 5.1%,
  10.2% and 20.8% for alpha = 0.05, 0.10 and 0.20, within sampling error of about one point. A single calibration
  draw can land above it (seed 0: 12.0% at alpha = 0.10), since the guarantee holds on average over draws.
