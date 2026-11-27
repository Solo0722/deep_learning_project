# Class-adaptive copy-paste pilot: validation read-out

Source: `COMPLETE.json` and `history.jsonl` of the P- and C- runs on CCAST, pulled on 2026-11-25.
The test evaluation was submitted as PBS job 817465. When it finishes, its `test_metrics.json` files replace this read-out for any claim.

Every number is the **best-epoch validation macro AP50**, as used for checkpoint selection, for Faster R-CNN with seeds 0, 1 and 2.

**Arms:**
- cprand: objects pasted at class frequency.
- cpbal: classes pasted uniformly.
- cpada: classes pasted by current reference-split AP50, with more pastes for classes with lower AP.

All three arms use the uniform edit policy. The control is the C- uniform run.

## dacl10k (reviewer's kill-switch pair)

| Arm | s0 | s1 | s2 | Mean |
|---|---|---|---|---|
| base | 20.69 | 20.34 | 20.31 | 20.44 |
| measured | 21.13 | 20.97 | 21.29 | 21.13 |
| uniform | 21.79 | 21.86 | 21.70 | 21.78 |
| shared2 | 22.31 | 21.85 | 21.94 | 22.03 |
| cprand | 21.91 | 21.80 | 21.53 | 21.75 |
| cpbal | 21.63 | 21.71 | 21.33 | 21.56 |
| cpada | 22.00 | 21.87 | 21.54 | 21.80 |

Paired per seed:

| Comparison | s0 | s1 | s2 | Mean |
|---|---|---|---|---|
| cpada − uniform | +0.21 | +0.01 | −0.15 | +0.02 |
| cpada − cpbal | +0.37 | +0.16 | +0.21 | +0.25 |
| cpada − cprand | +0.09 | +0.07 | +0.01 | +0.06 |

## RDD2022

| Arm | s0 | s1 | s2 | Mean |
|---|---|---|---|---|
| base | 51.03 | 51.03 | 51.05 | 51.04 |
| measured | 51.18 | 51.83 | 51.33 | 51.45 |
| uniform | 50.63 | 50.98 | 50.48 | 50.70 |
| cprand | 49.97 | 50.22 | 49.99 | 50.06 |
| cpbal | 50.05 | 50.28 | 50.42 | 50.25 |
| cpada | 50.05 | 50.02 | 49.83 | 49.97 |

cpada − uniform, paired per seed: −0.58, −0.95, −0.66 (mean −0.73).

## Reading against the reviewer's go/no-go rule

The rule asks cpada to beat both uniform and cpbal by at least 0.8 AP50.

- **dacl10k:** cpada ties uniform (+0.02). It is ahead of cpbal by only +0.25, although that lead holds in all three seeds. This is a **no-go** on validation.
- **RDD:** all three paste arms are below uniform and base.

The validation gaps are far from the 0.8 bar.

## Test results (job 817465, finished 2026-11-25)

Test macro AP50, Faster R-CNN, seeds 0, 1 and 2.

| Pair | Arm | s0 | s1 | s2 | Mean |
|---|---|---|---|---|---|
| dacl10k | uniform | 22.32 | 22.33 | 22.47 | 22.37 |
| dacl10k | cprand | 22.31 | 22.50 | 22.41 | 22.41 |
| dacl10k | cpbal | 22.94 | 22.80 | 22.85 | 22.87 |
| dacl10k | cpada | 23.00 | 22.77 | 22.82 | 22.86 |
| RDD | uniform | 50.57 | 50.53 | 50.54 | 50.55 |
| RDD | cprand | 49.96 | 49.84 | 50.27 | 50.02 |
| RDD | cpbal | 50.22 | 49.98 | 49.78 | 49.99 |
| RDD | cpada | 50.01 | 49.69 | 49.59 | 49.77 |

dacl10k, paired per seed, with the 95% seed t interval (df = 2). These intervals were computed by hand; the image bootstrap has not been run yet.

| Comparison | Mean | 95% seed t interval |
|---|---|---|
| cpbal − uniform | +0.49 | [0.19, 0.79] |
| cpada − uniform | +0.49 | [0.07, 0.91] |
| cpada − cpbal | −0.00 | per seed +0.06, −0.03, −0.03 |
| cprand − uniform | +0.04 | — |

**Reading.** On dacl10k test, class-balanced pasting beats uniform by about 0.5 AP50 in all three seeds, and adaptive pasting adds nothing over balanced pasting. This is the reviewer's "static class balancing, not online adaptation" outcome, and it falls short of the +0.8 bar. On RDD every paste arm is 0.5–0.8 below uniform, which is itself below base (51.15).

On dacl10k, validation and test disagree: on validation, cpbal trailed uniform.

### Paired image bootstrap (dacl10k test)

1,000 resamples of the 1,141 test images. Each resample is shared by both arms and all seeds, and the result is the mean of the per-seed differences. This is the same procedure as `cua.analysis.ap50.paired_bootstrap_seeds`.

| Comparison | Difference | 95% bootstrap interval | 95% seed t interval | Claim? |
|---|---|---|---|---|
| cpbal − uniform | +0.48 | [−0.06, 0.94] | [0.19, 0.79] | **no**: the bootstrap interval includes 0 |
| cpada − uniform | +0.49 | [−0.05, 0.99] | [0.07, 0.91] | no |
| cpada − cpbal | +0.01 | [−0.28, 0.27] | — | no |

Under the pre-registered rule, a claim needs both intervals to exclude 0. The class-balanced copy-paste gain on dacl10k is therefore consistent across seeds but **not significant** at the image level. Report it as a suggestive trend.

### Kill test: polygon-mask copy-paste (cpbalm, dacl10k Faster R-CNN, test)

Go rule (fixed before the runs): cpbalm − uniform ≥ +0.74 AP50, with both the seed t interval and the paired image bootstrap (1,000 resamples) excluding 0.

| Arm | Test AP50, seeds 0/1/2 | Mean |
|---|---|---|
| uniform | 22.32 / 22.33 / 22.47 | 22.37 |
| cpbal (rectangle) | 22.94 / 22.76 / 22.85 | 22.85 |
| cpbalm (mask) | 22.57 / 22.07 / 22.43 | 22.36 |

| Comparison | Difference | 95% seed t interval | 95% bootstrap interval |
|---|---|---|---|
| cpbalm − uniform | −0.02 | [−0.65, +0.62] | [−0.61, +0.55] |
| cpbalm − cpbal | −0.49 | [−0.91, −0.07] | [−0.97, +0.05] |
| cpbal − uniform | +0.48 | [+0.16, +0.79] | [−0.06, +0.94] |

**Verdict: no-go.** Mask pasting gives no gain over uniform, and it trails rectangle pasting on all three seeds; the bootstrap interval for that gap only just includes 0. Validation had ranked cpbalm first (21.90 vs 21.78 uniform and 21.56 cpbal), so validation and test disagree again.
