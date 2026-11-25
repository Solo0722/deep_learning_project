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

The test numbers are pending, and the validation gaps are far from the 0.8 bar.
