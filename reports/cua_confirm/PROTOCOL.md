# Confirmation of the three strength-pilot picks

Fixed before any run. Read-out: `cua/analysis/confirm.py`; its threshold is a constant there and is not changed
after the runs.

## Why

In the strength pilot (`reports/cua_strength/STRENGTH.md`), three per-class choices were picked in all three
cross-fitting folds:

| Class | Edit and level |
|---|---|
| Cavity | saturation @ x3 |
| Hollowareas | contrast @ x2 |
| Rockpocket | noise @ x2 |

They were found after looking at seeds 0–2, so they are hypotheses, not results. This test checks them on
seeds that played no part in finding them.

## Runs

dacl10k, Faster R-CNN, seeds 3, 4 and 5, with the strength-pilot settings unchanged:
- `A-dacl10k-fasterrcnn-s{3,4,5}-warm` (3-epoch warm start) and `-base` (3-epoch continuation, no edit)
- `S-dacl10k-fasterrcnn-s{3,4,5}-saturation-x3`, `-contrast-x2`, `-noise-x2` (edit on every image, p = 1)

15 runs (`python -m cua.ccast.make_jobs --family confirm --dataset dacl10k --detector fasterrcnn`).

## Test

For each pick (edit, level, class c) and seed k, from the final-epoch validation per-class AP50:
- **Target effect** d_k = AP50(c | arm) − AP50(c | base)
- **Specificity** s_k = d_k − mean over the other 11 classes of their arm − base delta

Each gets a one-sided paired t-test over the three seeds (H1: mean > 0), Holm-corrected over the three picks
separately for each statistic. **A pick is confirmed when both Holm-adjusted p-values are below 0.05.** All
three picks are reported whatever the outcome.

## How the result will be read

- **Confirmed pick:** that class benefits from its edit specifically, on new seeds. This is the per-class
  signal a class-aware policy needs; the next step is a policy that applies the confirmed edits to images of
  those classes only, compared with uniform sampling and with the same edits applied to all images.
- **Not confirmed:** the pick was a selection effect of the first three seeds.
- With three seeds and df = 2 the test has low power; a pick with a positive mean that misses the threshold is
  reported as not confirmed, with its interval.
