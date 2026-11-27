# Strength pilot: do per-class effects become usable when the edit strength varies?

Fixed before any S- run is trained. Thresholds below are not changed after the runs.

## Why

Every edit in the study has one moderate strength. At that strength, the class-specific part of the effects is mostly seed noise: the empirical-Bayes signal share λ is 0.18–0.43, and choosing each class's edit from two seeds gains +0.06 AP50 on the third (dacl10k, Faster R-CNN; whole-image pixel edits; `reports/cua_part_a_report/deltas.csv`). Shrinking the per-class choice toward the pooled one does not change this (+0.06). The selector is not the limit; the size of the class effects is.

Kirichenko et al. (NeurIPS 2023) find that class-level harm from augmentation grows with its strength, and that changing the strength for a few classes recovers their accuracy. CUDA (ICLR 2023) also sets a strength per class. If class effects grow with strength faster than seed noise, then a policy that chooses **edit and strength per class** has room that a fixed-strength menu does not. This pilot measures that, cheaply, before any full CUA run.

## Design

- **Pair:** dacl10k, Faster R-CNN. It has the strongest interaction, the only Holm-significant colour edit (contrast), and colour-defined classes (Rust, Efflorescence, Wetspot).
- **Edits:** the five pixel edits with one scalar strength: brightness, contrast, saturation, blur, noise.
- **Strength levels:** x1 is the current menu. The brightness, contrast and saturation factor ranges widen around the same centre (saturation is clipped at 0, i.e. greyscale); blur and noise σ grow. The widest ranges were set after rendering real dacl10k images (`previews.png`): a brightness or contrast factor of 0.1 blanks the image, which would measure destroying it rather than a class-specific effect, so the lowest factor is 0.3.

| Edit | x1 (existing) | x2 | x3 |
|---|---|---|---|
| Brightness factor | [0.7, 1.3] | [0.5, 1.5] | [0.3, 1.7] |
| Contrast factor | [0.7, 1.3] | [0.5, 1.5] | [0.3, 1.7] |
| Saturation factor | [0.5, 1.5] | [0.0, 2.0] | [−0.5, 2.5], clipped at 0 (1 in 6 draws greyscale) |
| Blur σ (px) | [0.5, 1.5] | [1.0, 3.0] | [1.5, 4.5] |
| Noise σ (of 255) | 0.03 | 0.06 | 0.10 |

  Each level consumes the same random draw as x1, so the strength draw for an image is the same quantile at every level and the arms stay paired.
- **Protocol:** as Part A, except that the edit is applied to **every** image (p = 1.0 instead of 0.5), so each arm's exposure to its edit is doubled; Part A's p = 0.5 halves the measured effects. Branch from `A-dacl10k-fasterrcnn-s{0,1,2}-warm`, continue 3 epochs, same data order and edit draws. The base arm is `A-dacl10k-fasterrcnn-s{k}-base` (it applies no edit, so p does not matter for it). Because p changes, x1 is rerun at p = 1.0; the Part A x1 arms (p = 0.5) are reported as a reference only.
- **Runs:** `S-dacl10k-fasterrcnn-s{0,1,2}-<edit>-x{1,2,3}`, 5 edits × 3 levels × 3 seeds = **45 short runs**, about 30 GPU-hours. No new warm or base runs.
- **Before submission:** render each edit at the low and high end of each level on real training images (done: `previews.png`), check that adding the strength setting leaves every existing config digest unchanged, and run the full test suite.

## Decision rule

The menu for the analysis is "none" plus the 15 (edit, level) arms.

| Check | Statistic | Pass |
|---|---|---|
| **S1 interaction** | (edit, level)×class F-test with seed×class term, as in Part A | p < 0.05 |
| **S2 usable (primary)** | Cross-fitted per-class selection gain over the 16-arm menu, as in `reports/cua_local/PROTOCOL.md` | mean ≥ **+0.50** AP50 and > 0 in all three folds |
| S3 descriptive | For each edit: spread of Δ across classes (SD) and seed SD, at x1, x2, x3; λ per level | reported only |
| S4 descriptive | Same gain with empirical-Bayes shrinkage toward the pooled effect | reported only |

The +0.50 bar is the same as the object-local pilot's. A smaller short-run gain is unlikely to survive the shrinkage from short runs to full training (ρ = 0.56) and reach the 0.74 AP50 minimum detectable effect of the full comparison.

## Step 2 (only if S1 and S2 pass)

CUA with the 16-arm menu (the per-class utility now selects edit and strength), against shared-utility augmentation with the same menu and against uniform augmentation over the same menu, on dacl10k with Faster R-CNN, full schedule, three seeds, test set once. Primary: CUA − shared, paired seed t and image bootstrap, both intervals above zero. Secondary: worst-class test AP50. A separate protocol is written before those runs.

## Implementation

- `TrainConfig.edit_strength: int = 1` (x1, x2, x3), validated to {1, 2, 3} and to the five scalable edits; x1 leaves every existing config digest unchanged.
- `cua/augment/pixel.py`: each of the five edits takes a strength level (ranges in the table above), drawing the same single random number at every level.
- `cua/experiments/specs.py`: `strength_specs()` (edit_p = 1.0, levels 1–3); `cua/ccast/make_jobs.py`: `--family strength` (external warm runs).
- `cua/analysis/strength.py`: S1–S4 from the S- runs and the A- base arm, plus the Part A x1 (p = 0.5) selection gain as a reference; writes `STRENGTH.md` and `strength.json`.
- Tests: x1 bit-identical to the current edits; x2/x3 ranges; paired draws; validation; specs and digests; make_jobs; the analysis on synthetic tables.
