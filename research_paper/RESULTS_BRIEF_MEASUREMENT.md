# Brief for the chat: method updates and measurement results (2026-11-18)

This brief gets the chat to do two things in `main.tex`:

1. Bring Sections III to V in line with what was actually built and run.
2. Write the first half of Section VI, the measurement-study results.

The Part C comparison is still running, so its results are not in this brief.

- Part 1 lists the files to upload.
- Part 2 is the message to paste.
- Part 3 is the data the message refers to. Paste it too, or upload this file.

## 1. Files to upload

| File (repo path) | Why |
|---|---|
| `research_paper/PAPER_CONTEXT.md` | Writing rules, terminology, citation map |
| `research_paper/main.tex` | Edit target (Sections III-V exist; VI is to be written) |
| `research_paper/references.bib` | The only citation keys allowed |
| `docs/superpowers/specs/2026-11-08-class-utility-augmentation-design.md` | Study design |
| `research_paper/RESULTS_BRIEF_MEASUREMENT.md` | This file (Part 3 holds the numbers) |
| `reports/cua_part_a_report/heatmap_*_*.png` (6 files) | Per-class effect heatmaps, one per dataset and detector |
| `reports/cua_part_a_report/tests.json`, `deltas.csv` | Exact test statistics and per-seed deltas |

## 2. Message to paste

> Read `PAPER_CONTEXT.md` first and follow §0 on every reply. The spec and the updates in Part 3 of
> `RESULTS_BRIEF_MEASUREMENT.md` override `PAPER_CONTEXT.md` and the current `main.tex` wherever they
> differ. Write in paper voice: no "Part A/B/C", "arm", "H1", "spec", "CCAST" or "ledger". Return
> drop-in LaTeX.
>
> **Task 1: update Sections III-V** to match Part 3, section "What changed in the method and setup".
> Only change what Part 3 says changed. Keep everything else. In particular:
>
> - Remove the class-targeted study (Section IV, "Class-Targeted Study"). It was not run. Move one
>   sentence about it to the limitations.
> - Make the epoch-budget rule match the one we used, and say plainly that it replaced the original rule
>   after the first RDD run.
> - Name RetinaNet as the second detector.
> - Fill in the estimator details: the shared stratified reference batch, the frozen backbone and float32
>   passes, the smoothing, and the warm-up count for the harm block.
> - Fill in the fixed settings and how they were chosen on validation, with the tuning numbers.
> - Fill in the dacl10k classes and splits, and the reported estimator overhead.
> - Replace every `\todo{}` that Part 3 answers. Leave the others.
>
> **Task 2: write Section VI.A, "Do augmentation effects depend on the class?"**, using Part 3, section
> "Measurement results". It needs:
>
> - one table of interaction tests for all six dataset × detector pairs;
> - one figure, a 2×3 grid of the heatmaps, or two heatmaps plus a reference to the rest in an appendix,
>   your choice;
> - a short paragraph per finding listed in Part 3.
>
> Report what the numbers show, including that the RetinaNet results do not show the effect on RDD and
> dacl10k. Do not explain the RetinaNet difference with a mechanism we did not test; offer at most one
> hedged possibility. Add the short-vs-full-run agreement check with its wide interval.
>
> Leave Section VI.B (the policy comparison) as a `\todo{}` with the table and figure placeholders listed
> in Part 3. After the LaTeX, list the open `\todo`s and any claim you were unsure about.

## 3. Data

### What changed in the method and setup

1. **Second detector.** RetinaNet ResNet-50-FPN (torchvision, COCO-pretrained). YOLOv8 is not used.

2. **Epoch budget rule (changed).**
   - The original rule compared the final epoch with the epoch at 80% of training. It failed on RDD only
     because of the jump after the last learning-rate drop.
   - The rule used compares the final epoch with the first epoch at the final learning rate. A budget
     passes when the gain is below 0.3 AP50.
   - Budgets: RDD 20 epochs (51.05 vs 50.96), VOC 20 epochs (81.57 vs 81.59), dacl10k 30 epochs (curve
     flat near 20.0 AP50 over the last 6 epochs).
   - LR schedule: ×0.1 from epoch round(0.6E)+1 and ×0.01 from round(0.8E)+1.

3. **dacl10k (final).**
   - 7,910 labelled images. Splits: train 4,609, reference 1,000, val 1,160, test 1,141.
   - 1,012 official-test images have no labels and are skipped.
   - 12 damage classes are kept: Crack, Wetspot, Efflorescence, Rust, Rockpocket, Hollowareas, Cavity,
     Spalling, Graffiti, Weathering, Restformwork, ExposedRebars.
   - Alligator crack (ACrack) is excluded because it has 282 training boxes, below the 300 gate.
   - Co-occurrence: 75% of target-bearing training images contain ≥ 2 classes.
   - No edit is excluded for dacl10k, so its menu has 13 edits including vertical flip. RDD and VOC have
     12 edits.

4. **VOC (final).**
   - Splits: train 13,694, reference 1,206, val 1,651, test 4,952 (VOC07 test).
   - Co-occurrence 35%.
   - Near-duplicate screen (64-bit difference hash, Hamming distance ≤ 4): 20 train-vs-test and 2
     val-vs-test pairs found among 4,952 test images. They were reported, not removed: the official VOC
     splits are kept as published so results stay comparable with other VOC work. That is under 0.5% of
     the test set; mention it in one sentence.

5. **RDD2022.** Co-occurrence 35%. Classes D00, D10, D20, D40.

6. **Estimator details as implemented.**
   - **Reference batch.** Every K steps, one shared reference batch of 64 images is drawn from the
     reference set, stratified so that every class appears at least once.
   - **Per-class loss.** For each class c, h_c is the gradient of the loss restricted to proposals
     (Faster R-CNN) or anchors (RetinaNet) matched to class-c boxes in that batch. The spec described a
     separate 64-image subset per class; the shared batch is what was used.
   - **Estimation batch.** 8 training images, the unedited version and each edit, with the same random
     draws so the comparison is paired.
   - **Parameters.** θ_s is:
     - Faster R-CNN: the ROI box head and predictor plus the top FPN level.
     - RetinaNet: the classification and regression heads plus the top FPN level.

     Every other parameter is frozen during estimation, and the estimator never changes the model.
   - **Precision.** Estimation passes run in float32 even though training uses mixed precision. Under
     float16 the summed class losses overflowed and gave non-finite utilities.
   - **Smoothing.** Exponential moving mean and variance, β = 0.9. The first estimate initialises the
     mean. se = sqrt(var·(1−β)/(1+β)).
   - **Harm block.** Active only after 3 estimates of a cell.
   - **Start.** Estimation starts after the first epoch; the first epoch uses the uniform policy.
   - **Measured-table policy.** Uses Part A's seed mean ΔAP50 and its seed standard error (s.d./√3) in
     place of ū and se, with the same scoring, harm block and sampling.

7. **Fixed settings, tuned on validation** (RDD, Faster R-CNN, seed 0, full schedule).
   - Score: mean validation macro AP50 over the 4 epochs at the final learning rate.
   - Overhead: estimator share of epoch wall time.
   - Rule, fixed before the results were read: a setting replaces the default only if it gains ≥ 0.5 AP50
     and its overhead is ≤ 20%.

   | Setting | Score | Overhead |
   |---|---|---|
   | K=200, head+FPN, ε=0.1 (default) | 49.61 | 15.3% |
   | K=100 | 49.84 | 26.7% (excluded) |
   | K=400 | 49.66 | 8.2% |
   | head only | 49.81 | 13.0% |
   | ε=0.05 | 49.78 | 15.0% |
   | ε=0.2 | 49.78 | 15.2% |

   No setting beat the default by 0.5 (largest gain 0.24), so the defaults were kept: K = 200,
   θ_s = head + top FPN level, ε = 0.1. The method is not sensitive to these choices in this range.

8. **Overhead.** At K = 200 the estimator takes about 15% of epoch wall time (training plus validation),
   which is about 18% extra relative to the rest of the epoch. One estimate takes about 8.6 s on an A10.

9. **Ablations.** No harm block, no identity option, inverse-AP class weights. Validation only; RDD and
   dacl10k; Faster R-CNN; 3 seeds. The tolerance-gated controller ablation and the class-targeted study
   were not run. List both as limitations.

10. **Statistics for the comparison** (for Section V's metrics paragraph).
    - Paired seeds.
    - A 95% seed-level paired t interval and a 95% paired image bootstrap: 1,000 resamples of test
      images, with the same images for every arm and seed, averaging the per-seed differences.
    - A claim requires both intervals to exclude 0 in the claimed direction.
    - "Best simple baseline" = base or uniform, whichever has the higher mean test macro AP50.
    - No correction across the six dataset × detector comparisons; say so.
    - 5 seeds if the H2 interval spans 0 and is wider than 1.0 AP50.
    - Test set used once, after all settings were fixed.

### Measurement results (edit-wise study, validation, 3 seeds, per-class ΔAP50 = edit − base)

**Interaction test** (two-way, seed as block; H0: an edit's effect is the same for every class):

| Dataset | Detector | Classes | Edits | F | df | p |
|---|---|---|---|---|---|---|
| RDD2022 | Faster R-CNN | 4 | 12 | 3.73 | (33, 88) | 4.5e-7 |
| RDD2022 | RetinaNet | 4 | 12 | 0.79 | (33, 88) | 0.77 |
| VOC | Faster R-CNN | 20 | 12 | 1.60 | (209, 440) | 2.4e-5 |
| VOC | RetinaNet | 20 | 12 | 1.91 | (209, 440) | < 1e-6 |
| dacl10k | Faster R-CNN | 12 | 13 | 2.52 | (132, 288) | 4e-11 |
| dacl10k | RetinaNet | 12 | 13 | 1.06 | (132, 288) | 0.34 |

**Per-edit class contrasts with Holm correction (p_Holm < 0.05):**
- dacl10k × Faster R-CNN: mosaic (p_Holm = 0.0003) and contrast (0.002). Contrast is positive in all 3
  seeds for Crack and Hollowareas, and negative in all 3 seeds for Restformwork, Rust, ExposedRebars and
  Weathering.
- VOC × RetinaNet: mosaic (p_Holm < 0.001). Mosaic helps on average there (+1.82), positive in all seeds
  for 16 of 20 classes, but by very different amounts across classes (spread 7.6 AP50).
- No other single edit survives Holm correction.

**Findings to write up, one paragraph each:**
1. **Class dependence with Faster R-CNN.** With Faster R-CNN, the effect of an edit depends on the class
   in all three domains. With RetinaNet it does on VOC only.
2. **Some edits help one class and hurt another.** "Helps" means positive in all 3 seeds and "hurts"
   means negative in all 3 seeds; counts are classes:
   - dacl10k × Faster R-CNN: contrast helps 2, hurts 6; mosaic helps 3, hurts 6; sharpen helps 3, hurts 2.
   - VOC × Faster R-CNN: rotate helps 4, hurts 3; sharpen helps 3, hurts 3; hue helps 3, hurts 3.
   - dacl10k × RetinaNet: vertical flip helps 3, hurts 3, but its interaction is not significant.
3. **Most edits cost a little on average in the short run. Mosaic stands out:**
   - strongly negative on RDD (−3.6 Faster R-CNN, −2.9 RetinaNet);
   - mildly negative on dacl10k (−0.6, −0.5);
   - negative on VOC with Faster R-CNN (−0.85);
   - positive on VOC with RetinaNet (+1.8).

   No edit is reliably positive across all classes of a dataset.
4. **Short-run proxy agreement** (RDD, Faster R-CNN). Four edits were re-run with the full 20-epoch
   schedule. The Spearman correlation between short-run and full-run per-class ΔAP50 over 16 (edit,
   class) cells is ρ = 0.56, 95% bootstrap CI [0.04, 0.86]. That passes the pre-set 0.5 threshold, but
   the interval is wide. Treat the short-run sizes as indicative and the signs as moderately reliable.
5. **RetinaNet on RDD and dacl10k.** There is no significant interaction, and the class spreads are
   similar in size to seed noise. Report this as found. Do not claim why.

**Mean ΔAP50 over classes per edit (for a compact table if wanted; F = Faster R-CNN, R = RetinaNet):**

| Edit | RDD F | RDD R | VOC F | VOC R | dacl F | dacl R |
|---|---|---|---|---|---|---|
| mosaic | −3.55 | −2.93 | −0.85 | +1.82 | −0.64 | −0.51 |
| blur | −1.27 | −0.73 | −0.34 | −0.59 | −0.27 | −0.02 |
| rotate | −0.75 | −0.56 | −0.10 | −0.06 | +0.05 | +0.03 |
| jpeg | −0.67 | −0.33 | −0.29 | −0.19 | −0.36 | −0.05 |
| noise | −0.52 | −0.93 | −0.32 | −0.21 | −0.23 | +0.13 |
| translate | −0.44 | −0.63 | −0.22 | −0.09 | −0.03 | +0.18 |
| contrast | −0.29 | +0.08 | −0.32 | −0.10 | −0.12 | +0.07 |
| erase | −0.25 | −0.51 | −0.37 | −0.12 | −0.24 | +0.04 |
| hue | −0.21 | −0.31 | −0.10 | −0.04 | −0.16 | −0.10 |
| brightness | −0.17 | −0.70 | −0.26 | −0.05 | −0.01 | +0.13 |
| saturation | −0.01 | −0.24 | −0.13 | −0.03 | −0.10 | −0.08 |
| sharpen | +0.14 | −0.39 | −0.08 | −0.17 | −0.01 | +0.35 |
| vflip | n/a | n/a | n/a | n/a | −0.31 | +0.19 |

### Placeholders for Section VI.B (fill when the comparison finishes)

- **Table:** test macro AP50, mean ± s.d. over seeds, for Base, Uniform, Measured, Shared and
  Class-utility, for each of the 6 dataset × detector pairs. Generated as `main_table.tex`.
- **Figure:** forest plot of Class-utility − Shared and Class-utility − best simple baseline. Thin line =
  seed t interval, thick line = image bootstrap.
- **Figure:** estimator agreement, a scatter of the mean estimated utility against measured ΔAP50, with
  Spearman ρ and CI.
- **Table:** ablations (validation): no harm block, no identity, inverse-AP weights.
- **Text:** secondary metrics (AP50:95, AP by size, false-positive images on damage-free RDD images) and
  overhead.
