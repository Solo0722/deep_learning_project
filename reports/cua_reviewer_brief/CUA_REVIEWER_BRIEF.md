# Class-Utility Augmentation (CUA): complete brief for an external reviewer

This brief has everything needed to review CUA: the exact method as implemented, every setting, every result so far, what has already been tested and ruled out, and what we want from you. Where the paper draft (*Class-Aware Augmentation Selection for Multi-Class Object Detection*) and this brief disagree, **this brief is correct**. §11 lists the known draft errors.

Notation follows the draft: edit menu 𝒪, identity o₀, class set S(x) of an image, reference batch R, parameter subset θ_s, smoothed utility ū(o, c) with standard error se(o, c), and measured per-class effect Δ̄(o, c) from the edit-wise study.

---

## 1. Goal and hypotheses

**Question.** In multi-class object detection, does choosing each training image's augmentation from the classes the image contains beat dataset-level augmentation?

**Hypotheses as pre-registered:**
- **H1:** An edit's effect depends on the class (the edit × class interaction in the edit-wise study).
- **H2:** CUA beats *shared-utility* augmentation on test macro AP50. Shared is the same estimator and policy with class information pooled away.
- **H3:** CUA beats the best simple baseline (base or uniform, whichever is higher).
- **H4:** CUA's estimated utilities rank-agree with the measured per-class effects Δ̄(o, c).

**Short answer so far:** H1 holds for Faster R-CNN on all three datasets and for RetinaNet on VOC only. H2, H3 and H4 have failed so far for CUA v1 and v2, a per-class population-based variant and the offline measured-effect table (§7–§9).

---

## 2. Data

| Dataset | Classes | Train | Reference | Val | Test | Images with ≥ 2 classes (train) |
|---|---|---|---|---|---|---|
| RDD2022 | 4: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole | 25,863 | 1,000 | 5,760 | 5,760 | 35% |
| dacl10k | 12: Crack, Wetspot, Efflorescence, Rust, Rockpocket, Hollowareas, Cavity, Spalling, Graffiti, Weathering, Restformwork, ExposedRebars | 4,609 | 1,000 | 1,160 | 1,141 | 75% |
| PASCAL VOC (07+12 trainval → 07 test) | 20 | 13,694 | 1,206 | 1,651 | 4,952 | 35% |

- **Reference set:** about 1,000 training images removed from optimisation in **every** policy. Only the CUA estimators read it, and PBT and adaptive copy-paste use it for scoring. All policies therefore train on identical data.
- **RDD2022:** fixed split. Exact duplicates and near-duplicates (64-bit difference hash, Hamming distance ≤ 4) are kept within one partition.
- **dacl10k:** public Hugging Face copy `Voxel51/dacl10k` (FiftyOne export; licence CC BY-NC 4.0). The official test images have no labels and are skipped. Labelled images are grouped by near-duplicate hash and re-split 70/15/15. Each polygon becomes one box (min/max over all rings, clipped to the image), and degenerate polygons are dropped. Only damage classes are kept. The gate (≥ 300 training boxes, not region-like) removed ACrack (282 boxes).
- **VOC:** official splits. "Difficult" objects are excluded from the training targets and act as ignore regions only in evaluation. During training, nothing uses them, so they are effectively background. The near-duplicate screen found 20 train–test and 2 val–test pairs; they were reported, not removed.
- **Dataset gate:** a dataset is used only if ≥ 20% of its object-bearing training images contain ≥ 2 classes.

---

## 3. Detectors and training recipe (identical in every policy)

- **Detectors:** torchvision Faster R-CNN ResNet-50-FPN (v1) and RetinaNet ResNet-50-FPN (v1), both COCO-pretrained. All backbone stages are trainable and BatchNorm statistics are frozen.
- **Input:** letterbox to 640 × 640 with padding value 114.
- **Optimiser:** SGD, momentum 0.9, weight decay 5e-4, batch size 4, base LR 0.0025. Linear warm-up over 1,000 iterations from 0.001× the base LR.
- **Schedule:** LR × 0.1 from epoch round(0.6E)+1 and × 0.01 from round(0.8E)+1. Mixed precision.
- **Epoch budgets E:** RDD 20, VOC 20, dacl10k 30. Rule: the final epoch must gain less than 0.3 val macro AP50 over the first epoch at the final LR.
- **Inference:** score > 0.001, NMS at IoU 0.5, at most 100 detections per image.
- **Checkpoint:** validated every epoch, and the best val macro AP50 checkpoint goes to test.
- **Seeds:** {0, 1, 2}, shared across policies. They fix head initialisation, data order and augmentation draws (keyed RNG per seed, epoch and image index).
- **Hardware:** single NVIDIA A10 per run. A 20-epoch Faster R-CNN RDD run takes about 12–14 h.
- **Primary metric:** test macro AP50 (COCO protocol at IoU 0.5, averaged over classes with test objects).
- **Secondary metrics:** AP50:95, per-class AP50, AP by size, and (RDD) the fraction of object-free test images with a confident detection.

---

## 4. Augmentation

**Base recipe (every policy, never adapted):** horizontal flip with p = 0.5, then isotropic scale jitter U(0.8, 1.2) about the centre. A draw that pushes a box out of the canvas is redrawn (up to 10 tries).

**Edit menu:** one edit per image, at one moderate strength range. Strength is sampled fresh per application.

| Edit | Strength (from `cua/augment/*`) | RDD | VOC | dacl10k |
|---|---|---|---|---|
| translate | up to ±10% of 640 px per axis | ✓ | ✓ | ✓ |
| rotate | U(−10°, +10°), boxes → axis-aligned hull | ✓ | ✓ | ✓ |
| vflip | — | ✗ | ✗ | ✓ |
| brightness | factor U(0.7, 1.3) | ✓ | ✓ | ✓ |
| contrast | factor U(0.7, 1.3) | ✓ | ✓ | ✓ |
| saturation | factor U(0.5, 1.5) | ✓ | ✓ | ✓ |
| hue | shift U(−0.05, +0.05) of the hue circle | ✓ | ✓ | ✓ |
| blur | Gaussian, σ U(0.5, 1.5) px | ✓ | ✓ | ✓ |
| noise | Gaussian, σ = 0.03 × 255 | ✓ | ✓ | ✓ |
| jpeg | quality U{30..60} | ✓ | ✓ | ✓ |
| sharpen | factor U(1.5, 2.5) | ✓ | ✓ | ✓ |
| erase | random-noise rectangle, 2–10% of canvas, aspect 0.3–3.3, covers ≤ 50% of any box | ✓ | ✓ | ✓ |
| mosaic | 2 × 2 tiles at half scale; boxes < 30% visible dropped | ✓ | ✓ | ✓ |
| **valid edits** | | 12 | 12 | 13 |

- Geometric edits transform the boxes. A draw that loses a box is redrawn up to 10 times, after which the image is left unedited.
- Mosaic uses the policy's choice for the anchor image. The other three tiles are drawn uniformly from the training set.
- **Images with no objects:** in every policy they get identity with p = 0.5, otherwise a uniform menu edit.
- **No copy-paste in Part C.** It was added later as a separate pilot (§9.3).

---

## 5. The edit-wise measurement study ("Part A", Section IV of the draft)

**Protocol.**
- For each dataset, detector and seed: train a warm checkpoint for 3 epochs on the base recipe.
- Branch |𝒪| + 1 continuations of 3 epochs each: one base-only, and one per edit with the edit applied with p = 0.5 on top of the base recipe.
- LR is at base for the first two continuation epochs and ÷ 10 for the last.
- Branches share the checkpoint, optimizer state, data order, base draws and edit coin, so they differ only in the edit.
- Δ_{o,c,k} = AP50_c(o, k) − AP50_c(base, k), on **validation**. 258 runs in total.

**Interaction test.** Model Δ = μ + α_k + β_o + γ_c + (αγ)_{kc} + (βγ)_{oc} + e. F-test on (βγ). Per-edit class contrasts are Holm-corrected.

| Dataset | Detector | F | df | p |
|---|---|---|---|---|
| RDD2022 | Faster R-CNN | 3.73 | (33, 88) | 4.5e-7 |
| RDD2022 | RetinaNet | 0.79 | (33, 88) | 0.77 |
| VOC | Faster R-CNN | 1.60 | (209, 440) | 2.4e-5 |
| VOC | RetinaNet | 1.91 | (209, 440) | < 1e-6 |
| dacl10k | Faster R-CNN | 2.52 | (132, 288) | 4e-11 |
| dacl10k | RetinaNet | 1.06 | (132, 288) | 0.34 |

- **Holm-significant single edits:** dacl10k FRCNN mosaic (p_Holm = 0.0003) and contrast (0.002); VOC RetinaNet mosaic (< 0.001). Contrast on dacl10k FRCNN is positive in all 3 seeds for Crack and Hollowareas and negative in all 3 for Restformwork, Rust, ExposedRebars and Weathering.
- **Edits that help some classes and hurt others** (all 3 seeds same sign):
  - dacl10k FRCNN: contrast (helps 2 classes, hurts 6), mosaic (3, 6), sharpen (3, 2).
  - VOC FRCNN: rotate (4, 3), sharpen (3, 3), hue (3, 3).
- **No edit is reliably positive across all classes of any dataset.**

**Mean ΔAP50 over classes, per edit** (F = Faster R-CNN, R = RetinaNet):

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

**RDD2022, per class** (mean ΔAP50 over 3 seeds):

| Edit | FRCNN D00 | D10 | D20 | D40 | RetinaNet D00 | D10 | D20 | D40 |
|---|---|---|---|---|---|---|---|---|
| blur | −0.67 | −1.20 | −1.25 | −1.98 | −0.23 | −0.76 | −0.79 | −1.15 |
| brightness | −0.15 | −0.46 | −0.26 | +0.20 | −0.28 | −0.20 | −0.20 | −2.10 |
| contrast | −0.04 | −0.46 | −0.23 | −0.45 | +0.20 | −0.34 | +0.70 | −0.22 |
| erase | −0.24 | −0.36 | −0.20 | −0.19 | +0.05 | −0.44 | −0.85 | −0.80 |
| hue | −0.07 | −0.35 | −0.37 | −0.07 | +0.36 | −0.34 | −0.29 | −0.97 |
| jpeg | −1.03 | −0.66 | −0.49 | −0.51 | −0.46 | −0.46 | +0.39 | −0.79 |
| mosaic | −2.37 | −3.74 | −3.61 | −4.48 | −1.31 | −4.53 | −2.06 | −3.82 |
| noise | −0.83 | −0.66 | −0.35 | −0.22 | −0.83 | −0.46 | −1.00 | −1.45 |
| rotate | −0.58 | −1.63 | −0.27 | −0.52 | −0.36 | −1.26 | +0.05 | −0.66 |
| saturation | +0.14 | −0.17 | −0.05 | +0.07 | +0.29 | −0.50 | −0.22 | −0.54 |
| sharpen | +0.45 | +0.30 | −0.30 | +0.09 | −0.25 | +0.06 | −0.18 | −1.21 |
| translate | −0.18 | −1.11 | −0.48 | −0.00 | +0.19 | −0.67 | −0.25 | −1.78 |

**Where the per-cell variance comes from** (decomposition of the seed-mean Δ̄ table; the residual includes seed noise, so it is an upper bound on the interaction):

| Pair | edit % | class % | residual % | edits with a class |t|>2 positive **and** another |t|>2 negative |
|---|---|---|---|---|
| RDD FRCNN | 87 | 2 | 11 | none |
| RDD RetinaNet | 59 | 19 | 22 | none |
| dacl10k FRCNN | 10 | 31 | 58 | 7 |
| dacl10k RetinaNet | 7 | 24 | 68 | 3 |
| VOC FRCNN | 6 | 36 | 58 | 12 |
| VOC RetinaNet | 30 | 30 | 40 | 9 |

**Key reading.**
- On RDD, harmful edits hurt **every** damage class, and blur hurts the pothole (D40) most. Class-specific selection has little to exploit there beyond dropping the bad edits globally.
- Real class conflict exists on VOC and dacl10k with Faster R-CNN. Even there, non-mosaic per-cell effects are about ±1–3 AP50, which is close to seed noise.

**Short-vs-full-run proxy check** (RDD, Faster R-CNN): 4 edits were re-run for the full 20 epochs. Spearman ρ between short-run and full-run per-class Δ is 0.56 over 16 cells, 95% bootstrap CI [0.04, 0.86]. Signs are moderately reliable; sizes are indicative only.

**Not run:** the class-targeted study (edit applied only inside one class's boxes) that the draft describes.

---

## 6. CUA v1 (the method in the draft)

**Estimator** (every K optimizer steps, starting after epoch 1):
- **Estimation batch B:** 8 random training images, rendered unedited and with every edit. The same base draws and seeds are used, so the comparison is paired.
- **Reference batch R:** 64 images from the reference set, stratified so every class appears at least once.
- **Per-class loss L_c(R):** classification + box loss over proposals (FRCNN) or foreground anchors (RetinaNet) matched to class-c GT. Background terms are excluded.
- **h_c = ∇_{θ_s} L_c(R):** one backward pass per class.
- **g₀, g_o:** gradients of the **full pooled training loss** on B (unedited / edited) w.r.t. θ_s.
- **u(o, c) = η ⟨g_o − g₀, h_c⟩ / (L_c(R) + δ)**, with η the current training LR and δ = 0.01.
- Every class with an object in R is updated every round, whether or not it is present in B.
- **θ_s:**
  - FRCNN: RoI box head (TwoMLPHead) + box predictor + top FPN level (`inner_blocks[-1]`, the C5 lateral conv; `layer_blocks[-1]`, the P5 output conv).
  - RetinaNet: classification and regression heads + the same top-FPN modules.
  - All other parameters are frozen during estimation. Estimation never changes the model.
- **Precision:** estimation passes always run in float32 (fp16 overflowed to NaN).
- **Smoothing:** EMA mean and variance, β = 0.9. The first estimate initialises the mean. se = √(var·(1−β)/(1+β)).

**Policy** (per image, from S(x)):
- **Score:** s(o | S) = mean over c ∈ S of ū(o, c), with equal weights. Identity scores 0.
- **Harm block:** drop edit o if, for any c ∈ S with ≥ 3 estimates, ū + 1.64·se < 0. Identity is never blocked.
- **Sampling:** p = (1 − ε)·softmax(s/τ) + ε/|A(S)|, with ε = 0.1, τ = std of the allowed scores (floor 1e-6). Before any estimate exists the policy is uniform over 𝒪 ∪ {o₀}.
- **Tuned once** (RDD, FRCNN, seed 0, validation; rule: change a default only for a gain ≥ 0.5 AP50 at ≤ 20% overhead). No setting qualified, so the defaults K = 200, θ_s = head + top FPN and ε = 0.1 were kept:

  | setting | score | overhead |
  |---|---|---|
  | default (K = 200) | 49.61 | 15.3% |
  | K = 100 | 49.84 | 26.7% |
  | K = 400 | 49.66 | 8.2% |
  | head only | 49.81 | 13.0% |
  | ε = 0.05 | 49.78 | — |
  | ε = 0.2 | 49.78 | — |

**Compared policies (Part C):** all on the base recipe, 3 seeds, test evaluated once.
- **base:** no edit.
- **uniform:** one edit uniform over 𝒪 ∪ {o₀}, TrivialAugment-style but with our fixed strength ranges.
- **measured:** the CUA policy with ū, se replaced by the Part A table Δ̄(o, c) and its seed standard error (s.d./√3), fixed for the whole run. This is the "oracle-ish" offline policy.
- **shared:** the same estimator, with the per-class u pooled to one value per edit before the EMA; the same harm block on the pooled estimate.
- **cua:** as above.

**Statistics:**
- Paired by seed.
- Two 95% intervals: a seed-level paired t interval, and a paired image bootstrap (1,000 test-image resamples, same images for every arm and seed).
- A **claim** needs both intervals to exclude 0 in the claimed direction.
- No multiplicity correction across the 6 pairs.

---

## 7. Part C results, v1 (test macro AP50, mean ± s.d. over 3 seeds)

| Pair | base | uniform | measured | shared | cua |
|---|---|---|---|---|---|
| RDD FRCNN | 51.15 ± 0.24 | 50.55 ± 0.03 | **51.47 ± 0.15** | 49.86 ± 0.19 | 50.13 ± 0.26 |
| RDD RetinaNet | **44.56 ± 0.22** | 43.82 ± 0.45 | 44.55 ± 0.47 | 43.32 ± 0.24 | 43.57 ± 0.13 |
| dacl10k FRCNN | 20.89 ± 0.12 | **22.37 ± 0.08** | 21.54 ± 0.09 | 20.53 ± 0.03 | 20.50 ± 0.27 |
| dacl10k RetinaNet | 9.26 ± 1.29 | 7.70 ± 0.85 | **9.80 ± 1.03** | 7.74 ± 1.33 | 8.66 ± 0.86 |
| VOC FRCNN | 85.60 ± 0.08 | **86.54 ± 0.10** | 86.30 ± 0.08 | 86.37 ± 0.06 | 86.49 ± 0.12 |
| VOC RetinaNet | 82.38 ± 0.16 | 83.44 ± 0.05 | 83.76 ± 0.24 | **84.08 ± 0.29** | 83.71 ± 0.16 |

Uniform is *below* base on RDD (both detectors) and on dacl10k RetinaNet.

**Paired comparisons** (difference = second − first; [t interval] / [bootstrap interval]):

| Pair | H2: cua − shared | H3: cua − best baseline | cua − measured |
|---|---|---|---|
| RDD FRCNN | +0.27 [−0.13, 0.67] / [0.07, 0.50] | −1.02 vs base [−1.20, −0.84] / [−1.48, −0.52] | −1.34 [−2.05, −0.62] |
| RDD RetinaNet | +0.24 [−0.44, 0.93] / [−0.01, 0.50] | −0.99 vs base [−1.24, −0.73] | −0.98 [−1.87, −0.08] |
| dacl10k FRCNN | −0.02 [−0.72, 0.68] | −1.87 vs uniform [−2.63, −1.11] | −1.03 [−1.74, −0.32] |
| dacl10k RetinaNet | +0.92 [−4.40, 6.24] / [0.73, 1.14] | −0.60 vs base [−5.91, 4.70] | −1.14 [−4.91, 2.63] |
| VOC FRCNN | +0.13 [−0.08, 0.33] | −0.04 vs uniform [−0.35, 0.26] | +0.20 [0.08, 0.31] / [−0.10, 0.48] |
| VOC RetinaNet | −0.37 [−0.79, 0.05] / [−0.55, −0.19] | **+0.28 vs uniform [0.02, 0.53] / [0.06, 0.50] (claim)** | −0.05 [−0.26, 0.16] |

**Measured policy vs best baseline:** no significant win on any pair. It is significantly *worse* than uniform on dacl10k FRCNN (−0.84 [−1.27, −0.41]) and VOC FRCNN (−0.24 [−0.45, −0.04]).

**H4** (Spearman ρ, run-mean ū vs Δ̄, over all cells, 95% bootstrap CI):

| Pair | ρ | 95% CI |
|---|---|---|
| RDD FRCNN | **−0.67** | [−0.82, −0.45] (anti-correlated) |
| RDD RetinaNet | −0.12 | [−0.42, 0.17] |
| dacl10k FRCNN | −0.12 | [−0.28, 0.05] |
| dacl10k RetinaNet | −0.01 | [−0.18, 0.16] |
| VOC FRCNN | −0.20 | [−0.33, −0.06] |
| VOC RetinaNet | +0.10 | [−0.05, 0.23] |

**Ablations** (validation, Faster R-CNN, RDD and dacl10k, 3 seeds; difference vs cua): none is distinguishable from cua.

| Ablation | RDD | dacl10k |
|---|---|---|
| no harm block | −0.12 [−0.61, 0.38] | +0.07 [−0.40, 0.55] |
| no identity option | +0.17 [−0.09, 0.43] | +0.15 [−0.43, 0.74] |
| inverse-AP class weights | −0.02 [−0.62, 0.58] | +0.00 [−0.39, 0.40] |

**Overhead:** the estimator takes 14–21% of epoch wall time (about 16–26% extra). One estimate takes about 8.6 s on an A10.

**Secondary AP50:95 (test), v1 arms:**

| Pair | base | uniform | measured | shared | cua |
|---|---|---|---|---|---|
| RDD FRCNN | 22.70 | 22.41 | 22.83 | 21.94 | 22.05 |
| RDD RetinaNet | 19.66 | 19.35 | 19.65 | 19.00 | 19.05 |
| dacl10k FRCNN | 9.39 | 10.39 | 9.95 | 9.55 | 9.58 |
| dacl10k RetinaNet | 4.32 | 3.60 | 4.56 | 3.60 | 4.02 |
| VOC FRCNN | 57.75 | 58.94 | 58.66 | 59.06 | 59.18 |
| VOC RetinaNet | 57.24 | 58.16 | 58.16 | 58.66 | 58.34 |

---

## 8. Why v1 failed: the estimator diagnosis

Offline, `cua/analysis/utility_check.py` recomputed estimator quantities on saved checkpoints: one warm Part A checkpoint and one Part C cua best checkpoint per pair, 20 rounds each. It scored several variants against the Part A table.

**Variants:**
- `dot`: the v1 utility.
- `normed`: g_o rescaled to ‖g₀‖.
- `cosdiff`: cos(g_o, h_c) − cos(g₀, h_c).
- `ratio`: ‖g_o‖/‖g₀‖.

**Findings:**
1. **No class signal within an edit.** Spearman ρ between estimate and Δ̄, computed within each edit across classes, is ≈ 0 for every variant and pair (range −0.21 to +0.09). Per-cell utilities are mostly noise: |t| > 2 in only about 5–30% of cells, and signs flip between the two checkpoints.
2. **Gradient magnitude is not the problem.** ‖g_o‖/‖g₀‖ ≈ 1 for all edits except mosaic (0.99–1.02 against 1.26 on dacl10k FRCNN). Norm-free variants do no better.
3. **Mosaic is the top-scored edit** under the v1 utility in 7 of 12 checks, though it is the most harmful edit in Part A on RDD and dacl10k. It packs about 4× the objects into the batch, so its pooled gradient carries more signal toward h_c.

---

## 9. What was tried after v1

### 9.1 CUA v2 (arms `shared2`, `cua2`)

Four fixes, aimed at noise and mosaic:
1. **Lookahead estimator.** Instead of first-order alignment, take **4 real SGD steps** on θ_s, at 10× the training LR with grad-norm clip 5, using the clean or the edited 16-image batch. Then measure each class's reference loss: u(o, c) = (L_c after clean steps − L_c after edited steps)/(L_c before + δ). θ_s is restored afterwards.
2. **Instance normalisation.** Each edited step's gradient is scaled by (#objects clean / #objects edited), applied after the clip. This removes mosaic's object-count advantage.
3. **Less noise.** Estimation batch 16, reference batch 32, K = 400, β = 0.98. A cell counts only once |ū| ≥ 2·se; otherwise it scores 0.
4. **Prior.** A cheap measured prior (Part A ΔAP50 after the first continuation epoch) is combined with the online estimate on the z scale (Stouffer).

**Offline check of the lookahead alone:** within-edit ρ is −0.27 to +0.15, and mosaic is still top-scored in 5 of 6 pairs.

**Test macro AP50 (mean ± s.d.):**

| Pair | shared2 | cua2 | (cua v1) |
|---|---|---|---|
| RDD FRCNN | 50.63 ± 0.14 | 50.88 ± 0.13 | 50.13 |
| RDD RetinaNet | 43.92 ± 0.38 | 43.92 ± 0.22 | 43.57 |
| dacl10k FRCNN | 22.42 ± 0.29 | 21.49 ± 0.22 | 20.50 |
| dacl10k RetinaNet | 7.01 ± 1.75 | 8.23 ± 0.82 | 8.66 |
| VOC FRCNN | 85.99 ± 0.51 | 86.46 ± 0.24 | 86.49 |
| VOC RetinaNet | 83.91 ± 0.24 | 83.96 ± 0.13 | 83.71 |

- **cua2 − cua:** significant gains on 4 of 6 pairs: RDD FRCNN +0.75, RDD RetinaNet +0.35, dacl10k FRCNN +0.99, VOC RetinaNet +0.25.
- **cua2 − shared2:** a claim only on RDD FRCNN (+0.25 [0.21, 0.29] / [0.03, 0.46]). On dacl10k FRCNN, cua2 is *worse* (−0.92).
- **cua2 − best baseline:** a claim only on VOC RetinaNet (+0.52 [0.30, 0.74] / [0.26, 0.78]). It is worse on dacl10k FRCNN (−0.88 vs uniform) and RDD RetinaNet (−0.64 vs base). RDD FRCNN is −0.27, not significant.
- **cua2 − measured:** no claim on any pair.

The prior and the gating mostly pull cua2 toward the measured table, so v2's gains over v1 do not show that the online class estimate adds information.

### 9.2 Per-class population-based training (PBT) pilot

- **Setup:** RDD, Faster R-CNN, seed 0. Four members, each with a logit table (1 + |𝒪|) × C. The policy is softmax of the mean over c ∈ S, with ε = 0.1.
- **Schedule:** 20 epochs cut into 10 windows of 2 epochs.
- **Selection after each window:**
  - Every member restarts from the member with the best reference macro AP50.
  - For each class c, every member takes column c from the member with the best reference AP50 for c.
  - Members 1–3 then add N(0, 1.0) noise.
- **Result (validation):** final val macro AP50 50.75, against base 50.72 and measured 51.04 on the same seed.
- **Per-class winners:**
  - They mostly follow whichever member is best overall: 64% agreement, against 25% expected by chance.
  - The unperturbed member 0 wins a class at chance rate (22%).
  - Weight-donor members across windows: 1, 3, 1, 3, 3, 2, 3, 0, 2, 3.
  - Reading: per-class AP differences between members are noise at this scale, and there is no stable per-class preference.

### 9.3 Class-adaptive copy-paste (CAP), implemented, no results yet

- **Mechanism:** a different lever. Objects are cut from training images (box rectangles, from a per-class bank) and pasted into other training images, *before* the base geometry and the uniform edit policy.
  - 1–3 objects per image with probability 0.5.
  - Placements cover ≤ 30% of any existing box, with a 4-px feather.
  - Each paste is at most half the content width and half the region height.
  - On RDD, pastes go into the lower half of the image (road region).
- **Arms:** `cprand` (objects at class frequency), `cpbal` (uniform over classes), `cpada` (class probabilities from the current reference-split AP50, lower AP → more pastes, ε floor 0.2, refreshed each epoch). The control is the Part C uniform run.
- **Status:** pilot specs and jobs exist, and no results are in the repo yet.

---

## 10. What is ruled out, and what is still open

**Ruled out by data:**
- The harm block starving multi-class images (null ablation).
- Collapse toward "no edit" (null no-identity ablation; mosaic is over-rated, not under-rated).
- Gradient-norm inflation (ratio ≈ 1 except mosaic).
- One-step first-order error as the main cause (the multi-step lookahead still has no class signal).
- Class weighting (null inverse-AP ablation).
- Per-class selection by direct population search at the 2-epoch, 4-member scale (PBT pilot).

**Structural facts a redesign has to live with:**
- **RDD:** edit harms are shared by all classes (87% of variance is the edit main effect), so the best class-aware policy is about as good as "drop the bad edits".
- **Sizes:** class-specific effects, where they exist (VOC, dacl10k with FRCNN), are about 1–3 AP50 per cell, near seed noise. The edit menu is moderate. Only mosaic, and on RDD blur, move AP by more than about 2.
- **Baseline ordering:** the strongest baseline varies by pair. Base wins on RDD and dacl10k RetinaNet, uniform on dacl10k FRCNN and VOC. Even the offline measured table does not beat the best baseline significantly anywhere.
- **Statistics:** 3 seeds; differences under about 0.3–0.5 AP50 are below what these tests resolve.

**Open questions we would like a view on:**
1. Is there a class-aware augmentation mechanism worth testing given these facts? If so, what is the cheapest diagnostic that could kill it before we build it (an oracle using Part A, as in our measured arm)?
2. Should the paper reposition as a measurement study? (Class dependence exists for FRCNN, but online per-class selection, including an offline oracle table, does not beat simple baselines.) Or is there a method contribution still reachable?
3. Is the evaluation design (paired seeds, double interval, no multiplicity correction) sound for the claims we can make?

**Constraints for any proposal:**
- Single A10 GPUs; about 12–14 h per 20-epoch FRCNN RDD run.
- Every new arm costs 3 seeds × 6 pairs unless scoped down.
- torchvision detectors. There is no ignore-region support in the training losses, and adding it needs changes to RPN matching, RoI sampling and the RetinaNet loss.

---

## 11. Known errors in the paper draft you may have seen

- dacl10k licence is **CC BY-NC 4.0**, not CC BY 4.0.
- "26,863 training images" for RDD includes the 1,000 reference images. The training split is 25,863.
- The **class-targeted study** (Section IV-D) and the **tolerance-gated ablation** were **not run**.
- Section III describes v1 only. v2, PBT and copy-paste (§9) are not in the draft, and neither are any results.
- θ_s for RetinaNet also includes the top FPN level (the draft says classification and regression subnets only).
- The estimator starts after epoch 1; the first epoch uses the uniform policy.
- The Fig. 2 utility-table colours are illustrative, not measured.

---

## 12. Files, if you need more

| What | Path |
|---|---|
| Design spec | `docs/superpowers/specs/2026-11-08-class-utility-augmentation-design.md` |
| Estimators | `cua/estimator/utility.py` (v1), `lookahead.py` (v2), `class_loss.py`, `params.py` |
| Policies | `cua/policy.py`; menu `cua/augment/menu.py`; edits `cua/augment/{pixel,geometric,mosaic,base,paste}.py` |
| Training and configs | `cua/train.py`, `cua/experiments/specs.py`, `cua/pbt.py` |
| Part A data | `reports/cua_part_a_report/` (`deltas.csv` per seed / edit / class, `tests.json`, heatmaps) |
| Part C v1 / v2 | `reports/cua_part_c_report/`, `reports/cua_part_c_v2_report/` (`results.json`, `main_table.csv`, `comparisons.csv`) |
| Estimator diagnosis | `reports/cua_utility_check/`, `reports/cua_utility_check_v2/` |
| PBT pilot | `reports/cua_pbt_pilot/` (`decisions.jsonl`, `final.json`) |
