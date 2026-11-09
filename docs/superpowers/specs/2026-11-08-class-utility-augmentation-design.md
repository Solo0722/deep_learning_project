# Class-Utility Augmentation for Multi-Class Object Detection — Study Design

Date: 2026-11-08 · Status: approved in conversation, awaiting written-spec review · Supersedes: the v1/v2 tolerance-gated controllers (historical evidence only)

## 1. Purpose

Find out whether different object classes in multi-class detection benefit from different augmentations, and, if they do, exploit it with a method that picks augmentations by **measured per-class training benefit** rather than by detector tolerance.

The study is designed so that either outcome is publishable:

- If classes differ: evidence of class-dependent augmentation benefit in detection + a method that exploits it + proof it beats a shared policy.
- If classes do not differ: a controlled, multi-domain negative finding with the measurement protocol that establishes it.

### Why the change from v1/v2

| Old problem | Fix in this design |
|---|---|
| Controller measured tolerance (does the detector still see the object), not training benefit | Benefit = predicted reduction in a class's reference-set loss from learning on the edited vs unedited image |
| Controller always applied an edit | "No edit" is always a candidate with benefit 0 |
| Edit menu had no net-positive operations | Menu includes operations known to help detection (mosaic, translation, occlusion, colour) |
| Premise (classes want different edits) never measured | Part A measures it directly before any method claim |
| One seed, differences within noise | 3 paired seeds, image-bootstrap intervals, pre-declared primary claim |
| Road damage only | Three domains, two detector families |

## 2. Claims and hypotheses

Paper framing: *class-adaptive augmentation for multi-class object detection*. Road damage is one of three domains, not the subject.

- **H1 — heterogeneity.** For at least one edit, the effect on per-class AP50 differs across classes (significant edit × class interaction; or effect reliably positive for one class and reliably negative for another).
- **H2 — method (primary claim).** The class-utility policy beats the shared-utility policy on test-set macro AP50.
- **H3 — practical value.** The class-utility policy beats the best of {base recipe, uniform-menu baseline} on test macro AP50.
- **H4 — estimator validity.** The method's online benefit estimates rank-agree with Part A's measured per-class AP changes (Spearman ρ > 0, CI excluding 0).

Claims we will not make: first class-adaptive augmentation; first online augmentation for detection; first use of validation/benefit signals; first object-aware augmentation (all have prior art — see the Nov 3 proposal, §4–5).

## 3. Datasets

| Domain | Dataset | Why | Splits |
|---|---|---|---|
| Road damage | RDD2022, classes D00/D10/D20/D40 | Thin shape-defined classes (cracks) beside blob-like (potholes); 35% of target-bearing images contain ≥2 classes (measured on frozen train/val) | Existing frozen `detection_splits_v1` (26,863 / 5,760 / 5,760) |
| Bridge damage | dacl10k (9,920 images, >100 bridges), damage classes only, polygons → boxes | Colour-defined classes (rust, efflorescence) beside shape-defined (cracks) | Official splits if test labels are available; otherwise our own group split (by bridge if metadata allows, else image-level with near-duplicate screening via `screen_near_duplicates.py`), 70/15/15 |
| General objects | PASCAL VOC (07+12 trainval → train/val; VOC07 test → test) | Known benchmark; frequent co-occurrence | Val = 10% of trainval, held out |

**Reference set (method only).** From each training split, hold out ~1,000 images (cluster/group-disjoint from the rest of train), stratified so every class has ≥100 objects where possible. Removed from training in **every** arm so arms see identical training data.

**Data gates (run once after download, before any training):**

1. **Co-occurrence gate:** ≥20% of target-bearing training images contain ≥2 classes. Failing dataset is replaced by CODEBRIM (if it passes), else the study runs on the remaining domains.
2. **dacl10k box conversion:** one box per connected polygon component per class. Classes with <300 training boxes, or whose boxes routinely cover >50% of the image (region-like "stuff"), are excluded and listed in the paper. Licence (non-commercial academic use) recorded.
3. **Leakage check:** near-duplicate screen between splits for the new datasets, as was done for RDD.

## 4. Data roles

| Data | Used for | Never used for |
|---|---|---|
| Train (minus reference set) | Detector optimisation | — |
| Reference set | Method's online benefit estimates (all online arms) | Training, checkpoint selection |
| Validation | Part A/B measurements, measured-table policy, ablations, hyper-parameter choices, checkpoint selection | Final claims |
| Test | Final H2/H3 evaluation, opened once after all choices are frozen | Anything else |

## 5. Detectors and training recipe

- **Faster R-CNN ResNet-50-FPN** (torchvision, COCO-pretrained) — primary.
- **RetinaNet ResNet-50-FPN** (torchvision, COCO-pretrained) — second family; matches the detectors reported by Zoph et al. and Scale-Aware AutoAug.
- 640×640 letterbox for all datasets, SGD (momentum 0.9, wd 5e-4), batch 4, linear LR warm-up, step decay at ~60% and ~80% of training.
- **Epoch budget per dataset** is fixed once, before Part C, from a base-recipe run: the smallest budget at which base-recipe validation AP50 has plateaued (gain < 0.3 AP50 over the last 20% of epochs). RDD starts from the existing 20-epoch schedule.
- **Mixed precision on**, TF32 allowed, no strict determinism mode. Repeatability comes from seeds, not bitwise equality.
- **Seeds:** {0, 1, 2}, shared across arms (head init, data order, augmentation RNG). Escalate to 5 seeds for the H2 comparison only if its 95% CI spans zero and its width exceeds 1.0 AP50.

## 6. Augmentation menu

**Base (every arm, never adapted):** horizontal flip (p = 0.5) + scale jitter (0.8–1.2, isotropic) with box-preserving rejection (no annotated object cropped).

**Adaptive menu (one moderate strength each):**

| Family | Edit | Strength |
|---|---|---|
| Geometric | Translation | ±10% of canvas |
| Geometric | Rotation | ±10°, boxes = rotated-corner hull |
| Geometric | Vertical flip | where valid only |
| Colour | Brightness | ×[0.7, 1.3] |
| Colour | Contrast | ×[0.7, 1.3] |
| Colour | Saturation | ×[0.5, 1.5] |
| Colour | Hue | ±0.05 (HSV, fraction of circle) |
| Image quality | Gaussian blur | σ ∈ [0.5, 1.5] px |
| Image quality | Gaussian noise | σ = 0.03 (RGB/1) |
| Image quality | JPEG compression | quality ∈ [30, 60] |
| Image quality | Sharpening | factor ∈ [1.5, 2.5] |
| Occlusion | Random erasing | 2–10% of canvas, not covering >50% of any box |
| Composition | Mosaic | 4 images, 2×2, boxes clipped, objects <30% visible dropped |

**Validity rules (hard constraints, fixed before training):**

| Domain | Excluded edits | Reason |
|---|---|---|
| RDD2022 | Vertical flip; any rotation beyond ±10° | Vehicle-camera perspective; large rotation turns longitudinal (D00) into transverse (D10) cracks |
| dacl10k | none expected; reviewed against class definitions at data-gate time | Close-up surfaces are orientation-free |
| VOC | Vertical flip | Upside-down scenes are not label-preserving in practice |

Copy-paste is out of scope (no instance masks for RDD; recipient-region curation is a separate project).

## 7. Part A — one augmentation at a time (ground truth)

**Question:** what is the effect of each edit on each class's AP50?

- **Shared warm checkpoint** per dataset × detector × seed: base recipe for 3 epochs.
- **Arms:** base, and base + each valid menu edit (applied to the whole image with probability 0.5). ~14 arms.
- **Short run:** continue 3 epochs from the warm checkpoint, with LR decayed ×0.1 for the last epoch.
- **Output:** per-class ΔAP50 (edit − base) on validation, per seed → the **edit × class benefit table**.
- **Coverage:** 3 datasets × 2 detectors × 3 seeds.

**Proxy check:** for RDD × Faster R-CNN, train 4 edits (chosen to span the A ranking: best, worst, two middle) with the full schedule, 1 seed. Report Spearman agreement between short-run and full-run per-class Δ. If ρ < 0.5, short-run conclusions are reported as indicative only and Part C's measured-table arm uses full-run evidence where available.

## 8. Part B — one class at a time (cross-class effects)

**Question:** when an edit is applied only to class *c*'s objects, what happens to class *c* (direct effect) and to the other classes (spill-over through shared parameters)?

- **Edits:** the 3 edits from Part A with the largest class disagreement on each dataset.
- **Datasets:** RDD2022 and dacl10k, Faster R-CNN, 3 seeds.
- **Application:** the edit is rendered inside class-*c* boxes (union, 8-px feathered mask) with probability 0.5 on images containing *c*. Geometric edits are excluded from B (they cannot be applied locally without moving boxes).
- **Seam control:** for 1 edit, a matched arm applies it to a random non-object region of the same area, to separate seam artefacts from the edit's effect.
- **Output:** a direct/spill-over matrix per edit. Decides whether local per-class editing (Option 3) is worth adding later: only if spill-over or conflict is large and the whole-image policy cannot resolve it.

## 9. Part C — the method: class-utility augmentation

### 9.1 Benefit estimator

Let θ_s be a parameter subset (Faster R-CNN: ROI box head + top FPN level; RetinaNet: classification and regression subnets). Every K = 200 optimiser steps:

1. Draw an estimation batch B of 8 training images. For the unedited batch and for each valid edit o, compute g_0 = ∇θ_s L(B) and g_o = ∇θ_s L(T_o(B)), where L is the detector's full training loss and T_o includes the base geometry with the same random draw.
2. For each class c, compute h_c = ∇θ_s L_c(R_c), where R_c is a 64-image subsample of the reference set containing c, and L_c is the classification + box-regression loss restricted to proposals/anchors matched to class-c ground truth.
3. Instantaneous benefit (first-order predicted extra reduction in class-c reference loss from one SGD step on the edited vs unedited batch):

   u(o, c) = η · ⟨g_o − g_0, h_c⟩ / (L_c(R_c) + δ), δ = 0.01

4. Smooth: ū(o,c) ← β ū(o,c) + (1 − β) u(o,c), β = 0.9; track a running variance to give a standard error se(o,c).

Benefit estimation starts after 1 base-recipe epoch; until then the policy is uniform over the menu plus identity. Estimator overhead (GPU time and % over base) is logged; target ≤ 20%.

### 9.2 Policy for an image with class set S

- Candidates: identity (benefit 0 by definition) + valid menu edits.
- Score: score(o) = (1/|S|) Σ_{c∈S} w_c ū(o, c), with w_c = 1 (equal weights; inverse-AP weights are an ablation).
- Harm block: drop o if, for any c ∈ S, ū(o,c) + 1.64·se(o,c) < 0 (reliably harmful at one-sided 95%).
- Sample: p(o) = (1 − ε) softmax(score/τ) + ε / |allowed|, with τ = std of current scores (floored at 1e-6) and ε = 0.1.
- Images with no target objects: identity or a uniformly sampled menu edit (p = 0.5), identical in all arms.
- Mosaic: chosen per anchor image using that image's S; the other three tiles are drawn uniformly.

### 9.3 Shared-utility variant

Identical machinery, but ū_shared(o) = mean_c ū(o, c) and the harm block uses the pooled estimate. Every image gets the same preferences. **The only difference from the class version is class information.**

### 9.4 Fixed settings to tune on validation only (RDD × Faster R-CNN, 1 seed), then freeze

K ∈ {100, 200, 400}; θ_s ∈ {head only, head + top FPN}; ε ∈ {0.05, 0.1, 0.2}. Chosen by validation macro AP50, then frozen for all datasets and detectors.

## 10. Part C — main comparison

| Arm | Description |
|---|---|
| Base | Flip + scale jitter only |
| Uniform menu | Base + one edit sampled uniformly from {identity} ∪ menu (TrivialAugment-style) |
| Measured-table | Base + fixed policy from Part A's validation table, using the same scoring/harm/sampling rule as §9.2 with ū replaced by measured ΔAP50 |
| Shared-utility | §9.3 |
| Class-utility | §9.1–9.2 |

Coverage: 3 datasets × 2 detectors × 3 seeds, full training schedule, checkpoint chosen by best validation macro AP50, then evaluated once on test.

**Ablations** (validation; RDD and dacl10k; Faster R-CNN; 3 seeds): no harm block; no identity option; inverse-AP class weights; the old tolerance-gated controller (v2) as a "tolerance instead of benefit" baseline.

## 11. Metrics and statistics

- **Primary:** test macro AP50 (COCO evaluation at IoU 0.5, averaged over classes).
- **Secondary:** AP50:95, per-class AP50, AP small/medium/large, false-positive images on target-free images (RDD), estimator overhead.
- **Paired seeds** across arms; report mean ± s.d.
- **Intervals:** for every arm difference, a 95% CI from a paired image-level bootstrap (1,000 resamples of test images, same resample for both arms, per seed, then pooled) alongside the seed-level paired t interval. Claims require both to exclude 0.
- **H1 test:** per dataset × detector, two-way analysis of per-seed ΔAP50 with edit × class interaction (seed as block); per-edit class contrasts with Holm correction across edits.
- **H4 test:** Spearman ρ between time-averaged ū(o,c) and Part A ΔAP50(o,c), bootstrap CI over (o,c) cells.
- **Pre-declared:** H2 on test macro AP50 is the primary claim. Test set opened once, after §9.4 and all arm settings are frozen.

### Outcome rules

| Result | What the paper says |
|---|---|
| H1 true, H2 true | Class-dependent benefit exists and the method exploits it |
| H1 true, H2 false | Classes differ, but online estimation does not exploit it (check H4: estimator vs policy failure) |
| H1 false | Classes do not need different augmentation in these domains; class-utility ≈ shared-utility is the expected consistency check |
| H3 false | Reported plainly; the method is not recommended over simple recipes |

## 12. Compute budget (~800 GPU-hours, A10)

Planning estimates; replaced by measured numbers after the first base runs.

| Part | GPU-h |
|---|---|
| Data gates, base-recipe budget runs | ~30 |
| A (both detectors) | ~160 |
| Proxy check | ~30 |
| B | ~60 |
| §9.4 tuning | ~40 |
| C main comparison (both detectors) | ~460 |
| Ablations | ~60 |
| **Total** | **~840** |

**If compute runs short, priority order:** (1) A Faster R-CNN all datasets → (2) C main Faster R-CNN → (3) proxy check → (4) B → (5) C RetinaNet → (6) A RetinaNet → (7) ablations.

## 13. Code structure (detail in the implementation plan)

New package alongside the existing audited code; reuse the existing RDD loader, letterbox and COCO evaluation where they already pass tests.

| Unit | Responsibility |
|---|---|
| `datasets/` | One adapter per dataset → common record format (image, boxes, labels, group id); data gates |
| `augment/` | Base recipe, menu edits, validity rules per domain; every edit is a pure function (image, boxes, rng) → (image, boxes) |
| `estimator/` | Gradient-based per-class benefit estimate (§9.1) |
| `policy/` | Class-utility, shared-utility, measured-table, uniform policies behind one interface |
| `train/` | Common trainer for both detectors; arm configs; checkpoint/seed handling |
| `analysis/` | Per-class Δ tables, bootstrap CIs, interaction tests, figures |

Each unit is tested independently (box correctness for every edit, policy sampling invariants, estimator sign on a toy problem).

## 14. Paper mapping

| Section | Source |
|---|---|
| Introduction | Class conflict in multi-class images; tolerance ≠ benefit |
| Related work | Existing review, re-angled: tolerance-based (CTAugment, CUDA) vs benefit-based (BDA, MADAug, DDAS); class-aware (AdaAug, Yoo–Kang, Tsuchiya, instance-switching) |
| Measurement study | Parts A, B, proxy check |
| Method | §9 |
| Experiments | §10–11 |
| Discussion | Outcome rules, cost, limitations (box-level labels, single strength, no copy-paste) |

## 15. Risks

| Risk | Mitigation |
|---|---|
| RDD label noise hides small effects | Two other domains; AP50 primary; bootstrap + seeds |
| Estimator too noisy | Smoothing, harm block uses se, H4 diagnoses it explicitly |
| dacl10k polygon→box conversion gives odd boxes | Exclusion rule in §3, documented |
| Short-run proxy disagrees with full runs | Proxy check; fallback rule in §7 |
| Compute overrun | Priority order in §12 |
| Course scope (original proposal: crop classification, from-scratch CNN) | Resolved 2026-11-08: instructor confirmed the pretrained-detection, multi-domain scope is acceptable |

## 16. Out of scope

Copy-paste and other compositional edits beyond mosaic; per-region local editing in the main method (gated on Part B); adapting edit strength; YOLO-family detectors; architecture changes.
