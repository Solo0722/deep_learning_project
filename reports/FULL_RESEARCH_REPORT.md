# Class-aware augmentation for multi-class object detection: complete research record

This report covers every phase of the research, from the first crop-classification study to the positive-control runs now queued on CCAST.

- **Sources.** Every number comes from a file in this repository, and that file is named next to the number. The numbers were re-read from those files while this report was written, not copied from memory.
- **Discrepancies.** Where two files disagree, or an earlier document stated something the data does not support, this is listed in Section 1 and not smoothed over.
- **Gaps.** Things that were planned but not done are listed in Section 23.

All AP values are on a 0–100 scale. "F" means Faster R-CNN and "R" means RetinaNet. Intervals are 95%.

---

## 1. Corrections and discrepancies found while compiling this report

These come first so that nothing is hidden.

| # | What | Detail | Effect on conclusions |
|---|---|---|---|
| 1 | **Two values for the rectangle copy-paste arm `cpbal` on dacl10k test** | `test_metrics.json` (pycocotools, the official metric) gives seeds 22.943 / 22.800 / 22.851, mean **22.87** (`reports/cua_stats/holm_mde.py`, `PASTE`). The bootstrap code's own AP computation (`cua/analysis/ap50.py`, `ap50_from_matches`) gives 22.94 / **22.76** / 22.85, mean **22.85**. That second set is the one in the kill-test table of `reports/cua_paste_pilot/VAL_SUMMARY.md`. Seed 1 differs by 0.04. Uniform is identical in both (22.32 / 22.33 / 22.47). | None. As a result, cpbal − uniform is +0.49 by pycocotools and +0.48 by the bootstrap code, and the seed t interval is [0.19, 0.79] or [0.16, 0.79]. The cause of the 0.04 difference has not been investigated. |
| 2 | **The mask-paste (`cpbalm`) test numbers come from the bootstrap code's AP computation, not pycocotools** | The values 22.57 / 22.07 / 22.43 (mean 22.36) were computed with `ap50_from_matches`. The official `test_metrics.json` files exist on CCAST (job 817673) but have not been read yet. | They could differ by a few hundredths, as in item 1. The go rule needed +0.74, and the measured gap is −0.02, so the verdict cannot change. The official values should still be read before they go in the paper. |
| 3 | **PBT agreement statistic** | Recomputed from `reports/cua_pbt_pilot/decisions.jsonl`: the member that wins a class equals the member that donates weights in 25 of 40 decisions, i.e. **62.5%**. The paper (`research_paper/main.tex`, line 524), the reviewer brief and `HANDOFF.md` say **64%**. Member 0 wins 9 of 40 (22.5%), which matches the "22%" in those documents. | The conclusion (per-class picks follow the overall-best member, far above the 25% chance level) stands. **The paper's 64% must be changed to 62.5%.** |
| 4 | **PBT comparison values not verified** | The paper compares PBT's 50.75 with "50.72 for base and 51.04 for measured on the same seed". The best-epoch validation values for seed 0 in `VAL_SUMMARY.md` are base 51.03 and measured 51.18. PBT's 50.75 is its *final* window. The 50.72 / 51.04 are probably final-epoch values, but they could not be checked locally because the C-run histories are on CCAST. | Unverified. Check on CCAST before the paper keeps these numbers, and state whether final or best epoch is meant. |
| 5 | **Gradient-norm claim was overstated** | The reviewer brief §8, `HANDOFF.md` and `reports/selective_supervision_review/REVIEW.md` say the edited-to-unedited gradient-norm ratio ‖g_o‖/‖g₀‖ is ≈ 1 for every edit except mosaic, on all pairs. Recomputed from `reports/cua_utility_check/summary.json`, that holds for every Faster R-CNN checkpoint (non-mosaic ratios 0.93–1.10) and for VOC RetinaNet (0.90–1.08). It does **not** hold for RDD RetinaNet (0.78–1.98 and 0.96–1.37) or dacl10k RetinaNet (0.79–1.34 and 0.65–1.90). The mosaic ratio itself ranges from 0.12 to 3.36 across the 12 checkpoints. | The main diagnosis (no class signal within an edit; norm-free variants do no better) is unaffected. The sentence "norm ≈ 1 except mosaic" must be restricted to Faster R-CNN and VOC RetinaNet. The paper does not currently state the ratio claim. |
| 6 | **The paper still says 26,863 RDD training images** | `research_paper/main.tex` line 236 says 26,863. Table `tab:datasets` (line 248) says 25,863 training + 1,000 reference. 26,863 includes the reference images. | Text fix needed in line 236. |
| 7 | **The reviewer brief said "no multiplicity correction"** | That was true when it was written. Holm correction was added afterwards (`reports/cua_stats/HOLM_MDE.md`), and the paper's statistics paragraph now describes it. | Superseded. Section 14 below gives the Holm results. |
| 8 | **The reviewer brief left out some intervals** | Its H2/H3 table omitted several bootstrap intervals. | All intervals are given in full in Sections 11 and 13 of this report. |
| 9 | **My own errors earlier in this work** | (a) I gave advice on the paper before reading it. (b) In Results VI.B I wrote that CUA was significantly worse than the simplest baseline on "four of six pairs"; the data says three. This was corrected in `main.tex`. (c) When first printing the mask kill-test results, I scaled AP by 100 twice (2237 instead of 22.37). This was caught and corrected before the numbers were reported, and it changed nothing. (d) My first progress-check script read the wrong field and printed empty validation AP. | (b) is fixed. The others did not affect any reported result. |
| 10 | **Page count** | The compiled `research_paper/main.pdf` is **15 pages** (`research_paper/main.log`). An earlier figure of 17 pages does not match this build. | — |

---

## 2. Timeline of phases

| Phase | What | Main sources |
|---|---|---|
| 1 | Crop classification of road damage (custom CNN) | `reports/experiment_report_20261027/pdf_text.txt` §2.1 |
| 2 | Detection data preparation and detector engineering checks on RDD | same report §2.2–§4; `reports/detection_*_v1/`, `DATA_AUDIT.md` |
| 3 | Five-condition Faster R-CNN comparison on RDD, including the first online class-adaptive local controller (v1) | same report §5–§7; `RDD_Prototype_Submission_2026-10-26/` |
| 4 | Revised class-adaptive controller pilots (RDD v2: four-arm pilot, strict two-arm run) | `reports/proposal_handoff_20261103/report_text.txt` §6; `reports/rdd_v2_*/` |
| 5 | Proposal, related-work limitations and scope reset (3 November) | `reports/proposal_handoff_20261103/` |
| 6 | Class-Utility Augmentation (CUA) study design | `docs/superpowers/specs/2026-11-08-class-utility-augmentation-design.md`, plans in `docs/superpowers/plans/` |
| 7 | Part A: edit-wise measurement study (258 runs) | `reports/cua_part_a_report/` |
| 8 | Part C: policy comparison, v1 (90 test-evaluated runs, 18 ablations, 6 tuning runs) | `reports/cua_part_c_report/` |
| 9 | Estimator diagnosis (offline) | `reports/cua_utility_check/` |
| 10 | CUA v2 (36 runs) | `reports/cua_part_c_v2_report/`, `reports/cua_utility_check_v2/` |
| 11 | Per-class population-based training (PBT) pilot | `reports/cua_pbt_pilot/` |
| 12 | Review of an external "selective supervision" redesign | `reports/selective_supervision_review/REVIEW.md` |
| 13 | Reviewer brief | `reports/cua_reviewer_brief/CUA_REVIEWER_BRIEF.md` |
| 14 | Class-adaptive copy-paste pilot (18 runs) | `reports/cua_paste_pilot/VAL_SUMMARY.md` |
| 15 | Holm correction and minimum detectable effects | `reports/cua_stats/` |
| 16 | Mask copy-paste kill test (3 runs) | `reports/cua_paste_pilot/VAL_SUMMARY.md` (last section) |
| 17 | Paper rewrite toward the measured results; real-image figures | `research_paper/main.tex`, `research_paper/figures/` |
| 18 | Positive control (15 runs, queued) | `reports/cua_control/PROTOCOL.md` |

---

## 3. Phase 1: crop classification

Source: `reports/experiment_report_20261027/pdf_text.txt`, §2.1 and Table 1.

- **Setup:** annotated road-damage crops as classification inputs. A randomly initialised CNN with 695,732 trainable parameters, trained with unweighted cross-entropy, AdamW, batch size 64 and a 20-epoch cosine schedule. Epoch 20 was selected by validation macro-F1.
- **Result:** 7,461 of 8,242 validation crops correct, i.e. **90.524% accuracy, macro-F1 0.891457**.

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| D00 | 3,898 | 0.9042 | 0.9472 | 0.9252 |
| D10 | 1,771 | 0.9524 | 0.9368 | 0.9445 |
| D20 | 1,592 | 0.8759 | 0.8021 | 0.8374 |
| D40 | 981 | 0.8686 | 0.8491 | 0.8588 |

- **Errors and controls:** the largest error was D20 predicted as D00 (223 crops, 14.01% of D20). A box-aspect-ratio comparator reached 67.03% accuracy (macro-F1 0.4439); the majority class reached 47.29% (0.1605).
- **Why it was dropped:** the project moved to full-image detection to study augmentation with localisation, multiple objects and empty images. Crop accuracy is feasibility evidence only.

---

## 4. Phase 2: RDD detection data and engineering checks

Source: same report, §2.2–§4, Tables 2–6.

**Raw pool (image and XML pairs):**

| Source | Pairs |
|---|---|
| China Drone | 2,401 |
| China MotorBike | 1,977 |
| Czech | 2,829 |
| India | 7,706 |
| Japan | 10,506 |
| Norway | 8,161 |
| United States | 4,805 |

**Exclusions:**
- `United_States_003917.jpg`: duplicate pixels with conflicting boxes.
- `India_006389.jpg`: sole label "D0w0".
- One zero-width D20 box in `Japan_001265.jpg` was removed; the image was kept.

The resulting pool has **38,383 images and 55,002 target boxes**.

**Frozen partitions:**

| Partition | Images | With targets | No targets | Boxes |
|---|---|---|---|---|
| Training | 26,863 | 16,630 | 10,233 | 38,530 |
| Validation | 5,760 | 3,567 | 2,193 | 8,225 |
| Reserved test | 5,760 | 3,569 | 2,191 | 8,247 |

Boxes per class (train / val / test): D00 18,226 / 3,888 / 3,899; D10 8,283 / 1,770 / 1,776; D20 7,435 / 1,592 / 1,589; D40 4,586 / 975 / 983. Similarity clusters stay within one partition, with zero group crossings.

**Engineering checks:**

| Stage | Evidence |
|---|---|
| Data audit | 8 tests |
| Training loader | 22 tests; 15 real images |
| Detector integration | 14 tests; both detectors ran on 15 images |
| Small learning check | 18 tests; 200 updates per detector |
| YOLO diagnostic | Fixed checkpoints, no optimiser steps |
| Class-adaptive package | 81 local tests |

**Small learning check (training subset, 200 updates):**

| Detector | AP50 at step 0 / 50 / 100 / 200 | AP50:95 at step 200 |
|---|---|---|
| Faster R-CNN | 0 / 7.306 / 19.021 / 35.517 | 17.554 |
| YOLOv8n | 0 / 0 / 0.518 / 0 | 0.000 |

Faster R-CNN missed the predeclared threshold (AP50 ≥ 50 with a gain ≥ 10) but showed learning. YOLO showed a loss spike on an all-negative batch and was not pursued.

---

## 5. Phase 3: five-condition Faster R-CNN comparison on RDD (class-adaptive v1)

Source: same report, §5–§7, Tables 7–18.

**Conditions:**

| Condition | What it does |
|---|---|
| Identity | No augmentation |
| Fixed image-level | Appearance edits on the whole image |
| Fixed local | The same edits restricted to one overlap group |
| Online shared local | One feedback-selected policy for every group |
| Online class-adaptive local | A policy per exact class set |

**Setup:** COCO-initialised Faster R-CNN ResNet-50 FPN, 640 px letterbox, 20 epochs, SGD at LR 0.0025, float32, single seed. The checkpoint was selected by maximum validation AP50:95.

**Validation at the selected checkpoint (Table 10):**

| Condition | Epoch | AP50:95 | AP50 | AP75 | AR100 |
|---|---|---|---|---|---|
| Identity | 14 | 22.823 | 51.142 | 17.091 | 43.836 |
| Fixed image-level | 14 | 22.810 | 50.687 | 17.295 | 44.339 |
| Fixed local | 18 | 22.555 | 50.160 | 16.570 | 44.078 |
| Online shared local | 14 | **23.059** | 51.033 | 17.302 | 44.281 |
| Class-adaptive local | 18 | 22.563 | 50.183 | 16.975 | 43.874 |

**Final epoch 20 (Table 11):**

| Condition | AP50:95 | AP50 | AP75 | AR100 |
|---|---|---|---|---|
| Identity | 22.392 | 50.200 | 16.800 | 42.813 |
| Fixed image-level | 22.575 | 50.541 | 16.519 | 43.682 |
| Fixed local | 22.487 | 49.814 | 16.867 | 43.805 |
| Online shared local | 22.588 | 50.350 | 16.956 | 43.438 |
| Class-adaptive local | 22.493 | 49.835 | 17.008 | 43.699 |

**Per class at the selected checkpoint (Tables 12 and 13; order: identity / image-level / local / shared / class-adaptive):**

| Class | AP50:95 | AP50 |
|---|---|---|
| D00 | 24.744 / 24.784 / 24.090 / 24.905 / 24.235 | 51.324 / 51.062 / 50.131 / 50.654 / 50.337 |
| D10 | 20.206 / 20.266 / 19.520 / 20.050 / 19.446 | 47.995 / 47.844 / 46.601 / 47.893 / 46.092 |
| D20 | 29.975 / 29.631 / 30.114 / 30.162 / 30.069 | 62.951 / 62.408 / 62.727 / 63.011 / 63.024 |
| D40 | 16.367 / 16.559 / 16.496 / 17.120 / 16.502 | 42.297 / 41.436 / 41.181 / 42.575 / 41.278 |

**Detections on the 2,193 no-target validation images at confidence 0.25 (Table 14):**

| Condition | Images with a detection | Boxes | Share of images |
|---|---|---|---|
| Identity | 858 | 1,637 | 39.12% |
| Fixed image-level | 867 | 1,640 | 39.53% |
| Fixed local | 817 | 1,538 | 37.25% |
| Online shared local | 886 | 1,707 | 40.40% |
| Class-adaptive local | 806 | 1,527 | 36.75% |

**Run time (Table 15):**

| Condition | Hours | PBS job |
|---|---|---|
| Identity | 11.09 | 794364 |
| Fixed image-level | 11.51 | 795553 |
| Fixed local | 13.26 | 795787 |
| Online shared local | 14.14 | 796146 |
| Class-adaptive local | 14.45 | 796544 |

The total is 64.46 run-hours. Probe time was 56.41 min (shared) and 57.19 min (class-adaptive).

**Controller exposure (Tables 16–18):**
- Strength level 3 was applied in 76.17% of class-adaptive edits, against 18.79% for online shared.
- 13,583 of 158,151 class-adaptive opportunities (8.59%) fell back to identity: 1,205 because the class set was unobserved, 12,378 because support was insufficient.
- 9 of the 15 possible class sets never activated. For example, D00+D20 appears in 440 training source clusters and was activated 0 of 10 times.

**Verdict at the time:** class-adaptive was 0.496 AP50:95 below online shared and 0.260 below identity, so the intended advantage was not shown. Each condition had one run.

---

## 6. Phase 4: revised controller pilots on RDD (v2)

Source: `reports/proposal_handoff_20261103/report_text.txt` §6, Table 2. Single seed, validation AP50:95.

| Comparison | Condition | AP50:95 |
|---|---|---|
| Four-arm pilot, epoch 4 | Geometry only | **19.8080** |
| | Fixed local | 18.8610 |
| | Online shared local | 19.3656 |
| | Class-adaptive local | 19.6076 |
| Strict two-arm run, epoch 8 | Geometry only | **19.6846** |
| | Class-adaptive local | 18.4369 |

- In the four-arm pilot, class adaptation beat fixed and shared local but stayed below geometry only.
- In the strict run it was 1.2477 below geometry only.
- A repeatability discrepancy was investigated, and the strict checks matched the examined trajectories (`reports/rdd_v2_repro_v1/`, `rdd_v2_repro_v2/`, `rdd_v2_strict_v1/`).

---

## 7. Phase 5: proposal and scope reset

`reports/proposal_handoff_20261103/report_text.txt` restated the problem without prescribing a method:
- A transformation can help one class and hurt another in the same image.
- Related work already covers class-adaptive (CUDA), online (CTAugment, TeachAugment), benefit-based (BDA, DDAS, MADAug) and object-aware detection augmentation.
- The defensible gap is therefore handling class-dependent effects jointly in multi-object detection images.
- The earlier results (Sections 5–6) were recorded as negative evidence for those specific recipes.

---

## 8. CUA study design

Sources: `docs/superpowers/specs/2026-11-08-class-utility-augmentation-design.md`; reviewer brief §1–§4; `research_paper/main.tex` Sections III–V.

**Hypotheses (pre-registered):**

| | Hypothesis |
|---|---|
| H1 | An edit's effect depends on the class |
| H2 | CUA beats shared-utility augmentation (same estimator, classes pooled) on test macro AP50 |
| H3 | CUA beats the best simple baseline (base or uniform, whichever is higher) |
| H4 | CUA's estimated utilities rank-agree with measured per-class effects |

**Datasets:**

| Dataset | Classes | Train | Reference | Val | Test | Train images with ≥ 2 classes |
|---|---|---|---|---|---|---|
| RDD2022 | 4 (D00, D10, D20, D40) | 25,863 | 1,000 | 5,760 | 5,760 | 35% |
| dacl10k | 12 (Crack, Wetspot, Efflorescence, Rust, Rockpocket, Hollowareas, Cavity, Spalling, Graffiti, Weathering, Restformwork, ExposedRebars) | 4,609 | 1,000 | 1,160 | 1,141 | 75% |
| PASCAL VOC (07+12 trainval → 07 test) | 20 | 13,694 | 1,206 | 1,651 | 4,952 | 35% |

**dacl10k preparation:**
- Source: the FiftyOne export `Voxel51/dacl10k` (CC BY-NC 4.0). The 1,012 official test images have no labels and were discarded, leaving 7,910 labelled images.
- Near-duplicate groups were split 70/15/15.
- Each polygon becomes one box.
- Class gate: ≥ 300 training boxes. It removed ACrack (282 boxes).

**VOC:** "difficult" objects are excluded from training and used as ignore regions in evaluation. The near-duplicate screen found 20 train–test and 2 val–test pairs, reported and not removed.

**Dataset gate:** a dataset is used only if ≥ 20% of object-bearing training images contain ≥ 2 classes.

**Training (identical for every policy):**
- Detectors: torchvision Faster R-CNN and RetinaNet with ResNet-50-FPN, COCO-pretrained, all stages trainable, BatchNorm statistics frozen.
- Input: 640 px letterbox (padding value 114).
- Optimiser: SGD, momentum 0.9, weight decay 5e-4, batch 4, LR 0.0025, 1,000-iteration warm-up from 0.001×.
- Schedule: ×0.1 from epoch round(0.6E)+1 and ×0.01 from round(0.8E)+1; mixed precision.
- Inference: score > 0.001, NMS 0.5, at most 100 detections per image.
- The best-validation checkpoint goes to test. Seeds 0, 1, 2. Single A10 GPU per run.

**Epoch budgets** (`reports/cua_part_a_report/budget.json`, Faster R-CNN budget runs):

| Dataset | Epochs | Final validation macro AP50 | Best | Plateau rule met |
|---|---|---|---|---|
| RDD | 20 | 51.05 | 51.19 | yes |
| VOC | 20 | 81.57 | 81.68 | yes |
| dacl10k | 30 | 20.10 | 20.32 | yes |

The rule: the final epoch gains < 0.3 over the first epoch at the final LR.

**Base recipe (all policies):** horizontal flip (p = 0.5) plus scale jitter U(0.8, 1.2).

**Edit menu:**

| Edit | Strength | Valid on |
|---|---|---|
| translate | ±10% per axis | all |
| rotate | ±10° | all |
| vflip | — | dacl10k only |
| brightness | factor 0.7–1.3 | all |
| contrast | factor 0.7–1.3 | all |
| saturation | factor 0.5–1.5 | all |
| hue | shift ±0.05 of the circle | all |
| blur | Gaussian σ 0.5–1.5 px | all |
| noise | σ 0.03 × 255 | all |
| jpeg | quality 30–60 | all |
| sharpen | factor 1.5–2.5 | all |
| erase | 2–10% of the canvas | all |
| mosaic | 2 × 2 tiles | all |

That gives 12 valid edits on RDD and VOC and 13 on dacl10k.

---

## 9. Part A: edit-wise measurement study

Source: `reports/cua_part_a_report/` (`tests.json`, `deltas.csv` with 2,664 rows of seed × edit × class, `proxy.json`, heatmaps).

**Protocol:**
- A 3-epoch warm checkpoint on the base recipe.
- Then |𝒪| + 1 three-epoch continuations: base only, and one per edit at p = 0.5.
- Branches share the checkpoint, data order and draws.
- Δ = AP50_c(edit) − AP50_c(base) on **validation**.
- **258 runs:** RDD and VOC 42 per detector each, dacl10k 45 per detector.

**Edit × class interaction F-test (`tests.json`):**

| Pair | F | df | p | Holm-significant single edits |
|---|---|---|---|---|
| RDD F | 3.73 | (33, 88) | 4.5e-7 | none |
| RDD R | 0.79 | (33, 88) | 0.77 | none |
| VOC F | 1.60 | (209, 440) | 2.4e-5 | none |
| VOC R | 1.91 | (209, 440) | 1e-8 | mosaic (p_Holm < 0.001) |
| dacl10k F | 2.52 | (132, 288) | 4.1e-11 | mosaic (0.0003), contrast (0.0021) |
| dacl10k R | 1.06 | (132, 288) | 0.34 | none |

So H1 holds for Faster R-CNN on all three datasets and for RetinaNet on VOC only.

**Mean ΔAP50 over classes, per edit (recomputed from `deltas.csv`):**

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

Every per-class cell for every pair is in Appendix A.

**Edits that help some classes and hurt others, with the same sign in all three seeds** (reviewer brief §5):
- dacl10k F: contrast (helps 2 classes, hurts 6), mosaic (3, 6), sharpen (3, 2).
- VOC F: rotate (4, 3), sharpen (3, 3), hue (3, 3).
- On dacl10k F, contrast is positive in all three seeds for Crack and Hollowareas, and negative in all three for Restformwork, Rust, ExposedRebars and Weathering.

No edit is reliably positive for all classes on any dataset.

**Variance decomposition of the seed-mean Δ table** (`reports/selective_supervision_review/REVIEW.md` §3; the residual is an upper bound on the interaction):

| Pair | Edit % | Class % | Residual % | Edits with a class \|t\| > 2 positive and another \|t\| > 2 negative |
|---|---|---|---|---|
| RDD F | 87 | 2 | 11 | none |
| RDD R | 59 | 19 | 22 | none |
| dacl10k F | 10 | 31 | 58 | 7 |
| dacl10k R | 7 | 24 | 68 | 3 |
| VOC F | 6 | 36 | 58 | 12 |
| VOC R | 30 | 30 | 40 | 9 |

**Proxy check** (`proxy.json`): 4 edits on RDD F were re-run for the full 20 epochs. Spearman ρ between the short-run and full-run per-class Δ is **0.56**, 95% CI [0.04, 0.86], over 16 cells.

**Not run:** the class-targeted study (an edit applied only inside one class's boxes).

---

## 10. CUA v1: method and tuning

Sources: `cua/estimator/utility.py`, `cua/policy.py`, `main.tex` Section III, reviewer brief §6.

**Estimator:**
- **Formula:** u(o, c) = η⟨g_o − g₀, h_c⟩ / (L_c(R) + δ), with δ = 0.01 and η the current LR.
- **Gradients:** g₀ and g_o are gradients of the pooled training loss on 8 images, unedited and edited. h_c is the gradient of class c's loss on a 64-image reference batch.
- **θ_s:** box head + predictor + top FPN level (Faster R-CNN), or the classification and regression heads + top FPN (RetinaNet).
- **Precision:** float32 (fp16 gave NaN, fixed in commit 84a36f3).
- **Smoothing:** EMA with β = 0.9.
- **Schedule:** estimates start after epoch 1 (epoch 1 uses uniform). One estimate takes about 8.6 s on an A10.

**Policy:**
- Score: the mean of ū over the classes in the image; identity scores 0.
- Harm block: an edit is blocked if ū + 1.64·se < 0 for any class present with ≥ 3 estimates.
- Sampling: (1 − ε)·softmax(s/τ) + ε/|A|, with ε = 0.1 and τ = the standard deviation of the scores.

**Tuning** (`.superpowers/sdd/2026-11-17-cua-plan2-class-utility-policy/progress.md`): RDD F, seed 0, mean validation macro AP50 over epochs 17–20. The rule, fixed in advance: change a default only for a gain ≥ 0.5 at ≤ 20% overhead.

| Variant | Score | Overhead |
|---|---|---|
| default (K = 200, head + top FPN, ε = 0.1) | 49.61 | 15.3% |
| K = 100 | 49.84 | 26.7% (excluded) |
| K = 400 | 49.66 | 8.2% (from the reviewer brief; not in the ledger) |
| head only | 49.81 | 13.0% (from the reviewer brief; not in the ledger) |
| ε = 0.05 | 49.78 | — |
| ε = 0.2 | 49.78 | — |

No variant qualified, so the defaults were kept.

**Compared policies:**

| Policy | What it does |
|---|---|
| base | no edit |
| uniform | one edit uniform over menu + identity, TrivialAugment-style |
| measured | CUA's policy fed the fixed Part A table |
| shared | the same estimator with classes pooled |
| cua | as above |

---

## 11. Part C results: test macro AP50, every arm, every seed

Source: `reports/cua_part_c_v2_report/results.json` (`arms`). It holds the v1 arms (also in `reports/cua_part_c_report/main_table.csv`) and the v2 arms. Each cell is the mean ± s.d. over seeds, followed by seeds 0 / 1 / 2.

| Pair | base | uniform | measured | shared | cua | shared2 | cua2 |
|---|---|---|---|---|---|---|---|
| RDD F | 51.15 ± 0.24 (51.26 / 51.32 / 50.88) | 50.55 ± 0.03 (50.57 / 50.53 / 50.54) | **51.47 ± 0.15** (51.64 / 51.38 / 51.39) | 49.86 ± 0.19 (50.02 / 49.92 / 49.64) | 50.13 ± 0.26 (50.17 / 50.37 / 49.85) | 50.63 ± 0.14 (50.49 / 50.77 / 50.62) | 50.88 ± 0.13 (50.75 / 51.00 / 50.89) |
| RDD R | **44.56 ± 0.22** (44.72 / 44.65 / 44.30) | 43.82 ± 0.45 (43.60 / 44.33 / 43.53) | 44.55 ± 0.47 (44.48 / 45.04 / 44.12) | 43.32 ± 0.24 (43.58 / 43.10 / 43.29) | 43.57 ± 0.13 (43.63 / 43.66 / 43.42) | 43.92 ± 0.38 (44.31 / 43.56 / 43.88) | 43.92 ± 0.22 (44.02 / 44.07 / 43.67) |
| dacl10k F | 20.89 ± 0.12 (20.85 / 20.79 / 21.03) | 22.37 ± 0.08 (22.32 / 22.33 / 22.47) | 21.54 ± 0.09 (21.57 / 21.61 / 21.43) | 20.53 ± 0.03 (20.53 / 20.56 / 20.49) | 20.50 ± 0.27 (20.81 / 20.27 / 20.44) | **22.42 ± 0.29** (22.12 / 22.71 / 22.42) | 21.49 ± 0.22 (21.61 / 21.63 / 21.23) |
| dacl10k R | 9.26 ± 1.29 (10.62 / 9.08 / 8.07) | 7.70 ± 0.85 (8.64 / 6.97 / 7.48) | **9.80 ± 1.03** (9.88 / 10.78 / 8.73) | 7.74 ± 1.33 (8.98 / 7.88 / 6.34) | 8.66 ± 0.86 (7.69 / 8.95 / 9.32) | 7.01 ± 1.75 (6.33 / 5.71 / 9.00) | 8.23 ± 0.82 (7.37 / 8.30 / 9.00) |
| VOC F | 85.60 ± 0.08 (85.52 / 85.69 / 85.58) | **86.54 ± 0.10** (86.65 / 86.49 / 86.47) | 86.30 ± 0.08 (86.34 / 86.34 / 86.20) | 86.37 ± 0.06 (86.35 / 86.43 / 86.33) | 86.49 ± 0.12 (86.53 / 86.59 / 86.36) | 85.99 ± 0.51 (85.63 / 86.58 / 85.76) | 86.46 ± 0.24 (86.42 / 86.24 / 86.72) |
| VOC R | 82.38 ± 0.16 (82.22 / 82.40 / 82.53) | 83.44 ± 0.05 (83.38 / 83.46 / 83.48) | 83.76 ± 0.24 (83.49 / 83.88 / 83.93) | **84.08 ± 0.29** (83.79 / 84.08 / 84.37) | 83.71 ± 0.16 (83.53 / 83.80 / 83.81) | 83.91 ± 0.24 (83.65 / 83.93 / 84.13) | 83.96 ± 0.13 (83.82 / 84.08 / 83.98) |

Uniform is below base on RDD (both detectors) and on dacl10k R. `results.json` lists no missing runs.

**v1 paired comparisons, before Holm** (`reports/cua_part_c_report/comparisons.csv`; difference = b − a; seed t interval / image bootstrap interval; "claim" means both intervals exclude 0):

| Pair | cua − shared (H2) | cua − best baseline (H3) | cua − measured |
|---|---|---|---|
| RDD F | +0.27 [−0.13, 0.67] / [0.07, 0.50] | −1.02 vs base [−1.20, −0.84] / [−1.48, −0.52] | −1.34 [−2.05, −0.62] / [−1.77, −0.86] |
| RDD R | +0.24 [−0.44, 0.93] / [−0.01, 0.50] | −0.99 vs base [−1.24, −0.73] / [−1.36, −0.63] | −0.98 [−1.87, −0.08] / [−1.30, −0.61] |
| dacl10k F | −0.02 [−0.72, 0.68] / [−0.30, 0.23] | −1.87 vs uniform [−2.63, −1.11] / [−2.53, −1.32] | −1.03 [−1.74, −0.32] / [−1.75, −0.36] |
| dacl10k R | +0.92 [−4.40, 6.24] / [0.73, 1.14] | −0.60 vs base [−5.91, 4.70] / [−0.96, −0.27] | −1.14 [−4.91, 2.63] / [−1.51, −0.80] |
| VOC F | +0.13 [−0.08, 0.33] / [−0.05, 0.30] | −0.04 vs uniform [−0.35, 0.26] / [−0.30, 0.23] | +0.20 [0.08, 0.31] / [−0.10, 0.48] |
| VOC R | −0.37 [−0.79, 0.05] / [−0.55, −0.19] | **+0.28 vs uniform [0.02, 0.53] / [0.06, 0.50] (claim)** | −0.05 [−0.26, 0.16] / [−0.25, 0.17] |

- H2: no claim on any pair.
- H3 positive claim: VOC R only (+0.28). It is lost after Holm (Section 14).
- H3, CUA reliably *worse*: RDD F, RDD R and dacl10k F, with both intervals below 0. That is **three** pairs.

**Measured − best baseline** (`reports/cua_part_c_v2_report/comparisons.csv`):

| Pair | Difference | Seed t | Bootstrap |
|---|---|---|---|
| RDD F | +0.32 vs base | [−0.26, 0.89] | [0.06, 0.59] |
| RDD R | −0.01 vs base | [−0.88, 0.86] | [−0.31, 0.28] |
| dacl10k F | −0.84 vs uniform | [−1.27, −0.41] | [−1.27, −0.44] |
| dacl10k R | +0.54 vs base | [−2.51, 3.58] | [0.30, 0.82] |
| VOC F | −0.24 vs uniform | [−0.45, −0.04] | [−0.40, −0.05] |
| VOC R | +0.33 vs uniform | [−0.14, 0.79] | [0.05, 0.58] |

The measured policy beats the best baseline on no pair. It is reliably worse on dacl10k F and VOC F.

**H4: estimated utility vs measured Δ (v1 cua, run-mean ū), Spearman ρ** (`results.json`, `h4`):

| Pair | ρ | 95% CI | Cells |
|---|---|---|---|
| RDD F | −0.67 | [−0.82, −0.45] | 48 |
| RDD R | −0.12 | [−0.42, 0.17] | 48 |
| dacl10k F | −0.12 | [−0.28, 0.05] | 156 |
| dacl10k R | −0.01 | [−0.18, 0.16] | 156 |
| VOC F | −0.20 | [−0.33, −0.06] | 240 |
| VOC R | +0.10 | [−0.05, 0.23] | 240 |

H4 was computed for v1 only. No H4 exists for cua2.

**What the estimator rated highest:** in the v1 run-mean utility table, the two top-rated edits are **mosaic, then blur, for every class** on RDD F (4 of 4 classes), dacl10k F (12 of 12) and VOC F (20 of 20), recomputed from `results.json` `h4.cells`. In Part A, mosaic is the most harmful edit on RDD and dacl10k, and blur the second most harmful on RDD.

**Ablations** (validation, Faster R-CNN, 3 seeds, difference vs cua, seed t interval):

| Ablation | RDD | dacl10k |
|---|---|---|
| no harm block | −0.12 [−0.61, 0.38] | +0.07 [−0.40, 0.55] |
| no identity option | +0.17 [−0.09, 0.43] | +0.15 [−0.43, 0.74] |
| inverse-AP class weights | −0.02 [−0.62, 0.58] | +0.00 [−0.39, 0.40] |

None is distinguishable from cua.

**Estimator overhead** (`results.json`, `overhead`; share of epoch wall time / extra time relative to the rest of the epoch):

| Pair | shared | cua | shared2 | cua2 |
|---|---|---|---|---|
| RDD F | 15.3% / 18.0% | 15.7% / 18.7% | 18.9% / 23.3% | 19.3% / 23.9% |
| RDD R | 16.6% / 19.9% | 16.2% / 19.3% | 18.7% / 23.0% | 18.6% / 22.9% |
| dacl10k F | 13.9% / 16.1% | 14.5% / 17.0% | 18.8% / 23.1% | 16.5% / 19.7% |
| dacl10k R | 17.5% / 21.2% | 17.1% / 20.7% | 18.1% / 22.0% | 17.4% / 21.1% |
| VOC F | 17.8% / 21.6% | 18.1% / 22.2% | 17.0% / 20.4% | 18.2% / 22.2% |
| VOC R | 20.8% / 26.2% | 20.8% / 26.2% | 15.7% / 18.6% | 15.7% / 18.7% |

**Secondary test metrics** (`results.json`, `secondary`). Each cell: AP50:95 / AP small / AP medium / AP large / false-positive image share in %. The false-positive share is computed only for test sets that contain object-free images. VOC test has none, so it shows "—".

| Pair | base | uniform | measured | shared | cua | shared2 | cua2 |
|---|---|---|---|---|---|---|---|
| RDD F | 22.70 / 8.98 / 18.28 / 24.83 / 22.08 | 22.41 / 8.66 / 18.14 / 24.37 / 20.02 | 22.83 / 8.94 / 18.47 / 24.84 / 22.53 | 21.94 / 8.20 / 17.83 / 23.91 / 17.48 | 22.05 / 8.22 / 17.79 / 24.12 / 18.30 | 22.34 / 8.65 / 18.02 / 24.40 / 18.53 | 22.43 / 8.79 / 18.04 / 24.67 / 19.05 |
| RDD R | 19.66 / 7.52 / 15.28 / 21.18 / 0.11 | 19.35 / 7.11 / 15.26 / 20.76 / 0.05 | 19.65 / 7.83 / 15.53 / 21.16 / 0.09 | 19.00 / 7.56 / 14.95 / 20.34 / 0.09 | 19.05 / 7.16 / 14.89 / 20.46 / 0.12 | 19.33 / 6.47 / 15.23 / 20.71 / 0.06 | 19.25 / 6.88 / 15.06 / 20.81 / 0.11 |
| dacl10k F | 9.39 / 3.01 / 6.10 / 10.69 / 68.86 | 10.39 / 2.87 / 5.76 / 11.77 / 69.23 | 9.95 / 2.61 / 6.24 / 11.23 / 70.33 | 9.55 / 2.47 / 4.48 / 10.99 / 67.40 | 9.58 / 2.60 / 4.50 / 10.99 / 68.13 | 10.34 / 3.00 / 6.06 / 11.68 / 67.03 | 10.02 / 2.61 / 4.93 / 11.52 / 66.30 |
| dacl10k R | 4.32 / 0.34 / 1.37 / 5.18 / 2.56 | 3.60 / 0.26 / 1.24 / 4.31 / 2.20 | 4.56 / 0.29 / 1.40 / 5.50 / 2.56 | 3.60 / 0.22 / 1.07 / 4.29 / 2.20 | 4.02 / 0.24 / 1.14 / 4.78 / 2.56 | 3.21 / 0.23 / 1.12 / 3.82 / 1.83 | 3.86 / 0.26 / 1.14 / 4.66 / 1.83 |
| VOC F | 57.75 / 25.94 / 42.08 / 63.14 / — | 58.94 / 26.98 / 44.29 / 63.94 / — | 58.66 / 28.88 / 43.52 / 63.70 / — | 59.06 / 33.13 / 45.27 / 63.88 / — | 59.18 / 32.43 / 45.52 / 64.03 / — | 58.34 / 30.46 / 44.80 / 63.33 / — | 58.79 / 32.83 / 45.09 / 63.60 / — |
| VOC R | 57.24 / 26.70 / 39.32 / 63.16 / — | 58.16 / 25.61 / 41.22 / 63.78 / — | 58.16 / 28.00 / 43.47 / 63.19 / — | 58.66 / 26.54 / 43.61 / 63.84 / — | 58.34 / 28.20 / 42.80 / 63.56 / — | 58.38 / 28.11 / 43.76 / 63.55 / — | 58.59 / 26.70 / 43.39 / 63.90 / — |

The false-positive share is the fraction of object-free test images with a detection above the score threshold set in `cua/test_eval.py` (`FP_SCORE` = 0.5). The dacl10k and RDD values are not comparable across detectors, because RetinaNet scores are calibrated differently.

---

## 12. Why v1 failed: offline estimator diagnosis

Source: `reports/cua_utility_check/summary.json`, made with `cua/analysis/utility_check.py`. There are 12 checks: for each pair, one Part A warm checkpoint and one Part C cua best checkpoint, each with 20 rounds.

**Variants scored against Part A:**
- `dot`: v1.
- `normed`: g_o rescaled to ‖g₀‖.
- `cosdiff`: cos(g_o, h_c) − cos(g₀, h_c).
- `ratio`: ‖g_o‖/‖g₀‖.

| Check | Within-edit ρ: dot / normed / cosdiff | Top-scored edit (dot) | Non-mosaic norm ratio | Mosaic ratio |
|---|---|---|---|---|
| dacl10k F, cua best | −0.05 / −0.04 / −0.11 | mosaic | 0.99–1.02 | 1.26 |
| dacl10k F, warm | +0.04 / −0.02 / −0.17 | mosaic | 0.97–1.10 | 3.36 |
| dacl10k R, cua best | −0.01 / −0.12 / −0.11 | contrast | 0.79–1.34 | 0.38 |
| dacl10k R, warm | −0.02 / −0.00 / +0.05 | blur | 0.65–1.90 | 0.24 |
| RDD F, cua best | −0.10 / −0.09 / −0.12 | mosaic | 0.98–1.02 | 1.00 |
| RDD F, warm | −0.08 / −0.19 / +0.08 | mosaic | 0.96–1.02 | 1.43 |
| RDD R, cua best | −0.21 / −0.12 / −0.01 | mosaic | 0.78–1.98 | 0.60 |
| RDD R, warm | +0.01 / −0.14 / +0.09 | blur | 0.96–1.37 | 0.12 |
| VOC F, cua best | −0.02 / +0.04 / +0.04 | mosaic | 0.93–1.04 | 0.68 |
| VOC F, warm | −0.01 / −0.06 / +0.01 | mosaic | 0.98–1.05 | 1.32 |
| VOC R, cua best | +0.08 / +0.04 / +0.02 | noise | 0.97–1.05 | 0.61 |
| VOC R, warm | −0.09 / −0.03 / −0.06 | erase | 0.90–1.08 | 0.27 |

**Findings:**
- **No class signal within an edit:** within-edit ρ runs from −0.21 to +0.09 across all checks and variants.
- **Mosaic dominates:** it is the top-scored edit under `dot` in 7 of 12 checks.
- **Norm-free variants do no better.**
- **Noise:** per-cell |t| > 2 in about 5–30% of cells, and signs flip between checkpoints. This is from `HANDOFF.md`; the per-cell t values are in the per-check JSON files.
- **The norm ratio** is near 1 for Faster R-CNN and VOC RetinaNet only (correction 5 in Section 1).

---

## 13. CUA v2

**Method** (`cua/estimator/lookahead.py`; `main.tex` Section III, subsection "A Lower-Variance Estimator"):
1. **Lookahead:** 4 SGD steps on θ_s at 10× LR, with grad-norm clip 5, on the clean and on the edited 16-image batch. Then u(o, c) = (L_c after clean − L_c after edited) / (L_c before + δ).
2. **Object-count scaling:** each edited step's gradient is scaled by the ratio of object counts, clean over edited.
3. **Less noise:** reference batch 32, K = 400, β = 0.98, and a cell counts only if |ū| ≥ 2·se.
4. **Prior:** the Part A first-continuation-epoch table (`probe_<det>.json`), combined with the online estimate by Stouffer's method on the z scale.

**Offline check of the lookahead alone** (`reports/cua_utility_check_v2/summary.json`, 10 rounds each, cua best checkpoint, seed 0):

| Pair | Within-edit ρ | All-cell ρ | Edit-level ρ | Top-scored edit |
|---|---|---|---|---|
| dacl10k F | −0.19 | −0.15 | −0.03 | mosaic |
| dacl10k R | +0.12 | +0.18 | +0.42 | noise |
| RDD F | +0.15 | +0.24 | +0.28 | mosaic |
| RDD R | −0.27 | −0.48 | −0.39 | mosaic |
| VOC F | −0.06 | −0.12 | −0.43 | mosaic |
| VOC R | +0.06 | +0.13 | +0.27 | mosaic |

Within-edit ρ runs from −0.27 to +0.15, and mosaic is still top-scored in 5 of 6.

The test results are in the Section 11 table (columns shared2 and cua2).

**v2 paired comparisons, before Holm** (`reports/cua_part_c_v2_report/comparisons.csv`; seed t / bootstrap):

| Pair | cua2 − shared2 | cua2 − best baseline | cua2 − measured | cua2 − cua (v1) |
|---|---|---|---|---|
| RDD F | **+0.25 [0.21, 0.29] / [0.03, 0.46] (claim)** | −0.27 vs base [−0.93, 0.39] / [−0.61, 0.07] | −0.59 [−1.26, 0.07] / [−0.90, −0.28] | **+0.75 [0.12, 1.37] / [0.34, 1.11] (claim)** |
| RDD R | +0.00 [−1.09, 1.09] / [−0.23, 0.25] | −0.64 vs base [−0.78, −0.50] / [−0.97, −0.34] | −0.63 [−1.38, 0.12] / [−0.92, −0.34] | **+0.35 [0.12, 0.57] / [0.04, 0.66] (claim)** |
| dacl10k F | −0.92 [−1.84, −0.01] / [−1.43, −0.47] | −0.88 vs uniform [−1.64, −0.12] / [−1.43, −0.43] | −0.04 [−0.37, 0.29] / [−0.68, 0.56] | **+0.99 [0.19, 1.78] / [0.56, 1.46] (claim)** |
| dacl10k R | +1.21 [−2.03, 4.46] / [0.97, 1.51] | −1.03 vs base [−6.27, 4.20] / [−1.30, −0.70] | −1.57 [−5.55, 2.41] / [−1.86, −1.25] | −0.43 [−0.90, 0.04] / [−0.68, −0.10] |
| VOC F | +0.47 [−1.28, 2.23] / [0.27, 0.71] | −0.08 vs uniform [−0.79, 0.63] / [−0.31, 0.16] | +0.17 [−0.64, 0.97] / [−0.11, 0.42] | −0.03 [−0.94, 0.87] / [−0.22, 0.18] |
| VOC R | +0.05 [−0.40, 0.50] / [−0.14, 0.24] | **+0.52 vs uniform [0.30, 0.74] / [0.26, 0.78] (claim)** | +0.20 [−0.15, 0.55] / [0.01, 0.40] | **+0.25 [0.08, 0.41] / [0.03, 0.45] (claim)** |

**Reading:**
- v2 is better than v1 on 4 of 6 pairs.
- v2 beats its own shared version only on RDD F (+0.25), and is reliably worse than it on dacl10k F (−0.92).
- v2 beats the best baseline only on VOC R (+0.52).
- v2 matches the measured table on no pair.
- The prior and the gate pull cua2 toward the measured table, so v2's gains over v1 do not show that the online class estimate adds information.

---

## 14. Holm correction and minimum detectable effects

Source: `reports/cua_stats/HOLM_MDE.md`, `holm_mde.json`, script `holm_mde.py`.

**Method:**
- Holm is applied within each family of 6 pairs, separately to the seed p and the bootstrap p.
- The seed p is exact for df = 2.
- The bootstrap p is approximated from the stored 95% interval under normality.
- A claim needs both Holm p < 0.05 in the same direction.

**Results that survive Holm on both tests:**

| Family | Pair | Difference | Holm p, seed / bootstrap |
|---|---|---|---|
| H3 v1 | dacl10k F | −1.87 vs uniform | 0.036 / < 0.001 |
| H3 v1 | RDD F | −1.02 vs base | 0.010 / < 0.001 |
| H3 v1 | RDD R | −0.99 vs base | 0.018 / < 0.001 |
| H3 v2 | RDD R | −0.64 vs base | 0.015 / < 0.001 |
| H3 v2 | **VOC R** | **+0.52 vs uniform** | **0.048** / < 0.001 |

**Nothing survives** in H2 (v1 or v2, all 6 pairs), in measured − best baseline, or in copy-paste − uniform.

**What changes under Holm:**
- The v1 VOC R H3 result (+0.28) is lost (seed Holm p 0.13).
- The v2 RDD F H2 result (+0.25) survives on the seed test (p 0.009) but fails on the bootstrap (p 0.076), so it is not a claim.
- The v2 VOC R H3 result (+0.52) is the only positive result left. It is not class information: cua2 − shared2 is +0.05 (p 0.66) on that pair, and shared (84.08) and measured (83.76) both score above uniform there.

**H4 under Holm:** RDD F ρ −0.67 (Holm p 2e-5); VOC F ρ −0.20 (0.012); the other four pairs ≥ 0.56. Both significant correlations are **negative**.

**Copy-paste − uniform under Holm:**

| Pair | Arm | Difference | Seed Holm p | Bootstrap Holm p |
|---|---|---|---|---|
| dacl10k | cpbal | +0.49 | 0.12 | 0.12 |
| dacl10k | cpada | +0.49 | 0.15 | 0.12 |
| RDD | cpbal | −0.55 | 0.15 | not run |
| RDD | cpada | −0.78 | 0.12 | not run |
| RDD | cprand | −0.52 | 0.15 | not run |

**Minimum detectable effect (80% power, both tests):**

| Pair | Seed s.d. of differences | MDE seed t | MDE bootstrap | Effective MDE |
|---|---|---|---|---|
| VOC F | 0.12 | 0.40 | 0.31 | **0.40** |
| VOC R | 0.17 | 0.55 | 0.31 | **0.55** |
| RDD F | 0.20 | 0.65 | 0.38 | **0.65** |
| dacl10k F | 0.23 | 0.74 | 0.72 | **0.74** |
| RDD R | 0.28 | 0.90 | 0.42 | **0.90** |
| dacl10k R | 2.11 | 6.87 | 0.39 | **6.87** |

The pre-registered rule called for two extra seeds when the primary interval spans 0 and is wider than 1.0. That applied to dacl10k R, and **the extra seeds were not run**.

---

## 15. Per-class population-based training (PBT) pilot

Sources: `reports/cua_pbt_pilot/` (`pbt.json`, `decisions.jsonl`, `final.json`), `cua/pbt.py`.

**Setup:** RDD F, seed 0, 4 members, 20 epochs in 10 windows of 2 epochs, σ = 1.0, ε = 0.1. After each window:
- All members restart from the member with the best reference macro AP50.
- Each class column is copied from the member best on that class.
- Members 1–3 add noise.

**Results:**
- Final validation macro AP50 of the four members: 50.68, **50.75** (member 1, the reported value), 50.69, 50.74.
- Weight donors per window: 1, 3, 1, 3, 3, 2, 3, 0, 2, 3.
- The per-class winner equals the weight donor in **25 of 40 (62.5%)**, against 25% by chance. Member 0 wins 22.5%.
- The comparison values in the paper (base 50.72, measured 51.04) are unverified (Section 1, item 4).

---

## 16. External redesign review ("Class-Aware Selective Supervision")

Source: `reports/selective_supervision_review/REVIEW.md`.

**Verdict:** do not adopt the full architecture.

- **On RDD, the conflict it targets is absent:** the edit main effect is 87% of the variance. The proposal's example also runs backwards: blur hurts the pothole class D40 most (−1.98) and the crack class D00 least (−0.67).
- **Its cosine estimator would repeat the failure already diagnosed.**
- **Point-by-point:**
  - Collapse toward "no edit": contradicted, since mosaic was over-rated.
  - Pooled gradient: correctly described.
  - θ_s blind to small objects: mostly wrong.
  - Norm-driven estimate: already tested (but see correction 5).
  - Harm block starves multi-class images: logically true, but the `noharm` ablation is null.

---

## 17. Class-adaptive copy-paste (CAP) pilot

Sources: `cua/augment/paste.py`, `cua/experiments/specs.py` (`paste_specs`), `reports/cua_paste_pilot/VAL_SUMMARY.md`.

**Mechanism:**
- Objects (box rectangles from a per-class bank) are pasted into training images before base geometry and the uniform edit policy.
- 1–3 objects with probability 0.5.
- A paste may cover ≤ 30% of any existing box, with a 4-px feather.
- Each paste is at most half the content width and half the region height.
- On RDD, pastes go into the lower half of the image only.

**Arms** (Faster R-CNN, seeds 0–2, on dacl10k and RDD; 18 runs; the control is the C-uniform run):

| Arm | How classes are chosen |
|---|---|
| cprand | at class frequency |
| cpbal | uniform over classes |
| cpada | by reference AP50 (lower AP gets more pastes), ε floor 0.2, refreshed each epoch |

**Best-epoch validation macro AP50:**

| dacl10k | s0 | s1 | s2 | Mean |
|---|---|---|---|---|
| base | 20.69 | 20.34 | 20.31 | 20.44 |
| measured | 21.13 | 20.97 | 21.29 | 21.13 |
| uniform | 21.79 | 21.86 | 21.70 | 21.78 |
| shared2 | 22.31 | 21.85 | 21.94 | 22.03 |
| cprand | 21.91 | 21.80 | 21.53 | 21.75 |
| cpbal | 21.63 | 21.71 | 21.33 | 21.56 |
| cpada | 22.00 | 21.87 | 21.54 | 21.80 |

| RDD | s0 | s1 | s2 | Mean |
|---|---|---|---|---|
| base | 51.03 | 51.03 | 51.05 | 51.04 |
| measured | 51.18 | 51.83 | 51.33 | 51.45 |
| uniform | 50.63 | 50.98 | 50.48 | 50.70 |
| cprand | 49.97 | 50.22 | 49.99 | 50.06 |
| cpbal | 50.05 | 50.28 | 50.42 | 50.25 |
| cpada | 50.05 | 50.02 | 49.83 | 49.97 |

**Test macro AP50** (pycocotools, job 817465):

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

**dacl10k test comparisons:**

| Comparison | Difference | Seed t | Bootstrap (1,000 resamples) | Claim |
|---|---|---|---|---|
| cpbal − uniform | +0.49 (pycocotools) / +0.48 (bootstrap code) | [0.19, 0.79] | [−0.06, 0.94] | no |
| cpada − uniform | +0.49 | [0.07, 0.91] | [−0.05, 0.99] | no |
| cpada − cpbal | −0.00 (seeds +0.06, −0.03, −0.03) / +0.01 | — | [−0.28, 0.27] | no |
| cprand − uniform | +0.04 | — | — | no |

**Reading:**
- Balanced pasting is about +0.5 on dacl10k in all three seeds, but not significant at the image level, and not after Holm.
- Adaptive pasting adds nothing over balanced.
- On RDD, every paste arm is 0.5–0.8 below uniform.
- On dacl10k, validation ranked cpbal below uniform, while test ranked it above.

---

## 18. Polygon-mask copy-paste kill test

**Code:**
- `cua/datasets/dacl10k_masks.py` builds `train_masks.json`: 4,609 images and 32,391 polygons, checked box by box against the training records.
- `cua/augment/paste.py` rasterises the polygon inside the crop and pastes only those pixels. The pasted box is fitted to the mask.
- The arm is `cpbalm`. Tests are in `tests/cua/test_paste_masks.py`.

**Go rule, fixed before the runs:** cpbalm − uniform ≥ +0.74 AP50 (the dacl10k F MDE), with both the seed t interval and the bootstrap excluding 0.

**Validation (best epoch):**

| Arm | s0 / s1 / s2 | Mean |
|---|---|---|
| uniform | 21.79 / 21.86 / 21.70 | 21.78 |
| cpbal | 21.63 / 21.71 / 21.33 | 21.56 |
| **cpbalm** | 21.82 / 22.13 / 21.76 | **21.90** |

**Test** (bootstrap code's AP computation; see Section 1, items 1 and 2):

| Arm | s0 / s1 / s2 | Mean |
|---|---|---|
| uniform | 22.32 / 22.33 / 22.47 | 22.37 |
| cpbal | 22.94 / 22.76 / 22.85 | 22.85 |
| cpbalm | 22.57 / 22.07 / 22.43 | 22.36 |

| Comparison | Difference | Seed t | Bootstrap |
|---|---|---|---|
| cpbalm − uniform | −0.02 | [−0.65, +0.62] | [−0.61, +0.55] |
| cpbalm − cpbal | −0.49 | [−0.91, −0.07] | [−0.97, +0.05] |
| cpbal − uniform | +0.48 | [+0.16, +0.79] | [−0.06, +0.94] |

**Verdict: no-go.** Masks give no gain over uniform and trail rectangles on all three seeds. Validation ranked masks first, and test reversed it.

---

## 19. Positive control (in progress)

Sources: `reports/cua_control/PROTOCOL.md`, `cua/analysis/control.py`, `cua/augment/menu.py` (`PLANT`, `drop_class`), `cua/train.py` (`plant`, `resolve_plant`), `tests/cua/test_positive_control.py`.

**Design:**
- A planted edit deletes every annotation of the most frequent dacl10k training class. Pixels are unchanged, and it is the identity on images without that class.
- It is added to the menu of five arms (uniform, shared, cua, shared2, cua2) × 3 seeds = 15 runs, `K-dacl10k-fasterrcnn-s{0,1,2}-<arm>`.
- Four checks are fixed in the protocol:

| Check | Pass condition |
|---|---|
| O0 harm under uniform | target-class test AP50 of K-uniform is below C-uniform, both intervals below 0 |
| O1 detection | t ≤ −2 and the target class has the most negative utility in the planted row, every seed; contrast interval below 0 |
| O2 action | planted-edit rate ratio, images with vs without the class, ≤ 1/3 in every seed |
| O3 accuracy | CUA beats K-uniform on target-class test AP50, both intervals above 0 |

**Status:**
- Submitted as jobs 817902–817916.
- At the last check, 2 of 15 had started (seed 0 uniform at epoch 13 of 30, seed 0 shared at epoch 6) and 13 were queued. None had finished, and there were no errors.
- **No results yet.**
- Local tests: 9 pass for this feature. The full fast suite gives 467 passed, 1 skipped, 18 deselected (slow).

---

## 20. The paper

File: `research_paper/main.tex`. Title: *Class-Aware Augmentation Selection for Multi-Class Object Detection*. IEEEtran conference format, 15 pages as compiled. Backup before the latest rewrite: `research_paper/drafts/main_backup_before_negative_results_1125.tex`.

**Done in the last rewrite:**
- Copy-paste is described as outside the menu.
- The measured-effect policy is described as not an upper bound.
- CUA-v2 is added (Section "A Lower-Variance Estimator").
- Holm and MDE are added to the statistics paragraph, together with the admission that the extra-seed rule was not run.
- Results VI.B is written (tables `tab:main`, `tab:comparisons`, `tab:ablations`).
- New Section VI.C on copy-paste (table `tab:paste`).
- A discussion paragraph on why class-specific selection fails, and updated limitations.

**Real-image figures** (`research_paper/figures/make_assets.py` was run on CCAST; `build_real_figures.py` was run locally):
- Fig. 2 (`cua_framework_real.pdf`): real dacl10k images, the detector schematic, and the real v1 utility table.
- The augmentation gallery (`augmentation_gallery.pdf`).
- The copy-paste figure (`copy_paste_examples.pdf`), rectangle vs mask.

**Open `\todo` items (12 markers):**
- Rewrite the intro paragraph, contributions and title.
- Fig. 1 image (two related markers: image choice and caption).
- Measured GPU-hours.
- Secondary metrics paragraph.
- Forest plot rebuilt for v1 + v2 with Holm marking, plus its caption.
- Estimator-agreement figure, plus its caption.
- The mask copy-paste paragraph.
- The headline discussion.

**Text errors still in the paper:** 26,863 (line 236), and 64% (line 524, should be 62.5%). The PBT comparison values (line 524) are unverified. See Section 1.

---

## 21. Code and infrastructure

- **Package:** `cua/`, with records, the three dataset adapters, augmentation, estimators, policies, trainer with exact resume, evaluation, analysis, CCAST job generation and PBT.
- **History:** 82 commits on branch `study/class-utility-augmentation`. The last commit is `a583cd6`.
- **Uncommitted work:**
  - Modified: `cua/augment/menu.py`, `cua/augment/paste.py`, `cua/ccast/make_jobs.py`, `cua/experiments/specs.py`, `cua/paste_preview.py`, `cua/train.py`, `reports/cua_paste_pilot/VAL_SUMMARY.md`, `research_paper/main.tex`, `research_paper/main.pdf`.
  - New: `cua/analysis/control.py`, `cua/datasets/dacl10k_masks.py`, `reports/cua_control/`, `reports/cua_stats/`, the figure scripts and assets, `tests/cua/test_paste_masks.py`, `tests/cua/test_positive_control.py`, the paper backup, and this report.
- **Config fingerprints:** new config fields are left out of the run fingerprint at their defaults, so existing run IDs are unchanged. Pinned in tests: C-dacl10k uniform `2ad52d017282`, cpbal `4a46d3daa0b0`.
- **CCAST:** PBS Pro, A10 GPUs. Every login costs a Duo push.

---

## 22. Run inventory (CUA study)

| Family | Runs | Status |
|---|---|---|
| Epoch-budget runs | 3 | done |
| Part A (A-) | 258 | done |
| Part C v1 (C-, 5 arms × 3 seeds × 6 pairs) | 90 | done, test-evaluated |
| Ablations (X-) | 18 | done, validation only |
| Tuning (T-) | 6 | done, validation only |
| CUA v2 (C-, shared2/cua2) | 36 | done, test-evaluated |
| PBT pilot | 4 members × 10 windows | done |
| Copy-paste (P-, rectangles) | 18 | done, test-evaluated |
| Copy-paste (P-, masks) | 3 | done, test-evaluated |
| Positive control (K-) | 15 | submitted; 2 running, 13 queued at the last check |

Before the CUA study: 1 crop-classification run, the small learning checks, 5 full RDD detection runs (64.46 h), and the RDD v2 pilots. Total GPU-hours for the CUA study have not been measured.

---

## 23. What the evidence supports, and what is not done

**Supported:**
- Class-dependent edit effects exist for Faster R-CNN on all three datasets and for RetinaNet on VOC. Outside mosaic (and blur on RDD), they are small, about 1–3 AP50 per cell.
- On RDD, harms are shared by all classes.
- No class-aware method tested beats the best simple baseline after Holm, except cua2 on VOC R (+0.52, seed Holm p 0.048). That gain is not explained by class information.
- This covers CUA v1, v2, the measured table, PBT, adaptive copy-paste and mask copy-paste.
- CUA beats its own pooled version nowhere after Holm.
- The estimator's utilities show no class signal within an edit and rank mosaic and blur highest for every class.
- Validation and test rankings disagreed twice for copy-paste.

**Not done or not verified:**
1. Positive control results (running).
2. Official pycocotools values for cpbalm (on CCAST, unread).
3. PBT comparison values (base 50.72, measured 51.04): which epoch.
4. The cause of the 0.04 AP difference between pycocotools and the bootstrap code's AP for cpbal seed 1.
5. Extra seeds for dacl10k RetinaNet (pre-registered rule, not run).
6. The class-targeted study and the tolerance-gated ablation described in early drafts (not run).
7. The image bootstrap for the RDD paste arms (not run).
8. H4 for v2 (not computed).
9. Total GPU-hours (not measured).
10. Paper `\todo` items (Section 20) and the three text fixes (Section 1, items 3, 4, 6).
11. Nothing has been committed since `a583cd6`.

---

## Appendix A. Part A: mean ΔAP50 per edit and class (3 seeds)

Generated from `reports/cua_part_a_report/deltas.csv`. Validation, short-run protocol of Section 9.

### rdd Faster R-CNN

| Edit | D00 | D10 | D20 | D40 |
|---|---|---|---|---|
| blur | -0.67 | -1.20 | -1.25 | -1.98 |
| brightness | -0.15 | -0.46 | -0.26 | +0.20 |
| contrast | -0.04 | -0.46 | -0.23 | -0.45 |
| erase | -0.24 | -0.36 | -0.20 | -0.19 |
| hue | -0.07 | -0.35 | -0.37 | -0.07 |
| jpeg | -1.03 | -0.66 | -0.49 | -0.51 |
| mosaic | -2.37 | -3.74 | -3.61 | -4.48 |
| noise | -0.83 | -0.66 | -0.35 | -0.22 |
| rotate | -0.58 | -1.63 | -0.27 | -0.52 |
| saturation | +0.14 | -0.17 | -0.05 | +0.07 |
| sharpen | +0.45 | +0.30 | -0.30 | +0.09 |
| translate | -0.18 | -1.11 | -0.48 | -0.00 |

Seeds per cell: [3].

### rdd RetinaNet

| Edit | D00 | D10 | D20 | D40 |
|---|---|---|---|---|
| blur | -0.23 | -0.76 | -0.79 | -1.15 |
| brightness | -0.28 | -0.20 | -0.20 | -2.10 |
| contrast | +0.20 | -0.34 | +0.70 | -0.22 |
| erase | +0.05 | -0.44 | -0.85 | -0.80 |
| hue | +0.36 | -0.34 | -0.29 | -0.97 |
| jpeg | -0.46 | -0.46 | +0.39 | -0.79 |
| mosaic | -1.31 | -4.53 | -2.06 | -3.82 |
| noise | -0.83 | -0.46 | -1.00 | -1.45 |
| rotate | -0.36 | -1.26 | +0.05 | -0.66 |
| saturation | +0.29 | -0.50 | -0.22 | -0.54 |
| sharpen | -0.25 | +0.06 | -0.18 | -1.21 |
| translate | +0.19 | -0.67 | -0.25 | -1.78 |

Seeds per cell: [3].

### dacl10k Faster R-CNN

| Edit | Crack | Wetspot | Efflorescence | Rust | Rockpocket | Hollowareas | Cavity | Spalling | Graffiti | Weathering | Restformwork | ExposedRebars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur | -0.39 | +0.27 | -1.14 | -0.44 | +0.02 | +0.17 | -0.29 | -0.26 | -0.69 | -0.01 | -0.42 | -0.03 |
| brightness | +0.17 | +0.09 | -0.40 | +0.01 | -0.02 | +0.36 | -0.01 | +0.16 | +0.18 | -0.07 | -0.87 | +0.23 |
| contrast | +0.47 | +0.08 | -0.33 | -0.51 | -0.03 | +0.74 | -0.03 | +0.06 | -0.16 | -0.43 | -0.90 | -0.42 |
| erase | -0.52 | -0.03 | -0.17 | +0.02 | -0.02 | -0.02 | -0.05 | +0.05 | -0.40 | -0.28 | -1.37 | -0.13 |
| hue | -0.01 | +0.24 | -0.13 | -0.55 | -0.05 | +0.23 | -0.05 | +0.01 | -0.62 | -0.52 | -0.62 | +0.13 |
| jpeg | +0.15 | +0.08 | -0.26 | -0.53 | -0.01 | -1.30 | -0.20 | -0.20 | -0.97 | -0.55 | -1.04 | +0.50 |
| mosaic | -1.98 | +0.65 | -1.11 | -1.85 | +0.13 | +0.83 | -1.19 | -0.25 | +0.43 | -0.74 | -3.12 | +0.47 |
| noise | -0.66 | +0.16 | +0.21 | -0.09 | +0.02 | -1.28 | +0.11 | +0.28 | -0.94 | -0.88 | +0.10 | +0.18 |
| rotate | -0.07 | -0.28 | -0.41 | -0.04 | -0.01 | +0.77 | -0.14 | -0.38 | +0.57 | +0.03 | -0.17 | +0.73 |
| saturation | +0.06 | -0.09 | -0.17 | -0.39 | -0.03 | +0.81 | +0.08 | -0.03 | -0.33 | -0.18 | -1.13 | +0.14 |
| sharpen | +0.32 | +0.18 | +0.44 | +0.05 | -0.02 | +0.06 | +0.01 | +0.03 | -0.52 | -0.10 | -0.96 | +0.37 |
| translate | +0.17 | +0.02 | -0.04 | -0.35 | -0.00 | +0.59 | +0.10 | -0.12 | -0.71 | -0.10 | -0.34 | +0.38 |
| vflip | -0.14 | -0.24 | -0.56 | -0.03 | -0.02 | +0.62 | -0.27 | -0.18 | +0.15 | -0.65 | -2.14 | -0.19 |

Seeds per cell: [3].

### dacl10k RetinaNet

| Edit | Crack | Wetspot | Efflorescence | Rust | Rockpocket | Hollowareas | Cavity | Spalling | Graffiti | Weathering | Restformwork | ExposedRebars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur | -0.08 | -0.19 | -0.26 | +0.08 | +0.04 | +0.25 | +0.01 | +0.29 | -0.69 | +0.31 | -0.03 | +0.02 |
| brightness | -0.18 | -0.31 | -0.28 | +2.26 | +0.01 | +0.90 | -0.07 | +0.56 | -0.53 | -0.84 | -0.01 | -0.01 |
| contrast | -0.18 | -0.15 | -0.23 | +1.62 | +0.00 | +0.42 | +0.29 | +1.12 | -0.91 | -1.06 | -0.02 | -0.07 |
| erase | -0.30 | +0.02 | -0.25 | +1.22 | +0.00 | +0.37 | +0.08 | +0.13 | -0.66 | -0.06 | +0.03 | -0.07 |
| hue | +0.00 | -0.04 | -0.14 | -0.19 | +0.01 | -0.26 | +0.06 | +0.03 | -0.09 | -0.57 | -0.03 | +0.08 |
| jpeg | -0.30 | +0.01 | -0.09 | +0.31 | +0.00 | -1.58 | +0.09 | +0.65 | +0.37 | +0.02 | -0.01 | -0.07 |
| mosaic | -0.45 | -0.60 | -0.47 | +1.40 | +0.00 | -4.39 | -0.14 | -0.70 | -0.31 | -0.79 | -0.03 | +0.30 |
| noise | -0.14 | -0.26 | -0.15 | +1.38 | +0.02 | +1.07 | -0.08 | +0.80 | -0.02 | -0.98 | -0.00 | -0.03 |
| rotate | -0.11 | -0.08 | +0.21 | -1.18 | +0.00 | +1.87 | +0.11 | -0.21 | -0.32 | +0.22 | -0.02 | -0.07 |
| saturation | -0.19 | -0.05 | -0.08 | -0.25 | +0.00 | +0.66 | -0.19 | +0.70 | -0.87 | -0.71 | -0.02 | -0.01 |
| sharpen | -0.08 | -0.13 | -0.25 | +2.32 | +0.05 | +0.65 | +0.32 | +0.51 | +0.53 | +0.07 | -0.03 | +0.26 |
| translate | -0.29 | -0.05 | -0.27 | +0.78 | +0.00 | +2.62 | +0.11 | -0.15 | +0.05 | -0.60 | +0.01 | -0.07 |
| vflip | +0.24 | -0.34 | -0.19 | +3.35 | +0.01 | -0.13 | +0.31 | +1.12 | -0.50 | -1.58 | -0.03 | +0.04 |

Seeds per cell: [3].

### voc Faster R-CNN

| Edit | aeroplane | bicycle | bird | boat | bottle | bus | car | cat | chair | cow | diningtable | dog | horse | motorbike | person | pottedplant | sheep | sofa | train | tvmonitor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur | -0.58 | -0.35 | -1.27 | +0.45 | -2.95 | -0.02 | -0.13 | -0.75 | -0.40 | -1.79 | -0.73 | +0.16 | +1.34 | +0.04 | +0.06 | -0.36 | -0.21 | -0.13 | +0.30 | +0.41 |
| brightness | -0.29 | -0.09 | -0.84 | -1.72 | -1.43 | -0.11 | -0.03 | +0.00 | +0.01 | -1.05 | -0.64 | +1.08 | +0.61 | -0.92 | -0.05 | -0.53 | +0.16 | +0.26 | -0.01 | +0.46 |
| contrast | -0.76 | +0.36 | +0.40 | -0.80 | -1.44 | -0.11 | -0.11 | -0.67 | -0.72 | +0.24 | -2.05 | +0.98 | -0.56 | -0.38 | -0.10 | -0.68 | -0.16 | -0.07 | +0.64 | -0.34 |
| erase | -0.24 | +0.26 | -0.47 | -0.75 | -1.81 | +0.27 | -0.79 | +0.22 | -0.66 | -1.43 | -0.88 | +0.88 | +0.04 | -0.41 | +0.10 | -1.35 | -0.37 | -0.22 | +0.51 | -0.25 |
| hue | -0.04 | +0.23 | +0.07 | +0.41 | +0.54 | -0.28 | -0.33 | -0.29 | +0.14 | -0.58 | -2.09 | +0.60 | -0.26 | -0.03 | -0.13 | +0.46 | -0.59 | -0.11 | +0.49 | -0.14 |
| jpeg | +0.02 | -0.21 | -0.76 | -0.11 | -0.94 | +0.20 | +0.01 | -0.85 | -0.26 | -3.38 | -0.78 | +0.63 | +0.83 | +0.28 | -0.12 | -0.61 | -0.45 | +0.04 | +0.20 | +0.47 |
| mosaic | -0.09 | -1.55 | -1.71 | -3.29 | +1.62 | -0.10 | -1.15 | -1.09 | +0.12 | +0.04 | -2.93 | +0.30 | +1.05 | +0.29 | -0.07 | -1.18 | -1.87 | -1.90 | -0.34 | -3.11 |
| noise | +0.04 | -1.11 | -0.51 | -0.70 | -1.96 | +0.65 | -0.04 | -0.14 | +0.22 | -1.49 | -1.62 | +0.39 | +0.08 | +0.33 | -0.03 | -1.06 | +0.27 | +0.34 | +0.35 | -0.44 |
| rotate | -0.53 | +0.62 | -0.11 | +0.09 | -1.46 | -0.33 | -0.39 | +0.05 | +0.20 | -2.06 | -1.56 | +1.45 | +0.39 | -0.35 | -0.10 | +1.24 | -0.25 | +0.42 | +0.62 | +0.16 |
| saturation | -0.11 | +0.14 | -0.38 | -0.19 | -1.00 | +0.05 | +0.17 | -0.22 | +0.31 | -0.64 | -0.69 | +0.82 | +0.02 | -0.37 | -0.10 | -0.74 | +0.19 | -0.16 | +0.58 | -0.21 |
| sharpen | +0.13 | -0.06 | -1.23 | -0.95 | +0.10 | +0.01 | -0.34 | -0.06 | +0.10 | +0.06 | -0.74 | +0.86 | +0.51 | -0.61 | -0.19 | +0.67 | -0.65 | +0.71 | +0.45 | -0.29 |
| translate | -0.51 | -0.10 | -0.35 | -0.16 | -0.25 | +0.33 | -0.17 | -0.17 | +0.52 | -0.76 | -1.19 | +0.75 | +0.56 | -0.57 | +0.06 | -0.65 | -0.05 | -1.08 | -0.10 | -0.47 |

Seeds per cell: [3].

### voc RetinaNet

| Edit | aeroplane | bicycle | bird | boat | bottle | bus | car | cat | chair | cow | diningtable | dog | horse | motorbike | person | pottedplant | sheep | sofa | train | tvmonitor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur | -0.03 | -0.61 | -0.00 | -0.65 | -0.85 | -0.62 | -0.66 | -0.12 | -0.62 | -3.21 | -0.99 | -0.17 | +0.66 | -0.01 | -0.01 | -0.73 | -1.85 | +0.18 | -0.17 | -1.33 |
| brightness | +0.44 | +1.24 | -0.08 | +0.95 | +0.28 | -0.64 | -0.52 | +0.49 | +0.40 | -1.18 | -1.98 | +0.06 | +0.24 | +0.09 | +0.09 | -0.32 | -1.07 | +1.07 | -0.08 | -0.56 |
| contrast | +0.23 | -0.22 | +0.44 | +1.35 | -0.84 | -0.11 | -0.53 | +0.14 | +0.39 | -1.81 | -1.08 | +0.20 | -0.36 | +1.83 | +0.23 | -0.12 | -0.68 | +0.52 | -0.25 | -1.25 |
| erase | +0.36 | +0.17 | -0.24 | -0.52 | +0.56 | -0.63 | -0.36 | +0.33 | -0.29 | -2.01 | +0.14 | +0.47 | -0.32 | +1.32 | +0.03 | -0.11 | -0.53 | +0.37 | -0.04 | -1.10 |
| hue | +0.76 | +0.52 | +0.89 | +0.78 | -0.48 | -0.25 | -0.56 | +0.37 | -0.26 | -0.74 | -0.97 | -0.44 | +0.45 | +1.50 | +0.18 | +0.09 | -1.63 | -0.53 | +0.41 | -0.95 |
| jpeg | -0.15 | +0.11 | +0.26 | -1.29 | +1.55 | -0.45 | -0.81 | +0.24 | +0.28 | -1.74 | -0.16 | -0.49 | +0.43 | +1.18 | -0.13 | -1.62 | -1.64 | +0.06 | +0.63 | -0.05 |
| mosaic | +0.83 | +2.10 | +1.00 | -0.06 | +6.96 | +0.68 | +1.32 | -0.05 | +2.75 | +2.67 | +0.84 | +0.71 | +2.14 | +2.97 | +1.42 | +3.83 | +0.98 | +4.18 | +1.83 | -0.66 |
| noise | -0.37 | -0.25 | -0.16 | +0.11 | -0.64 | +0.02 | -0.24 | +0.61 | +0.97 | -1.58 | -1.12 | -0.56 | +0.36 | +0.77 | -0.02 | -0.83 | -1.27 | +0.70 | +0.48 | -1.21 |
| rotate | +0.87 | +0.08 | +1.22 | -0.22 | -0.02 | -0.14 | -0.22 | +0.26 | -0.41 | +0.08 | -1.09 | +0.42 | +0.01 | +0.95 | -0.49 | -0.55 | -0.81 | +0.50 | -0.09 | -1.47 |
| saturation | +0.64 | -0.36 | -0.56 | -0.20 | +0.31 | -0.06 | -0.08 | +0.41 | +0.24 | -1.14 | -0.58 | +0.86 | -0.03 | +1.57 | -0.05 | +0.19 | -1.51 | +0.28 | +0.15 | -0.58 |
| sharpen | -0.12 | -0.90 | -0.05 | -0.06 | +0.55 | -0.10 | -0.56 | +0.34 | -0.15 | -0.26 | -0.77 | +0.17 | -0.63 | +0.11 | +0.12 | -1.23 | -0.97 | +1.73 | +0.00 | -0.61 |
| translate | +0.41 | +0.38 | -0.02 | -1.18 | +0.21 | +0.16 | -0.66 | +0.54 | +0.60 | -1.52 | -0.66 | -0.22 | +0.25 | +1.09 | +0.03 | -0.78 | -1.00 | +1.40 | -0.20 | -0.67 |

Seeds per cell: [3].
