# RDD v2 four-arm pilot: returned-results audit

Audit date: 2026-10-29. PBS job: 797282.bright04. Seed: 42.

The pilot completed and the numerical and controller audits passed. At the predeclared epoch-4 endpoint, geometry-only has the highest validation AP. Class-adaptive local augmentation improves over the two other appearance-augmentation arms, but does not improve over geometry-only. This establishes that the revised controller operates with adequate pooled support; it does not establish a detection advantage for the proposed method.

Recommendation: do not launch a four-arm, 20-epoch sweep from these results. Preserve the current recipe and investigate the already recorded smoke-run discrepancy before interpreting small differences. The next useful scientific test, if we continue, is a separately specified, bounded comparison of geometry-only and class-adaptive augmentation over a longer common horizon. Do not tune the retention threshold on these validation results. Independent paired seeds are required before claiming a repeatable advantage.

## What was compared

All four arms start from the identical completed epoch-1 identity checkpoint, restoring model parameters, optimizer momentum and RNG state. Each then trains for epochs 2, 3 and 4: 20,148 additional optimizer steps and 80,589 training-image presentations per arm. Each epoch uses all 26,863 frozen training images, including 10,233 images with no target-class objects. Validation uses the same 5,760 unaugmented images and 8,225 target boxes after every epoch. The test set is not part of this experiment.

The detector is Faster R-CNN with a ResNet-50 FPN backbone, initialized through the existing COCO-pretrained warmup. Images use a 640-pixel square prepared canvas. Training uses batch size 4, float32, SGD with learning rate 0.0025, momentum 0.9 and weight decay 0.0005. The original later learning-rate decays are not reached in this pilot.

Every arm uses identical keyed horizontal flips, small isotropic resizes and translations. Any candidate that would crop any known target or valid non-target annotation is rejected to identity. No target is clipped, dropped or relabelled. Consequently, this experiment compares appearance policies **on top of common geometry**; it cannot estimate the benefit of geometry relative to identity training.

| Arm | Local appearance policy |
|---|---|
| Geometry-only | No local appearance edits |
| Fixed local | Sample from the full fixed bank of 24 operation/level combinations |
| Online shared | One admissible action distribution shared by all classes, measured on the current detector |
| Class-adaptive | Route the distribution using the classes in the selected local overlap group |

Local edits have a common 0.5 application opportunity on target-bearing images, with the same selected overlap group. The bank contains brightness, contrast, gamma, blur and noise at three levels. Sampling is balanced across eligible families, then operations, then levels; it is not uniform over the 24 actions. The local region is a union of bounding-box masks, not a damage segmentation mask.

Before epochs 2 and 4, each online arm measures its own detector on the same preselected 256-image training panel. An action requires at least 16 reference-detected objects from 8 clusters and retention of at least 0.8 for the relevant class. Mixed groups intersect constituent-class permissions and apply supported direct-evidence vetoes. Missing direct mixed-group evidence is explicitly marked as borrowed. Detector tolerance remains a proxy for admissibility, not a measure of training utility.

## Final endpoint

Values are AP points on a 0–100 scale. Comparisons use epoch 4, not a retrospectively selected best epoch.

| Arm | AP50:95 | AP50 | AP50:95 difference from geometry-only |
|---|---:|---:|---:|
| Geometry-only | 19.8080 | 45.0466 | 0.0000 |
| Fixed local | 18.8610 | 43.4441 | -0.9470 |
| Online shared | 19.3656 | 43.8678 | -0.4424 |
| Class-adaptive | 19.6076 | 44.7577 | -0.2003 |

Class adaptation gains 0.7466 AP points over fixed local and 0.2421 over online shared. It remains 0.2003 points below geometry-only. The correct claim is partial recovery from the observed appearance-augmentation deficit in this run, not superiority over the strongest simple control.

| Arm | Epoch 2 AP50:95 | Epoch 3 AP50:95 | Epoch 4 AP50:95 |
|---|---:|---:|---:|
| Geometry-only | 17.7280 | 17.4890 | 19.8080 |
| Fixed local | 17.3650 | 16.8462 | 18.8610 |
| Online shared | 17.7438 | 17.5062 | 19.3656 |
| Class-adaptive | 17.0088 | 16.1053 | 19.6076 |

The common epoch-1 checkpoint has 16.5371 AP. Class adaptation trails substantially at epochs 2 and 3, then recovers at epoch 4. That trajectory motivates testing whether the deficit is transient, but does not prove it will disappear. Both the controller's measurement panel and the model change between updates; the epoch-4 recovery cannot be attributed solely to the policy refresh.

## Class and false-positive findings

| Arm | D00 longitudinal | D10 transverse | D20 alligator | D40 pothole |
|---|---:|---:|---:|---:|
| Geometry-only | 21.0168 | 17.2705 | 25.6914 | 15.2531 |
| Fixed local | 20.4964 | 16.4682 | 24.0580 | 14.4215 |
| Online shared | 21.2650 | 16.2790 | 25.9457 | 13.9725 |
| Class-adaptive | 20.8987 | 16.3008 | 25.8903 | 15.3408 |

The class-adaptive gain over online shared is concentrated in potholes: +1.3682 D40 AP points. Against geometry-only, however, D40 improves by only 0.0876, while transverse cracks lose 0.9697. Longitudinal cracks lose 0.1181 and alligator cracks gain 0.1989. These are descriptive, single-seed differences.

At the existing confidence threshold 0.25, the class-adaptive detector produces fewer false positives and fewer true detections. The following diagnostic independently matches predictions to annotated targets at IoU 0.5, in confidence order, one-to-one and with the correct class. Precision and recall are micro-averaged over all four classes; they are not COCO AP.

| Arm | No-target images with predictions / 2,193 | True positives | False positives | Precision (%) | Recall (%) |
|---|---:|---:|---:|---:|---:|
| Geometry-only | 1,038 | 5,238 | 15,274 | 25.54 | 63.68 |
| Fixed local | 838 | 5,085 | 14,158 | 26.43 | 61.82 |
| Online shared | 853 | 5,112 | 14,263 | 26.38 | 62.15 |
| Class-adaptive | 778 | 5,040 | 12,728 | 28.37 | 61.28 |

Relative to geometry-only, class adaptation reduces the number of no-target images with a prediction by 260 (25.0%), but loses 198 true-positive detections at this threshold. This is an interpretable precision/recall trade-off, not proof of superior detection. Score calibration can affect a fixed-threshold comparison. Annotation completeness also bounds the interpretation of apparent false positives.

Small-object AP is 9.1938 for geometry-only and 7.7948 for class-adaptive. This is a warning to retain in later evaluations, not a stable class-specific conclusion: the small-object subset contains 211 D00, 96 D10, only 2 D20 and 203 D40 objects. Size is defined in original-image pixels and therefore also depends on source resolution.

## Controller behavior and sparse support

All 16 pooled class/update cells across the two online arms exceed the frozen support requirements. The first update has exactly identical raw observation records in both arms, consistent with their common starting state and panel.

Entries below are reference-detected objects / distinct source clusters / admissible actions out of 24.

| Update | D00 | D10 | D20 | D40 | Global shared actions |
|---|---:|---:|---:|---:|---:|
| Both arms before epoch 2 | 61 / 58 / 16 | 54 / 50 / 14 | 55 / 55 / 23 | 42 / 36 / 13 | 9 |
| Online shared before epoch 4 | 81 / 73 / 22 | 65 / 61 / 18 | 59 / 57 / 23 | 58 / 47 / 18 | 13 |
| Class-adaptive before epoch 4 | 81 / 72 / 22 | 62 / 60 / 22 | 55 / 53 / 23 | 61 / 49 / 20 | 19* |

*The class-adaptive row's global distribution is a counterfactual computed from that arm's own observations; the arm actually uses its class-group routes. It is not the independently trained online-shared arm's policy.

The original sparse exact-class-set problem is reduced through pooling, but not eliminated for mixed groups. At the first update, none of the ten observed mixed-class sets has sufficient direct evidence for every constituent class. By epoch 4, two observed mixed sets, D00+D10 and D20+D40, do. Partially supported mixed cells can still veto actions: complete direct support of a group is not required for a supported constituent's veto.

Borrowed-evidence edit opportunities number 1,050/8,347 (12.58%) in epoch 2, 1,086/8,246 (13.17%) in epoch 3 and 384/8,280 (4.64%) in epoch 4. These routes are not independently certified mixed-group behavior.

Class adaptation does not collapse to identity or to the actual shared arm. Its sampled actions differ from the actual shared arm on 4,949, 4,926 and 4,787 opportunities in epochs 2, 3 and 4. However, comparison with the global shared policy computed from the **same class-adaptive model** shows the isolated routing distinction becoming smaller: different sampled actions fall from roughly 59% to 12.27% by epoch 4. Opportunity-weighted distribution distance falls from about 0.371 to 0.115. Thus the late policy is more permissive and less class-specific, although it remains distinct.

A leave-one-cluster-out sensitivity check changes two of the 96 pooled action gates at the first update, two for the shared arm's second update and one for the class-adaptive arm's second update. This limited sensitivity check is not a confidence interval or evidence of semantic safety. Reference recall is still only 52–65% across classes at the initial update; retention is conditional on already detected objects and does not directly protect reference misses.

## Exposure, integrity and cost

| Epoch | Images per arm | Appearance opportunities | Accepted geometry | Geometry rejected to identity |
|---|---:|---:|---:|---:|
| 2 | 26,863 | 8,347 | 21,801 | 5,062 |
| 3 | 26,863 | 8,246 | 21,744 | 5,119 |
| 4 | 26,863 | 8,280 | 21,674 | 5,189 |

All three appearance arms apply an admissible action at all 24,873 opportunities; there are no no-admissible-action fallbacks in training. Recorded changed-image counts are 24,803 for fixed local, 24,682 for shared and 24,688 for class-adaptive. An applied action can leave quantized pixels unchanged, particularly on very small regions. These pixel-change totals were not rerendered in this audit because the raw images are not in the archive.

The audit verified:

- The downloaded archive's SHA-256, all 77 internal file checksums and all 29 released source checksums. Configuration, pinned environment records and frozen train/validation/split-config hashes agree.
- Identical recorded model/optimizer/RNG fork state across the four arms, correct step/image counts and matching per-epoch input-pairing hashes.
- Independent official COCOeval recomputation for the common warmup and all four final endpoints from saved predictions and independently constructed validation ground truth. Every checked aggregate and per-class AP metric matches exactly; maximum numerical discrepancy is zero.
- Independent matching of all 29,696 saved controller views, affine transformation of all their target/non-target annotations, and recomputation of pooled gates, direct mixed evidence, vetoes and routed distributions.
- Controller state records match the appropriate starting fork or preceding epoch-3 validation state; recorded model/optimizer/RNG preservation and repeated-inference parity checks pass.
- Metadata-only replay over all 80,589 common image/epoch cases. Every non-pixel exposure counter agrees for all four arms, including source/class operation counts, opportunities, selected-group object counts, geometry rejection and borrowed-evidence usage. Logged pixel-dependent pairing hashes also agree across arms, but were not reconstructed without the raw images.

The main pilot invocation took 29,587 seconds, or 8.22 hours on one NVIDIA A10. This excludes the preceding test/smoke period. Each online arm uses 14,856 additional detector-image forwards for probing, including repeated-inference checks, costing approximately 13.9 minutes per arm in recorded controller elapsed time. The comparison equalizes training steps, not total compute. Sequential arm runtimes are not a controlled hardware benchmark.

The records archive contains no full checkpoint tensors or new image previews. Therefore this audit does not claim a fresh visual inspection of the pilot's actual rendered pixels or local re-inference of remote checkpoints. The earlier geometry/controller preview audits remain the visual evidence.

## What remains unresolved and what to do next

Update, 2026-10-29: the separate smoke archive has now been downloaded, verified and audited in `smoke_audit/RESULTS.md`. All eight smoke evaluations recompute exactly, and the online arms correctly use identity appearance. Nevertheless, the three arms with identical recorded treatment already have different model and gradient hashes after the first seven smoke updates. The archive cannot identify the first divergent operation. The original smoke gate did not test training equivalence and resumed only the class-adaptive arm. The immediate next action is a short controlled GPU diagnostic separating ordinary repeat-run variation, probe execution and resume; the proposed epoch-8 extension below remains deferred. The full-pilot measurements above are unchanged.

The following paragraphs preserve the recommendation made before the separate smoke archive was available.

The earlier GPU smoke log showed different short training trajectories for geometry-only and online arms even when the tiny probe forced appearance identity. The returned archive contains the smoke PASS summary, not its raw histories/checkpoint tensors. Its exact-state restoration and inference checks do not resolve that training-trajectory difference. CUDA nondeterminism is a plausible explanation, but has not been established as the cause. The approximately 0.60-point range on a 24-image smoke validation subset must not be used as an error bar for this 5,760-image comparison.

First inspect the already saved smoke records on CCAST at `runs/rdd_v2_pilot_smoke_797282.bright04/`; compare model/optimizer/RNG state, data-pairing records, resume boundaries and training trajectories. This uses existing evidence before requesting additional computation. If a software mismatch is identified, fix it and repeat only the necessary small verification. If the inputs/state agree and the remaining difference is numerical, record that limitation and quantify training variation with independent paired runs rather than claiming deterministic training.

My proposed next scientific stage is a separately frozen extension of geometry-only and class-adaptive training to a common epoch-8 endpoint, retaining the existing transforms, thresholds, learning-rate schedule and clean validation. Preselect any additional training probe panels before running. This is a practical bounded horizon to test the observed delayed recovery, not a statistically justified optimum. Do not resume the completed pilot in place; use a new stage with explicit checkpoint provenance. Keep the test set closed.

That two-arm stage asks whether the added controller can beat the simpler control over a longer horizon. It cannot, by itself, establish that class conditioning causes any advantage over shared online control. A final method claim would still require the shared comparator, paired independent seeds, a declared primary endpoint and repeatable gains. If the proposed method remains worse without a compelling declared benefit, stop scaling this recipe and revisit the intervention. Do not reinterpret tolerance, successful execution or favorable secondary diagnostics as evidence that the main accuracy hypothesis succeeded.

## Evidence files

Downloaded archive: `rdd_v2_pilot_seed42_v1_records_7297d6591f74.tar.gz`

SHA-256: `70ace2ca31c6e118e4d705cca0fb21f31eeecca70dee57cb246fcc9dca8fcfe5`

This folder contains `archive_audit.json`, `metric_recomputation.json`, `controller_audit.json`, `exposure_replay.json`, `endpoint_results.csv`, `class_results.csv` and the read-only extracted evidence under `returned_records/`. Audit programs are `audit_returned.py`, `recompute_metrics.py`, `audit_controller.py` and `replay_exposure.py`. The latter two reuse the earlier independent matching/affine audit helpers; the source experiment package was not modified.
