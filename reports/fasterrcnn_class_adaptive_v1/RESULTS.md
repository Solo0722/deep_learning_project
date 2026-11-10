# Online class-adaptive local augmentation: completed-run audit

## Finding

The controller executed its declared exact-class-set policy, but the primary comparison did not show an accuracy gain. Selected validation AP50:95 is 22.563: -0.496 percentage points relative to online shared, -0.260 relative to identity, and +0.008 relative to fixed local. These are descriptive results from one training run per condition, not an equivalence or statistical significance result.

## Controlled comparison

Each condition started fresh with the same COCO initialization recipe, seed 42, frozen manifests, detector, optimizer schedule and evaluation. All completed 20 epochs, 134,320 optimizer steps and 537,260 image presentations on 26,863 training images. The 5,760-image validation set contains 8,225 retained target boxes. Maximum validation AP50:95 selects a checkpoint; earliest ties win. The test partition was not evaluated.

| Condition | Selected epoch | AP50:95 | AP50 | Final AP50:95 |
|---|---:|---:|---:|---:|
| Identity | 14 | 22.823 | 51.142 | 22.392 |
| Fixed image-level | 14 | 22.810 | 50.687 | 22.575 |
| Fixed local | 18 | 22.555 | 50.160 | 22.487 |
| Online shared local | 14 | 23.059 | 51.033 | 22.588 |
| Online class-adaptive local | 18 | 22.563 | 50.183 | 22.493 |

All scores use a 0-100 scale; differences are percentage points.

## Class-level results

| Class | Targets | Identity AP50:95 | Online shared AP50:95 | Class-adaptive AP50:95 | Adaptive minus shared |
|---|---:|---:|---:|---:|---:|
| D00 (Longitudinal cracks) | 3888 | 24.744 | 24.905 | 24.235 | -0.670 |
| D10 (Transverse cracks) | 1770 | 20.206 | 20.050 | 19.446 | -0.604 |
| D20 (Alligator cracks) | 1592 | 29.975 | 30.162 | 30.069 | -0.093 |
| D40 (Potholes) | 975 | 16.367 | 17.120 | 16.502 | -0.618 |

## Controller behavior and exposure

All ten updates used the same prospective 256-image/group panels as online shared. The budget was 71,680 additional image forwards, plus auxiliary ROI computation. All 15 exact-set states per update, including unsupported states, were independently reconstructed from saved per-object observations.

Of 158,151 matched application opportunities, 144,568 applied an edit and 13,583 (8.59%) used identity. The recorded changed-image count is 144,418. Online shared applied 158,151 edits, of which 156,819 changed pixels. Thus opportunity counts match, while actual edit exposure does not.

Fallback reasons: unobserved class set = 1,205, insufficient support = 12,378, no admissible family = 0.

Applied strength distribution: level 1 = 12,443 (8.61%), level 2 = 22,002 (15.22%), level 3 = 110,123 (76.17%).

| Exact class set | Active updates / 10 | Opportunities | Applied edits | Identity |
|---|---:|---:|---:|---:|
| D00 | 10 | 63,509 | 63,509 | 0 |
| D10 | 10 | 25,353 | 25,353 | 0 |
| D20 | 10 | 34,471 | 34,471 | 0 |
| D40 | 9 | 14,109 | 12,645 | 1,464 |
| D00+D10 | 9 | 9,081 | 8,142 | 939 |
| D00+D20 | 0 | 2,933 | 0 | 2,933 |
| D00+D40 | 0 | 498 | 0 | 498 |
| D10+D20 | 0 | 2,250 | 0 | 2,250 |
| D10+D40 | 0 | 191 | 0 | 191 |
| D20+D40 | 1 | 4,452 | 448 | 4,004 |
| D00+D10+D20 | 0 | 719 | 0 | 719 |
| D00+D10+D40 | 0 | 97 | 0 | 97 |
| D00+D20+D40 | 0 | 313 | 0 | 313 |
| D10+D20+D40 | 0 | 126 | 0 | 126 |
| D00+D10+D20+D40 | 0 | 49 | 0 | 49 |

Exact-set support constraints and class-specific strengths are observable implementation behavior. Neither passing the 80% retention gate nor increasing the auxiliary training loss establishes useful regularization or label preservation. The saved results do not identify whether stronger edits, missing support, family weighting, or training variability caused the downstream difference.

The same-snapshot pooled-policy diagnostic in `controller_verification.json` is exploratory arithmetic on the class-adaptive detector snapshots. It is not an alternative trained detector and supplies no counterfactual AP estimate.

## False alarms and runtime

| Condition | No-target images with detections / 2,193 | Fraction | Full-run hours |
|---|---:|---:|---:|
| Identity | 858 | 39.12% | 11.09 |
| Fixed image-level | 867 | 39.53% | 11.51 |
| Fixed local | 817 | 37.25% | 13.26 |
| Online shared local | 886 | 40.40% | 14.14 |
| Online class-adaptive local | 806 | 36.75% | 14.45 |

Detections here use confidence 0.25 and are false positives relative to retained annotations; negative images were not visually re-audited. Full-run time includes validation and probes but excludes tests and smoke; between-run timing does not isolate controller overhead. Recorded probe time was 57.19 minutes for class-adaptive and 56.41 minutes for online shared.

## Verification and limits

The returned archive SHA-256 is `c4817b3cb3bc69813776d06662a9070a7415e9552cab7c6b331c5237414d9b74`. All 123 archive members and checksum coverage were verified. Archived source bytes match the released package. The frozen training/validation manifests, original-coordinate validation ground truth, version/configuration records, training budgets, all 20 prediction files, epoch histories and best/final checkpoint metadata were checked. Direct COCO evaluation independently recomputed epochs 18 and 20, including per-class metrics; maximum discrepancy was zero. All 20 epochs of keyed training decisions, order, group geometry, fallbacks and exposure counters were replayed. Pixel counters and state-preservation flags were checked for internal consistency; raw pixels, probe logits/boxes, live detector state and full checkpoints were not re-executed.

Before augmentation began, epoch-1 AP50:95 already differed across conditions: 16.537 / 15.892 / 16.424 / 15.733 / 16.518 (table order). The cause is unestablished. These differences are not an estimate of final-run variance and must not be subtracted from later scores. One run per condition cannot establish repeatability or equivalence; validation-based selection also means these are development results. The first four conditions retain their earlier independent metric audits; this analysis rechecks their original archive hashes and comparison records.

## Decision

The planned five-condition development sequence is complete. Report a working, audited class-adaptive prototype with no observed gain over its primary online-shared comparator under this protocol. Keep this result rather than changing thresholds after observing it. Before further GPU work, inspect whether the retention/loss criterion selects useful augmentation strengths and whether exact-set support is adequate for the intended cases. A matched fixed-strength or uniform-family ablation can address attribution; matched seed repetitions are needed to assess reliability. Any revised controller is a new experimental condition. No additional training, detector inference, test evaluation, manuscript change or publication was performed by this audit.

Reproduce with `python reports/fasterrcnn_class_adaptive_v1/audit_ccast_results.py` then `python reports/fasterrcnn_class_adaptive_v1/compare_baselines.py`. The audit records Python/library versions and source hashes. Plot creation refuses existing figures; preserve or choose a fresh output directory when regenerating them. Primary outputs: `ccast_verification.json`, `controller_verification.json`, `comparison.json`, CSV tables, `comparison.png` and `controller_strengths.png`.
