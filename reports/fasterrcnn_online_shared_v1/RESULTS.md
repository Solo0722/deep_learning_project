# Online shared-local audit and four-condition comparison

Audit completed 2026-10-26. All scores below use the 0–100 scale; changes are percentage points.

**Verdict:** online shared local has the highest observed validation AP50:95 of the four runs, with a small gain over identity and a larger gain over the fixed-local recipe. The saved controller records are internally consistent and its decisions reproduce. These results do not establish a reliable improvement or isolate the benefit of online adaptation from using gentler augmentation strengths.

## Primary comparison

Each condition independently selects its checkpoint by maximum full-validation AP50:95, with the earliest epoch winning ties. This is the declared rule; the final epoch is a separate descriptive comparison.

| Condition | Selected epoch | AP50:95 | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|---:|
| Identity | 14 | 22.823 | 51.142 | 17.091 | 43.836 |
| Fixed image-level | 14 | 22.810 | 50.687 | 17.295 | 44.339 |
| Fixed local | 18 | 22.555 | 50.160 | 16.570 | 44.078 |
| Online shared local | 14 | **23.059** | 51.033 | 17.302 | 44.281 |

Online shared local gains **+0.236 AP50:95 points over identity**, **+0.249 over fixed image-level**, and **+0.504 over fixed local**. AP50 remains 0.108 points below identity. It therefore does not dominate every metric.

At the final epoch, AP50:95 is 22.392 / 22.575 / 22.487 / 22.588 in the same order. Online shared remains numerically highest, but its final-epoch advantage over fixed image-level is only **0.013 points**. Its selected AP50:95 falls by 0.472 points between epochs 14 and 20. Checkpoint selection matters; the selected-checkpoint rule has not been changed after observing these results.

The four conditions use the same declared pretrained initialization, seed 42, frozen training/validation manifests, detector, optimizer settings, image size, data order convention, and evaluation. Each completed 20 epochs, 134,320 optimizer steps and 537,260 image presentations. There are 26,863 training images and 5,760 validation images with 8,225 retained targets. Ground truth and nonaugmentation configuration match exactly across the saved records. No test data were read.

## Class results and false alarms

| Class | Identity AP50:95 | Fixed image-level | Fixed local | Online shared local | Online minus identity | Online minus fixed local |
|---|---:|---:|---:|---:|---:|---:|
| D00: longitudinal cracks | 24.744 | 24.784 | 24.090 | 24.905 | +0.161 | +0.816 |
| D10: transverse cracks | 20.206 | 20.266 | 19.520 | 20.050 | −0.156 | +0.530 |
| D20: alligator cracks | 29.975 | 29.631 | 30.114 | 30.162 | +0.187 | +0.048 |
| D40: potholes | 16.367 | 16.559 | 16.496 | 17.120 | +0.753 | +0.624 |

Online shared improves every class relative to fixed local, although the alligator-crack difference is only 0.048 points. Relative to identity, potholes supply the largest improvement and transverse cracks decline. These class differences are descriptive, not separately confirmed effects.

At confidence ≥0.25, among the same 2,193 validation images with no retained target annotations:

| Condition | Images with at least one detection | Fraction | Detection boxes |
|---|---:|---:|---:|
| Identity | 858 | 39.12% | 1,637 |
| Fixed image-level | 867 | 39.53% | 1,640 |
| Fixed local | 817 | 37.25% | 1,538 |
| Online shared local | 886 | 40.40% | 1,707 |

The AP50:95 gain does not come with fewer false alarms at this fixed threshold. These are false positives relative to retained annotations; annotation completeness and score calibration were not re-audited.

## What the controller actually did

All ten scheduled updates occurred before epochs 2, 4, …, 20. Each used 256 distinct training images from 256 distinct source clusters, one selected local overlap group per image, and the entire declared grid: eight operations, three strengths and two draws for noise. Each update made 7,168 image forwards; the run used 71,680 probe image forwards in total. These are image-view counts, not additional training updates.

All four classes met the minimum reference support at every update. The full run had **zero controller identity fallbacks**. This differs from the earlier small smoke check, where insufficient support intentionally triggered identity.

| Update before epoch | Photometric strength | Blur/noise strength | Photometric probability |
|---:|---:|---:|---:|
| 2 | Unavailable | 1 | 0.00% |
| 4 | 1 | 1 | 51.51% |
| 6 | 1 | 1 | 49.95% |
| 8 | 2 | 1 | 50.11% |
| 10 | 3 | 1 | 50.55% |
| 12 | 1 | 1 | 49.07% |
| 14 | 1 | 1 | 48.67% |
| 16 | 3 | 1 | 51.17% |
| 18 | 3 | 1 | 50.76% |
| 20 | 3 | 1 | 49.55% |

Strength 1/2/3 means mild/medium/strong. Fixed local uses strength 2. For blur, the radii are 0.4/0.8/1.2; for noise, the standard deviations are 0.01/0.02/0.03 on normalized intensities.

**Observed:** the gate always selected mild blur/noise. At every update, medium Gaussian blur failed the 0.8 retention threshold for at least one class. Photometric strength changed over training, including being unavailable initially. Whenever both families were eligible, their probabilities were nearly uniform: photometric probability ranged from 48.67% to 51.51%.

**Interpretation:** most visible policy adaptation was in eligibility and strength selection. This run supplies little evidence that strong family-probability reweighting is necessary. It does not isolate whether the gain requires online feedback; a fixed mild recipe could potentially explain it and has not been tested.

### Evidence relevant to class conditioning

An exploratory reconstruction of maximum admissible strengths from the pooled class measurements finds different class limits in six of ten updates for each family. At epochs 12 and 14, photometric level 3 passed for D00, D10 and D20, but D40 contrast reduction retained only 0.692 and 0.625 of the previously detected potholes. The common policy selected level 1 for everyone. At epoch 14, 40 potholes across 33 clusters were reference-detected, so this was not an insufficient-support fallback.

This is a measured shared-policy constraint and a reason to test class conditioning. It is **not** a class-set-specific policy estimate: the measurements pool objects across different group contexts. Allowing a stronger edit on one class does not itself establish a training benefit, semantic label preservation, or transfer to unseen examples. Probes are refreshed over time, so strength changes can reflect both detector changes and panel sampling.

### Exposure and edits that changed no pixels

Online shared and fixed local have exactly the same 158,151 application opportunities, selected groups and mask-area totals over all 20 epochs. The union masks cover an aggregate 7.255% of applied-image content; alpha-weighted coverage is 7.173%.

The online run reports 156,819 applications that changed pixels, leaving **1,332 applications with no pixel change (0.842%)**, including 79 in epoch 20. This is not controller identity fallback. In saved probes, 73 of 2,560 level-1 blur views also have identical reference/transformed RGB hashes; every other tested operation/strength has a nonzero recorded change. These observations are consistent with mild blur sometimes producing no byte-level change, but aggregate training records cannot assign every unchanged training edit to a particular operation.

Reported changed pixels are 3,546,431,748 online versus 4,100,790,428 for fixed local: 5.797% versus 6.703% of applied-image content. Changed-pixel count measures edit extent, not perceptual magnitude. The intervention has matched masks and opportunities but different operations, strengths and actual pixel exposure.

## Compute cost

| Condition | Full-run hours | A10 host |
|---|---:|---|
| Identity | 11.09 | gpu0035 |
| Fixed image-level | 11.51 | gpu0021 |
| Fixed local | 13.26 | gpu0020 |
| Online shared local | 14.14 | gpu0020 |

The online run took about 53 minutes longer than fixed local. Its ten recorded probe attempts total **56.41 minutes**, about 6.65% of full-run elapsed time. Whole-run differences include training, validation, I/O and machine variation; they are not a controlled measurement of controller overhead. No inference-time controller is added to the detector.

## Audit evidence and limits

The transferred archive SHA-256 is `b30df4e003be14cf8a53059341451c5101261d98d32e58cdc94866c793f0a231` (407,751,939 bytes). The audit checked all 119 files, all 118 declared file checksums and all 30 released package files. The archived code is byte-identical to the frozen release.

Checks include frozen train/validation separation by image, source cluster and decoded-RGB fingerprint; validation ground truth; all 20 prediction files' geometry, labels, counts and image coverage; run/history/CSV consistency; complete training progress; and checkpoint metadata/selection pointers. Independent direct COCO evaluation of saved detections at selected epoch 14 and final epoch 20 reproduced all aggregate and per-class metrics with **zero numerical difference**. The prior three result archives and analysis-source hashes were freshly checked; their existing independent metric recomputations were retained.

All ten controller policies were independently rebuilt from saved per-object reference/transformed detection indicators, IoUs and losses using a separate array-based implementation. This covers all 24 operation/strength cells, four class retention gates, support counts, normalized difficulties, probabilities and selected strengths. Panel choices match the prospective frozen plan. All 20 epochs of augmentation choices, order, group geometry and exposure were replayed and matched. Executed source checks and saved invariants report unchanged model/buffer/gradient/RNG state during probes.

The archive does not contain full checkpoint tensors, raw images, raw probe detection boxes/scores or ROI logits. Consequently this is an independent audit of saved predictions, observations and metadata, not a fresh detector-inference or pixel audit. Probe matching/loss computation cannot be re-executed from these records alone. Only selected/final AP metrics were recomputed; all 20 epochs received geometry/coverage/record checks.

There is one run per condition. The unaugmented first epoch already differs: AP50:95 is 16.537 / 15.892 / 16.424 / 15.733. This demonstrates that nominally matched seed/settings did not yield identical trajectories, but does not estimate final-result variance or justify subtracting warmup differences. The cause is not established. Repeated seeds are needed to judge reliability. Validation supplied checkpoint selection and these comparisons; there is no untouched test estimate. The large improvement after epoch 12 also appears in all four curves and coincides with the scheduled learning-rate reduction, so it is not unique evidence for the controller.

## Recommendation

Retain online shared local as the completed stage-4 comparator. Proceed to the planned class-adaptive condition with the same operation/strength grid, probe schedule and budget rules, support and retention gates, training budget and checkpoint-selection rule. The class-specific constraint observed above provides a concrete hypothesis; its downstream benefit is still untested.

Before claiming that online adaptation is responsible for a gain, add a matched fixed mild-strength control. Before claiming that induced-loss weighting contributes, compare against uniform weighting over eligible families. Before claiming reliability, repeat matched conditions across seeds. These are separate attribution questions, not grounds to discard the completed run. No further training or inference was launched by this audit.

Reproducible outputs: `ccast_verification.json`, `controller_verification.json`, `comparison.json`, the accompanying CSV tables, and `comparison.png/.pdf` plus `controller_policy.png/.pdf`. Analysis scripts are `audit_ccast_results.py`, `audit_online_controller.py` and `compare_baselines.py`; they preserve existing outputs. Source hashes are recorded in the JSON evidence files. Frozen releases and earlier results were not modified.
