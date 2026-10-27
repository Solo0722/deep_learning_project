# Working prototype: class-adaptive local augmentation for road-damage detection

**Milestone:** working prototype of the proposed algorithm; baseline comparison and documented debugging evidence.
**Submission snapshot:** October 26, 2026. Start with this README.

## What is implemented

The proposed component is an online augmentation controller around a standard COCO-pretrained Faster R-CNN ResNet-50-FPN detector. It groups connected overlapping damage boxes and applies one consistent local edit to a selected group. Policies are conditioned on the exact set of damage classes in the group. Paired training-image probes measure detection retention and auxiliary object loss. Each represented class must pass the retention constraint; unsupported or inadmissible policies use identity. The detector trains on complete images and all valid targets. Inference uses the ordinary detector without the controller.

The implemented sequence is identity, fixed image-level, fixed local, online shared local, and online class-adaptive local. The first four full training runs are complete. The class-adaptive implementation has passed local checks, and its CCAST job was observed running in user-supplied terminal output. Its completed RDD evaluation is not available in this submission.

## Baseline comparison

Each completed condition uses 20 epochs, batch size 4, seed 42, 640-pixel inputs, 26,863 training images, and 5,760 validation images. Every condition starts from the same declared pretrained initialization. Select the checkpoint by the highest validation AP50:95, with earlier epochs winning ties. Scores below are on a 0-100 scale.

| Condition | Selected epoch | Validation AP50:95 | Validation AP50 |
|---|---:|---:|---:|
| Identity | 14 | 22.823 | 51.142 |
| Fixed image-level | 14 | 22.810 | 50.687 |
| Fixed local | 18 | 22.555 | 50.160 |
| Online shared local | 14 | 23.059 | 51.033 |

The online shared controller has the highest observed AP50:95, 0.236 points above identity, while AP50 is slightly lower. This is a descriptive one-seed development result. Validation also selects the checkpoints, and pre-augmentation training trajectories already differ. The online-versus-fixed contrast changes both control and access to strength levels. These results do not establish a reliable improvement or the efficacy of the proposed class-adaptive controller. The reserved test set is not used for model evaluation.

Full comparison and limitations: `reports/fasterrcnn_online_shared_v1/RESULTS.md`. Machine-readable scores: `reports/fasterrcnn_online_shared_v1/comparison.json`. The figure is `reports/fasterrcnn_online_shared_v1/comparison.pdf`.

## Working-prototype and debugging evidence

- **81 local tests passed**, with zero failures, errors, or skips. See `reports/fasterrcnn_class_adaptive_v1/unit_tests.txt` and `unit_tests.json`.
- Tests cover independent class-set support, mixed-class gates, no cross-set borrowing, identity fallback, valid local edits, unchanged pixels outside the group, checkpoint restoration, and interrupted-update recovery.
- An actual pretrained detector passed a synthetic-image CPU integration check, including auxiliary loss observation, unchanged ordinary predictions, preservation of model/buffer/gradient/random states during probes, and fresh-model checkpoint/policy replay. See `actual_model_check.json` and `actual_model_log.txt` in the same folder. This is software evidence, not RDD accuracy evidence.
- The completed online shared run has audited controller records and saved detection metrics. Selected- and final-epoch COCO metrics were independently recomputed with zero numerical difference; all ten policy updates were reconstructed from saved observations. See `reports/fasterrcnn_online_shared_v1/ccast_verification.json` and `controller_verification.json`.
- Class-adaptive CCAST launch evidence is in `CCAST_LAUNCH_EVIDENCE.txt`. Completion, remote test results, and class-adaptive accuracy are not claimed. The original release JSON retains its earlier pre-launch status as historical metadata.

These validation logs were produced during implementation. Packaging rechecks the shipped source checksums; it does not claim to rerun the detector experiments.

## Files to inspect first

| Requirement | Main evidence |
|---|---|
| Proposed algorithm | `fasterrcnn_class_adaptive_v1/controller.py`, `class_set_feedback.py`, `local_augmentation.py` |
| Detector feedback | `fasterrcnn_class_adaptive_v1/probe_detector.py`, `probe_runner.py` |
| Training and validation | `fasterrcnn_class_adaptive_v1/baseline.py`, `models.py`, `metrics.py`, `checkpoints.py` |
| Reproduction settings | `fasterrcnn_class_adaptive_v1/protocol.json`, `README.md`, `run_job.sh`, `job.pbs` |
| Tests and actual-model check | `fasterrcnn_class_adaptive_v1/tests/`, `reports/fasterrcnn_class_adaptive_v1/` |
| Completed comparisons | `reports/fasterrcnn_online_shared_v1/RESULTS.md` and the other baseline report folders |
| Method and experimental setup | `research_paper/main_manuscript.pdf` and `main.tex` |

## How to run

Use the documented Python/PyTorch environment. `rdd_detection_v1_environment.txt` records the CCAST environment; `protocol.json` contains the pinned package versions. The local validation environment and its limits are recorded in `LOCAL_VALIDATION.json`.

From an environment with the required dependencies, the package tests can be run without the full RDD dataset:

```bash
cd fasterrcnn_class_adaptive_v1
python -m unittest discover -s tests -v
```

For actual training, follow the package README and CCAST commands. They require the external RDD2022 images, pretrained weights, the frozen split manifests, and a GPU allocation. CCAST paths in the scripts are specific to the existing project installation. The included PBS script repeats the target-environment tests and a real-data smoke check before the full run. Do not launch a duplicate job while the current job is running.

Raw images, pretrained weights, full checkpoints, and large prediction archives are excluded to keep this hand-in compact. Frozen split manifests, source checksums, audit summaries, and result tables are included. This submission documents a functioning implementation and completed comparisons; it does not replace the final class-adaptive evaluation or a fully completed paper.

## Integrity

`SHA256SUMS` covers every included file other than itself. `SUBMISSION_MANIFEST.json` lists the rechecked package member counts and evidence scope. The original frozen code packages are copied without modification.
