# RDD v2: bounded four-arm training pilot

This package runs three additional full-data epochs (epochs 2–4) in each of four arms. Every arm forks the **same completed identity epoch-1 checkpoint**, including model weights, SGD momentum, step and all RNG states. It does not start a 20-epoch experiment.

| Arm | Geometry | Local appearance |
|---|---|---|
| `geometry_only` | Common keyed transform | None |
| `fixed_local` | Same transform | Fixed distribution over all 24 audited actions |
| `online_shared` | Same transform | One global admissible action set, updated online |
| `class_adaptive` | Same transform | Admissibility for the classes in the selected overlap group, updated online |

## Frozen comparison

Training uses all 26,863 frozen training images, including 10,233 images without target-class boxes. Full validation uses all 5,760 validation images without augmentation after every epoch. The test manifest and test pixels are never opened. One seed (42), 640-pixel letterbox, batch size 4 (retain last batch), SGD learning rate 0.0025, momentum 0.9, weight decay 0.0005; the inherited learning-rate schedule is continued, not restarted. No AMP or TF32. cuDNN benchmark is off and deterministic is on in every arm. This is exact checkpoint restoration, not a universal promise of bitwise CUDA training reproducibility.

The 20-epoch schedule's decays at epochs 12 and 16 are not reached. The first-epoch learning-rate warmup is already complete. All backbone stages are trainable with FrozenBatchNorm2d, epsilon zero. The starting checkpoint is fixed by filename, byte count and SHA-256 in `protocol.json`; no alternative checkpoint or automatic weight download is allowed. Each arm starts with identical model, optimizer and RNG hashes. The new job does not modify the identity checkpoint.

A clean validation pass of the common warmup state is saved once. The primary descriptive comparison is **epoch 4**, with all intermediate curves and per-class AP retained. A best-epoch file is kept as checkpoint metadata, but is not the pilot endpoint. Report AP50:95, AP50, AP75, AR100, per-class AP, COCO area diagnostics and predictions on no-target images. Area thresholds use original-image pixels; they are resolution-dependent, not an invariant measure of crack size.

## Augmentation and controller

Pipeline: stored-orientation image → existing 640-pixel letterbox → optional existing local appearance edit → whole-image geometry. Appearance uses the audited union of target-box overlap groups, with the existing one-pixel inward taper. It is not segmentation; background and occluders within boxes may be edited. No new mask, copy-paste, auxiliary ROI loss, rotation, vertical reflection or class-label remapping is introduced.

Geometry is a uniform keyed draw over the 54 audited reflection/scale/translation combinations. It applies to **all training images**, including negatives. Every target and valid non-target annotation follows the same continuous-edge map. Any annotation outside the canvas rejects the entire geometric candidate to identity, without clipping, dropping boxes or resampling. Invalid non-target annotations disable geometry for that image and are counted. Fractional boxes are retained; a float32 collapse is a hard error. Horizontal reflection preserves the crack's relationship to the reflected road and does not swap D00/D10 labels.

For target-bearing images, the three appearance arms have a common 0.5 application opportunity and the same keyed uniformly selected overlap group. The fixed arm admits the entire 24-action bank. Sampling is uniform over eligible families, then operations, then tested levels; this is not uniform over all 24 actions and does not prefer the strongest level. Online identity fallbacks are retained as fallbacks, not re-drawn actions. Actual edit rates can differ even though opportunities are paired.

Controllers update immediately before epochs 2 and 4, using **their own current detector** and the same frozen 256-image panel at each update (64 anchor groups per class, one known source cluster per image). Plans are generated without pixels or predictions: the audited planner with fresh planning seeds 4202001 and 4204001, using the predeclared panel-A label, not a panel selected for favorable results. Training and geometry seeds remain 42. At epoch 4 the arms have different learned models, so their probe predictions must be measured separately. Overlap across update panels is allowed and documented; these are training-only observations, not an independent accuracy test.

Each update measures clean, geometry-only reference, and 27 appearance views: 24 actions with a second noise draw at each of three noise levels. The 29 views per image plus four repeat-parity forwards total 7,428 detector images/update; four updates across two online arms total 29,712. Ordinary predictions use no GT proposal injection or target argument. Matching after inference requires correct class, confidence ≥0.25 and IoU ≥0.5, one-to-one. Only reference-detected selected-group objects enter retention denominators. Each noise pair is averaged within its object. Eligibility requires retention ≥0.8 and at least 16 detected objects from 8 clusters. Class evidence is pooled across class sets; supported direct mixed-set failures veto actions. Sparse mixed sets are explicitly marked as borrowed evidence. Unsupported constituent classes route to identity. No stale carry, threshold relaxation or adaptive panel replacement.

Probe contexts verify unchanged parameters, buffers, gradients, optimizer, modes and training RNG. All raw predictions, matches, transformed boxes, pixel hashes, support and adopted policies are saved. Partial/interrupted attempts remain recorded and never become active policies. Two fixed noise draws are reused for a given probe image/level as in the audited diagnostic; geometry uses the current epoch index.

## Pairing, checkpoints and cost

Training image order, geometry, reference pixels, target boxes, local group and application opportunities form an epoch hash that must agree across arms. Separate hashes and counters record actual actions, identity fallbacks, borrowed-evidence opportunities, source/class exposure and changed pixels. These summaries are reconstructable from frozen manifests, panels and adopted policies. A changed-pixel count is not evidence that an edit was useful.

Each arm processes 80,589 additional training images and 20,148 optimizer steps; all four total 322,356 images and 80,592 steps. There are 12 full post-epoch validations, plus the common warmup validation. Online measurement cost is additional; this is an equal-training-budget comparison, not an equal-total-compute comparison.

Checkpoints retain model, optimizer, RNG, epoch, next batch, counters and controller policy/evidence references. Save every 1,000 steps and around probes/validation; retain completed epochs and latest intermediate. Resume verifies code, config, data, environment, checkpoint and evidence hashes. It does not repeat already committed training or adopted probes. An interrupted uncommitted probe is rerun from its authoritative checkpoint and its earlier cost remains in attempt records. Validation files are hashed. Raw images and historical experiments are never overwritten.

One PBS job runs arms sequentially on one A10, 4 CPUs, 32 GB RAM. PBS ceiling: 12 hours. In-process administrative cap: 11 hours including the job's test/smoke period, allowing time to save and archive. Earlier full-run timing suggests roughly 7–9 hours, but geometry, IO and checkpoint overhead make that an estimate, not a guarantee. An incomplete job saves and stops; resumption is explicit and cannot extend the four-epoch endpoint. Total attempt cost, including partial probes, should be summed when interpreting runtime.

## Run

Upload/extract this package, then from the CCAST login node:

```bash
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_pilot_v1
bash submit.sh start
```

The PBS script activates `pytorch` and the existing `rdd_detection_v1` overlay. Do not request an interactive GPU or activate an environment on the login node. The job verifies files, runs unit tests, then a real GPU smoke check with four tiny arms, positive/negative images, actual probe collection, unaugmented validation, checkpoint reload/inference parity and a mid-epoch continuation. The full pilot starts only after that gate passes, reloading the original warmup checkpoint. Smoke panels are intentionally tiny and may have identity fallbacks; their metrics are software checks, not research results.

Check the latest submitted job:

```bash
bash /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_pilot_v1/status.sh
```

Disconnecting SSH does not cancel a PBS batch job. Reconnect and check status; do not submit a duplicate. If a run ends at its cap, inspect `runs/rdd_v2_pilot_seed42_v1/status.json` and the log before explicitly running `bash submit.sh resume`. A completed pilot is never resumed. A failure before the pilot output exists requires inspection; `resume` is not a way to bypass failed tests or the smoke gate.

Live log: `runs/pilot_job_logs/<PBS_JOBID>.log`. Main output: `runs/rdd_v2_pilot_seed42_v1/`. The completion log prints a unique records archive and SHA-256. Full checkpoint tensors remain on CCAST and are excluded from the download archive. GPU smoke outputs remain at `runs/rdd_v2_pilot_smoke_<PBS_JOBID>/`; the pilot archive includes its gate result and source path. Read `comparison.json`, all arm histories, augmentation counts and raw controller evidence together.

## Interpretation and next decision

This pilot tests short-horizon training behavior and whether the controller remains supported and class-specific as the models change. It does not establish convergence, statistical significance, or superior final accuracy. All arms include geometry, so it cannot estimate geometry's effect relative to identity. Validation is development data already used in this project; the test set remains closed.

Before scaling, inspect final-epoch and class-specific changes (especially D40), trajectories, negative-image predictions, support, collapse to shared/identity behavior, actual edit exposure and cost. If the class-adaptive route collapses, stalls, or worsens performance without an interpretable benefit, do not automatically scale this recipe. A favorable short pilot can motivate a separately frozen longer comparison. There is no automatic 20-epoch continuation and no post-hoc numeric success threshold.

`LOCAL_VALIDATION.json` states what was tested locally and the version difference from CCAST. `baseline.py` is inherited internal training/checkpoint infrastructure; the supported entry point is `run_job.sh` / `pilot.py`, not the inherited baseline CLI.
