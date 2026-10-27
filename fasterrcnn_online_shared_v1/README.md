# Online shared local augmentation: Faster R-CNN

Stage 4 of the agreed sequence: identity, fixed image-level, fixed local,
**online shared local**, then online class-adaptive local. This package learns
one common augmentation policy from training feedback. It does not maintain
class-set-specific policies. All conditions start independently from the same
COCO weights; the fixed-local checkpoint is not continued.

## Question and limits

Does this common online policy improve selected-checkpoint validation AP50:95
over the completed fixed-local recipe under the same detector training budget?
The three controls have no reliable established augmentation gain in the
available seed. They also diverged during the unaugmented first epoch. Small
differences remain descriptive, not causal or statistically established.

This run changes family probabilities and selects from a wider strength grid.
It tests that complete online policy, not adaptation isolated from a fixed
policy optimized over the same grid. The fixed baselines were prospective
recipes, not validation-optimized competitors. Probe computation is additional
and reported; detector optimizer steps are matched. This is a seed-42
development comparison, not a final efficacy claim or an untouched test result.

## Controller protocol

All numerical choices are frozen prospectively in `protocol.json`; they are
engineering choices for this first run, not claimed optimal settings.

- Train epoch 1 without augmentation. Update the controller immediately before
  epochs 2, 4, 6, ..., 20, holding its policy fixed between updates.
- Each update deterministically hash-ranks 256 target-bearing training images,
  preferring distinct source clusters. It selects one group uniformly using
  exactly the fixed-local grouping and group-selection namespace. Images are
  not resampled because the detector misses a class. Panel identities, classes,
  source clusters and selected object indices are saved.
- Probe every operation/strength on that same panel and model state, one local
  group edit per image. Probes use eval/inference mode, no backward pass, and
  separate keyed randomness. Parameter, buffer, gradient and global RNG hashes
  are checked before/after each complete or interrupted update.
- Reference detections use the unchanged detector output, confidence >= 0.25,
  IoU >= 0.5, class-correct score-ordered one-to-one matching to all GT. Retention
  counts selected-group objects detected in the reference that remain detected
  after the edit. New detections cannot compensate for losing reference ones.
- Each class needs at least 8 reference-detected target objects across at least
  4 source clusters. These minima avoid decisions from one or two detections;
  they do not guarantee precise retention estimates. Each class and each
  operation in a family must reach retention 0.8. Choose the strongest tested
  admissible level separately per family, without assuming monotonicity.
  A failed family is unavailable. If no family qualifies, use identity.
- Compute supervised classification/localization losses for every selected
  object, including those missed at the detection threshold. Average losses
  within object, draws within object, then objects within class. Normalize the
  *net class-mean* increase by reference mean + 0.1 and clip to [0,5]. Average
  classes and operations equally within each family. Outside-group losses and
  background/RPN losses never drive the family score.
- Sample eligible families with `0.8 * softmax(difficulty / 0.25) + 0.2 / K`;
  choose an operation uniformly within the selected family. This is one policy
  shared across all groups. Class-wise checks protect each class but do not
  create separate class-conditioned policies.

| Operation | Level 1 | Level 2 (fixed reference) | Level 3 |
| --- | --- | --- | --- |
| Brightness down/up | 0.9 / 1.1 | 0.8 / 1.2 | 0.7 / 1.3 |
| Contrast down/up | 0.9 / 1.1 | 0.8 / 1.2 | 0.7 / 1.3 |
| Gamma brighten/darken | 0.9 / 1.1 | 0.8 / 1.2 | 0.7 / 1.3 |
| Gaussian blur radius, resized pixels | 0.4 | 0.8 | 1.2 |
| Gaussian noise std on RGB/255 | 0.01 | 0.02 | 0.03 |

Noise uses two independent fields per image per update, reusing each field
across strengths. Retention and losses average the two draws; object/cluster
support counts do not double. Training uses fresh fields from the same noise
distribution. Brightness, contrast, gamma, blur and masking code are copied
unchanged from the controls. Descriptions in `augmentation_policy` distinguish
the fixed middle-strength reference bank from the online selection rule.

## Object-loss measurement

The normal detector forward produces predictions without adding GT proposals.
A temporary ROI pre-hook captures its backbone features and RPN proposals.
For the auxiliary loss observation only, the pinned ROI IoU matcher assigns
positive RPN proposals to objects. One extra GT ROI per object is explicitly
self-assigned so duplicate GT boxes and objects without a positive proposal
still receive a supervised observation. GT ROIs never enter reported detections
or retention. There is no random ROI subsampling during probes.

For each assigned ROI, the observer uses cross-entropy plus class-specific box
SmoothL1 (beta=1/9, sum over coordinates), with weights 1 and 1. It averages
these contributions within each object. This is an auxiliary positive-ROI
diagnostic, not the detector's aggregate training loss: the standard training
loss includes background, RPN terms and its original normalization. The same
auxiliary procedure is applied to both views. View-specific proposals can
change assignments; per-object ROI counts are recorded.

Implementation reference: [TorchVision 0.18 ROI heads and loss definitions](https://raw.githubusercontent.com/pytorch/vision/v0.18.0/torchvision/models/detection/roi_heads.py).
The actual-model check verifies exact ordinary prediction parity and that
probes leave detector state and RNG unchanged. Local versions differ from
CCAST, so the PBS job repeats tests and an actual GPU smoke in pinned versions.

## Matched training and additional compute

The complete 640-square image, all annotations, negative images, raster-overlap
groups and inward taper are preserved. Augmentation opportunities use q=0.5
and the same application/group random keys as fixed local. Eligible families
can be absent, so actual edited-image counts can be lower than fixed local;
both opportunities and identity fallbacks are recorded.

The detector uses the same fresh initialization, seed 42, 26,863 training
images, 5,760 validation images, 20 epochs, batch size 4, 134,320 optimizer steps,
SGD recipe, FP32 precision, preprocessing, and epoch-wise unaugmented validation.
Validation selects the best AP50:95 checkpoint, with earlier ties. The test
manifest and test pixels are never opened. Validation metrics are not inputs
to the controller.

Each update has 256 x (1 reference + 3 x [7 deterministic operations + 2 noise
draws]) = 7,168 backbone image forwards. Ten updates add 71,680, plus auxiliary
ROI computations. The GPU smoke has an additional standard-prediction parity
forward on its first reference batch. Every attempt records views and elapsed
time, including an interrupted attempt. A hard process kill can leave the
last two-image chunk's costs incomplete; it cannot adopt a partial policy.

## Checkpoints and records

Each completed update writes a uniquely named compressed observation record
under `controller/`, then commits its hash and policy together with the
detector/optimizer/RNG checkpoint. Training starts with that policy only after
the checkpoint is committed. Policy records can be recomputed from the saved
observations, and their hashes are checked at resume and completion.

Mid-epoch resume restores the active policy without reprobing. An interrupted
uncommitted update restarts from the last committed detector state. Old attempts
are retained and not counted as adopted policies. A complete observation file
without a committed model/policy checkpoint is also unadopted. The records
archive includes policies, observations, attempt costs, validation predictions,
history, configuration, code and checksums. Full `.pt` checkpoints remain on
CCAST, as in earlier stages.

## CCAST execution

From the extracted package on a login node, `bash submit.sh start` submits one
A10, four CPU cores, 32 GB RAM, 24 hours. The job verifies checksums, activates
the existing environment, and runs all focused tests. Then it runs a separate
24-train/24-validation-image GPU smoke: all eight fixed-bank operation/mask
checks, all three probe strengths, object losses, ordinary prediction parity,
controller evidence recomputation, training, validation and checkpoint replay.
For smoke only, augmentation warmup is zero and the panel uses all positive
images in the existing training smoke subset. The production retention and
support thresholds are unchanged. Insufficient support is a valid identity
policy, not a reason to relax the gate.

Only a successful smoke starts the full run, in a fresh process with the full
one-epoch warmup and 256-image panels. Its directory is
`runs/fasterrcnn_online_shared_seed42_v1`; smoke has a separate directory. The
live log is `runs/baseline_job_logs/<PBS job ID>.log`. SSH disconnection does
not stop the submitted job.

The full process saves and stops at its 81,000-second budget if unfinished;
completion in one allocation is not guaranteed. Resume is explicit with
`bash submit.sh resume` after inspecting the run and confirming the previous
job has ended. Resume does not increase the 20-epoch optimization budget.
There is no automatic resubmission. Existing releases and run directories are
preserved; start refuses an existing submission/run/smoke output.

## Result review rule

First verify the returned archive, all update records, selected/final metrics,
and the matched training budget. Report each update's support, admissible
families, strengths, probabilities and identity rate; report per-class AP,
negative-image false positives, changed pixels and probe/training time.

The primary development comparison is selected validation AP50:95 against
fixed local; identity/image-level remain reference controls. Do not claim an
online gain from training loss or retention alone. If the controller stays at
identity, report that the common gate had insufficient support/admissibility;
that run does not establish the benefit of active online augmentation. If it
is active but gains are small, a single seed and existing warmup divergence
cannot establish efficacy. No thresholds are changed during this run. The
next class-conditioned comparison should retain this operation grid, observer,
support rule and training/probe budget; any change requires a matched shared
control and an explicit protocol revision.
