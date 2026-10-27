# Fixed shared-augmentation Faster R-CNN baseline

This run compares a fixed image-level appearance policy with the completed
`fasterrcnn_identity_seed42_v1` control. It starts afresh from the same COCO
weights and seed; it does not fine-tune the identity run's selected checkpoint.
The architecture remains the existing Faster R-CNN ResNet50-FPN. The experimental
change is the training augmentation.

This is the first fixed recipe, chosen before this comparison is run. It is not
yet a validation-selected augmentation policy, an adaptive controller, or evidence
that augmentation improves detection. The paper's broader comparisons remain
future work.

## Fixed policy

Epoch 1 uses no augmentation. During epochs 2–20, each target-bearing training
image has probability 0.5 of receiving one operation. Images with no retained
target boxes remain unchanged, matching eligibility for the planned local-edit
control. All images still contribute to training. Validation is unaugmented.

The two families are sampled equally. Operations are sampled uniformly within
the chosen family; therefore the eight operations are not equally probable.
Probabilities and strengths remain fixed throughout training and across classes.

| Family | Operation | Fixed strength |
| --- | --- | --- |
| Photometric | Brightness down / up | Multiply RGB by 0.8 / 1.2 |
| Photometric | Contrast down / up | Factor 0.8 / 1.2 around full-content mean luminance |
| Photometric | Gamma brighten / darken | Exponent 0.8 / 1.2 |
| Blur/noise | Gaussian blur | Pillow radius 0.8 resized pixels |
| Blur/noise | Additive Gaussian noise | Standard deviation 0.02 on RGB scaled to [0,1] |

Edits cover the entire resized image content. Letterbox padding stays at 114;
bounding boxes, labels, image geometry and stored orientation are unchanged.
Contrast uses mean luminance `0.299 R + 0.587 G + 0.114 B`. Arithmetic uses stored
RGB values divided by 255, without conversion to linear light. Outputs are
clipped to [0,1] and rounded back to uint8; Pillow blur returns uint8 directly.
Noise is independent across pixels and channels. These modest strengths are a
declared starting recipe, not empirically established safe limits for every box.

Decisions are keyed by policy version, seed, zero-based epoch, image path and
purpose using SHA-256. Noise uses its own NumPy PCG64 generator. Augmentation
does not consume the detector's Python, NumPy or Torch random stream. Decisions
are independent of worker scheduling and checkpoint resume. Library versions
are pinned; cross-version pixel identity and bitwise CUDA training replay are
not promised.

## Matched training and evaluation

`identity_protocol.json` is a byte-for-byte copy of the verified control's
protocol. Startup rejects any unintended difference in non-augmentation
configuration. The following are unchanged:

- Frozen 26,863 training and 5,760 validation images; same source separation.
- COCO initialization, fresh five-output ROI classifier, all backbone stages
  trainable and FrozenBatchNorm behavior.
- Seed 42, 640-pixel letterboxing, batch size 4, same image order and final batch.
- Twenty epochs and 134,320 optimizer updates; FP32 without AMP or TF32.
- SGD learning rate 0.0025, momentum 0.9, weight decay 0.0005; 1,000-update linear
  learning-rate warmup and reductions after epochs 12 and 16.
- Full validation every epoch, AP50:95 checkpoint selection, earlier ties win.
- Score threshold 0.001, NMS IoU 0.5, at most 100 predictions per image;
  original-coordinate COCO metrics, per-class AP and negative-image counts.

Augmentation warmup and learning-rate warmup are separate. The former lasts one
epoch; the latter lasts 1,000 updates. No test manifest or test pixels are read.
Existing identity packages, records and checkpoints are preserved.

## Execution and records

Run `bash submit.sh start` from this package on a CCAST login node. It requests
one A10, four CPUs, 32 GB RAM and 24 hours in PBS. The job activates the existing
`pytorch` and `rdd_detection_v1` environments, verifies checksums and runs all
focused tests before a real-data GPU smoke check. The smoke uses 24 training
and 24 validation images at 640 pixels, disables augmentation warmup, and must
exercise augmentation and exact checkpoint prediction replay. It also saves a
reference plus all eight operations on one training image in
`augmentation_preview.png`, with measurements in
`augmentation_operation_check.json`. These are software/geometry checks, not a
semantic-safety certificate.

Only after the smoke passes does a separate process start the full training run
from fresh COCO initialization. The full run reinstates the one-epoch augmentation
warmup. The first full epoch reporting zero augmented images is expected.

Paths under `/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/`:

- Package: `fasterrcnn_shared_v1/`
- Smoke: `runs/fasterrcnn_shared_smoke_v1/`
- Full run: `runs/fasterrcnn_shared_seed42_v1/`
- Live log: `runs/baseline_job_logs/<PBS_JOBID>.log`

Each epoch records applied-operation/family counts, actual changed-image counts
and a hash of ordered decisions in `history.json`. The script verifies the
decision counts and ordering against the fixed plan. Checkpoints retain the
partial-epoch trace as well as model, optimizer, RNG and minibatch progress.
The `run.json` field `augmentation_applied_images` totals completed epochs;
partial-epoch counts at an interrupted run remain in checkpoint progress.

All completed-epoch checkpoints plus the latest intermediate checkpoint remain
on CCAST. Returned records archives exclude full checkpoint binaries; smoke
records and the live log are separate from the full-run archive. Budget stops
preserve progress and require an explicit resume command after checking job
status. There is no automatic resubmission or extension of the 20-epoch budget.
Submission refuses an existing start receipt or output; do not delete results
to bypass these guards.

## Validation status and interpretation

`LOCAL_VALIDATION.json` records the completed local checks: 25 focused tests,
unchanged comparator components, frozen manifest separation, Python 3.12 grammar,
Bash syntax, and a synthetic CPU check using the actual COCO model. The latter
checks identical initial model weights, all eight operations, finite updates,
an entirely negative batch, validation isolation and checkpoint prediction replay.

Local libraries differ from CCAST. No RDD image pixels or GPU training were used
locally. CCAST tests, GPU smoke and full training are pending at package release.
The single-seed comparison will estimate the effect of this fixed recipe under
this training setup. It cannot establish controller benefit or broad statistical
robustness. Compare validation-selected checkpoints under the same rule, and
keep the test partition reserved.

Implementation references: [Pillow 10.3 GaussianBlur source](https://raw.githubusercontent.com/python-pillow/Pillow/10.3.0/src/PIL/ImageFilter.py)
and [NumPy 1.26 random Generator documentation](https://numpy.org/doc/1.26/reference/random/generator.html).
The full prospective configuration is `protocol.json`.
