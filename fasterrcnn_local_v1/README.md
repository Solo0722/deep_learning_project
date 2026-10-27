# Fixed shared local augmentation: Faster R-CNN

This is stage 3: identity -> fixed shared image-level -> fixed shared local ->
online shared local -> online class-adaptive local. It starts from the same
fresh COCO initialization as both completed controls. This is full-image
detector training; all target boxes and labels are retained.

## What changes

The appearance recipe is unchanged from `fasterrcnn_shared_v1`: one unaugmented
epoch, then probability 0.5 of augmentation on target-bearing training images.
The two families are equally likely, with uniform operations within a family:

| Family | Operations | Strengths |
| --- | --- | --- |
| Photometric | Brightness down/up | RGB factor 0.8/1.2 |
| Photometric | Contrast down/up | Factor 0.8/1.2 around full-content mean luminance |
| Photometric | Gamma brighten/darken | Exponent 0.8/1.2 |
| Blur/noise | Gaussian blur | Pillow radius 0.8 resized pixels |
| Blur/noise | Gaussian noise | Standard deviation 0.02 on RGB/255 |

`shared_augmentation.py` is copied unchanged. Its policy randomization namespace
stays `fixed_shared_v1`: application, family, operation and noise draws are
identical to the completed shared run for each image and epoch. A separate
`fixed_local_v1` namespace selects one group uniformly, without resampling.
Probabilities and strengths are fixed and common across classes. There is no
detector feedback, online update or class-specific policy in this experiment.

The spatial application changes as follows:

1. Prepare the complete 640-square letterboxed image and its float64 boxes using
   the existing loader. Clip box edges to the resized content, floor minima and
   ceil maxima to obtain half-open pixel rectangles. Keep fractional/thin boxes.
2. Form connected components of positive-area overlap between these raster
   rectangles. This includes transitive overlaps. Touching half-open edges do
   not overlap; continuous boxes sharing a raster pixel belong to one group.
   Canonical ordering uses original object indices, independently of class.
3. Select one group uniformly. Construct the union of its rectangles, including
   holes where no rectangle covers a pixel. A group's bounding rectangle is
   only an implementation crop; it is not the edit mask.
4. Calculate the same reference appearance transform on the full resized
   content, preserving shared contrast/blur context and Gaussian-noise draws.
   Composite it once into the selected union. The inward taper reaches full
   weight at the second pixel layer: alpha is 0.5 on the outer 8-neighbour
   boundary and 1 in the one-pixel-eroded union. Alpha is zero outside. A
   one-pixel-wide support retains weight 0.5; it is never dropped.
5. Compute `round(x + alpha*(candidate-x))` once in float64, clip to uint8, and
   retain all boxes/labels. Overlap pixels never receive repeated transformations.
   Unselected groups, pixels outside the union, and letterbox padding are
   byte-identical to the reference. Negative images and validation stay unchanged.

These are bounding-box regions, not segmentation masks: they include pavement
context inside the boxes. Local edits affect fewer pixels and have a tapered
boundary. The comparison measures this complete spatial policy; it does not
isolate location from edit area or boundary blending. Coverage and changed
pixels are recorded explicitly. The taper and strengths were not selected by
tuning validation performance and are not guaranteed to preserve every defect.

## Matched training and evaluation

`identity_protocol.json` and `shared_protocol.json` are frozen copies. Startup
rejects unrelated training/evaluation differences. The following are unchanged:

- Frozen 26,863 training and 5,760 validation images; no test data access.
- COCO Faster R-CNN ResNet50-FPN, fresh five-output ROI head, all backbone stages
  trainable, existing FrozenBatchNorm settings.
- Seed 42, 640 pixels, batch size 4, exact sampler and retained final batch.
- Twenty epochs, 134,320 optimizer updates, FP32 without AMP/TF32.
- SGD learning rate 0.0025, momentum 0.9, weight decay 0.0005, 1,000-update
  learning-rate warmup, reductions after epochs 12 and 16.
- Full unaugmented validation each epoch. Best AP50:95 checkpoint; earlier ties.
- Score threshold 0.001, NMS IoU 0.5, at most 100 detections per image;
  original-coordinate COCO metrics, per-class AP and negative-image counts.

The model, metrics, optimizer update, validation, sampler and checkpoint helpers
remain unchanged. Augmentation consumes no global Python/NumPy/Torch random
stream. Local choices replay across worker scheduling and checkpoint resume.
All conditions are independent runs from the initial weights, not a chain of
fine-tuning the preceding run's best checkpoint.

## CCAST execution

Run `bash submit.sh start` from this extracted package on a CCAST login node.
It submits one A10, four CPUs, 32 GB RAM and 24 hours through PBS. The job:

1. Verifies package checksums and activates `pytorch` plus `rdd_detection_v1`.
2. Runs all 39 focused tests in the pinned CCAST environment.
3. Runs a separate 640-pixel GPU smoke on the existing 24 training and 24
   validation images. It forces all eight operations, checks targets/padding/
   outside-union invariants, exercises sampled local training, validates, and
   checks exact checkpoint prediction reload.
4. Only if the smoke passes, starts the separate full 20-epoch training run from
   fresh COCO weights. The full run restores the one-epoch augmentation warmup.

The smoke writes `augmentation_preview.png` (eight operations and reference),
`local_geometry_preview.png` (overlap, separated and smallest available groups),
and `augmentation_operation_check.json`. Cases are selected from training data
by geometry, not model performance. These are implementation checks, not proof
of semantic preservation. Full training does not wait for manual preview review.

Paths below the project directory on CCAST:

- Package: `fasterrcnn_local_v1/`
- Smoke: `runs/fasterrcnn_local_smoke_v1/`
- Full run: `runs/fasterrcnn_local_seed42_v1/`
- Log: `runs/baseline_job_logs/<PBS_JOBID>.log`

The job continues if the SSH connection disconnects. Submission refuses an
existing start receipt or output. Existing identity/shared packages and results
are preserved. Do not delete these guards to resubmit a running or completed job.

Each epoch saves operation/family counts, ordered local decision hashes, selected
groups/boxes, union pixel area, alpha-weighted area (`local_alpha_units/2`),
reference-content area and actual changed-pixel counts in `history.json`.
`shared_decision_sha256` excludes spatial metadata and is checked against the
completed shared run's frozen `shared_exposure.json` for all completed epochs.
Counts and full decision hashes are also checked against a metadata replay.
Applied operations can change zero pixels on locally constant or saturated
content; actual changes are measured separately, without retries or resampling.

Checkpoints retain partial-epoch trace, model, optimizer, RNG and next minibatch.
Completed-epoch checkpoints and the latest intermediate checkpoint stay on
CCAST. Returned records archives exclude checkpoint binaries. Smoke records
and live logs are separate from the full-run archive. The full process stops
near 22.5 hours if unfinished and preserves progress. After verifying job status,
`bash submit.sh resume` explicitly resumes; there is no automatic resubmission.

## Evidence and limits

`LOCAL_VALIDATION.json` records local software tests, unchanged components,
metadata replay against all 20 shared epochs, Python 3.12 grammar, Bash syntax,
and a synthetic CPU check with the actual COCO model. The model check covers all
eight operations, finite updates, a negative-only batch, identical starting
weights, validation isolation and exact checkpoint prediction replay.

Local library versions differ from CCAST. No RDD image pixels or GPU training
are used in these local checks. CCAST tests, smoke and training are pending at
release; no local-augmentation performance result exists yet.

The completed identity/shared controls already diverged during their first
unaugmented epoch. The cause is not established: matched seed/configuration
did not give an identical realized CUDA trajectory. This package preserves that
training setup rather than changing determinism in only one condition. The new
single-seed result will be descriptive; small changes cannot establish robust
benefit without replication. It will not establish online-controller efficacy.
Use the same validation checkpoint-selection rule and reserve the test set.

The prospective configuration is `protocol.json`. The sequence and interpretation
are recorded in the project's `DETECTION_DATA_RUNBOOK.md`.
