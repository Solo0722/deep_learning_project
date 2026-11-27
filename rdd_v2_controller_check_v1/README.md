# RDD v2: frozen-detector controller check

This is a bounded diagnostic of class support and policy sensitivity. It does
not train a detector, measure validation AP, prove improved accuracy, or start
another experiment automatically.

## Frozen design

Use the saved identity baseline's first completed epoch, pinned by file size and
SHA-256 from its verified records archive. This is a common observer for both
panels, not the validation-selected best checkpoint. The full checkpoint stays
on CCAST; weights are loaded strictly and no optimizer is instantiated.

The two panels contain 256 training images each, with 64 anchor groups for each
of D00, D10, D20 and D40. Hash-ranked selection uses annotations, not predictions.
There is one image per source cluster across both panels. Images may contain
other classes and all targets in the chosen overlap group contribute. These
quotas enrich potholes but do not guarantee sufficient detectable potholes.
The exact plan is generated and frozen locally before remote inference, then
recomputed and checked on CCAST. Neither panel is pooled into the other to make
an unsupported class pass. Class enrichment and known-cluster separation do
not imply independent or representative road-scene samples.

Each image has 29 views: clean, geometry-only reference, and 27 local appearance
draws (24 actions; Gaussian noise has two draws per level). The same geometry
candidate is used for every view in an image's reference/action set. Four extra
image forwards check repeated inference, for **14,852 image forwards total**.

The pipeline order is explicit: existing letterbox, existing local appearance,
then whole-image geometry. This keeps the already tested local mask and content
statistics unchanged. Geometry is the audited horizontal-flip/scale/translation
bank; a candidate cropping any known annotation returns to identity geometry,
without resampling. That fallback does not cancel a local appearance edit.
The geometry-only reference measures local retention conditional on the chosen
geometry. The separate clean view exposes geometry-related detection misses.
This is not evidence for geometry-before-local or a new blending mask.

Ordinary detections are matched after inference, at score >=0.25 and IoU >=0.5,
class-correct and one-to-one. GT never enters the detector. Noise outcomes are
averaged per object, not counted as extra support. Model parameters, buffers,
gradients, mode and global RNG are checked unchanged. No RPN/GT auxiliary loss
is calculated. Labels are zero-based in saved predictions and manifest records.

The preflight controller is reused: per-class retention >=0.8 with at least 16
detected objects from eight clusters; intersection for classes sharing a group;
supported mixed-set vetoes; otherwise explicitly borrowed marginal evidence.
Sampling is uniform eligible family, operation, then tested level. Identity is
the fallback for unsupported/empty routes. If every class allows the same bank,
the adaptive policy is reported as shared; differences are not forced.

## Outputs and interpretation

The archive contains each view's predictions, matches, pixel digest, target
geometry and rendering decision; both panel policies; between-panel action and
probability differences; class support; cluster/source sensitivity; resource
counts; checksum/provenance records; and eight full-resolution example sheets.
The thin Norway D10 case has additional raw-crop and 640-pixel prepared/scaled
RGB and box images. This extra inspection does not modify the frozen panels.

Read `result.json` and inspect the previews after download. A complete run is
only technical completion. Both-panel support, instability, class differentiation
and direct mixed-set evidence remain scientific review questions. There is no
validated numerical stability cutoff or automatic go signal. Unsupported panels
must not be rescued by lowering thresholds or resampling after detector outcomes.
Conditional bootstrap frequencies are sensitivity diagnostics, not confidence
guarantees. The resampling summaries assess pooled class gates; they do not
certify the joint mixed-set veto. Between-panel routed distributions are also
reported for all 15 nonempty class sets. Retention measures detector tolerance,
not learning utility.

If this check is informative and the implementation passes review, the next
step is a bounded, matched four-arm training pilot from a common warmup state:
geometry only, fixed local appearance, online shared local, and online
class-adaptive local. Its duration must be fixed using measured cost before AP
is inspected. Copy-paste, a new blending renderer, fixed-ROI auxiliary loss and
full 20-epoch runs remain outside this package.

## Run on CCAST

Upload the release archive, verify its checksum and extract once. On a login
node, run `bash submit.sh` from this package. It verifies package files and the
remote warmup checkpoint, then submits one A10 GPU, 4 CPUs, 32 GB, one hour.
No interactive allocation or manual environment activation is needed. The job
activates the existing `rdd_detection_v1` environment and runs tests first.

Saved job ID: `submission_start.id`. Live log:
`../runs/controller_job_logs/<PBS_JOBID>.log`. Results and archive:
`../runs/rdd_v2_controller_check_v1{,.tar.gz}`. The final log gives the archive
SHA-256. Disconnecting SSH does not stop this batch job. Do not resubmit it.

The program has a 50-minute administrative cap including its own setup, leaving
time for the one-hour job wrapper and output. An interrupted/failed check keeps
partial observations and a failure record when possible, but produces no
complete results archive. No automatic retry or training is implemented.

Local test evidence is in `LOCAL_VALIDATION.json`. The local runtime has no
PyTorch, CUDA or raw dataset, so actual checkpoint loading, GPU inference,
observer guards and real-pixel review still require CCAST. Unit tests that need
PyTorch are explicitly skipped locally and run in the target environment.

## Provenance

`controller.py`, `geometry.py`, and `rdd_data.py` are unchanged copies from
`rdd_v2_preflight_v1`. `local_augmentation.py` and `shared_augmentation.py` are
unchanged copies from `fasterrcnn_class_adaptive_v1`. The new code composes these
components and collects ordinary predictions. It does not alter prior runs,
splits, checkpoints, raw images, or the manuscript. No validation/test manifest
or pixels are opened by the runner.
