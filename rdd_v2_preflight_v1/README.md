# RDD v2: bounded preflight, release 1

This package checks the revised controller logic and whole-image geometry before
new training. It does not train, choose a checkpoint, read validation/test data,
implement copy-paste, or establish improved detector accuracy.

## What is implemented

- `controller.py`: per-class support and retention gates, intersection for the
  classes in a group, supported mixed-set vetoes, explicit borrowed-evidence
  flags, balanced sampling across eligible families/operations/strengths, and
  cluster sensitivity diagnostics. Loss is not a selection reward.
- `geometry.py`: horizontal reflection, isotropic 0.95/1/1.05 canvas scaling,
  translations of -16/0/16 pixels at 640 resolution. It maps every known target
  and non-target annotation and rejects any candidate cropping an annotation.
  Rejection returns the exact input. The 54-candidate bank has one uniform draw;
  no resampling until success. Labels and difficult/truncated flags are retained.
- `replay.py`: analyzes the saved training-only panels from both v1 online
  trajectories using the same new decision rules. Archive and individual record
  hashes are checked. Noise draws are averaged within objects, never counted as
  extra objects. Replaying a policy cannot predict its training outcome.
- `geometry_check.py`: checks all training annotations and renders all 54
  candidates on a deterministic 56-image subset. It exports black/white overlay
  sheets and numerical evidence. The immutable v1 data reader is included.

`protocol.json` freezes the scope and numerical preflight choices. These choices
are engineering proposals, not empirically optimal augmentation magnitudes or
statistical confidence guarantees. Local numerical tests use the local runtime;
the remote job explicitly verifies CCAST NumPy 1.26.4 and Pillow 10.3.0.

## CCAST execution

Upload the checksum-verified release and extract it once in the project root.
On a login node, `cd rdd_v2_preflight_v1` then `bash submit.sh`. The batch job
requests **2 CPUs, 8 GB, one hour, no GPU**; activates the existing environment;
runs the focused tests; then checks the real training pixels. It uses the PBS
`default` CPU queue described in the [CCAST FAQ](https://kb.ndsu.edu/133763).
This is CPU image/annotation work, so reserving an A10 would not accelerate it.
The wrapper refuses duplicate submissions and existing result directories.

Live log: `../runs/preflight_job_logs/<PBS_JOBID>.log`.
Output: `../runs/rdd_v2_geometry_check_v1/` and the matching `.tar.gz` archive.
The last line prints the archive SHA-256. An SSH disconnect does not stop the
batch job; inspect the saved job ID and log instead of submitting a duplicate.

The archive contains annotation acceptance counts, any invalid non-target boxes,
raw-image verification outcomes, transformed annotations, pixel hashes and
56 representative contact sheets. Numerical completion intentionally leaves
visual review pending. Accepted geometry preserves known boxes, not a guarantee
that all road damage was annotated or that the augmentation improves learning.
Some images can legitimately disable geometry because a non-target annotation
is malformed; those images must not be silently removed from training.

## Local evidence reproduction

From the project root, using Python with NumPy and Pillow:

```text
python -m unittest discover -s rdd_v2_preflight_v1/tests -p test_preflight.py -v
python rdd_v2_preflight_v1/replay.py --project . --out reports/rdd_v2_preflight_v1/replay_NEW
python rdd_v2_preflight_v1/geometry_check.py --splits detection_splits_v1 --out reports/rdd_v2_preflight_v1/annotation_check_NEW --manifest-only
```

New output paths are required. The replay uses the local copies of both archived
online v1 runs and their existing audit files. It does not need a GPU or raw
images. Historical inputs are read without modification.

## Interpretation and next stage

Read the companion `reports/rdd_v2_preflight_v1/RESULTS.md` for measured local
findings. Archived observations use v1 pixels, so they cannot validate the new
geometry or a different local blending mask. The two detector trajectories and
ten changing snapshots do not constitute independent repeated panels at one
fixed detector. Disjoint halves and cluster resampling only assess sensitivity
within the recorded panels. Their support and agreement denominators are kept
separate. Conditional bootstrap frequencies are not a calibrated safety bound.

If class-specific action sets coincide, report that the method has become a
shared policy at that snapshot. Do not create artificial class differences.
If pothole evidence is insufficient, improve the prospective training-only
panel allocation; do not lower the support/retention gate to obtain a pass.
Any changed allocation needs fresh observations on a common frozen detector.

Copy-paste remains disabled pending reviewed donor/recipient regions and
annotation/context checks. A smoother local renderer and fixed/mapped-ROI
auxiliary diagnostics need their own fresh checks. Full training is not bundled
here. After reviewing this preflight, prepare fresh, paired training-only probes
and a bounded four-arm pilot from one common warmup checkpoint; freeze the
pilot budget before examining its accuracy. Repeated full runs come later.

## File provenance

`rdd_data.py` is copied unchanged from `fasterrcnn_class_adaptive_v1/rdd_data.py`.
All package inputs are listed in `SHA256SUMS`, written with Unix LF endings.
`LOCAL_VALIDATION.json` records the exact local tests and their limits.
