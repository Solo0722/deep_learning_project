# Faster R-CNN full-data baseline

This is the first full-image **identity-augmentation control** for the RDD
augmentation study. It trains the existing pretrained Faster R-CNN ResNet50-FPN
with a fresh four-class output head. It is not an augmentation-controller run
or a detector comparison. This one-seed development baseline does not establish
method efficacy, and the fixed 20-epoch budget is not a convergence claim.

The frozen split has 26,863 training images and 5,760 validation images. The
runner opens only `train.jsonl` and `val.jsonl`; there is no test CLI option.
It checks the pinned manifests, split configuration, pretrained weight digest,
and train/validation image, source-cluster and decoded-RGB separation. Every
loaded image and XML is checked against its recorded hash; decoded RGB,
dimensions and orientation must also match. Empty-target images remain in both
partitions. Difficult targets are ordinary ground truth under the frozen policy.

## Training and validation

- Complete images are letterboxed to 640 x 640 using the verified stored-pixel
  geometry. No EXIF rotation is applied. No stochastic augmentation is enabled.
- COCO_V1 weights, FrozenBatchNorm2d with epsilon zero, all five backbone stages
  trainable, new classification and box-regression predictors for background
  plus D00/D10/D20/D40. No automatic model downloads.
- Twenty epochs; seed 42; batch size four; the final three-image batch is kept.
  Each epoch visits all training images exactly once in a reproducible shuffle.
  Total: 134,320 updates and 537,260 image presentations.
- SGD, learning rate 0.0025, momentum 0.9, weight decay 0.0005. Linear warmup
  over the first 1,000 updates from a factor of 0.001. Multiply learning rate
  by 0.1 after completed epochs 12 and 16. FP32, AMP and TF32 disabled.
  These are declared initial settings, not settings tuned on validation results.
- Full unaugmented validation after every epoch, with no gradient recording or
  optimizer update. COCO AP50:95 is the checkpoint-selection metric; ties retain
  the earlier epoch. Also save AP50/AP75/AR100, per-class AP and no-target-image
  false-positive counts at confidence 0.25.
- Predictions use score threshold 0.001, class-wise NMS IoU 0.5 and at most 100
  retained boxes per image. Restore original coordinates, clip at boundaries,
  and discard only collapsed predictions. These are the existing metric rules.

The SGD/scheduling pattern is informed by the pinned [TorchVision 0.18 detection
reference](https://raw.githubusercontent.com/pytorch/vision/v0.18.0/references/detection/train.py).
This project uses its own documented budget and learning rate; it is not a claim
to reproduce the reference's COCO training recipe.

## Launch on CCAST

The package uses the existing `pytorch` Conda environment plus
`.venvs/rdd_detection_v1`. No installations are performed. After uploading the
archive, compare its SHA-256 to the release record, extract with
`tar --keep-old-files`, enter this directory, then run:

```bash
bash submit.sh start
```

The submitted PBS job requests one A10, four CPUs, 32 GB host memory and 24 hours
in `gpus`, using project `x-ccast-prj-aaboah`. The queue/resource fields follow
the [CCAST job-submission guide](https://kb.ndsu.edu/137530) and
[queue policies](https://kb.ndsu.edu/it/page.php?id=107680). Submission returns a
job ID; the job runs independently of the SSH connection. The start receipt
prevents accidental resubmission of the same start command.

Inside the allocation the job verifies checksums, activates the environments,
runs the focused tests, then runs a 24-training/24-validation-image smoke check
at the real 640-pixel size. This check exercises updates, validation and exact
checkpoint prediction replay. Its metrics are software-check outputs. Only a
complete smoke result with checkpoint replay lets the full baseline start.
The full baseline initializes afresh from COCO weights with seed 42.

Outputs under the project root:

- `runs/fasterrcnn_baseline_smoke_v1/`: distinct smoke outputs.
- `runs/fasterrcnn_identity_seed42_v1/`: actual full-data run.
- `runs/baseline_job_logs/JOBID.log`: live stdout and stderr, including warnings.
- `runs/fasterrcnn_identity_seed42_v1_records_ATTEMPT.tar.gz`: returned records
  archive after a complete or time-budget-stopped invocation. Full checkpoints
  stay on CCAST; the archive contains their hashes and metadata.

`history.json` and `history.csv` contain the canonical epoch results.
`epoch_NNN_validation.json` and compressed prediction files preserve validation
outputs. `best.json` identifies the selected epoch/checkpoint. `last.json` is the
committed resume pointer. `run.json` and `attempts/` distinguish complete,
budget-stopped and failed invocations. Live job logs are separate from the
records archive and must be retrieved separately when diagnosing a warning.

## Interruptions and resumption

Checkpoints include model, optimizer, RNG states, epoch, next minibatch,
accumulated losses and selection history. They are committed through an atomic
pointer after writing and hashing the checkpoint. An interrupted write leaves
the previous committed checkpoint available. All 20 completed-epoch checkpoints
are retained (about 6.6 GB for model plus momentum), together with the latest
intermediate checkpoint. Replaced intermediate checkpoints from this new run
are retired only after the new commit succeeds. Original inputs and earlier
experiment outputs are never overwritten.

The fresh job's full run stops after 22.5 hours of its own invocation; resumption
uses 23 hours, leaving room for saving and job overhead within the 24-hour
allocation. Time limits are checked between batches. SIGTERM/SIGINT request a
checkpointed stop; SIGKILL/power loss resumes from the last committed checkpoint,
normally at most 1,000 updates earlier. Validation interrupted mid-pass is
repeated without retraining its completed epoch. No job is automatically
resubmitted, and the epoch budget stays fixed on resume.

After checking `qstat` and the saved status to ensure the previous job ended:

```bash
bash submit.sh resume
```

Resume rejects changed code, configuration, data, environment or checkpoint
bytes. A kernel lock prevents simultaneous writers. A complete run cannot be
resumed. If no committed checkpoint exists, preserve the failed directory and
inspect the error before using a new run location. Do not delete outputs just
to make the start command pass. No CUDA bitwise rerun guarantee is made.

The small training-set AP50 review threshold from the earlier feasibility
check does not gate this full-data run: the project has moved to a proper
training/validation baseline. Numerical and data failures still stop execution.
Review the saved learning and validation curves before altering the recipe or
starting the augmentation comparisons.
