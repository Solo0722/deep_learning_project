# RDD v2: strict paired comparison through epoch 8

This package compares `class_adaptive` local appearance augmentation against `geometry_only`. Both independently restart the same completed identity epoch-1 checkpoint and train through the fixed epoch-8 endpoint. This is a one-seed development experiment. Passing software checks is not evidence of improved accuracy.

## Fixed comparison

The question is whether the unchanged v2 class-adaptive recipe improves clean-validation detection over the same geometry without appearance edits at a longer matched horizon. The primary endpoint is **class-adaptive minus geometry-only epoch-8 AP50:95**, in percentage points. Earlier curves, per-class AP, recall, negative-image predictions, augmentation exposure and controller cost are secondary diagnostics. Do not select the most favorable epoch or tune thresholds during the run.

Training uses all 26,863 frozen training images, including 10,233 without target-class boxes. Validation uses all 5,760 frozen validation images without augmentation after every epoch. Test manifests and test pixels remain closed. Seed 42, image order, geometry, target boxes, overlap-group selection and appearance opportunities are paired across arms and checked by epoch hashes. Actual edits can differ because unsupported actions fall back to identity.

The existing Faster R-CNN ResNet-50-FPN has all backbone stages trainable and FrozenBatchNorm2d. The recipe remains 640-pixel letterbox, batch size four with the last smaller batch retained, float32 SGD with learning rate 0.0025, momentum 0.9 and weight decay 0.0005, without AMP or gradient clipping. First-epoch warmup is complete. The inherited learning-rate decays at epochs 12 and 16 are not reached. This recipe has not been established as converged or optimal.

Both arms restore `epoch_001_step_00006716_04dccd11bbd2.pt`, SHA-256 `dbd619334a57ad4422e929b90b327f80d1ca3bfc5b86f6f37594714ec138e858`, including model, optimizer momentum and RNG. The epoch-4 pilot checkpoints are not used. Local testing caught optimizer tensors aliasing the in-memory warmup payload on CPU; this package deep-copies optimizer state before loading and checks that training cannot mutate the payload. This finding does not establish a defect in prior GPU trajectories, which copied CPU checkpoint tensors to CUDA.

Each arm performs seven additional full epochs: 47,012 optimizer updates and 188,041 training-image presentations. Totals are 94,024 updates and 376,082 presentations, excluding the shared warmup and software gate. Fourteen post-epoch validations and one common warmup validation are saved. Best-checkpoint metadata is retained only for provenance; it does not select this experiment's endpoint.

## Unchanged geometry and local controller

Pipeline: stored-orientation image, letterbox, optional local appearance edit, whole-image geometry. The 54-candidate geometry bank combines horizontal reflection, isotropic canvas sizes 608/640/672 and translations -16/0/16 on each axis. A keyed uniform candidate applies to every training image, including negatives. All target and valid non-target annotations follow the same map. Any annotation leaving the canvas rejects the whole candidate to identity, without resampling, clipping or dropping boxes. Invalid non-target annotations disable geometry and are counted. Horizontal reflection preserves a crack's relation to the reflected road and does not swap longitudinal/transverse labels.

Positive images have a common keyed 0.5 appearance opportunity and uniformly selected overlap group. Edits use the existing union of target boxes with its inward taper. This is not segmentation: background within boxes may change. The eight audited appearance operations and three tested levels are unchanged. Selection is uniform over eligible families, then operations, then levels. There is no strongest-level preference or loss-hardness weighting.

Class-adaptive updates occur before epochs 2, 4, 6 and 8 using the current detector. Each uses a frozen 256-image training panel, with 64 anchor groups per class and distinct source clusters within the panel. Planning seeds are 4202001, 4204001, 4206001 and 4208001. The first two panels are unchanged from the original pilot; the two new panels are fixed before execution. Panels can overlap across updates and with training; they are not independent evaluation data.

Each update measures 29 views/image plus four repeated parity views: 7,428 detector images/update, 29,712 across four updates. Ordinary predictions use no ground-truth proposal injection. Post-inference reference matching requires correct class, confidence >=0.25, IoU >=0.5 and one-to-one matches. Only reference-detected selected-group objects enter retention denominators. Reference misses remain recall diagnostics. Two noise draws are averaged within each object. Eligibility requires retention >=0.8 with at least 16 detected objects in eight clusters. Pooled class evidence, supported mixed-set vetoes, borrowed-evidence flags and unsupported-class identity fallback are unchanged. No threshold relaxation, adaptive panel replacement or stale-policy carry is introduced.

Admissibility measures detector tolerance. It does not guarantee semantic validity or training utility. Actual edits and identity fallbacks must be reported separately from application opportunities.

## Strict execution and GPU gate

This carries forward settings that passed strict diagnostic PBS job 799047: deterministic algorithms with `warn_only=False`, cuDNN deterministic enabled, benchmark disabled, both TF32 flags disabled, uninitialized-memory filling enabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before Python. Installed TorchVision RoIAlign source is pinned to the audited hash. No permissive fallback is provided. The earlier check established short-path equality, not accuracy or universal determinism across devices/versions.

Every job performs package/environment checks, unit tests and a fresh GPU production-loop gate before training or resume. The gate uses actual `pilot.fit`, loading, `baseline.update`, validation and checkpoint paths at 640 pixels and batch four. It uses 26 training images (old smoke selection plus the first two panel rows) and 24 validation images. Both branches perform seven updates from the original warmup. The class branch includes a real two-image probe that must remain unsupported/identity, then saves, rebuilds and restores after its first update.

Inputs, model, optimizer, gradients, loss, RNG and modes must agree exactly at corresponding updates; validation predictions must also match. A mismatch, nonfinite update, changed strict setting, unsupported deterministic operation or failed restore stops the job. This tests side effects under identity appearance, not augmentation efficacy. Raw gate records, excluding full checkpoints, are copied into the main results archive.

## Order, resources and recovery

Full training runs class-adaptive first, then geometry-only, on one A10 with four CPUs and 32 GB host RAM. This measures the first full controller cost before paying for the control arm. The short gate uses the opposite order to establish its uninterrupted reference. Each full arm independently restores the same warmup, so learned state is not transferred.

The first 128 normal optimizer-update calls per fit invocation record synchronized wall time and GPU allocator peaks. Timing excludes loading, probes, validation and checkpoint writes. Every full probe records its own time and peaks with a declared 30-minute operational limit. A probe reaching that limit is not adopted; its pre-probe checkpoint and partial evidence are retained. GPU allocation is capped at 90% of visible memory. These are operational limits, not accuracy-based stopping rules.

Each PBS allocation has a 12-hour ceiling; an 11-hour administrative deadline includes tests and the gate and allows time to save/archive. Total strict-run duration is unknown; more than one allocation may be needed. There is no automatic resubmission or extension beyond epoch 8. A deadline stop checkpoints safely. Explicit `bash submit.sh resume` is available after reviewing the log; the next job repeats the gate and verifies immutable code/config/data/environment/evidence. Do not use resume to bypass a gate or resource failure.

Checkpoints save every 1,000 updates, around probes, before validation and at completed epochs. They retain model, optimizer, RNG, epoch, next batch, counters, policy and evidence references. Completed-epoch checkpoints and the latest intermediate remain on CCAST. Committed updates are not repeated. Interrupted uncommitted probes rerun from the authoritative checkpoint, preserving earlier attempt records. Full checkpoints are excluded from downloadable records archives. Historical runs and raw data are not overwritten.

## Entry points and outputs

Use `submit.sh`, `run_job.sh`, `gate.py` and `pilot.py`. Inherited modules such as `baseline.py` provide implementation; their old standalone CLI modes are not this experiment's entry points.

- Package: `/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_pair8_v1`
- Results: `../runs/rdd_v2_pair8_seed42_v1`
- Live log: `../runs/pair8_job_logs/<PBS_JOBID>.log`
- Login-node status: `bash /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_pair8_v1/status.sh`
- Initial submission from the package directory: `bash submit.sh start`
- Explicit continuation after reviewing a deadline stop: `bash submit.sh resume`

The job activates `pytorch` and the existing `rdd_detection_v1` environment. No interactive GPU is needed before submission. SSH disconnection does not cancel a submitted batch job. The local `reports/rdd_v2_pair8_v1/CCAST_COMMANDS.md` contains sealed upload/submission blocks.

## Interpretation

Audit the fixed endpoint, paired budgets, per-class metrics, negative-image predictions, support, edit rates, raw gate records and added compute. A favorable one-seed result motivates independent paired seeds and a matched online-shared comparator. A persistent deficit without a declared compensating benefit argues against scaling this recipe. This comparison does not isolate class conditioning, quantify seed variance or establish test generalization. Validation has informed development; the earlier permissive-execution pilot is a separate execution regime.

`LOCAL_VALIDATION.json` records local CPU implementation checks and limits. It is not a CCAST GPU result or an accuracy measurement from this package.
