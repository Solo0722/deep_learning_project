# Full-image detection data preparation

This is step 2 of the agreed implementation plan. The first deliverable is a
full-pool audit and grouping report. Detection splits are created only after
that report has been inspected. The old crop manifests, raw files, trained
checkpoints, and previous audit bundles remain unchanged.

## Verified starting point

The user's CCAST terminal output confirmed 38,385 image/XML filename pairs in
seven source folders, with no missing filename partners. The four transferred
audit-record checksums match the local copies: annotations, fingerprints,
candidate pairs, and the complete image inventory. Matching metadata files do
not establish that the current raw files still match those records; the new
script checks every raw image and XML byte hash.

The prior screen covered 23,766 distinct target-bearing images. The full-image
audit also includes 11,724 empty-XML images and 2,894 images containing only
other annotation labels. Neither category is called a verified healthy-road
class. All original annotation labels are preserved without remapping.

## Audit implementation

`audit_detection_data.py` uses the existing, unchanged screening utilities.
It pins the input records and historical image split to SHA-256 checksums,
checks the current raw file membership and byte hashes, and recalculates every
image fingerprint in one current Pillow runtime. This avoids mixing old and
new fingerprint implementations. It also records differences from cached
fingerprints, if any.

The global radius-8 dHash screen includes target-bearing and target-negative
images and is a heuristic candidate search, not proof of duplication or of
complete geographic independence. All historical candidate edges, recorded
source groups, and exact-RGB duplicate links remain grouping constraints.
New components that join old train/validation/test assignments are reported;
the script does not resolve them by moving images or changing manifests.

Detector targets preserve continuous XYXY coordinates without an origin shift.
Clipping and nonpositive-area targets are explicitly reported. The two-pixel
crop-extraction cutoff is not imposed on detection targets. Difficult/truncated
flags and EXIF orientation are recorded. Pixels retain stored orientation;
there is no automatic EXIF transpose, crop generation, or preview creation.
Target boxes with nonpositive area are flagged, never silently converted into
background-only images. The next manifest step must document their treatment.

The historical duplicate exclusion is retained as a record for review; this
audit does not delete any image or decide whether to remove new duplicates.
Candidate limits, changed inputs, invalid metadata, or processing failures
cannot result in a complete audit. A failed attempt uses a new output directory
on retry; there is no implicit overwrite or resume mode.

## CCAST execution

Use a CPU compute allocation. No GPU or detector packages are needed. The
existing `pytorch` environment supplies Python 3.12.3 and Pillow 10.3.0, as
reported in step 1. Do not install packages into the shared environment for
this audit. The program rejects execution on a login node.

The transfer package contains the audit code, unchanged helper code, synthetic
tests, a frozen copy of the old image assignments, pinned reference hashes,
and `SHA256SUMS`. Verify the package and its members before running. Extract
only into a new `detection_preparation_v1` directory inside the project.

From that directory, the audit invocation is:

```bash
python -u audit_detection_data.py \
  --root /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/raw \
  --inventory /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/image_audit_20261012T035733_790780Z/inventory.jsonl \
  --review /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/preparation_review \
  --previous-images frozen_crop_images.csv \
  --reference reference_inputs.json \
  --out /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_audit_v1
```

Commands are supplied to the user one at a time. A complete screen ends with
`DETECTION AUDIT COMPLETE`, saves a review archive, and prints its checksum.
An incomplete or failed screen must not be used to assign a detection split.

## What to inspect before assigning splits

- Complete status, expected coverage, and source-file integrity.
- Empty-XML versus other-label-only images and target annotation flags.
- Invalid/clipped targets, stored orientations, and exact-copy annotations.
- Candidate graph size and links involving target-negative images.
- Components crossing historical partitions, including previous training and
  validation exposure. Do not reclassify previously used development images as
  an untouched final test set.

The output `source_clusters.csv` is a grouping report, not a new split manifest.
Any conflict resolution and the eventual detection manifests must be versioned
and documented. Automated pixel reading for integrity and split-leakage checks
does not constitute detector evaluation; no held-out image previews or model
scores are produced here.

## Local validation and remaining work

Eight focused synthetic tests cover a negative-image bridge between old train
and test groups, preservation of inputs, candidate-cap failure, changed images
and XML, metadata tampering, raw-file membership changes, overwrite protection,
and continuous small-box handling. The local metadata preflight is recorded in
`reports/detection_preparation_v1/verification.json`.

These checks establish the tested software behavior, not annotation correctness
or model effectiveness. The CCAST audit has now completed: all 38,385 images
and their XML files matched the pinned audit records. Its returned archive and
all ten internal member hashes were verified locally. The immutable evidence
is `detection_audit_v1.tar.gz`, SHA-256
`07963cfdc3e14d2032dd1a80a1240a19c415598ef449bde737fe422902dd4f7b`.

## Detection manifests prepared after review

`reports/detection_split_review_v1/review.md` records the audit findings and
the decisions used by `make_detection_splits.py`. Two full-pool similarity
components connect previous training and validation assignments. Both remain
whole in training; six previous validation images move into training in the
new detection manifest. All 3,569 previous test images keep their assignment.
No test/development conflict was found by this screen.

The earlier exact-copy exclusion is retained. India_006389 is excluded from
this version because its only annotation is the ambiguous label `D0w0`; it is
neither relabeled nor used as a target-negative example. The one nonpositive
target box in Japan_001265 is omitted while its three valid targets remain.
All other previously omitted images with no target annotation are included.
New unanchored groups are allocated within source toward the existing 70/15/15
ratios, balancing image counts and the two no-target annotation categories.

The canonical manifest uses continuous XYXY boxes and zero-based foreground
class indices D00=0, D10=1, D20=2, D40=3. A detector adapter must translate class
indices explicitly if its API reserves zero for background. Original non-target
annotations and difficult/truncated flags are retained. All valid targets,
including difficult=1, are included as ordinary ground truth under this study's
documented convention. Stored image orientation is required.

| Partition | Images | Target boxes |
| --- | ---: | ---: |
| Training | 26,863 | 38,530 |
| Validation | 5,760 | 8,225 |
| Test | 5,760 | 8,247 |
| Total | 38,383 | 55,002 |

Eight focused synthetic tests passed for the new manifest builder. A separate
saved-file verification checked image/target coverage, annotation preservation,
historical exposure, all 1,107 new and 557 historical candidate edges, group
separation, exact hash separation, and archive contents. No screened group or
retained candidate edge crosses partitions. This does not establish complete
road/sequence independence. No raw pixels were opened locally during this step.

The generated `detection_splits_v1/` directory contains one JSONL file per
partition, membership CSV, configuration, decisions, exclusions, generator,
and checksums. The transfer package is `detection_splits_v1.tar.gz`, SHA-256
`89d1c16954187ef0126f69c670b83875c8b4ff8863aa5a1864c264df91b0cce4`.
Its checksum list has LF line endings; the CRLF error in the earlier audit
transfer package was corrected for this package.

CCAST deployment is complete according to the user's pasted terminal output:
the archive hash matched, extraction into `dataset/detection_splits_v1`
completed without reported errors, and all 13 entries in `SHA256SUMS` passed.
`reports/detection_split_review_v1/deployment.json` records this evidence and
its source. Step 2 is complete. Commands continue to be supplied one at a time.
The next stage is a small detector feasibility comparison, beginning with a
project-specific Python environment that reuses the verified PyTorch stack.
No detector training has started here.

The project environment `rdd_detection_v1` now passed the required package
imports, exact-version checks, and `pip check`, according to the user's CCAST
terminal output. In PBS job `794021.bright04` on `gpu0021`, it sees exactly
one assigned NVIDIA A10; TorchVision CUDA NMS and RoI Align forward/backward
both passed. `reports/detector_feasibility_v1/environment_setup.json` records
these checks. A Faster R-CNN ResNet50-FPN smoke check also passed one optimizer
step and inference with random weights on two synthetic 320x320 images: one
with all four foreground labels, one with no boxes. YOLOv8n subsequently
passed the equivalent random-weight synthetic optimizer/inference/NMS check.
The user exited the GPU allocation. These are software checks and do not
select the detector or establish detection accuracy.

`detection_loader_v1/` supplies the bounded data-interface check. It uses
15 training images selected from metadata for engineering case coverage,
checks source hashes and stored orientation, prepares the same full-image
pixels for both model adapters, preserves empty targets, and verifies class
indices and inverse box coordinates. Its README documents the provisional
640-pixel resize/padding convention. All 22 unit tests passed on CCAST, followed
by the 15-image CPU check in PBS job `794025.bright04` on `gpu0020`. The returned
archive has SHA-256
`8133f0f9c069ee3f9301afd662d4a4b2f43c4eaf1705de6cbf72b12cda07549e`.
Local inspection verified the archive hash, all nine saved code/package files
against the transferred release, all eight package checksum entries, and every
sample record against the frozen training manifest and selection. All 16
target boxes were retained; the largest reported coordinate round-trip error
was 0.0000832053 original-image pixels, below the 0.002-pixel tolerance.
`reports/detection_loader_v1/ccast_verification.json` records this verification.
No validation/test images or model outputs were used by this package. This
completes the selected loader checks; real-image detector integration and the
bounded detector feasibility experiment remain pending.

The downloaded `rdd_detection_v1_environment.txt` now matches its CCAST source
checksum, `f07b523b024d61f46fe407df674fa2361a2f9332ed489f794f7b8b20024d77c8`.
Its 127 entries include the pinned detector packages. Inherited Conda packages
often appear as build-directory file URLs, so this freeze snapshot alone is
not a portable environment installation file. Imported core versions are
recorded separately in the environment record.

`detection_integration_v1/` is the next prepared GPU check. It reuses the four
unchanged loader files and processes the same 15 training images with each
random-weight detector: eight optimizer steps and eight inference batches per
model, including an all-empty batch. Both adapters' image tensors and per-batch
signatures must agree. Losses, gradients, parameter updates, prediction schemas,
coverage, and partial failures are recorded. It computes no accuracy metrics
and does not choose an architecture. Its README gives the bounded protocol.
Fourteen focused local tests passed, and the actual Faster R-CNN integration
path passed an update/inference check on synthetic CPU inputs using the local
stack. The transfer
archive SHA-256 is
`c1a42e5e771f8a7ec0f810cc5352c9fcf00509c5e35084bb404ee5caac1a6b18`;
`reports/detection_integration_v1/release.json` records its verified contents.

The 14 tests subsequently passed on CCAST, followed by the real-image check
in GPU job `794043.bright04` on `gpu0020`. Both detectors completed all eight
updates and inference batches, including the all-empty sixth batch and final
single-image batch. The returned archive has SHA-256
`aa63883e24fecdbf7f68f993562310935f9b935247940e27edaafbd46e98e0a1`.
Local verification matched all ten saved package files to the release, checked
the nine checksum entries, and inspected all sixteen step records. Both models
received exactly matching recorded inputs and signatures. Every image ID,
target count, class label, geometry, and reconstructed prepared-box hash matches
the pinned training manifest. All recorded losses and update magnitudes are
finite; every step reports an actual classifier update. The detailed record is
`reports/detection_integration_v1/ccast_verification.json`.

The observed peak allocated tensor memory was 2.067 GiB for Faster R-CNN and
0.402 GiB for YOLOv8n at batch size two and 640 pixels. These measurements
establish that both tested paths fit the assigned A10, not an architecture
ranking or a throughput benchmark. The terminal reported nonfatal cuDNN
execution-plan warnings; their cause and performance impact were not established.
No validation/test images, accuracy metrics, or reusable checkpoints were
produced by this integration check.

Next, prepare a bounded training-only learning check with pretrained weights,
validated detection metrics, and checkpoint handling before a larger baseline.
The pretrained weights will be kept under the CCAST project directory
`weights/detection_v1/`, with source and checksum provenance recorded. Training
settings and stopping rules must be saved before execution. A same-architecture
random-initialization follow-up remains part of the broader plan; this next
check does not select the final detector or replace that control.

Both official pretrained checkpoints were downloaded on the CCAST login node,
according to the user's terminal output. Faster R-CNN COCO_V1 has SHA-256
`258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`,
matching the published filename prefix. YOLOv8n from the official `v8.4.0`
assets release has SHA-256
`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`,
which exactly matches the release's published full digest. Source URLs, sizes,
and evidence limits are saved in
`reports/detector_feasibility_v1/pretrained_weights.json`. These weights have
not yet been loaded into the four-class training path. The bounded learning
protocol and training package remain to be prepared before allocating a GPU.

The user confirmed creation of the checkpoint `SHA256SUMS` on CCAST. Independent
local downloads produced the same full hashes. The new `detection_learning_v1/`
package is prepared for 200 updates per pretrained detector on the same 15
training images. It evaluates training AP at steps 0, 50, 100 and 200, includes
all no-target images, records original-coordinate predictions, and saves/reloads
model, optimizer and RNG state at steps 100 and 200. The package README and
protocol specify the engineering review rule and its limitations. Neither
validation nor test data enters this check.

All 18 focused local tests passed. Both real pretrained models also passed
synthetic CPU updates, inference, checkpoint restoration, continued updates and
all-empty updates. Those checks use 128-pixel synthetic inputs and a newer local
PyTorch/TorchVision stack; Ultralytics is the pinned 8.4.182 wheel imported from
an isolated temporary directory. CCAST execution at 640 pixels is still pending.
The frozen transfer archive is `detection_learning_v1.tar.gz`, 34,716 bytes,
SHA-256 `26c7e771bd58856d1fbbb81b3b2d4929d89dbafeffe0f221511ea03b05ef3022`.
All 14 archive files and 13 checksum entries were verified with LF line endings.
`reports/detection_learning_v1/release.json` records the release. The next action
is uploading this package from Windows PowerShell; commands remain one at a time.

The user subsequently confirmed package transfer, checksum verification and
deployment. All 18 tests passed in CCAST job `794071.bright04` on `gpu0020`.
The 200-update learning run completed for both models, with exact scheduled
model/optimizer/RNG restoration reported at steps 100 and 200. Terminal training
AP50 values at steps 0/50/100/200 were 0/0.073063/0.190208/0.355172 for Faster
R-CNN and 0/0/0.005178/0 for YOLOv8n. Final training AP50:95 was 0.175538 and 0,
respectively. Both fall below the predeclared final AP50 review threshold of 0.50.
Faster R-CNN shows increasing training AP; YOLO's final score requires diagnosis.
This does not establish a detector ranking or held-out performance.

`reports/detection_learning_v1/ccast_terminal_result.json` preserves these
user-reported results, their precision and evidence limits. The reported result
archive SHA-256 is
`0ab14951c60bf566e2585ea66648d3bc743e2c427558b1f85b762ed50dd9693a`.
Local transfer and independent record verification remain pending. Retrieve and
inspect the recorded losses, per-class metrics, predictions, configuration and
checkpoint records before changing settings or starting further training.

The user returned to the login node and downloaded the learning archive. Local
verification matched its SHA-256, all 38 result checksum entries and all 14
released package files. The audit verified all 400 update records, matched the
input signatures to the earlier integration run, and reconstructed the ground
truth from the frozen training manifest. Independent direct COCO API evaluation
reproduced all eight scalar and per-class evaluations; the largest scalar
difference was zero. The reproducible audit and results are saved under
`reports/detection_learning_v1/`, including `audit_ccast_results.py`,
`ccast_verification.json`, `metrics.csv` and `loss_by_pass.csv`.

Faster R-CNN's retained predictions cover every target at IoU 0.5 with the
correct class, although this candidate-coverage statistic is not AP or one-to-one
recall. YOLO's final 1,307 retained predictions have no IoU-0.5 match even when
class is ignored. Its no-target batch at step 182 also produced a classification
loss spike and a maximum classifier update of 0.368820, compared with 0.007745
on the preceding step. These findings establish a failure of this training
configuration, not a cause or a detector ranking. The reviewed source uses a
simplified SGD loop without the standard Ultralytics trainer's gradient clipping;
normalization behavior and optimization stability remain separate hypotheses.
`post_learning_review.json` records the findings, source references and limits.

The next prepared package, `detection_diagnostics_v1`, evaluates the existing
YOLO checkpoints at steps 100 and 200 under running statistics, batch statistics,
then running statistics again. It takes zero optimizer steps, verifies unchanged
parameters/buffers/RNG/checkpoint bytes, requires prediction replay, and records
raw box coverage, score filtering, assignments and AP. This is an exploratory
normalization diagnostic on the same training cases, not a new benchmark.
Eight focused tests and the actual YOLO path on synthetic CPU inputs passed.
The six-file transfer archive is 14,878 bytes with SHA-256
`a6457377df59fc47a6bf77a87b4719665f90c0cd189e204799f62553626702b1`.
CCAST transfer and execution remain pending. The prior releases and result
archives are preserved, and commands continue one at a time.

The evaluation-only diagnostic subsequently completed in CCAST job
`794142.bright04` on `gpu0026`. After reconnecting, the user found its complete
records and downloaded `detection_diagnostic_check_v1.tar.gz`, SHA-256
`14dd16aad80c0dbf546994aa4c2aad80c6dfb215ca5d7f7a222e218b5557193f`.
The archive contains 22 files. Local verification matched all 21 result checksum
entries, all six diagnostic code files and the pinned learning dependencies.
The ground truth matches the frozen training manifest, and the input records
match the earlier learning run exactly. All six scalar and per-class COCO
evaluations reproduce with zero difference. Running-stat predictions match the
original learning run byte for byte and repeat unchanged after the intervention.
No CCAST unit-test transcript was supplied for this package; completion does not
establish a separate target-host test count.

At checkpoint 100, training AP50 is 0.005178 with running statistics and
0.171971 with current-batch statistics. At checkpoint 200, it is 0 and 0.080771,
respectively. This isolates normalization sensitivity on the fixed batches but
does not validate a remedy. At step 200 with batch statistics, raw candidate
boxes cover 13 of 16 targets at IoU 0.5; only three targets have a candidate
meeting both IoU 0.5 and the correct-class score threshold of 0.001. These
coverage counts are not one-to-one recall or AP. Weak class scores remain, and
the earlier update spike's cause is unresolved. No held-out data were used.

The reproducible record audit, metrics, coverage and assignment tables are in
`reports/detection_diagnostics_v1/`. The interpretation and proposed next action
are in `post_diagnostic_review.json`: compare the original normalization behavior
with fixed BatchNorm running statistics during training and evaluation, while
holding initialization, data, optimizer and the 200-update budget constant.
This comparison is proposed, not executed or packaged. A normalization-only
failure should stop that repair attempt, not trigger an automatic extension.

The user then flagged that work was drifting from the main research objective.
The proposed YOLO normalization-training repeat is therefore deferred. Its
diagnostic results remain preserved, but it is no longer the next action.
The recommended primary path is the existing pretrained Faster R-CNN
implementation, which showed a training learning signal and still requires a
proper baseline training/validation run. This is a sequencing recommendation,
not evidence that Faster R-CNN is superior or a user-confirmed final selection.
The next deliverable is the main training/validation pipeline supporting an
identity control, conventional shared augmentation, a matched fixed local
policy and the proposed adaptive controller. Later detector and initialization
comparisons remain separate work; course requirements are not waived.

The user then explicitly selected Faster R-CNN and authorized preparation of the
full training/validation loop and grouped launch commands. The resulting package
is `fasterrcnn_baseline_v1.tar.gz`, 33,032 bytes, SHA-256
`2d8ba5748350360e4d933e542bb350c4c45a9df06ed3febb73025fba2529e675`.
Its 16 archive files and 15 checksum entries were verified. The release record
is `reports/fasterrcnn_baseline_v1/release.json`.

This first run is an identity-augmentation control: COCO-pretrained Faster R-CNN
ResNet50-FPN, a new four-class damage head, 640-pixel letterboxing, seed 42,
batch size four, and 20 complete epochs over 26,863 training images. It retains
the final minibatch and all empty-target images. Full validation on 5,760 images
runs after every epoch; AP50:95 selects the best checkpoint, with earlier epochs
winning ties. AP50, AP75, AR100, per-class AP and negative-image false positives
are also recorded. No test manifest or pixels are opened. This is an initial
development recipe, not a convergence or augmentation-efficacy result.

Thirteen focused local tests passed, including mid-epoch stochastic resume,
validation isolation, incomplete-validation recovery, negative images and
atomic checkpoint failure. The actual pretrained model also passed synthetic
CPU updates, validation and exact prediction replay after loading a checkpoint
into a fresh model. Local library versions differ from CCAST, and no RDD pixels
or CUDA training were used locally. The complete train/validation metadata
passed hash and separation checks; all retained boxes stayed valid after the
declared resize. Python 3.12 grammar and all three Bash scripts' syntax passed.

Submission uses PBS rather than an interactive allocation: one A10, four CPUs,
32 GB RAM and a 24-hour limit in `gpus`. The queued job activates the existing
environments, repeats the focused tests and requires a successful real-data
24-training/24-validation-image GPU smoke check before starting the full run
from fresh pretrained initialization. Full-run checkpoints include optimizer,
RNG and minibatch progress. A time-budget stop preserves the fixed 20-epoch
plan for explicit resumption. No job is submitted or resubmitted automatically
by local preparation.

CCAST upload, tests, smoke and full training are pending. The immediate handoff
is two blocks: Windows upload, then CCAST checksum verification, protected
extraction and `bash submit.sh start`. Each block stops on failure. Record the
returned job ID before monitoring; do not infer scientific performance from
submission or a passed software check. The earlier tiny-subset AP50 review
threshold is no longer a prerequisite for this proper full-data baseline.

On 2026-10-23, the full baseline completed in PBS job `794364.bright04` on
`gpu0035` with one NVIDIA A10. The user-pasted launch log reported all 13 tests
passed and a complete real-data smoke check with exact checkpoint prediction
replay. The full run completed 20 epochs, 134,320 updates and 537,260 image
presentations in 39,927.58 seconds (11.09 hours), without a resumed invocation.
The SSH disconnects did not interrupt the recorded full training trajectory.

The downloaded records archive is
`fasterrcnn_identity_seed42_v1_records_b07d6daa294f.tar.gz`, 391,877,600 bytes,
SHA-256 `4d16ae384cc8f8663b47a00f84df331ea32c44f3742aebaa71e114b55fc0026f`.
Local analysis verified all 85 files, 84 result checksum entries and 16 released
code/package files. Configuration and environment match the released protocol.
The validation ground truth reconstructs exactly from the frozen validation
manifest. All 20 epochs have complete ordered validation coverage, valid saved
prediction geometry, consistent prediction/negative-image counts, epoch history
and checkpoint metadata. The train/validation image, cluster and RGB-hash
separation checks passed. The audit never opens the test manifest or raw pixels.

Epoch 14 is the declared AP50:95-selected checkpoint. Its validation AP50:95 is
0.2282311872580586, AP50 is 0.5114178506885583, AP75 is 0.17090964508633316 and
AR100 is 0.4383598835859107. Its per-class AP50:95 values are D00 0.24744259,
D10 0.20206224, D20 0.29975413 and D40 0.16366578. The final epoch has AP50:95
0.22392050310292788 and AP50 0.5020009921812068. Independent direct COCO API
evaluation of saved predictions reproduces all aggregate and per-class AP/AR
metrics at epochs 14 and 20 with zero difference. The other epochs' AP values
were checked for record consistency but were not independently recomputed.

Training loss falls from 0.25308 to 0.14424. Validation AP50:95 rises sharply
after the first scheduled learning-rate reduction and peaks at epoch 14; later
epochs remain slightly lower. This trajectory does not establish a causal
effect of the schedule or prove overfitting. At the selected epoch, 858 of 2,193
images with no retained target annotation have at least one prediction at
confidence 0.25. These are false positives relative to the frozen annotation
policy; annotation completeness and operational threshold calibration have not
been newly assessed. No conclusion about augmentation benefit follows from
this single identity run.

Audit code, exact verification scope, CSV tables, returned run/selection/context
records and learning-curve figures are under `reports/fasterrcnn_baseline_v1/`.
Full model checkpoint binaries and live stdout/stderr are outside the returned
records archive. The epoch-14 checkpoint's recorded SHA-256 is
`5d2310d78533456edbea7d43aeee48b89817d4e920a3d764243859f5dda62a41`;
the local audit checked this metadata and its references, not the remote bytes
or a fresh inference replay. The release record remains a preparation-time
snapshot; `ccast_verification.json` records the completed-run evidence.

The recommended next experiment is the planned fixed shared-augmentation
control using matched initialization, splits, sampling, training budget,
validation and selection rules. Preserve the identity baseline and keep test
data unopened while developing the comparisons. Do not extend this baseline
just because the final epoch is below the selected epoch. The new comparison
has not been implemented or launched by this analysis.

## Fixed shared-augmentation comparator prepared (2026-10-23)

The user authorized the next fixed shared-augmentation baseline and grouped
CCAST commands. Its separate package is `fasterrcnn_shared_v1.tar.gz`, 43,788
bytes, SHA-256
`4769ae879e222a5fd79382b4ccbb9900b358a7dfce27d2fbfb42377b0d471f38`.
All 20 archive files and 19 checksum entries were checked. The release record
is `reports/fasterrcnn_shared_v1/release.json`; the exact configuration and
usage are in `fasterrcnn_shared_v1/protocol.json` and its README.

This is the first fixed recipe, not yet the paper's validation-selected fixed
policy. Epoch 1 is unaugmented. During epochs 2â€“20, target-bearing training
images receive one image-level edit with probability 0.5. Empty-target images
remain unchanged to match eligibility for the future local-edit comparison.
The photometric and blur/noise families have equal probabilities; operations
are uniform within each family. Brightness and contrast factors are 0.8/1.2,
gamma exponents are 0.8/1.2, blur radius is 0.8 resized pixels, and Gaussian
noise standard deviation is 0.02 on [0,1] RGB. All resized image content is
eligible for the edit, while letterbox padding, boxes and labels stay fixed.
These values are a prospective initial recipe, not established class-specific
safe limits. There is no feedback controller or policy tuning in this run.

The script starts afresh from the same COCO initialization and seed as the
identity baseline. All non-augmentation training and evaluation settings are
checked against a frozen copy of that baseline's protocol. Eight dependency
files and eight critical training/evaluation/checkpoint functions remain
unchanged. Full training retains 26,863 images, 20 epochs and 134,320 updates,
with complete validation on 5,760 images each epoch and the same AP50:95
checkpoint selection rule. No test data is opened.

Augmentation draws use an independent deterministic image/epoch hash stream,
with a local PCG64 generator for noise. The training loop logs actual operation
counts and a hash of decisions, checks them against the fixed plan, and
preserves partial-epoch progress on resume. Metadata-only calculation predicts
158,151 augmented presentations across the complete run; this is a planned
exposure count, not executed training or a guarantee that every edit changes
pixels. The full counts are in
`reports/fasterrcnn_shared_v1/expected_exposure.json`.

All 25 focused local tests passed with no failures or skips. Tests cover known
operation effects, geometry/padding/target preservation, validation and negative
image isolation, global RNG isolation, worker-count invariance and exact
mid-epoch continuation on a stochastic CPU model. The actual pretrained
Faster R-CNN also passed a separate synthetic CPU check: initial weights exactly
match the identity implementation, all eight operations support finite training
updates, entirely negative batches remain supported, validation preserves the
model, and a fresh model restores checkpoint predictions exactly. Python 3.12
grammar and Bash syntax checks passed. Local versions differ from CCAST;
these checks do not establish augmentation efficacy or real-data CUDA behavior.

Submission requests one A10, four CPUs, 32 GB and 24 hours. The queued job
repeats the tests and requires a 24-training/24-validation-image GPU smoke
check before fresh full training. The smoke also saves a train-only preview
of all operations. Full output goes to `runs/fasterrcnn_shared_seed42_v1/`,
smoke output to `runs/fasterrcnn_shared_smoke_v1/`, and live logs to
`runs/baseline_job_logs/<PBS_JOBID>.log`. Existing identity artifacts are
preserved. CCAST submission, target-environment tests, GPU smoke and full
training are pending; the user receives Windows upload and CCAST submission
blocks that stop on failure.

## Shared-versus-identity results verified (2026-10-24)

The shared run completed 20 epochs in PBS job `795553.bright04` on `gpu0021`
with one NVIDIA A10, 134,320 optimizer updates and 537,260 image presentations.
The user-pasted launch log showed all 25 tests passing in 11.983 seconds. Its
records archive is `fasterrcnn_shared_seed42_v1_records_31a774d9401a.tar.gz`,
412,104,336 bytes, SHA-256
`28b28bee01376cc10fe0a60dca7ddfb9cd97ec9b548597fe316d94db164bfd7c`.
Local verification checked all 89 archive files, 88 result checksums, 20 released
package files, all 20 epochs' validation coverage/counts/prediction geometry,
configuration and environment, and checkpoint metadata references. The direct
COCO API independently reproduced aggregate and per-class metrics at the selected
epoch 14 and final epoch 20 with zero numerical difference. Full checkpoint
binaries and raw pixels were not reopened; no test data was read.

All 20 augmentation decision sequences and their ordering replay exactly from
the frozen training metadata. There were 158,151 planned/applied presentations;
CCAST reports that all of these changed pixels. Epoch 1 was unaugmented. The
recorded identity archive and its prior verification source were rechecked;
its previous direct COCO recomputation was retained without rerunning it.

Both conditions selected epoch 14 using the same AP50:95 rule. On the 0â€“100
scale, identity versus shared results are: AP50:95 22.8231 versus 22.8099
(shared minus identity -0.0132 percentage points); AP50 51.1418 versus 50.6873
(-0.4544); AP75 17.0910 versus 17.2951 (+0.2041); AR100 43.8360 versus 44.3389
(+0.5030). Per-class AP50:95 changes are D00 +0.0400, D10 +0.0593, D20 -0.3449
and D40 +0.1929 percentage points. Every class has lower AP50 in the shared run.
At confidence 0.25, images without retained target annotations containing at
least one prediction increased from 858/2,193 to 867/2,193. These counts depend
on the frozen annotation policy and do not establish annotation completeness.

The shared run's final-epoch AP50:95 is 22.5746 versus identity's 22.3921
(+0.1825 points). The declared primary comparison remains the selected
checkpoints; the last-epoch improvement does not change that result. Elapsed
time was 11.51 versus 11.09 hours (+25.31 minutes, +3.80%), measured on different
A10 nodes, so it is not an isolated augmentation-overhead estimate.

An important limitation is visible before augmentation begins: epoch-1
AP50:95 was 15.8920 for shared versus 16.5371 for identity (-0.6451 points),
although both used unaugmented inputs under the matched declared configuration.
The cause of this divergence is not established. Matching a seed and settings
did not produce identical realized training trajectories; very small later
differences cannot be attributed exclusively to augmentation from this pair.
One run per condition does not estimate training variability or demonstrate
statistical equivalence. Both checkpoint selection and comparison use the
development validation set.

The supported conclusion is no observed primary-metric gain from this initial
fixed shared recipe, with small mixed class-level changes. This is neither a
general rejection of augmentation nor evidence that an adaptive controller is
needed or effective. Preserve both controls. The next planned ablation is
fixed local application with the same operation bank, strengths, eligibility
and probability; repeat matched runs before claiming a reliable small gain.
No additional experiment was launched during this analysis.

The audit, exact comparison JSON/CSVs and inspected PNG/PDF learning curves are
under `reports/fasterrcnn_shared_v1/`, including `ccast_verification.json`,
`augmentation_verification.json`, `comparison.json`, `compare_with_identity.py`
and `comparison.png`. Released package and preparation records remain unchanged.

## Agreed experiment sequence (2026-10-24)

The user moved the online shared-local control into the main sequence before
the class-adaptive condition. The current order is:

| Stage | Condition | Main question relative to the previous stage | Current status |
| --- | --- | --- | --- |
| 1 | Identity | What does the detector achieve with no appearance augmentation? | One seed completed and verified |
| 2 | Fixed shared image-level | Does this fixed image-level recipe improve on identity? | One seed completed and compared; no primary-metric gain |
| 3 | Fixed shared local | Does restricting the same recipe to one selected damage group change performance? | One seed completed, verified and compared; no primary-metric gain |
| 4 | Online shared local | Does updating one common policy using detector feedback improve on the fixed local recipe? | One seed completed, records verified and compared; selected epoch 14, AP50:95 23.059 |
| 5 | Online class-adaptive local | Does conditioning the policy on the selected group's class set improve on one common online policy? | Package prepared and locally checked; CCAST submission pending |

Here, "shared" means one policy is used across classes. Both fixed controls
share their policy across classes; image-level versus local specifies its pixel
support. "Fixed" holds the operation probabilities and strengths constant,
while individual operations can still be sampled randomly. Online shared local
updates one common policy during training. Online class-adaptive local updates
policies conditioned on the class set represented in the selected group.
The shared online control may still use class labels for the matched retention
and loss measurements; it does not select a different policy by class set.

All local conditions retain full-image detector training and all annotations.
They use the same overlap grouping, one-group selection and blended union edit.
Match detector initialization, frozen splits, image order, optimization budget,
application probability, eligibility, warmup, validation and checkpoint selection
within paired seeds. Each condition starts from the same declared initial
weights. The sequence is an order of separate experiments, not successive
fine-tuning of one run's checkpoint.

For stages 4 and 5, match candidate operation/strength definitions, probe data
and schedule, budget rules, support requirements, retention criteria and
class-balanced loss definitions. The intended difference is whether the
resulting policy is common or class-set-conditioned. Record realized probe cost
and edit exposure so differences in resources remain visible. Numerical online
controller settings must be specified before those experiments are launched.

Keep the one-seed comparison and unaugmented warmup divergence as limitations;
this sequence does not replace replication or final held-out evaluation. The
immediate next work is stage 3, fixed shared local augmentation. No new training
was launched as part of this sequence update.

## Fixed shared local baseline prepared (2026-10-24)

The user authorized preparing stage 3 and grouped CCAST commands. The separate
`fasterrcnn_local_v1/` package and `fasterrcnn_local_v1.tar.gz` are now prepared.
Archive SHA-256:
`1aa06a73f778bdb06ff8bc2128ccd668cdb820c48e31cbb491d9440caf22569c`
(57,734 bytes; 24 archive files, 23 checksum entries). This is a preparation
record; no local-augmentation CCAST job has been submitted by the assistant.

The detector, fresh COCO initialization, frozen train/validation data, image
order, optimizer, 20-epoch budget, validation loop, checkpoint selection and
checkpoint/resume helpers match the completed controls. It retains full-image
training, all target boxes, negative images and unaugmented validation.

The appearance operations, strengths, warmup, probability and random draws are
the completed shared recipe. Only spatial application changes: rasterize the
prepared boxes with floor/ceil half-open bounds, group connected positive-area
raster overlaps, select one group uniformly, then composite the reference
appearance transform once into its union. Alpha is 0 outside, 0.5 on the outer
8-neighbour boundary and 1 in the eroded interior. Fractional and thin supports
are retained. Other groups, outside-union pixels, padding and labels are
unchanged. No adaptive policy, detector feedback or class-specific selection is
implemented in this stage. Exact numerical conventions are in
`fasterrcnn_local_v1/protocol.json` and `README.md`.

Completed local checks:

- All 39 tests passed, including transitive/raster overlap, union holes,
  single composition, padding, fractional boxes, unchanged targets, shared
  appearance decisions, global RNG isolation, worker independence and exact
  stochastic mid-epoch resume.
- The actual COCO Faster R-CNN passed a synthetic 128-pixel CPU check with all
  eight operations, three finite updates including a negative-only batch,
  classifier updates, unchanged validation state and exact checkpoint replay.
- Eight dependency files and eight core training/evaluation/checkpoint functions
  match the identity control. The shared appearance module is byte-identical.
- Frozen training-metadata replay matched every ordered appearance-decision
  hash and operation/family count from all 20 completed shared epochs: 158,151
  planned augmented presentations across the same 537,260 total presentations.
- Python 3.12 grammar, LF line endings, all three Bash scripts and archive bytes
  were checked. Git Bash required the known read-only retry outside the Windows
  sandbox; no scripts or remote jobs were executed by that syntax check.

At 640 pixels, the frozen training geometry contains 30,744 groups in 16,630
target-bearing images; 4,908 groups have multiple boxes and 3,558 contain multiple
classes. These counts use the current detection manifests and raster rule.
The planned selected union covers a mean 6.94% (median 2.76%) of content per
augmented image. These are prospective geometry/exposure measurements, not
observed pixel changes or model results. Changed pixels are measured during
the CCAST run without resampling operations that happen to change none.

The prepared PBS job requests one A10, four CPUs, 32 GB and 24 hours. It runs
the tests in the pinned environment, then the existing 24-train/24-validation
GPU smoke, including all-operation and overlap/separated/small-group previews.
Full training starts in a fresh process only after the smoke gate passes.
Outputs are `runs/fasterrcnn_local_smoke_v1/` and
`runs/fasterrcnn_local_seed42_v1/`; logs remain in `runs/baseline_job_logs/`.
The job survives SSH disconnection. Progress is checkpointed; budget stops
require explicit resumption, with no automatic extension of the epoch budget.

Local tests use different library versions from CCAST and synthetic pixels.
CCAST tests, GPU smoke, full training and scientific comparison remain pending.
This spatial comparison also changes edit area and boundary blending, so it
cannot isolate location alone. Preserve the one-seed limitation and the already
observed unaugmented-warmup divergence of the existing controls; no accuracy gain
or controller benefit is claimed. Keep the test set reserved.

Evidence: `reports/fasterrcnn_local_v1/release.json`, `unit_tests.txt`,
`local_model_check.json`, `comparator_checks.json`, `expected_exposure.json`,
`shell_checks.json`, and the packaged `LOCAL_VALIDATION.json`.


## Fixed local results verified and three-way comparison (2026-10-25)

PBS `795787.bright04` on `gpu0020` completed 20 epochs,
134,320 optimizer updates and 537,260 image presentations without resumption.
Elapsed full-run time was 13.26 hours on an A10;
this excludes tests/smoke and is not an isolated augmentation-overhead estimate.

Returned archive: `fasterrcnn_local_seed42_v1_records_b047925ba90a.tar.gz`
SHA-256: `6f1efc305780b1ab018a807bd728460b04582586166f6aeb616e693c6cfdcbac`.
Its 413,370,098 bytes, all 93 members and
92 checksum entries were verified locally.
All 24 released package files, execution configuration,
frozen split separation and validation ground truth matched. All 20 epochs had
complete validation coverage and consistent prediction geometry, counts, history
and checkpoint metadata. Epochs 18 (selected) and 20 (final) were independently
recomputed through direct COCO evaluation with maximum metric difference 0.
Full checkpoint binaries remain on CCAST; raw images and inference were not rerun.

The primary comparison uses each run's highest full-validation AP50:95, with
earlier ties retained. Scores below are on the 0–100 scale:

| Condition | Selected epoch | AP50:95 | AP50 | AP75 | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 14 | 22.8231 | 51.1418 | 17.0910 | 43.8360 |
| Fixed image-level | 14 | 22.8099 | 50.6873 | 17.2951 | 44.3389 |
| Fixed local | 18 | 22.5550 | 50.1602 | 16.5704 | 44.0782 |

Fixed local is **0.2681 percentage points below identity** and **0.2549 points
below fixed image-level** on selected AP50:95. Its selected AP50 is lower by
0.9816 and 0.5272 points, respectively. The final local epoch has AP50:95
22.4871 and AP50 49.8136. Final-epoch comparison is kept separate; choosing the
final epoch because it changes a comparison would violate the declared rule.

Per-class selected AP50:95 changes (percentage points):

| Class | Local minus identity | Local minus image-level |
| --- | ---: | ---: |
| D00 (Longitudinal cracks) | -0.6546 | -0.6945 |
| D10 (Transverse cracks) | -0.6861 | -0.7454 |
| D20 (Alligator cracks) | +0.1390 | +0.4839 |
| D40 (Potholes) | +0.1292 | -0.0637 |

At confidence 0.25, 817/2,193 local negative validation images had detections,
versus 858 for identity and 867 for image-level. Negative-image prediction boxes
were 1,538 versus 1,637 and 1,640. These are counts relative to retained annotations,
not a new visual review of annotation completeness. This secondary change does
not overturn the primary AP result.

Local group choices, geometry, exposure and decision ordering replayed exactly
for all 20 epochs against the prospective metadata plan. The separate appearance
hashes and operation/family counts matched the completed image-level control in
every epoch: 158,151 applied presentations, all reported changed on CCAST.
The recorded 4,100,790,428 changed pixels were 92.40% of the selected union pixels
and 6.70% of content pixels across augmented presentations. Union area was 7.25%
of that aggregate content area; alpha-weighted area was 7.17%. These are aggregate
area ratios, distinct from the earlier per-image mean of 6.94%. Actual pixel
changes were reported by CCAST and checked for consistency, not recomputed from
raw pixels locally.

**Finding:** this fixed local recipe did not improve the primary detection metric
in this seed. It is a completed control, not evidence that adaptive augmentation
will work. The class responses and negative-image counts are descriptive.

**Limits:** one run per condition; validation used for both checkpoint selection
and reporting; no untouched test evaluation. The unaugmented first epoch already
differed: local AP50:95 16.4239, identity 16.5371, image-level 15.8920. The cause is
not established, so small later changes cannot be attributed exclusively to
augmentation. Local application also changes edit area and boundary blending.
No statistical equivalence or reliable gain is established.

**Next:** preserve all three controls. Stage 4 remains online shared local, with
its feedback rule and numerical protocol specified before implementation or
another full run. Compare it directly to this fixed local control. Stage 5 then
tests class-set conditioning against the common online policy. No next-stage
training has been launched by this analysis.

Evidence and reproducible analysis are under `reports/fasterrcnn_local_v1/`:
`audit_ccast_results.py`, `ccast_verification.json`, `recomputed_metrics.json`,
`augmentation_verification.json`, `compare_baselines.py`, `comparison.json`,
CSV tables and `comparison.png` / `comparison.pdf`. The original archives and
released packages, including their preparation-time status snapshots, are unchanged.


## Online shared-local controller prepared (2026-10-25)

The user authorized implementation of stage 4 and command blocks. The separate
`fasterrcnn_online_shared_v1` package is prepared; no remote submission or result
has been observed. Stage 5 class conditioning remains unimplemented.

The controller pools selected-group feedback into one common policy. After one
unaugmented epoch it probes 256 source-diverse training images before epochs
2,4,...,20. Each image contributes one uniformly selected overlap group. All
8 operations at 3 strengths are measured on the same images/model; noise has
2 paired draws reused across strengths. Reference detections use confidence
0.25 and IoU 0.5. Every class needs 8 detected objects across 4 clusters, and
every operation/class in a family must retain at least 0.8. Unsupported families
are unavailable; no eligible family means identity. Class-balanced normalized
object-loss increases drive a softmax/exploration distribution (delta 0.1, cap 5,
temperature 0.25, exploration 0.2). These are prospective numerical choices,
not validation-selected optimal settings.

Ordinary detector predictions are unchanged. Auxiliary object losses use
positive eval RPN proposals plus explicitly self-assigned GT ROIs, CE plus
class-specific SmoothL1, averaged within objects. GT ROIs never enter retention
detections. Probes check unchanged model/buffer/gradient/RNG state. Complete
policy evidence is committed with model/optimizer/RNG checkpoints and recomputed
from saved records on resume. Interrupted/unadopted attempts retain cost records.

The detector budget remains 20 epochs, batch 4, seed 42 and 134,320 optimizer
steps. There are 158,151 matched application opportunities, but identity fallback
can reduce actual edits. Ten updates add 71,680 backbone image forwards plus
auxiliary ROI work. The strength grid expands around the fixed-local level, so
the comparison tests the complete online policy, not online updating isolated
from an optimized fixed equal-grid policy. Existing one-seed and warmup-divergence
limitations remain. The README specifies the evidence review rule and the
requirement to keep these probes/settings matched in the later class-set ablation.

Local validation passed 63 tests (including policy/gate/object-loss and interrupted
probe/mid-epoch resume checks), an actual pretrained-model synthetic CPU probe
sweep with 112 views, exact prediction/policy checkpoint replay, immutable-core
comparison, Python 3.12 grammar and Bash read-only syntax checks. Local libraries
differ from CCAST. The PBS job repeats tests and a real-data GPU smoke before
fresh full training. The small smoke panel may legitimately choose identity;
production retention/support thresholds are not lowered to force augmentation.

Archive: `fasterrcnn_online_shared_v1.tar.gz` (80,667 bytes; 30 files).
SHA-256: `fa427b760cc53514cac9b7de5d5ef63609d47810845585bb2cc5584a5b6f5a89`.

Evidence: `reports/fasterrcnn_online_shared_v1/release.json`,
`comparator_checks.json`, `panel_plan.json`, `unit_tests.json`,
`actual_model_check.json`, and `shell_checks.json`. Complete protocol, CCAST
execution and restart details are in `fasterrcnn_online_shared_v1/README.md`.
Full output: `runs/fasterrcnn_online_shared_seed42_v1`; smoke is separate.


## Online shared-local CCAST start observed

User-supplied qstat/log output for PBS `796146.bright04` shows state `R` in the
GPU queue. All 63 focused tests passed in 4.584 seconds. The eight local
operation checks and the actual GPU probe/train/validation/policy-checkpoint
gate passed. The smoke completed 420 probe image forwards over 15 positive
training images and selected identity because reference detection support was
insufficient. It had detected-object counts D00=0, D10=1, D20=0, D40=0; support
thresholds were not changed. Its zero AP is a tiny implementation check, not a
full experiment result. The earlier repeated AP=0.520833 lines are synthetic
unit-test output.

The fresh full run subsequently reached the last reported epoch 1/20, step 400.
No full-run validation or controller-update result is present yet. Epoch 1 is
the declared unaugmented warmup; the first full-panel update is before epoch 2.
The fork/cuDNN warnings did not prevent the checks passing or training starting
in this snapshot. Keep the submitted job running; no restart or protocol change
is indicated by the supplied output.

Terminal evidence is saved in `reports/fasterrcnn_online_shared_v1/ccast_start_796146.txt`
and summarized in `ccast_start_796146.json`. The reported smoke archive is
`fasterrcnn_online_shared_smoke_v1_records_e70b5792a173.tar.gz`, SHA-256
`50afb76c53abad9d309176ff411cfcfc5b7bf710d9033c66de147774f008e858`.
It has not been downloaded or independently verified. Frozen release records,
package contents and earlier completed controls are unchanged.


## Online shared-local completion reported (2026-10-26)

The user supplied `FASTER R-CNN FULL: COMPLETE` after all 20 epochs. The terminal
reports best epoch 14. Final epoch-20 validation AP50:95 is 0.225878 and AP50 is
0.503499 (rounded console values); these are not the selected-checkpoint metrics.
No efficacy comparison is established until the returned records are verified.

At epoch 20, 8,378 augmentation applications and 8,299 pixel-changing edits were
reported. Preserve this difference of 79 for the exposure/operation audit; the
terminal excerpt alone does not establish its cause. Full policy histories,
retention/support checks, selected epoch-14 metrics and runtime remain unaudited.

Records archive: `fasterrcnn_online_shared_seed42_v1_records_437f03a837f3.tar.gz`.
Reported SHA-256: `b30df4e003be14cf8a53059341451c5101261d98d32e58cdc94866c793f0a231`.

The archive is not yet present locally. Next: download with checksum
verification, inspect the controller evidence and recompute selected/final
metrics, then compare identity, image-level, fixed-local and online-shared runs.
No class-adaptive run is launched by this completion update. Completion evidence
is saved in `reports/fasterrcnn_online_shared_v1/ccast_completion_reported.json`;
released packages and preparation-time records are unchanged.


## Online shared-local audit completed (2026-10-26)

The preceding completion snapshot is superseded by the verified returned-archive
audit in `reports/fasterrcnn_online_shared_v1/ccast_verification.json`. SHA-256
matches `b30df4e003be14cf8a53059341451c5101261d98d32e58cdc94866c793f0a231`.
All 119 files, 118 declared checksums and 30 released source/package files pass.
All 20 prediction files pass geometry/coverage/count checks. Independent COCO
recomputation at selected epoch 14 and final epoch 20 matches exactly. The run
completed 134,320 optimizer steps and 537,260 image presentations without resume.

Scores below use the 0–100 scale and each run's declared maximum-validation
AP50:95 checkpoint selection. Identity: 22.823 AP50:95 / 51.142 AP50 (epoch 14);
fixed image-level: 22.810 / 50.687 (14); fixed local: 22.555 / 50.160 (18);
online shared local: **23.059 / 51.033 (14)**. Online selected AP50:95 improves by
0.236 / 0.249 / 0.504 points over these three controls. Its AP50 remains slightly
below identity. False alarms on images without retained targets increase at
confidence 0.25: 886/2,193 images online versus 858 identity and 817 fixed local.

All ten controller policies reproduce from saved per-object observations, and
all 20 epochs of application/order/group/exposure records replay exactly.
Every update has sufficient class support; the full run has zero controller
identity fallback. Blur/noise always uses mild level 1. Photometric strength
varies; when both families are eligible, photometric probability stays between
48.67% and 51.51%. Online and fixed-local masks/application opportunities match
exactly. Of 158,151 applications, 156,819 report changed pixels; 1,332 unchanged
applications are distinct from identity fallback. Mild blur is the only probe
operation/strength with unchanged pixel observations, but aggregate training
records do not establish the cause of every unchanged training application.

The run lasts 14.14 hours; recorded probes total 56.41 minutes and 71,680 image
forwards. Pooled class feedback differs: at epochs 12 and 14, pothole retention
alone blocks strong photometric edits that pass for the other three classes.
This motivates a class-conditioning hypothesis but does not demonstrate its
downstream value. These pooled measurements are not class-set-specific policies.

Verdict: the online-shared condition leads observed AP50:95 in this seed, but
reliable or specifically online gains are unestablished. The four trajectories
already differ during unaugmented warmup, there is one run per condition, and
validation is used for checkpoint selection. A matched fixed mild-strength
control is missing. The test set remains untouched. Raw image/checkpoint tensors
and probe detection boxes/ROI logits were not independently re-evaluated.

Full findings, limits and next-step recommendation:
`reports/fasterrcnn_online_shared_v1/RESULTS.md`. Structured comparison:
`comparison.json`; controller evidence: `controller_verification.json` in that
directory. PNG/PDF comparison and policy plots accompany the CSV tables.

Stage 4 is complete for this seed. Stage 5 remains the planned class-adaptive
local comparison with the same grid, probe budget rules, gates and training
settings. Fixed mild and uniform-family ablations address mechanism attribution;
repeated seeds address reliability. No new experiment was launched by this
audit, and frozen releases and earlier result archives remain unchanged.


## Online class-adaptive package prepared (2026-10-26)

The user authorized stage 5 and grouped CCAST commands after the stage-4 audit.
The separate `fasterrcnn_class_adaptive_v1` release implements the paper's exact
group-class-set feedback: up to 15 policies, observations pooled within each
exact set and then by represented class. Mixed groups have one union edit, with
every represented class and operation constrained by the same retention rule.
Unobserved/unsupported/no-admissible sets use identity; they do not borrow a
shared policy, retain stale eligibility, resample a group, or receive extra probes.

The matched comparator is `fasterrcnn_online_shared_seed42_v1`, whose archive was
verified before preparation. The new run starts fresh from the same COCO weights,
not the selected stage-4 checkpoint. Numeric settings, 20-epoch detector budget,
operation/strength grid, object observer, panel/group/noise selection and keyed
training draws match online shared. Only class-set pooling and policy routing
change. Parameter, buffer, gradient and RNG preservation and atomic policy/model
checkpoint adoption remain enforced. Per-set support, choices, fallback reasons
and changed-pixel counters are saved.

Local checks passed **81 tests**. An actual pretrained Faster R-CNN CPU check
used synthetic images, distinct constructed class policies for a training update,
all **112 paired probe image views**, exact ordinary prediction parity, unchanged
probe state/RNG and fresh-model checkpoint replay. This is implementation
evidence, not RDD performance. Python 3.12 grammar and all three shell scripts
passed syntax checks. Git Bash initially failed to start in the Windows sandbox;
a read-only `bash -n` retry outside it passed without executing/submitting scripts.
Local versions differ from CCAST; pinned-environment tests and the GPU smoke
are mandatory job gates before full training.

All ten prospective panels exactly match the shared run's planned images/groups.
Total probe budget remains **71,680 image forwards**, and fixed application
opportunities remain **158,151**. Training annotation metadata alone implies
at least **8,798 (5.56%)** identity opportunities from sparse class-set support,
even if every annotated probe object were detected. Actual detection/retention
can create further fallback. These are support upper bounds, not a prediction
of active policies or efficacy. Full-image training data and target coverage
are unchanged; actual edit exposure is allowed to differ and will be audited.

Archive: `fasterrcnn_class_adaptive_v1.tar.gz`, **93,081 bytes**, 34 files and
33 checksum entries. SHA-256:
`3e4abe51cbee1fb3d609fd3626e9ea96bbc402aa61c0e59dfe692ac6df48b661`.
Protocol SHA-256:
`651ebda915cf3a9d97eddf624ccebda423cefb4b8a7a472dc6e01971db494f1d`.

Full output: `runs/fasterrcnn_class_adaptive_seed42_v1`; smoke is separate.
`bash submit.sh start` requests one A10, four cores, 32 GB RAM and 24 hours,
activates the existing environment, verifies sources, runs tests and a real-data
GPU smoke, then starts full training in a fresh process. A failed gate stops
the job. Existing outputs are preserved; resume is explicit after inspecting
status and confirming the prior job has ended. No automatic resubmission occurs.

Evidence and commands: `reports/fasterrcnn_class_adaptive_v1/release.json`,
`comparator_checks.json`, `panel_plan.json`, `actual_model_check.json`,
`unit_tests.json/.txt`, `shell_checks.json`, `CCAST_COMMANDS.md`, and the package's
`README.md` and `LOCAL_VALIDATION.json`. CCAST submission, target-environment
tests, GPU smoke and full scientific results are **pending**. The primary
development comparison is selected validation AP50:95 against online shared;
the single-seed and warmup-divergence limitations remain. No test data were read.
