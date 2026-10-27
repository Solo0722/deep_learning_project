# Online class-adaptive local augmentation: Faster R-CNN

Stage 5 follows identity, fixed image-level, fixed local and online shared local.
This release tests **exact overlap-group class-set conditioning** against the
completed `fasterrcnn_online_shared_seed42_v1` run. It starts fresh from the same
COCO weights and seed; it does not continue the selected stage-4 checkpoint.

## Question and evidence boundary

Does conditioning the common online policy on each group's complete class set
improve selected validation AP50:95 with the same detector and probe budget?
The online-shared comparator achieved 23.059 AP50:95 and 51.033 AP50 at epoch 14.
Those are one-seed development measurements, not a reliable or final-test gain.
The prior four runs already differ during unaugmented warmup. This fifth run
does not resolve that variation; it is one controlled development condition.

The paper defines feedback separately for each exact class set. Pooling all
potholes regardless of group context and taking a class-wise minimum would be
a different estimator. Here, {D00}, {D40}, and {D00,D40} each use only their own
probe-group observations. All 15 nonempty class sets have explicit policy states.

## The only experimental change

The probe panel, selected group, paired candidate pixels, noise seeds, observer,
strength grid, timing and numeric gates match online shared. The controller now
partitions that panel by the selected group's exact class set. Within each set:

1. For every represented class, require at least 8 reference-detected target
   objects from at least 4 source clusters. Repeated noise views do not add
   objects or clusters. Other groups and other class sets cannot supply support.
2. A family/strength passes only if every represented class and every operation
   in that family retains at least 0.8 of reference detections. Select its
   strongest passing tested level; do not assume monotonicity.
3. Compute mean object losses within each class before normalizing the net
   increase by reference mean + 0.1 and clipping to [0,5]. Average classes and
   operations equally. Use 0.8*softmax(score/0.25)+0.2/K over eligible families.
4. Use identity for unobserved sets, insufficient support or no admissible
   family. Record these as separate reasons. Rebuild every set at each update:
   no borrowing from a pooled shared policy, no stale-policy carry-forward,
   no extra probes and no resampling a group to force augmentation.

Training selects the same image application opportunity (q=0.5) and local group
as the controls, then routes to that group's policy. It uses the same keyed
family/operation/noise uniforms as online shared. Identical policy entries
produce identical edits. A mixed-class overlap group receives one union edit;
no per-box sequential transformation or majority-class shortcut is used.

The complete image and every valid annotation remain detector inputs. Empty
images remain in training. Validation is unaugmented; neither its metrics nor
test data enter the controller. The test manifest is never opened. Inference
uses the ordinary detector without the controller or ground-truth annotations.

## Matched settings and cost

`protocol.json` is the authoritative prospective configuration. It retains:

- One unaugmented epoch; updates before epochs 2,4,...,20. Each update uses the
  **same 256 training images/groups** as the stage-4 panel at the same epoch.
- Eight operations, three strength levels, two paired noise draws. Each panel
  uses 256*(1+3*(7+2)) = 7,168 image forwards; ten panels total 71,680, plus
  auxiliary ROI computations. Partitioning the results needs no extra forwards.
- Photometric factors/gammas: down 0.9/0.8/0.7, up 1.1/1.2/1.3; blur radius
  0.4/0.8/1.2 pixels; noise std 0.01/0.02/0.03 on RGB/255. Fixed local uses level 2.
- Ordinary class-correct one-to-one predictions at confidence 0.25 and IoU 0.5.
  Per-object auxiliary losses use positive RPN ROIs plus a self-assigned GT ROI,
  CE plus class-specific SmoothL1 beta=1/9. GT ROIs never enter detections.
- Eval/inference-only probes; parameters, buffers, gradients and global RNG
  remain unchanged. Noise fields are reused across strengths within each draw.
- 640-square letterbox, same mask/taper and random namespaces; seed 42; 26,863
  training and 5,760 validation images; 20 epochs, batch 4, 134,320 optimizer
  steps and 537,260 image presentations. Same SGD, precision and LR schedule.
- Unaugmented full validation each epoch; select maximum AP50:95, earliest tie.

The fixed budget creates a genuine support constraint: some rare combinations
have too few annotated objects even if every object were detected. Training
annotation support upper bounds and exact planned opportunities are recorded
in `LOCAL_VALIDATION.json`. Actual detection support can be lower. Identity
fallback is part of this declared method, not a reason to lower the gate after
seeing outcomes. Actual edit counts may therefore differ despite matched
application opportunities. Report both, and do not claim equal edit exposure.

## Implementation and checks

The frozen `shared_controller.py` is byte-identical to the completed comparator.
`class_set_feedback.py` retains its numeric formulas, restricted to one exact
class-set subset. `controller.py` builds the table, routes groups, validates
states and records per-set opportunities/fallbacks/families/strengths/pixels.
Detector/data/loss/metrics/checkpoints and the training/validation loop are
preserved. The shared protocol and comparator hashes are shipped and checked.

Focused tests cover independent set support, joint mixed-class gates, no
cross-set borrowing, outside-group exclusions, equal class weighting, noise
averaging, nonmonotone levels, keyed pixel parity, corruption, and exact resume
after interruption or a failed controller commit. The actual pretrained-model
CPU check exercises active routing using explicitly constructed feedback, all
probe views using real synthetic-image predictions, state preservation and
fresh-model checkpoint replay. These are implementation checks, not RDD results.

Local Python/library versions differ from CCAST. The batch job repeats the
tests in the pinned environment and performs a real-data GPU smoke before full
training. The smoke retains the production support/retention thresholds; its
small panel can legitimately select identity. Do not relax thresholds merely
to make the smoke apply an edit.

## CCAST execution and recovery

On a login node, enter the checksum-verified package and run `bash submit.sh start`.
The PBS job requests one A10, four CPU cores, 32 GB RAM and 24 hours. It activates
`pytorch` and `.venvs/rdd_detection_v1` itself. No interactive allocation or
manual environment activation is required for submission. SSH disconnection
does not stop the batch job.

The job verifies package checksums, runs all tests, runs a separate 24-image
training/validation smoke, checks its saved gate, then starts the full run in a
fresh process. Smoke: `runs/fasterrcnn_class_adaptive_smoke_v1`.
Full run: `runs/fasterrcnn_class_adaptive_seed42_v1`.
Live log: `runs/baseline_job_logs/<PBS_JOBID>.log`.

Start refuses an existing submission or output. Never delete an earlier run
to retry. If the time budget ends, inspect the saved status and wait for the
previous job to end, then explicitly use `bash submit.sh resume`. The full run
stops/saves at 81,000 seconds initially; a resume has an 82,800-second budget.
It still trains only the declared 20 epochs. There is no auto-resubmission.

Each complete update writes compressed observations and commits all class-set
policies with model, optimizer and RNG state. Mid-epoch resume does not re-probe.
Interrupted uncommitted probes restart from the committed detector; cost and
orphan evidence remain recorded. All policies recompute from saved observations.
Records archives include predictions, observations, histories, source and hashes;
full `.pt` checkpoints remain on CCAST. Raw probe boxes/logits are not archived,
so independent later reconstruction is limited to saved matching/loss observations.

## Review and continuation rule

Verify the returned archive, controller decisions, exact panel/budget matching,
selected/final COCO metrics and per-set support/exposure before interpreting
results. Primary comparison: selected validation AP50:95 versus online shared.
Report AP50, per-class AP, negative-image false alarms, final metrics, identity
reasons, unchanged edits and runtime as secondary diagnostics.

An observed gain is descriptive until replicated; it does not prove semantic
preservation, reliable class conditioning or the value of loss weighting. If
most policies lack support, report a support-limited test rather than claiming
active class adaptation failed or changing thresholds mid-run. If policies are
active but AP does not improve, report no observed gain under this protocol.
Fixed mild-strength and uniform-family ablations remain separate attribution
questions. No seed repetitions, test evaluation or other experiment is launched
by this package.
