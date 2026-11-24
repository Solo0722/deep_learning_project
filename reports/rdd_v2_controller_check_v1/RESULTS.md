# RDD v2 controller diagnostic audit

**Verdict: proceed to preparation of a short, controlled four-arm training pilot.**
The diagnostic has passed the implementation and basic-support checks. It has
not established robust policy invariance, semantic safety of every edit, or an
accuracy improvement. The remaining central question is whether the revised
policy helps learning under matched training conditions. That requires a pilot.
The evidence does not justify skipping directly to full 20-epoch comparisons.

## Evidence and audit scope

PBS job `797244.bright04` ran on one NVIDIA A10 on `gpu0025`. The model was the
pinned identity baseline epoch-1 checkpoint, not its validation-selected best
checkpoint. The two 256-image panels contain 512 distinct source clusters and
64 primary anchor-group selections per class in each panel. All images belong
to the frozen training split. The first panel contains 409 selected-group
objects and the second 392; primary-group quotas are not object or detection
quotas.

Returned archive: `rdd_v2_controller_check_v1_results.tar.gz`.
SHA-256: `333154238321e50b2b32dc94710a55ffe0480d53096cded9b57114da14d3c2ae`.

The audit verified all 531 internal checksums, all 16 package-file hashes, the
training-manifest hash, and the exact frozen panel plan. Independent matching
recomputed every saved observation: 14,848 views containing 1,467,058 saved
predictions. All detection indicators, matched IoUs and false-positive counts
agreed. Independent homogeneous-coordinate mapping checked all 1,709 target
and non-target annotation instances in the panel images. The maximum coordinate
disagreement was 1.14e-13 pixels. No annotations were lost or labels changed.

Independent class/direct-set gates, mixed-set vetoes, routed action sets and
probability distributions matched the saved policies. Re-executing the analysis
also reproduced every saved summary and sensitivity field. This latter replay
uses the shipped sensitivity implementation; it is distinguished from the
independent matching, coordinate and policy calculations.

The job recorded unchanged model, gradient and global-RNG hashes, and zero
difference in its repeated inference check. The final model hash equals the
loaded checkpoint's model hash. These are recorded runtime invariants, not a
new local GPU execution. The four additional parity image forwards bring the
total to 14,852. The main program took 744.44 seconds (12.41 minutes), including
371.48 seconds of timed detector forwards. Queue time and wrapper tests are not
included in that elapsed measure. The full GPU checkpoint and 512 raw source
images remain on CCAST. The archive does not contain unit-test stdout, although
the verified job wrapper runs tests with fail-on-error before this diagnostic.

## Class support and decision differences

The unchanged requirement is at least 16 reference-detected objects from eight
distinct clusters, with per-action retention at least 0.8. All eight full-panel
class cells meet the support requirement.

| Class | Detected objects A / B | Detected clusters A / B | Allowed actions A / B | Same allow/reject decisions |
|---|---:|---:|---:|---:|
| D00 longitudinal cracks | 64 / 69 | 57 / 62 | 14 / 17 | 21/24 (87.5%) |
| D10 transverse cracks | 59 / 55 | 55 / 53 | 8 / 11 | 21/24 (87.5%) |
| D20 alligator cracks | 69 / 63 | 67 / 63 | 21 / 23 | 22/24 (91.7%) |
| D40 potholes | 41 / 27 | 34 / 25 | 16 / 12 | 20/24 (83.3%) |

Overall, 84/96 class-action decisions agree. This is a descriptive comparison
of two selected panels at one detector snapshot, not a calibrated confidence
level or proof of convergence. The class sampling distributions differ between
panels by total-variation distances 0.111, 0.233, 0.153 and 0.167, respectively.
The shared policy changes from seven to ten actions, with distribution TV 0.25.

Some switches are near the threshold: three D00 actions have retention 0.796875
in panel A, just below 0.8. Other differences are substantive. D40 contrast
reduction at level 2 retains 34/41 objects (82.9%) in A but 17/27 (63.0%) in B.
The controller cannot claim every admitted action reliably transfers to another
panel. The threshold should not be relaxed to hide these differences.

An exploratory cross-panel calculation applies one panel's class-marginal
action distribution to the other's already recorded outcomes. Its mean
reference-conditioned retention ranges from 89.1% to 97.6% across the eight
class/direction combinations. This average does not repair the individual
action failure above, certify mixed groups, or measure training benefit. It is
an additional descriptive use of the same diagnostic records.

## Does class adaptation remain distinct from a shared policy?

Yes, for this snapshot and the prespecified shared comparator. Class-specific
routing differs from the shared distribution on 256/256 selected groups in A
and 250/256 in B. The group-weighted TV distance to shared is 0.319 and 0.214.
The class-adaptive rule has therefore not collapsed to a shared policy here.
The difference partly reflects the deliberately conservative global
intersection used by the shared comparator; it is not evidence that the more
permissive class-adaptive rule improves AP.

The new strength rule is working as specified: it samples eligible families,
operations and independently tested levels, without automatically selecting
the strongest level or maximizing auxiliary loss. Noise draws are averaged
within each object and do not inflate support.

## Remaining support and sensitivity weaknesses

Only two of the 122 admitted pooled class-action cells lose admission under
some single-cluster deletion: D40 brightness reduction level 2 in A and D10
gamma brightening level 1 in B. Other rejected cells can become admitted after
a deletion. This shows that admitted actions are mostly resistant to that
small perturbation; it does not establish source or panel invariance.

D40 support remains the weak point. Panel A's deterministic cluster halves have
26 and 15 detected potholes; B's halves have 15 and 12. Three of the four halves
therefore fail the unchanged 16-object gate. Removing Japan from B leaves only
10 detected potholes and disables that class. Retaining the full panel size is
necessary; these results do not support halving the probe budget. Class and
source effects are not disentangled by these two panels.

Exact class-set support is still sparse:

- A has 80 selected mixed-class groups. Of these, 31 use borrowed pooled-class
  evidence; 49 belong to directly supported D00+D10 or D20+D40 sets.
- B has 74 mixed-class groups, all using borrowed pooled-class evidence.
- Both panels have only 11 reference-detected objects in D40-only groups.
  Their singleton route therefore also depends on pooling D40 across mixtures.
- The supported D20+D40 set in A vetoes two otherwise available actions. B has
  no effective mixed-set veto. There is no basis to claim mixed-set safety has
  been solved simply because pooled class support passes.

These are disclosed limits of the proposed marginal-pooling method. Do not
silently replace them with a claim of direct evidence for every class mixture.
All policies remain nonempty on the observed groups in these full panels.

## Geometry-related detection changes

Known annotations are preserved, but the frozen detector is not completely
invariant to the geometry. These are selected-group object counts at confidence
0.25 and IoU 0.5, not AP or population recall.

| Class | Selected objects | Clean detections | Geometry detections | Clean detections lost | New detections found |
|---|---:|---:|---:|---:|---:|
| D00 | 222 | 135 | 133 | 11 | 9 |
| D10 | 184 | 119 | 114 | 10 | 5 |
| D20 | 209 | 135 | 132 | 5 | 2 |
| D40 | 186 | 76 | 68 | 11 | 3 |

Panel B is the strongest caution: pothole matches drop from 34 to 27, with seven
losses and no gains. Across both panels, 10 of the 11 lost potholes still have a
saved D40 prediction with IoU at least 0.5 but confidence below 0.25. The other
case has a confident D40 prediction at IoU about 0.486. This identifies the
immediate matching failures in the saved predictions; it does not prove why
the detector changed or whether training on these views will help.

Geometry rejects 141/512 candidates (27.5%) to exact identity geometry, without
resampling, because a known annotation would be cropped. The rejected draw
does not cancel the appearance edit. Twelve additional accepted candidates are
the identity transform itself; 153 clean/reference pixel hashes are identical.
The rate differs by source, so all pilot arms must use the same keyed geometry
draws and rejection rule. Monitor realized geometry exposure and class metrics.

The four planned arms all include this same geometry and can compare appearance
policies conditional on it. They will not independently establish that geometry
is better than identity training. That would require a contemporaneous matched
identity arm; the old 20-epoch v1 run is not a substitute for that causal claim.

## Image review

All eight full-resolution sheets were inspected: 32 displayed views spanning
the four anchor classes in both panels. All five additional thin-case PNGs were
also inspected, including the unannotated RGB images. No visible box displacement
or unexplained edit outside the selected region was found. A pixel comparison
of the 24 displayed appearance tiles found no differences outside the selected
mask transformed with geometry, allowing a one-pixel interpolation halo and
excluding the intentional title bars. Overlays can hide pixels, so this is not
an exhaustive audit of the unavailable raw candidate images.

The strongest darkening and noise examples have visible rectangular boundaries.
Strong blur weakens fine crack texture. The current one-pixel boundary taper
does not remove those seams. This is an existing design limitation and a
plausible shortcut for a detector, not an observed cause of an AP change. The
renderer edits entire annotated box groups, including background and occasional
occluding objects; it is not a segmentation of damaged pavement pixels.

The sheets display strong candidates regardless of admission. All eight shown
strong-blur cases and seven of eight strong-darkening cases are rejected by
their panel's routed policy. Some noise and intensity edits remain admitted,
so rejection does not eliminate the mask-realism limitation. A fixed control
using the same renderer is necessary for interpreting the pilot.

One existing D40 annotation in `China_MotorBike_001766.jpg` resembles elongated
crack-like damage. This is a label-quality flag, not a verified relabeling. The
source label was preserved and no image was removed. Do not infer that passing
the numerical audit certifies the semantic correctness of every RDD annotation.

The thin Norway D10 target is 1.049 pixels high after ordinary 640 letterboxing
and 0.997 pixels after the additional 0.95 scale. Its positive-area box is
retained. Recomputing the saved scaled RGB from its prepared RGB gives exact
pixel agreement. Fine detail is already difficult to resolve at 640, and this
inspection does not certify its learnability. Preserve the frozen annotation;
do not add a post-hoc one-pixel filter.

Eighteen of 13,824 appearance renderings are pixel-identical to their reference;
all are Gaussian-blur draws. They remain recorded as attempted, unchanged
actions and are not replaced by another draw. They must not be counted as
effective changes in the pilot's exposure summaries.

## Pilot recommendation and constraints

Proceed with the existing four-arm design: geometry only, fixed local
appearance, online shared local appearance, and online class-adaptive local
appearance. The next implementation must freeze a short duration and resource
cap before inspecting pilot AP. This audit does not submit a job or specify an
untested training-time estimate from inference timing.

All arms must branch from the same warmup model, optimizer and training state,
with the same image order, base optimizer schedule, prepared resolution,
augmentation application opportunities and keyed geometry. Use the same
full-size class-balanced panel selection rule and scheduled panel identities
for both online arms; each arm must measure its own current detector after
training diverges. Do not pick A instead of B because its observed pothole
support is better, or merge panels after seeing their outcomes to rescue a gate.

Keep the measured controller thresholds, operator bank and renderer unchanged
for this first pilot. No copy-paste, new mask or fixed-ROI loss has been validated
here. Any subsequent renderer revision needs its own version and check. Log
class support, source composition, direct/borrowed decisions, identity fallbacks,
action strengths, unchanged-pixel draws and time. Unsupported updates must use
the declared fallback; they must not relax thresholds or reuse stale policies.

Check per-class clean-validation AP/recall as well as aggregate AP, particularly
D40 and small/thin damage. A short pilot is for learning behavior, measurement,
state recovery and cost; it is not a final winner-selection experiment. The
existing validation split is development data. Keep test images unopened until
the settled final evaluation. Require a review before extending pilot length or
starting full/multi-seed runs.

The numerical and visual evidence reveals no implementation blocker requiring
another identical frozen-detector job. The source sensitivity, borrowed support,
geometry difficulty and mask seams are real risks that a bounded, matched
training pilot should now evaluate. The appropriate conclusion is readiness to
test the revised method's learning behavior, not a prediction of better results.

## Reproduction

From the project root with the local NumPy/Pillow runtime:

```text
python reports/rdd_v2_controller_check_v1/audit_returned.py
python reports/rdd_v2_controller_check_v1/interpret_results.py
```

The scripts preserve the returned archive and refuse to replace extracted or
audit records that disagree. `numerical_audit.json` records the integrity and
independent calculations; `follow_up_analysis.json` contains the descriptive
transfer and pixel checks; `visual_review.json` records the inspection scope and
observations. Neither script performs new model inference or reads held-out
manifests. Original experiment packages, raw data, splits and checkpoints were
not modified.
