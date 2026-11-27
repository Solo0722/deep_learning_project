# RDD v2 preflight: implementation and retrospective feasibility

Date: 2026-10-28. Status: local numerical checks complete; real-image CCAST
rendering and visual review pending. No new detector was trained or evaluated.

**Verdict:** proceed with the bounded real-image geometry check. Do not launch a
full v2 comparison yet. Pooling largely removes the exact-class-set support
bottleneck, but does not guarantee sufficient pothole support, stable gates,
useful class differentiation, or better AP.

## Scope and provenance

Package: `rdd_v2_preflight_v1/`. Its `protocol.json` was written before running
the analyses below. Existing v1 outcomes informed that protocol, so this is a
retrospective feasibility analysis, not a prospective efficacy experiment.

The analysis reads the frozen 26,863-image training manifest and 20 archived
probe rounds: ten from each v1 online trajectory. Both archive hashes and each
compressed observation member were verified against their prior audits. The
archived observer configuration hashes and confidence/IoU/noise settings were
also checked. No validation/test manifests or raw held-out pixels were used.

Authoritative replay output: `replay_verified/summary.json`, `summary.csv` and
the 20 detailed epoch files. `source_protocol.json` preserves the exact replay
configuration. The earlier `replay/` results are retained; adding explicit
observer-settings provenance checks did not change any round outcome.

The immutable v1 reader is vendored unchanged. The new controller implements
class pooling, constituent-class intersections, supported mixed-set vetoes and
uniform eligible family -> operation -> level sampling. It does not optimize
loss or automatically select the largest passing strength. Support requires
16 reference-detected objects in eight source clusters; retention is at least
0.8. These cutoffs are engineering settings, not statistical confidence bounds.

## Local verification

The focused suite covers class/cluster support, reference misses, incomplete or
misidentified observations, noise-draw accounting, nonmonotone strength gates,
mixed vetoes, class/shared routing, keyed RNG isolation, explicit identity,
horizontal-flip pixel/box alignment, all four classes, fractional boxes, empty
images, non-target annotations, rejection fallback, input preservation,
known scale/translation coordinates, inverse maps, raw-image hashes, archive
creation, failure records and refusal to overwrite prior results.

Final test count and code hashes are in the packaged `LOCAL_VALIDATION.json`.
Local Python 3.12.14, NumPy 2.3.5 and Pillow 12.3.0 differ from CCAST; therefore
the remote job reruns the focused checks and requires NumPy 1.26.4/Pillow 10.3.0.
Each of the three shell scripts passed a separate Bash syntax check. A missing
`source` field in a newly added synthetic test fixture was fixed before the
final suite; no historical input or experiment result was changed.

## Controller findings

All counts below are **probe-group appearances across saved snapshots**, not
independent experimental samples, training exposure or predicted accuracy.

| Measurement | Result |
|---|---:|
| Panel groups examined | 5,120 |
| Groups with at least one admissible class-routed action | 5,034 / 5,120 (98.32%) |
| Mixed-class groups with an action | 607 / 632 (96.04%) |
| Active groups borrowing evidence because exact-set support was incomplete | 598 / 5,034 (11.88%) |
| Rounds supporting all four pooled classes | 17 / 20 |
| Round/class/action cells admitted | 1,700 |
| Admitted cells that change after at least one single-cluster deletion | 37 / 1,700 (2.18%) |
| Class/round pairs with both disjoint halves sufficiently supported | 65 / 80 |
| Mean action agreement on those supported half pairs | 95.96% |
| Pothole rounds with both halves sufficiently supported | 5 / 20 |

The three missing pooled-support cases are all D40: 12 and 13 detected objects
in the two epoch-2 records, and 15 in the shared trajectory's epoch-8 record.
On the class-adaptive epoch-8 record, D40 has exactly 16 detected objects in 16
clusters: removing one cluster makes 22 admitted actions unsupported. Aggregate
stability percentages would hide that important boundary case.

Class policies are not always distinct. At class-adaptive snapshots 10, 18 and
20, every class admits all 24 actions. The class-adaptive and shared policies
then coincide with the fixed balanced distribution over this action bank. Do
not manufacture differences merely to preserve the method's name.

Across all rounds, 3,572/5,120 routed group distributions differ from the shared
distribution, with mean total-variation distance 0.2059. However, this includes
three rounds where the shared all-four-class gate is forced to identity by
missing D40 support. Restricting to the 17 fully supported rounds, the distance
is only 0.0855 and 2,890/4,352 group distributions differ. A difference can be
small; neither statistic measures learning utility.

Supported mixed-set measurements veto four otherwise admitted set/action
cells in three snapshots, all for D00+D10. Many mixed cases still borrow
marginal evidence. Pooling fixes data fragmentation, not the logical gap
between class-average tolerance and safety of every mixture/context.

Cluster bootstrap frequencies, disjoint halves and source withholding are
saved per class/action. They are sensitivity diagnostics, not calibrated
confidence guarantees. Objects in a common image/cluster are kept together.
The screened source clusters themselves do not prove complete geographic or
capture-sequence independence. Different epochs/models are not independent
repeated panels at one frozen detector.

## Geometry findings

The bank contains 54 combinations: horizontal reflection on/off, square canvas
scale 0.95/1/1.05 and x/y translations -16/0/16 at 640 resolution. All annotations
move together. A horizontal flip maps x edges to `[W-x2, W-x1]`; it does not turn
a longitudinal crack into a transverse one relative to the reflected road.
No rotations, local geometric warps or class-label swaps are implemented.

All 26,863 training records were inspected against the full bank:

- 1,450,602 candidate/image combinations examined.
- 1,174,644 accepted (80.98%); 275,958 rejected (19.02%) because a known
  annotation would leave the canvas.
- No image disabled by invalid non-target annotation geometry.
- Zero positive-area or inverse-coordinate failures; maximum inverse error
  was 1.71e-13 prepared-image pixels.
- No clipping/deletion of annotations is used to make a candidate pass.

| Collection | Rejected candidate fraction |
|---|---:|
| China Drone | 32.12% |
| China MotorBike | 46.64% |
| Czech | 1.94% |
| India | 10.66% |
| Japan | 29.12% |
| Norway | 3.78% |
| United States | 28.40% |

These are bank-enumeration frequencies, not realized epoch exposure. Rejecting
to identity without resampling produces different actual geometry exposure by
source/class. Every comparison arm must use the same geometry draws and guard,
and report actual rejection/application counts. Do not silently resample until
success, which would change the declared distribution.

The annotation census alone cannot establish that the images look sensible.
CCAST must now verify the original image/XML/RGB hashes, render every candidate
on 56 selected training images, and export the representative overlays. Empty
target lists are not certified healthy road regions. Known-box preservation
does not establish annotation completeness or downstream learning benefit.

## Decision before training

1. Run and review the prepared CPU geometry job. Inspect preserved labels,
   padding, road orientation, small cracks and rejected-candidate fallbacks.
2. Replace unstratified fresh probe selection with an explicitly class-balanced,
   detection-independent allocation. A concrete next-stage candidate is two
   disjoint 256-image panels, each with 64 primary group selections per class,
   avoiding repeated source clusters. Report source composition and incidental
   classes; a class quota still cannot guarantee 16 correct reference detections.
   This recommendation is made **after** seeing this replay and requires a
   separately frozen fresh-probe plan. It has not been executed.
3. Use the same frozen detector for both fresh panels and both controller rules.
   Test actual panel-to-panel support/action stability. If decisions collapse
   to a shared policy, accept that outcome rather than force class differences.
4. Before a training pilot, check the final local renderer and any fixed-ROI
   diagnostic on fresh observations. This replay validates neither a new mask
   nor geometric/copy-paste counterfactuals.
5. Then prepare the bounded four-arm pilot from one common warmup checkpoint,
   with identical geometry, image order and nominal appearance opportunities.
   Freeze its duration/cost before viewing AP. Early AP is not a final winner.

Copy-paste is disabled. No full or multi-seed job is authorized by completion of
this package alone. The present result supports continued controlled testing,
not a claim that version 2 is better than version 1.
