# CCAST geometry preflight audit

Date: 2026-10-28. PBS job: `797236.bright04`.

**Verdict: the geometry implementation passes the returned-record audit and the
available visual inspection. Proceed to a bounded GPU controller diagnostic.
The evidence does not yet justify a full version-2 training comparison.**

## Verified evidence

The downloaded archive has SHA-256
`a6465d65ec2a0756b7574678184b8b088fcd73eff880b5b8796194258c20413f`
and contains 26,299,116 bytes. All 59 files covered by its internal checksum
manifest matched. Archive members were checked for duplicate/unsafe paths,
links and unexpected file types before extraction into `returned_records/`.
The original archive and extracted evidence were preserved unchanged.

The returned protocol hash, five Python-source hashes, frozen training-manifest
hash and target NumPy/Pillow versions match the submitted package. The user-
supplied CCAST log reports 24 passing tests and successful job completion.
The returned records establish that the geometry run used Python 3.12.3,
NumPy 1.26.4 and Pillow 10.3.0. Its timed checking phase took 103.49 seconds;
that excludes some job setup and is not a total training-time estimate.

An independent audit uses homogeneous corner-coordinate matrices rather than
calling the submitted forward-box implementation. It reproduces the full
training-manifest acceptance counts and verifies the returned sample boxes.

| Check | Audited result |
|---|---:|
| Training images in annotation census | 26,863 |
| Candidates per image | 54 |
| Image/candidate combinations | 1,450,602 |
| Accepted combinations | 1,174,644 (80.98%) |
| Rejected because a known annotation would leave the canvas | 275,958 (19.02%) |
| Invalid-annotation images disabling geometry | 0 |
| Real training images rendered | 56, eight per source collection |
| Rendered candidates | 3,024 |
| Accepted rendered candidates | 2,226 |
| Rendered candidates returning exact identity | 798 |
| Returned annotation instances independently checked | 6,210 |
| Non-target annotation instances included in that check | 918 |
| Maximum independent box-coordinate disagreement | 1.14e-13 pixels |

Every returned object ID, class label, difficult/truncated flag and original
annotation record is preserved. There are no added/dropped annotation records
or positive-area failures. For rejected candidates, saved pixel hashes and
annotations match the identity view, and changed-pixel counts are zero.

The 56-image subset contains 98 target objects: D00=48, D10=19, D20=14 and
D40=17. Thirteen images have no target boxes; some retain non-target road-feature
annotations. The subset is for coverage and inspection, not an accuracy sample.

## Visual inspection

All 56 returned PNG sheets were directly inspected, covering 336 display panels
(original plus five representative candidate views per image). No obvious
image/box misalignment, unintended geometric distortion or accepted-view
cropping of a displayed annotated region was found at the saved resolution.

Observed behavior:

- Horizontal reflections move the entire scene and all boxes together.
  Longitudinal and transverse class labels stay unchanged. A transverse crack
  remains transverse relative to the reflected road.
- Modest isotropic resize and translation move boxes with their visible road
  features. Overlapping annotations keep their relative placement.
- Boundary-touching target and non-target annotations trigger identity
  fallback when a proposal would crop them. This includes Repair, D43, D44,
  D50 and D60 annotations, not only the four detector classes.
- Gray margins agree with the declared fixed-canvas padding. The large bands
  in wide Norway images come from the existing letterbox preparation.
- Reflected lettering, road handedness and watermarks are expected effects of
  a whole-scene horizontal flip. They are not box-coordinate defects and do not
  by themselves establish that flips help detection.

Representative audited sheets:

- [D00/D10 orientation and alignment](returned_records/overlays/01_China_Drone_002195.png)
- [Boundary repair annotations and rejected crops](returned_records/overlays/06_China_Drone_001882.png)
- [Longitudinal and transverse cracks under reflection](returned_records/overlays/10_China_MotorBike_000200.png)
- [Overlapping damage boxes](returned_records/overlays/25_India_006613.png)
- [Wide-view images and small annotations](returned_records/overlays/44_Norway_004153.png)

The review is bounded: display tiles are grayscale and 320 by 320 pixels, with
annotation labels sometimes covering very thin defects. Only five candidate
views plus the original are displayed per image, although all 54 have numerical
records. Original RGB arrays remain on CCAST, so local audit cannot independently
rerender every pixel variant or certify all subtle crack detail. No original
semantic labels were relabeled or independently certified by this review.

## Issues to retain in the next check

**Realized geometry exposure differs by source.** Candidate rejection ranges
from 1.94% in Czech to 46.64% in China MotorBike. This is caused by annotation
locations and the deliberate crop guard. It is not a failure of that guard.
All future arms must share the same keyed geometry draws and rejection rule,
and report actual exposure. Do not resample until an edit succeeds or drop
edge-touching images merely to equalize application rates.

**One very thin target merits native-resolution inspection.** Object 12 (D10)
in `Norway/train/images/Norway_005967.jpg` has prepared dimensions approximately
10.708 by 1.049 pixels. Scaling by 0.95 gives 10.173 by 0.997 pixels. Eighteen
accepted candidates involve this crossing; they are 18 presentations of one
object, not 18 damaged annotations. Coordinates stay valid and positive-area.
This image was outside the 56-image subset. A one-pixel boundary is not a
semantic cutoff, so this is not evidence that the object disappeared. Include
it in the next check, export full-resolution prepared RGB views and inspect the
thin crack. Do not silently filter or relabel the target. The exact image/box
and hashes are saved in `small_target_follow_up.json`.

**The controller still needs fresh evidence.** This CPU job did not run a
detector, test the new class-pooled policy on fresh panels or measure AP. The
earlier replay found D40 support shortfalls and some snapshots where all class
policies coincide. Passing geometry checks does not resolve either issue.

## Next step and decision boundary

Prepare a bounded GPU diagnostic using one frozen detector and two disjoint,
training-only, class-balanced probe panels. The previously proposed allocation
is 256 images per panel with 64 primary group selections per class, avoiding
reused source clusters and never selecting by detector success. Freeze and log
the actual allocation and compute budget before inference. A quota does not
guarantee adequate correctly detected potholes.

Apply the shared and class-adaptive decision rules to the same observations.
Check support for every class, action/distribution agreement between panels,
mixed-group vetoes, fallbacks and whether class routing differs meaningfully
from the shared policy when both have sufficient support. Include the thin
Norway case as an explicitly separate renderer-inspection case if it is not
chosen by the frozen panel rule; do not quietly alter panel statistics.

After that diagnostic, a matched short training pilot can test behavior,
checkpoint recovery, convergence and cost. Full 20-epoch comparisons remain
premature until those checks pass. Copy-paste remains disabled. The new blending
renderer and fixed-ROI auxiliary diagnostic remain unverified and must not be
described as tested by this geometry job.

## Reproduction and saved outputs

Run from the project root:

```text
python reports/rdd_v2_geometry_check_v1/audit_returned.py
```

The audit requires NumPy/Pillow and the local downloaded archive plus frozen
training manifest. It reads no validation/test manifest, trains no detector and
uses no GPU. Existing extracted bytes/audit results must agree or it stops.

- `numerical_audit.json`: verified hashes, independent coordinate/census checks,
  sample coverage, limits and audit-script hash.
- `visual_review.json`: explicit inspection coverage and per-source notes.
- `image_inventory.json`: image IDs, class/annotation coverage and rejections.
- `small_target_follow_up.json`: the single thin-target case for follow-up.
- `returned_records/`: unchanged result records and overlay sheets.
