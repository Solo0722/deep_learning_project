# Object-local edits: is there a class-specific effect worth exploiting once edits stop being shared by the whole image?

Fixed before any L- run is trained. Thresholds below are not changed after the runs.

## Why

Every policy in the study so far applies one edit to the whole image, so all objects in an image get the same edit. Class-specific choice then has to trade classes off against each other inside each image. The measured-effect policy failed, and a cross-fitted check on the Part A table explains why (computed from `reports/cua_part_a_report/deltas.csv`, before this pilot):

**Cross-fitted per-class selection gain, whole-image edits** (short-run validation AP50). For each held-out seed, pick each class's best edit (or none) from the other two seeds, and separately pick one best edit for all classes. The gain is the macro AP50 of the per-class choice minus that of the global choice, on the held-out seed.

| Pair | Pixel edits (per fold) | Mean | All edits, mean |
|---|---|---|---|
| dacl10k F | +0.02, +0.12, +0.04 | **+0.06** | +0.09 |
| dacl10k R | −0.60, −0.96, −0.38 | −0.65 | −0.24 |
| RDD F | −0.25, +0.16, +0.08 | −0.00 | −0.00 |
| RDD R | +0.19, −0.27, −0.13 | −0.07 | −0.16 |
| VOC F | +0.24, +0.16, −0.24 | +0.05 | +0.02 |
| VOC R | +0.56, +0.57, +0.42 | +0.52 | −0.04 |

On dacl10k with Faster R-CNN, the pair with the strongest edit×class interaction, knowing each class's best whole-image edit is worth 0.06 AP50 on new seeds. The interaction is real, but it does not carry over to new seeds well enough to choose by. This pilot asks whether that changes when an edit touches only the objects.

## Design

- **Pair:** dacl10k, Faster R-CNN. It has the strongest interaction, the most multi-class images (75%), and polygon masks (`train_masks.json`, 32,391 polygons).
- **Edit scope:** an object-local edit renders the usual whole-image edit and copies only the pixels inside the polygons of annotated objects (1-px Gaussian feather), leaving the background unedited. Boxes are unchanged.
- **Edits:** the 8 pixel edits (brightness, contrast, saturation, hue, blur, noise, jpeg, sharpen). Geometric edits, erasing and mosaic have no local version.
- **Protocol:** identical to Part A. Branch from the existing warm checkpoints `A-dacl10k-fasterrcnn-s{0,1,2}-warm`, continue 3 epochs with p = 0.5, same data order, same edit-strength draws as the whole-image arm. The base arm is the existing `A-dacl10k-fasterrcnn-s{k}-base`. So each object-local arm is paired with both the base arm and its whole-image twin.
- **Runs:** `L-dacl10k-fasterrcnn-s{0,1,2}-<edit>`, 8 × 3 = **24 short runs**. No new warm or base runs.
- **Note on crosstalk:** all objects get the edit, so Δ(o, c) measures the effect on class c of editing every object, not only class-c objects. Overlapping objects of different classes share edited pixels. This is the cheapest design that removes background edits. A class-targeted design (edit only class c's objects) needs 96 arms per seed and is deferred to step 2.

## Pre-registered checks

| Check | Statistic | Go if |
|---|---|---|
| **L1 dependence** | Edit×class interaction F-test on object-local Δ (Part A model, seed×class term) | p < 0.05 |
| **L2 usable** | Cross-fitted per-class selection gain on object-local Δ, as in the table above, pixel edits + none | mean ≥ **+0.50** AP50 and > 0 in all three folds |
| L3 descriptive | Object-local minus whole-image Δ, per edit and class; size of the interaction component (residual SD of the (βγ) term) local vs whole-image | reported only |

**Go to step 2 only if L1 and L2 both pass.** The +0.50 bar is about 8× the whole-image value on this pair. Anything smaller is unlikely to survive the shrinkage from short runs to full training (ρ = 0.56) and reach the 0.74 AP50 minimum detectable effect of the full-training comparison.

**If L2 fails:** the negative result extends to object-local edits. One paragraph and the table go into the paper's discussion, and the class-aware direction is closed for this menu.

## Step 2 (only on go)

Full 30-epoch training on dacl10k, Faster R-CNN, seeds 0–2, scored once on test:

| Arm | What it does |
|---|---|
| uniform (existing `C-…-uniform`) | whole-image uniform edits |
| uniform-local | uniform edit choice per image, applied object-locally (class-blind control) |
| measured-local | each object gets its class's selected edit from the pilot (chosen on all 3 pilot seeds), applied inside its own polygon, p = 0.5 |

Primary comparison: measured-local − uniform-local, and measured-local − uniform. Both intervals (seed t, image bootstrap), Holm over the two. Claim a gain only if both corrected tests pass and the mean is ≥ 0.74 AP50. 6 new full runs.

## Code needed

1. `cua/augment/local.py`: `local_edit(sample, rng, edit, polygons)` renders `EDITS[edit]` on the whole image and composites it inside the rasterised polygons. It reuses the polygon rasteriser in `cua/augment/paste.py`.
2. `cua/train.py`: `edit_scope: str = "image"` ("image" | "object"), added to `_PLAN2_FIELDS` so existing digests are unchanged. It loads `train_masks.json` via `cua/datasets/dacl10k_masks.load_masks` when the scope is "object", and allows only pixel edits.
3. `cua/experiments/specs.py`: `local_specs()` and an `L-` resolver. Each L- spec equals its A- twin except `run_id` and `edit_scope`, pinned by a test like `test_control_specs_and_digests`.
4. `cua/ccast/make_jobs.py`: `--family local`.
5. `cua/analysis/local.py`: L1 to L3, reading A- base and whole-image arms and L- arms. It reuses the Part A interaction code.
6. Tests: the composite changes only in-mask pixels and is the identity with no objects; the same rng gives the same strength as the whole-image edit; L-spec digests; a fast trainer smoke test.

## Before submitting

In the same CCAST login as the submission, check that `A-dacl10k-fasterrcnn-s{0,1,2}-warm` checkpoints and the A- base histories still exist. If the warm checkpoints are gone, the warm runs (3) must be retrained first and the A- base arms re-run for exact pairing (3 more).

## Clarification after submission (no threshold changed)

A check of 41 training images on CCAST (before any L- run started) found 6 with no change under the object-local edit. All 6 have no annotated objects, so the object-local edit is the identity on them, by design, while the whole-image twin edits them. In that sample, 15% of images (6 of 41) are of this kind. L3 (object-local minus whole-image) therefore also includes the effect of not editing object-free images.
