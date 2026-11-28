# Positive control 2: can a per-class policy beat its class-blind twin when a per-class effect exists?

Fixed before any K2- run is trained. Read-out: `cua/analysis/control2.py`. Its thresholds are constants in
that file and are not changed after the runs.

## Why a second control

The first control (`reports/cua_control/`) planted an edit that only deletes the labels of one class. CUA-v2
found it and stopped applying it to that class's images (O1, O2 passed), but it could not show an accuracy
advantage for two reasons:
1. The planted edit only hurt. A class-blind policy loses nothing by dropping it everywhere, and shared-v2 did
   exactly that, so there was no per-class advantage to win.
2. At the uniform rate (1/15) the harm was too small to measure (O0 failed: Rust −0.29 test AP50, n.s.).

This control fixes both. The planted edit **helps most classes and hurts one**, so a class-blind policy must
either accept the harm or give up the help, while a per-class policy can keep the help and avoid the harm.

## Pair and planted edit

- **Pair: PASCAL VOC with RetinaNet.** Mosaic is the one edit with a large positive effect there (+1.82 AP50
  on average in Part A; shared-v2 and CUA-v2 apply it to about 55% of images in the C- runs and gain about
  +0.5 macro test AP50 over uniform).
- **Planted edit `plant_mosaic`** (`cua/augment/menu.py`): mosaic exactly as in the menu (same tiles, same
  centre, same random draws), except that the boxes of the target class c* are deleted **from the image's own
  tile** (the anchor) before the tiles are combined. The pixels are unchanged, so the anchor's c* objects are
  trained as background. c* objects in the three other tiles keep their labels.
  - On an image without c*, `plant_mosaic` is identical to mosaic.
  - On an image with c*, it is mosaic plus a harm to c*.
  - So the harm depends only on the classes of the image the edit is applied to, which is what a per-class
    policy conditions on. Deleting c* from all four tiles would add a harm no image-level policy can avoid
    except by dropping the edit, so it would not test the per-class mechanism.
- **Menu:** `plant_mosaic` **replaces** mosaic in its menu slot. If clean mosaic stayed in the menu, a
  class-blind policy could simply use it and there would be no dilemma. Because the slot is the same, the
  uniform arm draws exactly the same edit for every image and epoch as C-uniform.
- **Prior (v2 arms):** the probe prior row of mosaic is used for `plant_mosaic`. The prior is a measurement of
  mosaic, and on images without c* the planted edit is mosaic. It is also the realistic case: the prior says
  the edit helps every class, and only the online estimate can show that it now hurts c*. This makes the test
  harder for CUA-v2, not easier: for c* its online estimate must overturn a prior that favours the edit.
- **Target class:** c* = the class with the **second most** training boxes (`plant="second"`, ties go to the
  lowest index; the trainer records the resolved name as `plant_class` in each run's `config.json`). Not the
  most frequent class, person: it is in about 37% of training images, so a per-class policy that withholds
  mosaic from them gives up much of mosaic's gain, and the control could fail by design. The second class is
  in roughly a tenth of the images, often enough that most estimation batches of 16 images contain it.

## Runs

VOC, RetinaNet, seeds 0, 1, 2, full C- schedule; each arm is its C- run with `plant="second"` and
`plant_kind="mosaic"` and nothing else changed.

| Run | Policy | Paired reference |
|---|---|---|
| `K2-voc-retinanet-s{0,1,2}-uniform` | uniform | `C-voc-retinanet-s*-uniform` |
| `K2-voc-retinanet-s{0,1,2}-shared2` | shared-v2 | `C-voc-retinanet-s*-shared2` |
| `K2-voc-retinanet-s{0,1,2}-cua2` | CUA-v2 | `C-voc-retinanet-s*-cua2` |

9 runs (`python -m cua.ccast.make_jobs --family control2 --dataset voc --detector retinanet`). All are scored
once on test with `cua.test_eval --prefix K2-`. As in the first control, accuracy checks use test only.

## Pre-registered checks

Intervals: the seed t interval (95%, df = 2) of the per-seed difference, and the seed-pooled paired image
bootstrap (1000 resamples of the test images). "Above 0" means both intervals lie above 0.

| Check | Statistic | Pass |
|---|---|---|
| **O4 per-class advantage (primary)** | Macro test AP50, K2-cua2 − K2-shared2 | Both intervals above 0 |
| **O3 mechanism** | Test AP50 of c*, K2-cua2 − K2-shared2 | Both intervals above 0 |
| **O0 manipulation** | Test AP50 of c*, K2-shared2 − C-shared2 | Both intervals below 0 |
| **O1 detection** (cua2) | Raw per-estimate utility u(plant_mosaic, ·) in `utility.jsonl` | In every seed t of u(plant_mosaic, c*) ≤ −2 and c* is the most negative class of the row; seed t interval of the contrast with the other classes below 0 |
| **O2 action** (cua2) | Rate of `plant_mosaic` on images with c* ÷ rate on images with other classes only (target-free images removed by exact replay, as in the corrected first control), epochs ≥ 5 | Ratio ≤ 1/3 in every seed |
| **O5 vs the default** (secondary) | Macro test AP50, K2-cua2 − K2-uniform | Both intervals above 0 |

Reported without a pass rule: the same statistics for the other arms (O1 and O2 for shared2, which logs the
per-class table; O0 for uniform and cua2), the rate of `plant_mosaic` on images without c* for each arm (does
the policy keep using the helpful edit?), and the cost of the plant to each arm (K2 − C, macro test AP50).

## Power

In the C- runs on this pair, the per-seed difference cua2 − shared2 in macro test AP50 has SD ≈ 0.18. With
three seeds the seed t interval excludes 0 only if the mean difference is above about 0.45 AP50. The expected
advantage is the c* harm a class-blind policy must accept, divided by 20 classes, minus the mosaic gain CUA
gives up on c* images. So O4 can pass only if the planted harm under shared-v2 costs c* several AP50 points.
O3 and O0 show whether it did.

## How the result will be read

- **O4 passes:** given a class-specific effect of this kind, CUA's per-class estimator and policy convert it
  into a test-accuracy gain over the same method with pooled classes. The main null result then reflects the
  real edits (no class-specific effects of usable size), not a method that could never win.
- **O4 fails, O1 and O2 pass, O0 passes:** CUA finds and avoids the harm and shared-v2 suffers it, but the
  difference is too small for three seeds; report the size against the 0.45 threshold.
- **O4 fails, O0 fails:** shared-v2 either avoided the edit or the harm was too small; check its
  `plant_mosaic` rate. If shared-v2 dropped the edit and lost mosaic's gain, O4 should still pass; if not,
  report it as such.
- **O1 or O2 fails:** with a prior that favours the edit, CUA-v2's online estimate did not overturn it for c*;
  report the size of the c* cell and of its prior.
