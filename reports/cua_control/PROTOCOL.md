# Positive control: can CUA find and use a class-specific effect when one exists?

Fixed before any K- run is trained. The read-out script is `cua/analysis/control.py`. Its thresholds are constants in that file and are not changed after the runs.

## Why

The main result is negative: class-specific edit selection does not beat uniform sampling. A reviewer can reply that the estimator or the policy could never have found a class-specific effect. This control plants an effect whose size and target class are known by construction, then checks each link in the chain: the effect exists, the estimator sees it, the policy acts on it, and accuracy improves.

## Planted edit

`plant` (in `cua/augment/menu.py`) deletes every annotation of one target class c* from the image. The pixels are unchanged, so c*'s objects are trained as background. On an image without c* it changes nothing. It harms c* and, by construction, no other class.

- **Target class:** c* = the class with the most training boxes (`plant="frequent"`, ties go to the lowest index). The trainer records the resolved name as `plant_class` in each run's `config.json`.
- **Menu:** the planted edit is added as the last item of the normal menu. Uniform sampling picks it with probability 1/15 on dacl10k (identity + 13 edits + plant).

## Runs

dacl10k, Faster R-CNN, seeds 0, 1 and 2. Each arm uses exactly the settings of its C- run, with `plant="frequent"` added.

| Run | Policy |
|---|---|
| `K-dacl10k-fasterrcnn-s{0,1,2}-uniform` | uniform |
| `K-…-shared` / `K-…-cua` | v1 estimator (shared / per-class) |
| `K-…-shared2` / `K-…-cua2` | v2 estimator (shared / per-class; the planted row gets no probe prior) |

15 runs, about the same cost as 15 C- runs. All arms are scored once on test with `cua.test_eval --prefix K-`.

## Pre-registered checks

| Check | Statistic | Pass |
|---|---|---|
| **O0 manipulation** | Test AP50 of c*, K-uniform − C-uniform | Both the seed t interval and the paired image bootstrap lie below 0 |
| **O1 detection** (cua, cua2) | Raw per-estimate utility u(plant, ·) in `utility.jsonl` | In every seed: t of u(plant, c*) ≤ −2, and c* has the most negative u in the plant row. Also: the seed t interval of the contrast u(plant, c*) − mean over other classes lies below 0 |
| **O2 action** (cua, cua2) | Planted-edit rate on images with c* ÷ rate on images without c*, epochs 5–30 | Ratio ≤ 1/3 in every seed |
| **O3 benefit** (cua, cua2) | Test AP50 of c*, arm − K-uniform | Both intervals lie above 0 |

Shared arms are reported under the same statistics. They are expected to pass O1 on the per-class table, which they log, but to fail O2. Their policy pools classes, so it cannot treat images with and without c* differently. That contrast isolates the per-class part of CUA.

## How the result will be read

- **O0 fails:** the planted harm is too small to matter even under uniform sampling. O1 and O2 are still informative about the estimator and the policy, but O3 is moot. Report this as such.
- **O0, O1 and O2 pass:** the machinery finds and acts on a class-specific effect of known size. The main null result then reflects the absence of such effects among the real edits, not a blind estimator. Compare the size of the planted cell with the largest real cells of the C-run utility tables.
- **O1 or O2 fails:** CUA could not have exploited class-specific effects of this size. The paper must say so and state the size of effect the estimator can detect.

Validation and test disagreed twice for copy-paste, so O0 and O3 are judged on test only.

## Correction to the O2 statistic (made after the runs, before any test result)

The first read-out divided by the planted-edit rate on all images without c*, including images with no
target object at all. Those images never reach the policy: `cua.policy.choose` gives them the identity with
probability 1/2 and otherwise a uniform menu edit, so they receive the planted edit at 1/(2 × 14) ≈ 0.036
under every arm. Because the policy arms apply it far less often to images with objects, this fixed rate
made the "without c*" group look much more planted-on than the policy made it, and the ratio fell below
1/3 even for the class-blind shared arms, whose policy cannot tell the two kinds of image apart.

The corrected statistic restricts the "without c*" group to images that contain other target classes,
i.e. to images the policy decides. The draws on target-free images are replayed exactly from the keyed
RNG (`target_free_draws` in `cua/analysis/control.py`) and subtracted; the replay is checked against
the logged counts. The threshold (ratio ≤ 1/3 in every seed) and every other check are unchanged. The
uncorrected ratio is still reported, in its own column.
