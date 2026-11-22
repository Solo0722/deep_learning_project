# Per-class population-based augmentation (PBT) — design and pilot

## Why
CUA v1/v2 estimate per-class utility online from a few batches. The estimate is mostly noise
(effects of 0.5–1.5 AP50 over a full run; 16 images, 4 steps, 32 reference images per estimate).
PBT instead measures the real quantity, per-class AP, after several epochs of training, on
detectors that differ only in their edit tables (paired comparison).

## Method
- Population of N members (pilot N=4), same seed, same data order, same initial weights.
- Each member has a logit table L of shape (1 + |menu|, C): row 0 is "no edit".
  Image with class set S: p(o) = (1-eps) * softmax(mean_{c in S} L[o, c]) + eps / (1 + |menu|).
  The pooled control ("shared PBT") has C = 1: one column for every class.
- Training is cut into windows of W epochs (pilot W=2, so 10 windows over 20 epochs).
- Each window, each member trains W epochs from the window's start weights (via `init_run`, so
  optimizer state, LR schedule and keyed RNG continue exactly), then is scored on the reference
  split: per-class AP50 and macro AP50. Reference is never used for val-based model selection or test.
- Select (after window w):
  - weights: every member of window w+1 starts from the member with the best reference macro AP50;
  - tables, per class c: column c of every member is set to column c of the member with the best
    reference AP50 for c (ties → lowest member index);
  - explore: member 0 keeps the selected table unchanged; members 1..N-1 add N(0, sigma) noise to
    every logit (pilot sigma = 0.5), seeded by (seed, window, member).
- Window 1 tables: member 0 all zeros (uniform), others N(0, sigma).
- Final model: after the last window, the member with the best validation macro AP50 at its last
  epoch (the same split the other policies use to pick best.pt).

## Cost
About N full runs per (dataset, detector, seed), plus N reference evaluations per window.

## Implementation
- `TrainConfig.policy = "table"` with `table_path` (JSON: menu, classes, logits) and `table_eps`.
  New fields are out of the digest at their defaults, so existing digests are unchanged.
- `cua/pbt.py`: `PBTConfig`, run-id layout `PBT-<ds>-<det>-s<seed>-<variant>/w<k>-m<j>` (member runs
  live under `runs_root/PBT-.../`), and steps that are each idempotent:
  - `init` — write window-1 tables and state.json;
  - `member --window k --member j` — train the window (exit 3 at the deadline, resumable), then
    write ref_metrics.json;
  - `select --window k` — read members' ref_metrics.json, write window k+1 tables and
    decisions.jsonl; after the last window write final.json;
  - `run` — everything in order in one process (tests, small jobs).
- CCAST: the whole chain is submitted up front — N member jobs per window (afterok on the previous
  select), one CPU-light select job per window (afterok on its N members).

## Pilot (decides whether to go further)
RDD, Faster R-CNN, seed 0, N=4, W=2, per-class variant, validation and reference only (no test).
Read-outs:
1. Winner consistency: for each class, how often the winning member's table for that class is
   the same direction as in the previous window, and the rank correlation of the selected table's
   per-class edit preferences between consecutive windows.
2. Reference per-class AP spread between members in a window versus the spread from seed noise.
3. Final validation macro AP50 against base / uniform / measured on the same pair and seed.
Go on to the full matrix only if (1) shows selections that are stable beyond chance.
