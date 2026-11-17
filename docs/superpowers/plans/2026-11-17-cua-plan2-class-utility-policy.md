# CUA Plan 2 — Class-Utility Policy (Part C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train detectors under the five main-comparison policies (base recipe, uniform, measured-effect, shared-utility, class-utility/CUA), with the online class-utility estimator, plus the tuning and ablation run families.

**Architecture:** Each training image's edit is chosen in the main process by a policy inside a batch sampler, which yields `(index, edit)` pairs that the dataset renders. Every K optimiser steps, the estimator scores every valid edit for every class: it takes the gradient of the full training loss on edited and unedited copies of an 8-image estimation batch, and the gradient of per-class masked losses on a 64-image stratified reference batch, both restricted to a small parameter subset θ_s. It then updates EMA utility tables that the policy reads. All new state lives in `last.pt`, so the existing exact-resume and deadline logic keeps working.

**Tech Stack:** Python 3.12, PyTorch 2.3 / torchvision 0.18 on CCAST (local dev has 2.10 / 0.25; use only APIs present in both), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-11-08-class-utility-augmentation-design.md` (§4, §6, §9, §10). Plan 1: `docs/superpowers/plans/2026-11-08-cua-plan1-foundation-and-part-a.md`.

## Decisions taken after the spec (binding, 2026-11-16/17)

1. **Reference batch:** each estimation step draws one shared reference batch R of 64 images from `reference.jsonl`, stratified so every class appears at least once. h_c = ∇θ_s L_c(R), where L_c is masked to class-c matches. This replaces 64 images per class.
2. **Backbone without gradients:** during estimation passes, every parameter outside θ_s has `requires_grad=False`, so only θ_s is backpropagated.
3. **Smoothing:** a cell's first update sets ū = u, v = 0. Later updates, with d = u − ū_old: ū = ū_old + (1−β)d, v = β(v + (1−β)d²), se = sqrt(v(1−β)/(1+β)), β = 0.9. The harm block uses a cell only after `min_estimates = 3` updates.
4. **Measured-effect policy:** ū = mean ΔAP50 over Part A seeds, se = s.d./√n_seeds, with the same scoring, harm and sampling rule.
5. **Shared-utility:** per estimation step, u_shared(o) = mean over classes present in R of u(o,c), and it gets its own EMA tracker (shape |O|×1). Every image uses that one row.
6. **Second detector:** RetinaNet. "Top FPN level" means `backbone.fpn.inner_blocks[-1]` and `backbone.fpn.layer_blocks[-1]` for both detectors.
7. **Out of scope here (Plan 3):** test-set evaluation, comparison statistics, estimator-agreement analysis, figures, and the tolerance-gated ablation.

## Global Constraints

- Base recipe in every policy: hflip p = 0.5 + scale jitter 0.8–1.2 (`base_geometry`), never adapted.
- The menu comes from `valid_menu(dataset)` (vflip invalid for rdd/voc). Identity is the extra candidate `None`, with utility 0 by definition.
- Estimator: K = 200 steps, |B| = 8, |R| = 64, δ = 0.01, η = current LR. It starts at epoch 2 (epoch 1 is uniform over menu ∪ {identity}).
- Policy: score(o) = (1/|S|) Σ_{c∈S} w_c ū(o,c), w_c = 1. Harm block: drop o if, for any c ∈ S, ū(o,c) + 1.64·se(o,c) < 0. p(o) = (1−ε) softmax(score/τ) + ε/|allowed|, τ = std of the allowed scores floored at 1e-6, ε = 0.1.
- Images with no target objects: identity with p = 0.5, else a uniform menu edit. This is identical in every policy, base recipe included.
- Mosaic: chosen per anchor image from its own S; the other three tiles are uniform (existing `ArmTransform` behaviour).
- Part C runs use the full schedule `full_schedule(BUDGET_EPOCHS[dataset])` with no warm start. Checkpoint selection stays best validation macro AP50 (existing trainer).
- The trainer never reads `test.jsonl`. It reads `reference.jsonl` only for the `shared` and `cua` policies.
- Existing run configs must keep their digests (RetinaNet Part A jobs are running and resuming on CCAST).
- LF line endings for every new file.

## Review Focus

1. **Digest drift:** adding TrainConfig fields must not change the digest of any existing budget/Part A/proxy spec. Otherwise resumed CCAST jobs refuse to continue. Pinned in Task 1.
2. **Class with no matches in an estimation step** (0 matched proposals/anchors in R): skip that class's update. Never divide by zero or write NaN. Pinned in Task 3.
3. **Every candidate harm-blocked with identity disabled** (`noid` ablation): fall back to identity and count the fallback. Never raise or sample from an empty set. Pinned in Task 2.
4. **Non-finite utility** (inf/NaN from a degenerate batch): raise with epoch and global iteration, like the trainer's non-finite-loss check. Pinned in Task 4.
5. **Resume mid-run under `cua`:** a run stopped at a deadline and resumed must match an uninterrupted run (CPU) in history, utility.jsonl and the final model. Pinned in Task 5.

---

### Task 1: Config fields with stable digests

**Files:** Modify `cua/train.py` (TrainConfig, digest, `_validate`). Test `tests/cua/test_train.py`.

**Interfaces — Produces:** `TrainConfig` gains these fields, with defaults:
`policy: str = "arm"` ("arm" | "uniform" | "measured" | "shared" | "cua"), `est_every: int = 200`, `est_batch: int = 8`, `ref_batch: int = 64`, `theta: str = "head_fpn"` ("head" | "head_fpn"), `epsilon: float = 0.1`, `beta: float = 0.9`, `est_start_epoch: int = 2`, `weights: str = "equal"` ("equal" | "inv_ap"), `harm_block: bool = True`, `allow_identity: bool = True`.
`"arm"` is the existing Part A behaviour (`edit`, `edit_p`). The base recipe is `policy="arm", edit=None`.

- [ ] Step 1: Tests:
  - `test_digest_unchanged_for_existing_specs`: assert `part_a_specs("rdd","fasterrcnn",(0,))[0].digest() == "4c63658d0428b1daf190a09da78e843cf9ed44a266b29e0af61a0814f3946200"` and `[2]` (translate arm) `== "0b5522a1478db708c5f7ffb88ea424ae6f6bdd8b5ca29d913bf3b1dad41f7f88"`.
  - `test_digest_changes_with_new_fields`: `tiny_cfg(policy="cua").digest() != tiny_cfg().digest()`.
  - `test_validate_rejects_bad_policy_fields`: unknown policy, `policy!="arm"` with `edit` set, `epsilon` outside [0,1], `theta` unknown, `est_every < 1` each raise ValueError.
- [ ] Step 2: Run `pytest tests/cua/test_train.py -k "digest or validate" -v`. Expect FAIL.
- [ ] Step 3: Implement. `digest()` hashes `asdict(self)` with every field in `_PLAN2_FIELDS` dropped when it equals its dataclass default. `_validate` checks the rules above.
- [ ] Step 4: Run the full `pytest tests/cua -q -m "not slow"`. Expect PASS.
- [ ] Step 5: Commit `feat(cua): policy config fields with stable digests for existing runs`.

### Task 2: Policies

**Files:** Create `cua/policy.py`. Test `tests/cua/test_policy.py`.

**Interfaces — Produces:**
- `class UtilityTable` holds `mean: np.ndarray (|O|, C)`, `se: np.ndarray (|O|, C)` and `count: np.ndarray (|O|, C) int`.
- `class Policy(Protocol): def probs(self, class_set: frozenset[int]) -> tuple[tuple[str | None, ...], np.ndarray]` (candidates, probabilities summing to 1).
- `UniformPolicy(menu)`: identity ∪ menu, uniform.
- `IdentityPolicy()`: always `None` (base recipe).
- `UtilityPolicy(menu, table_fn: Callable[[], UtilityTable], *, shared: bool, epsilon, harm_block, allow_identity, weights_fn: Callable[[], np.ndarray] | None, min_estimates=3, z=1.64)`. When `shared`, the table has one column, used for every S.
- `choose(policy, class_set, rng) -> str | None`. No-object images (`class_set` empty) bypass the policy: identity with p = 0.5, else a uniform menu edit.
- `fallbacks: int` counter on `UtilityPolicy`.

- [ ] Step 1: Tests, with exact values:
  - `test_uniform_before_any_estimate`: all counts 0 → p uniform over identity ∪ menu (cua and shared).
  - `test_identity_scored_zero_and_never_blocked`.
  - `test_score_softmax_epsilon`: menu ("a","b"), C=2, mean [[1,3],[−1,−1]], se 0, count 5, S={0,1}, ε=0.1, harm_block False. Scores: id 0, a 2, b −1. τ = np.std([0,2,−1]). Assert p equals `0.9*softmax(scores/τ) + 0.1/3`.
  - `test_harm_block_uses_any_present_class`: ū(a,1) = −0.5, se 0.1, count 3, S = {0,1} → "a" absent. S = {0} → "a" present. count 2 → not blocked.
  - `test_all_blocked_without_identity_falls_back`: allow_identity False, every edit blocked → candidates (None,), p [1.0], `fallbacks == 1`.
  - `test_shared_ignores_class_set`: same probs for S={0} and S={1}.
  - `test_inv_ap_weights_change_score`: `weights_fn` returns [2,0] → only class 0 counts.
  - `test_no_object_image`: `choose` over 10,000 draws gives identity frequency 0.5 ± 0.02.
  - `test_tau_floor_with_equal_scores`: all scores equal → uniform p, no NaN.
- [ ] Step 2: Run `pytest tests/cua/test_policy.py -v`. Expect FAIL.
- [ ] Step 3: Implement. Vectorise with numpy. Weights are normalised to mean 1 over the classes in S.
- [ ] Step 4: Run. Expect PASS.
- [ ] Step 5: Commit `feat(cua): uniform, identity and utility policies`.

### Task 3: Per-class masked losses and θ_s

**Files:** Create `cua/estimator/__init__.py` (empty), `cua/estimator/class_loss.py` and `cua/estimator/params.py`. Test `tests/cua/test_class_loss.py`.

**Interfaces — Produces:**
- `theta_params(model, detector: str, theta: str) -> list[Parameter]`. "head" = `trainable_head_parameters`. "head_fpn" adds `backbone.fpn.inner_blocks[-1]` and `layer_blocks[-1]`.
- `only_theta(model, params)`: a context manager that sets `requires_grad=False` on every other parameter and restores the original flags on exit, also on exceptions.
- `@dataclass ClassLosses: sums: Tensor (C,)` (differentiable, class index 0-based), `counts: Tensor (C,) int` (matched proposals/anchors per class), `n_norm: int` (torchvision's normaliser: sampled proposals for Faster R-CNN, `max(1, num_foreground)` for RetinaNet).
- `class_losses(model, detector, images, targets, num_classes) -> ClassLosses`, run in train mode.
  - **Faster R-CNN:** `transform` → `backbone` → `rpn` → `roi_heads.select_training_samples` → `box_roi_pool` / `box_head` / `box_predictor`. Per proposal: `F.cross_entropy(reduction="none")` + `smooth_l1(beta=1/9)` summed over the 4 coords of its own class. Sum into the class of its label (background excluded).
  - **RetinaNet:** `transform` → `backbone` → `head` → `anchor_generator` → `proposal_matcher(box_iou)`. For foreground anchors: `sigmoid_focal_loss` summed over classes, plus the L1 box loss (`box_coder.encode_single`). Sum into the matched gt's class.
  - Labels are 1-based in targets. Class index = label − 1.

- [ ] Step 1: Tests. CPU, `pretrained=False`, size 320, 2 synthetic images, 3 classes:
  - `test_frcnn_box_part_matches_torchvision`: `torch.manual_seed(0)` before both calls. `sum_c box_part_c / n_norm == model(images, targets)["loss_box_reg"]` (rtol 1e-5). Expose the box part via `class_losses(..., parts=True)` returning `(cls_sums, box_sums)`.
  - `test_retinanet_box_part_matches_torchvision`: same against `["bbox_regression"]`.
  - `test_counts_match_labels`: counts equal the number of sampled proposals per foreground class.
  - `test_absent_class_has_zero_count_and_zero_sum`.
  - `test_only_theta_restores_flags`: flags restored after a raise inside the context, and only θ_s gets `.grad` after `backward` inside it.
- [ ] Step 2: Run `pytest tests/cua/test_class_loss.py -v`. Expect FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: Run. Expect PASS.
- [ ] Step 5: Commit `feat(cua): per-class masked detection losses and theta_s selection`.

### Task 4: Utility estimator

**Files:** Create `cua/estimator/utility.py`. Test `tests/cua/test_utility.py`.

**Interfaces:**
- **Consumes:** `class_losses`, `theta_params`, `only_theta`, `UtilityTable`.
- **Produces:**
  - `utility_from_grads(g0: Tensor, go: Tensor (|O|,P), h: Tensor (C,P), Lc: Tensor (C,), lr: float, delta=0.01) -> np.ndarray (|O|,C)`: u = lr·(go−g0)·hᵀ / (Lc+δ).
  - `class EmaTracker(shape, beta)`:
    - `update(u, valid: np.ndarray bool)` applies decision 3 to valid cells only.
    - `table() -> UtilityTable`.
    - `state_dict()` / `load_state_dict()`.
  - `stratified_reference(records, n, rng) -> list[int]`. One image per class, classes visited in rng order, then a uniform fill without replacement. Raises if C > n.
  - `class Estimator(model, detector, theta, menu, num_classes, est_batch, ref_batch, delta, beta, amp)` with `estimate(train_render, ref_render, train_records, ref_records, rng, lr) -> dict`. It returns u (|O|,C), Lc, valid, u_shared and seconds, and updates the `cua` and `shared` trackers.
    - `train_render(i, edit)` and `ref_render(i)` return `(image_tensor, target)`. They are rendered with a `ThreadPoolExecutor(max_workers=4)`.
    - The same `torch.manual_seed(derived)` is set before each g pass, so RPN sampling is paired.
    - Under AMP, each loss is multiplied by 2**10 before `autograd.grad` and the grads are divided back. Grads are flattened over θ_s in fp32.
    - A class with `count == 0` in R is invalid for this step.
    - Any non-finite u raises `RuntimeError` naming epoch and global iteration (passed in).

- [ ] Step 1: Tests:
  - `test_utility_from_grads_exact`: hand-computed 2×2 case.
  - `test_ema_and_se`: updates u = 1 then u = 3 with β = 0.9 give ū = 1.2, v = 0.36 and se = sqrt(0.36·0.1/1.9) ≈ 0.137649, count 2. Invalid cells unchanged.
  - `test_state_roundtrip`.
  - `test_stratified_reference_covers_all_classes`: every class is present, size n, deterministic for a fixed rng.
  - `test_identity_edit_row_is_zero`: tiny CPU model, menu `(None, "brightness")` → row for None is exactly 0.
  - `test_estimate_shapes_finite`: tiny model, 3 classes, a menu of 2 edits → u shape (2,3), finite, `seconds > 0`.
  - `test_nonfinite_raises`: monkeypatch `utility_from_grads` to return NaN → RuntimeError mentioning "global iteration".
- [ ] Step 2: Run. Expect FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: Run. Expect PASS.
- [ ] Step 5: Commit `feat(cua): gradient-alignment class-utility estimator`.

### Task 5: Trainer integration

**Files:**
- Modify `cua/augment/menu.py`, `cua/data.py` and `cua/train.py`.
- Create `cua/policy_sampler.py`.
- Test `tests/cua/test_train_policy.py`, plus edit `tests/cua/test_train.py::test_never_reads_test_or_reference_jsonl`.

**Interfaces:**
- **Consumes:** Tasks 1–4.
- **Produces:**
  - `ArmTransform.apply(sample, epoch, index, fetch, n_items, edit: str | None) -> Sample`: base + edit, no coin. `__call__` keeps its coin and now calls `apply`, so Part A output is byte-identical (existing tests must stay green).
  - `DetectionDataset.__getitem__` accepts `int` (existing) or `(int, edit)`. The tuple form calls `apply`.
  - `PolicyBatchSampler(records, policy_fn, batch_size, order, seed, epoch)` yields lists of `(i, edit)`. Each choice uses `keyed_rng(seed, epoch, i, "policy")`. It accumulates `edit_counts` and `edit_by_class` for the history row.
  - Run-dir artefact `utility.jsonl`: one row per estimate, `{"epoch","global_iter","lr","u","mean","se","count","Lc","valid","shared_mean","shared_se","seconds"}`.
  - The history row gains `"policy": {"edit_counts","edit_by_class","estimator_seconds","estimates","fallbacks"}` when `policy != "arm"`.
  - `last.pt` gains `"policy_state"`: the trackers and the inverse-AP weights.
  - `measured`: loads `data_root/<dataset>/measured_<detector>.json` (Task 6). It raises if the file is missing or its classes or edits differ from the run's.

  Training loop: for `uniform/measured/shared/cua`, build the loader with `batch_sampler=PolicyBatchSampler(...)`. Before `est_start_epoch` the policy is `UniformPolicy`. For `shared/cua`, after each optimiser step with `epoch >= est_start_epoch and global_iter % est_every == 0`, call `Estimator.estimate`. `inv_ap` weights update at each epoch end from `metrics["per_class"][cls]["ap50"]` (w ∝ 1/max(AP,1)). On resume, `utility.jsonl` is truncated to rows with `epoch <= ckpt epoch` (same rule as history).

- [ ] Step 1: Tests. CPU, `tiny_data` with `augment=True`, `size=640`, mark `slow` if > 20 s:
  - `test_arm_policy_output_unchanged`: existing Part A tests unchanged, plus `apply(..., edit=None)` equals `base`.
  - `test_uniform_policy_trains_and_logs_counts`: the history row has `edit_counts` summing to the number of training images.
  - `test_cua_writes_utility_rows`: `est_every=1`, `est_start_epoch=1`, `ref_batch=2`, `est_batch=2` → `utility.jsonl` rows with u shape (|menu|, C).
  - `test_cua_resume_equals_uninterrupted` (Review Focus 5): mirror `test_resume_is_equivalent_to_uninterrupted_run`.
  - `test_reference_read_only_for_shared_and_cua`: monkeypatch `read_records` to record paths. arm/uniform/measured never read `reference.jsonl`. No policy reads `test.jsonl`.
  - `test_measured_requires_matching_table`.
- [ ] Step 2: Run. Expect FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: Run the full `pytest tests/cua -q -m "not slow"`, then the new slow tests. Expect PASS.
- [ ] Step 5: Commit `feat(cua): train with uniform, measured, shared and class-utility policies`.

### Task 6: Measured-effect table, run specs and job families

**Files:**
- Create `cua/measured.py`.
- Modify `cua/experiments/specs.py` and `cua/ccast/make_jobs.py`.
- Tests: `tests/cua/test_measured.py`, `tests/cua/test_specs.py`, `tests/cua/test_make_jobs.py`.

**Interfaces — Produces:**
- `python -m cua.measured --deltas <deltas.csv> --data-root <cua_data>` writes `measured_<detector>.json` per (dataset, detector) found: `{"edits","classes","mean","se","n_seeds"}`, with se = s.d./√n. It refuses fewer than 2 seeds.
- `CUA_SETTINGS = {"est_every": 200, "theta": "head_fpn", "epsilon": 0.1}`, a module constant updated after tuning. Part C and ablation specs pass it explicitly (never via TrainConfig defaults).
- `part_c_specs(dataset, detector, seeds=(0,1,2), policies=("base","uniform","measured","shared","cua"))`. Run id `C-<ds>-<det>-s<seed>-<policy>`, `full_schedule(BUDGET_EPOCHS[ds])`, no `init_run`, `edit_p` unused.
- `tune_specs()` → `T-rdd-fasterrcnn-s0-<name>` for name in `default, k100, k400, head, eps0.05, eps0.2`: one factor at a time around the defaults; the spec's full grid is out of the GPU budget.
- `ablation_specs(dataset, seeds=(0,1,2))` for dataset ∈ {rdd, dacl10k}, Faster R-CNN only. `X-<ds>-fasterrcnn-s<seed>-<noharm|noid|invap>`, built on the `cua` config.
- `resolve()` handles the C-, T- and X- ids. `make_jobs --family part_c|tune|ablation` with `--policies` (csv). Walltime is 24 h for these (resume covers longer runs).

- [ ] Step 1: Tests:
  - `test_measured_table_mean_and_se` (3 seeds of known deltas).
  - `test_measured_refuses_one_seed`.
  - `test_part_c_ids_and_schedule`: `C-voc-retinanet-s1-cua` resolves with 20 epochs and the `CUA_SETTINGS` values.
  - `test_base_policy_is_arm_without_edit`.
  - `test_tune_has_six_unique_runs`.
  - `test_ablation_flags` (noharm → `harm_block=False`; noid → `allow_identity=False`; invap → `weights="inv_ap"`).
  - `test_make_jobs_part_c_writes_15_pbs_and_no_dependencies`.
- [ ] Step 2: Run. Expect FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: Run the full suite. Expect PASS.
- [ ] Step 5: Commit `feat(cua): measured-effect table, Part C/tuning/ablation specs and jobs`.

### Task 7: Overhead smoke check on CCAST (operator)

- [ ] Upload the code (one login). Run `make_jobs --family part_c --dataset rdd --detector fasterrcnn --seeds 0 --policies cua --smoke` and submit.
- [ ] Check (one login): `utility.jsonl` has finite rows, and `estimator_seconds / epoch seconds` is in the history row. Record the overhead. If > 20 %, raise `est_every` or reduce `ref_batch` before tuning, and log it in the spec.
- [ ] Commit any setting change with the measured overhead in the message.
