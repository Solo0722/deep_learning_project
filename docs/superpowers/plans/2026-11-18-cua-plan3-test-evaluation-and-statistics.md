# CUA Plan 3 — Test Evaluation, Statistics, Tables and Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn finished Part C runs into the paper's evidence. This means one test-set evaluation per main-comparison run, the H2/H3/H4 statistics and ablations, and the tables and figures.

**Architecture:**
- **`cua/test_eval.py`** evaluates each complete `C-` run's `best.pt` on `test.jsonl` exactly once. It writes `test_predictions.json` and `test_metrics.json` into the run directory.
- **`cua/analysis/part_c.py`** reads those files plus `history.jsonl` and `utility.jsonl`. It computes:
  - paired seed t intervals;
  - a seed-pooled paired image bootstrap (a new function in `cua/analysis/ap50.py`);
  - the H4 Spearman agreement, reusing `proxy_agreement`;
  - validation-only ablation contrasts.

  It then writes `results.json`, CSV/LaTeX tables and PNG figures.
- **A static PBS script** runs the test evaluation on CCAST.

**Tech Stack:** Python 3.12, PyTorch/torchvision (CCAST: torch 2.3 / torchvision 0.18), numpy, scipy, matplotlib, pycocotools, pytest.

**Spec:** `docs/superpowers/specs/2026-11-08-class-utility-augmentation-design.md` (§2, §4, §10, §11). Plans 1 and 2: `docs/superpowers/plans/2026-11-08-cua-plan1-foundation-and-part-a.md`, `docs/superpowers/plans/2026-11-17-cua-plan2-class-utility-policy.md`.

## Global Constraints

- **Test-set use:**
  - The test set is used only by `cua/test_eval.py`, only for run ids matching `C-<dataset>-<detector>-s<seed>-<policy>` (spec §4: "opened once, after §9.4 and all arm settings are frozen").
  - Tuning (`T-`), ablation (`X-`), Part A (`A-`), budget and proxy runs never touch test.
- **Checkpoint:** the checkpoint is the run's `best.pt`, which is the best validation macro AP50 (spec §10).
- **Primary metric:** test macro AP50, which is COCO evaluation at IoU 0.5 averaged over classes with ≥1 non-ignored GT. This is the same rule as `coco_metrics`.
- **Secondary metrics:** AP50:95, per-class AP50, AP small/medium/large, false-positive images on target-free test images, and estimator overhead (spec §11).
- **Seed pairing:** seeds are paired across arms. Report mean ± s.d. over seeds (sample s.d., ddof=1).
- **Intervals:**
  - For every arm difference, report a 95% seed-level paired t interval and a 95% paired image-level bootstrap with 1,000 resamples.
  - The bootstrap uses the same resampled test images for both arms and for every seed. Each resample's difference is the mean over seeds of the per-seed macro AP50 difference.
  - A claim requires both intervals to exclude 0.
- **H2:** `cua` − `shared` on test macro AP50, per dataset × detector.
- **H3:** `cua` − the better of {`base`, `uniform`}. "Better" means the higher mean test macro AP50 over seeds, which is the conservative choice.
- **Escalation flag (spec §5):** set `escalate_to_5_seeds` when the H2 t interval spans 0 and its width exceeds 1.0 AP50.
- **H4 (spec §11):**
  - Spearman ρ between the time-averaged utility ū(o,c) and Part A ΔAP50(o,c), with a bootstrap CI over (o,c) cells.
  - Time-averaged ū = mean of the instantaneous `u` over all `utility.jsonl` rows where `valid` is true, averaged over the `cua` run's seeds.
  - Part A ΔAP50 = mean over seeds from `deltas.csv`.
- **Ablations:** validation only. Each of `noharm`, `noid` and `invap` minus `C-<ds>-fasterrcnn-s<seed>-cua`, on best validation macro AP50, with a paired t interval, for rdd and dacl10k.
- **False-positive images:** a test image with no non-ignored GT that has ≥1 detection with score ≥ 0.5. Report the count, the number of target-free images, and the fraction. The fraction is `None` when there are no target-free images.
- **Estimator overhead:** Σ `policy.estimator_seconds` / Σ epoch `seconds` over `history.jsonl`, for `shared` and `cua` runs.
- **Report figures:** plain black and white (user preference), with no colour fills.
- **Out of scope:** Part B (spec §8, not run) and the tolerance-gated (v2) controller ablation (spec §10, deferred from Plan 2). Porting it needs a new policy and 6 more runs. This plan leaves both out, and the paper lists them as limitations unless the user decides otherwise.

## Review Focus

1. **Missing or incomplete arm for a seed:**
   - Behaviour: comparisons pair only seeds present in both arms, record `n_seeds`, and set `claim=False` with `note="fewer than 2 paired seeds"` when n < 2. Never compare unpaired means or raise mid-report.
   - Pinned in Task 3 (`test_unpaired_seed_is_dropped`).
2. **Re-running test evaluation:**
   - Behaviour: a run with an existing `test_metrics.json` is skipped, not overwritten. A non-`C-` run id is refused.
   - Pinned in Task 1 (`test_skips_already_evaluated_run`, `test_refuses_non_part_c_runs`).
3. **`best.pt` whose run config does not match the current spec digest** (code changed under a finished run):
   - Behaviour: refuse that run with a clear message, and continue with the others.
   - Pinned in Task 1 (`test_refuses_digest_mismatch`).
4. **A class with zero GT in the test split or in a bootstrap resample:**
   - Behaviour: excluded from macro AP50, both in `test_metrics.json` (as `coco_metrics` does) and in the bootstrap (`nanmean`), never 0 and never NaN in the output.
   - Pinned in Task 2 (`test_pooled_bootstrap_ignores_absent_class`).
5. **A dataset with no target-free test images:**
   - Behaviour: the false-positive image fraction is `None`, with no division by zero.
   - Pinned in Task 1 (`test_fp_images_none_without_target_free_images`).

---

### Task 1: One-time test evaluation of Part C runs

**Files:**
- Create: `cua/test_eval.py`, `cua/ccast/test_eval.pbs`
- Test: `tests/cua/test_test_eval.py`

**Interfaces:**
- Consumes:
  - `resolve(run_id) -> TrainConfig` (`cua/experiments/specs.py`).
  - `TrainConfig.digest()`.
  - `build_detector(name, num_classes, size, pretrained)` (`cua/models.py`).
  - `predict(model, dataset, device, batch_size, num_workers)` and `coco_metrics(records, predictions, classes)` (`cua/evaluate.py`).
  - `DetectionDataset(records, image_root, None, size)` (`cua/data.py`).
  - `read_records`, `read_classes` (`cua/records.py`).
  - `_predictions_json` (`cua/train.py`).
  - Run layout: `config.json` has key `digest`; `best.pt` has keys `model`, `epoch`, `best_macro`, `best_epoch`; `COMPLETE.json`.
  - Image root from `<data_root>/<ds>/source.json` (`image_root`, relative to the dataset dir), as `train()` reads it.
- Produces:
  - `fp_images(records, predictions, threshold=0.5) -> dict` with keys `count`, `n_target_free`, `fraction`.
  - `evaluate_run(run_dir: Path, data_root: Path, device: str = "cuda", batch_size: int = 8, workers: int = 4) -> dict`, which returns the `test_metrics.json` content.
  - `main(argv) -> int`: CLI `python -m cua.test_eval --runs-root DIR --data-root DIR [--device cuda] [--prefix C-]`. It returns 0 when every complete run is evaluated or skipped, and 1 if any run was refused. It prints one line per run.
  - `test_metrics.json` = `coco_metrics(...)` output plus `"fp_images": fp_images(...)`, `"best_epoch"`, `"best_val_macro_ap50"`, `"n_test_images"`, `"run_id"`, `"digest"`.

- [ ] **Step 1: Write the failing tests** in `tests/cua/test_test_eval.py`.
  - **Fixture:** reuse the tiny synthetic voc layout from `tests/cua/test_train_policy.py::data`, but write a valid `test.jsonl` with 3 records, one of them target-free (empty boxes). Train for one epoch a `TrainConfig` with `run_id="C-voc-fasterrcnn-s0-base"`, `policy="base"`, `epoch_lr=(1.0,)`, `pretrained=False`, CPU only, and the other fields as `cfg()` in `test_train_policy.py`. Monkeypatch `cua.test_eval.resolve` to return that config. This test is marked `slow`.
  - `test_evaluates_best_checkpoint_once` (slow):
    - `evaluate_run` writes `test_predictions.json` and `test_metrics.json`.
    - `metrics["best_epoch"] == 1` and `metrics["n_test_images"] == 3`.
    - `metrics["fp_images"]["n_target_free"] == 1`.
    - `"macro_ap50" in metrics`.
  - `test_skips_already_evaluated_run`: a run dir that already has `test_metrics.json` (content `{"sentinel": 1}`) plus `COMPLETE.json`. `main` leaves the file byte-identical and prints `already evaluated`.
  - `test_refuses_non_part_c_runs`: `evaluate_run` on a dir named `T-rdd-fasterrcnn-s0-default` raises `ValueError` matching `"Part C"`. `main(["--prefix", "T-", ...])` raises `SystemExit` with code 2 (argparse `choices=("C-",)`).
  - `test_refuses_digest_mismatch`:
    - Setup: `config.json` with `{"digest": "0"*64}` plus `COMPLETE.json` and `best.pt` stub, in a dir named `C-voc-fasterrcnn-s0-base`.
    - `evaluate_run` raises `RuntimeError` matching `"digest"`.
    - `main` over a runs root holding this dir and an already-evaluated dir returns 1 and still prints the skip line for the other.
  - `test_incomplete_run_is_reported_not_evaluated`: a `C-` dir without `COMPLETE.json` makes `main` print `incomplete` and write nothing.
  - `test_fp_images_none_without_target_free_images`: `fp_images` on records that all have GT gives `{"count": 0, "n_target_free": 0, "fraction": None}`.
  - `test_fp_images_counts_confident_detections_only`: two target-free images, one with a 0.6-score detection and one with only a 0.4-score detection, give `{"count": 1, "n_target_free": 2, "fraction": 0.5}`.

- [ ] **Step 2: Run them; all fail with ModuleNotFoundError `cua.test_eval`.**
  Run: `python -m pytest tests/cua/test_test_eval.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement `cua/test_eval.py`.**
  - **Build:** `build_detector(cfg.detector, len(classes), cfg.size, pretrained=False)`. The weights come from `best.pt`, so nothing is downloaded on compute nodes. Then `load_state_dict(torch.load(best.pt, weights_only=True)["model"])`.
  - **Write:** use the atomic tmp file + `os.replace` pattern, as in `cua/train.py` `_write_json`.
  - **Order:** check `COMPLETE.json`, then the id pattern, then the digest, before loading any test records. Use regex `^C-([^-]+)-([^-]+)-s(\d+)-([^-]+)$`.

- [ ] **Step 4: Write `cua/ccast/test_eval.pbs`.**
  - Resources: one A10 GPU, `ncpus=4`, `mem=32gb`, walltime 8 h, group `x-ccast-prj-aaboah`, log to `$P/cua_runs/_logs/test_eval.log`.
  - Environment: the same conda/venv activation as `job_template.pbs` (`P=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research`, `TORCH_HOME`, and `PYTHONPATH=$P/cua_code`).
  - Include the bad-GPU guard line from `job_template.pbs`.
  - Then run `python -m cua.test_eval --runs-root $P/cua_runs --data-root $P/cua_data`.
  - The script is idempotent, so resubmitting continues where it stopped.

- [ ] **Step 5: Run the fast tests, then the slow one.**
  Run: `python -m pytest tests/cua/test_test_eval.py -q -p no:cacheprovider`
  Expected: all pass.

- [ ] **Step 6: Commit** `feat(cua): one-time test evaluation of Part C runs`.

### Task 2: Seed-pooled paired image bootstrap

**Files:**
- Modify: `cua/analysis/ap50.py` (add one function after `paired_bootstrap`)
- Test: `tests/cua/test_evaluate.py` (append; the existing `paired_bootstrap` tests live there; check with `grep -n paired_bootstrap tests/cua/*.py` and append to that file)

**Interfaces:**
- Consumes: `Matches`, `_macro`, `image_matches` (`cua/analysis/ap50.py`).
- Produces: `paired_bootstrap_seeds(pairs: list[tuple[Matches, Matches]], n_boot: int = 1000, seed: int = 0) -> dict` with keys `diff`, `lo`, `hi` and `n_seeds`.
  - `diff` = mean over pairs of (macro(b) − macro(a)) on all images.
  - Each resample draws one index vector (`rng.integers(0, n, n)`) and uses it for every pair, which keeps the images paired across arms and seeds. The resample difference is the mean of the per-pair differences.
  - `ValueError` if the pairs differ in `n_images` or `gt_count`. All arms and seeds share the same test GT.

- [ ] **Step 1: Write the failing tests.**
  - `test_pooled_bootstrap_single_pair_matches_paired_bootstrap`: with one pair and the same seed, `diff`, `lo` and `hi` equal `paired_bootstrap`'s exactly.
  - `test_pooled_bootstrap_averages_seeds`:
    - Two pairs. In pair 1, b is a copy of a with one extra true positive, so its diff is > 0. In pair 2, b equals a (diff 0).
    - `diff` equals the mean of the two single-pair diffs (`pytest.approx`, abs 1e-12), and `n_seeds == 2`.
  - `test_pooled_bootstrap_ignores_absent_class`: a class with zero GT in every image gives a finite `diff`, `lo` and `hi`.
  - `test_pooled_bootstrap_rejects_different_ground_truth`: Matches built from different records raise `ValueError`.
  - Build Matches with `image_matches` on small hand-made records, as the existing bootstrap tests do.

- [ ] **Step 2: Run them; they fail with ImportError.**
- [ ] **Step 3: Implement `paired_bootstrap_seeds`.**
- [ ] **Step 4: Run** `python -m pytest tests/cua/test_evaluate.py -q -p no:cacheprovider -k bootstrap`. Expected: pass.
- [ ] **Step 5: Commit** `feat(analysis): seed-pooled paired image bootstrap`.

### Task 3: Part C statistics

**Files:**
- Create: `cua/analysis/part_c.py`
- Test: `tests/cua/test_part_c_analysis.py`

**Interfaces:**
- Consumes:
  - `paired_bootstrap_seeds`, `image_matches` (Task 2).
  - `proxy_agreement(short, full, n_boot, seed)` and `read_history` (`cua/analysis/part_a.py`). For H4, pass the Part A rows as `short` (`{"edit", "cls", "delta"}`, several seeds) and the time-averaged ū rows as `full` (one per cell).
  - `test_metrics.json` and `test_predictions.json` (Task 1).
  - `utility.jsonl` rows (`u`, `valid`: [edit][class] lists in `valid_menu(dataset)` × classes.json order).
  - `history.jsonl` rows (`seconds`, `policy.estimator_seconds`, `metrics.macro_ap50`).
- Produces (all in `cua/analysis/part_c.py`):
  - `load_runs(runs_root: Path) -> list[dict]`: one row per complete `C-` or `X-` run. Keys:
    - from the run id: `kind` ("C"/"X"), `dataset`, `detector`, `seed`, `arm` (policy for C, ablation name for X);
    - `best_val` (max `metrics.macro_ap50` in history), `overhead` (None for runs without estimator seconds);
    - `test` (the parsed `test_metrics.json`, or None if not evaluated).
  - `seed_t_interval(diffs: list[float]) -> dict` with keys `mean`, `lo`, `hi`, `n`. The 95% t interval uses n−1 df. With n < 2, `lo` and `hi` are None.
  - `compare(rows, dataset, detector, arm_a, arm_b, metric="test", matches=None) -> dict`:
    - Pairs seeds present in both arms.
    - The metric is `test.macro_ap50` or `best_val`.
    - Keys: `a`, `b`, `n_seeds`, `seeds`, `t` (seed_t_interval of b−a), `boot` (paired_bootstrap_seeds dict, or None when `matches` is None), `claim`, `note`.
    - `claim` = both intervals exclude 0 on the same side. For validation comparisons only the t interval counts.
    - `matches` maps `(arm, seed)` to a Matches object.
  - `h3_baseline(rows, dataset, detector) -> str`: `"base"` or `"uniform"`, whichever has the higher mean test macro AP50.
  - `h4_agreement(runs_root, deltas_rows, dataset, detector, n_boot=2000, seed=0) -> dict`: `proxy_agreement` output plus `n_runs`.
  - `summarize(runs_root: Path, data_root: Path, deltas_csv: Path, n_boot: int = 1000) -> dict`. It returns the full `results.json` structure:
    - `{"arms": {ds-det: {arm: {"mean", "sd", "n", "per_seed"}}}`;
    - `"h2": {ds-det: compare(... "shared", "cua") + {"escalate_to_5_seeds": bool}}`;
    - `"h3": {ds-det: compare(... baseline, "cua") + {"baseline": str}}`;
    - `"vs_measured": {ds-det: compare(... "measured", "cua")}`;
    - `"h4": {ds-det: ...}`;
    - `"ablations": {ds: {name: compare(... "cua", name, metric="val")}}`;
    - `"overhead": {ds-det: {arm: mean}}`;
    - `"secondary": {ds-det: {arm: {"ap50_95", "ap_small", "ap_medium", "ap_large", "fp_fraction"} means}}`;
    - `"missing": [run ids expected by part_c_specs/ablation_specs but absent or unevaluated]`.
  - Matches for the bootstraps are built once per (arm, seed) from `test_predictions.json` and the dataset's `test.jsonl` with `image_matches`.

- [ ] **Step 1: Write the failing tests.**
  - Write the run dirs synthetically: `config.json`, `COMPLETE.json`, `history.jsonl`, `test_metrics.json`, `test_predictions.json`, and `utility.jsonl` for cua. Use 3 test records and 2 classes under `tmp_path/data/voc`. No training.
  - `test_seed_t_interval_known_values`: `seed_t_interval([1.0, 2.0, 3.0])` gives `mean == 2.0` and `lo`/`hi` == 2 ∓ 4.302653 · 1/√3 (`approx` 1e-5).
  - `test_compare_pairs_seeds_and_claims`:
    - Setup: cua is 1.0 above shared on seeds 0, 1 and 2, with spread ±0.1. The predictions give b strictly more true positives on every image.
    - `claim is True`, `n_seeds == 3`, and the t interval is entirely > 0.
  - `test_unpaired_seed_is_dropped`: shared has seeds {0, 1, 2} and cua has {0, 1}, so `n_seeds == 2` and `seeds == [0, 1]`. With only seed 0 in cua: `claim is False` and `note == "fewer than 2 paired seeds"`.
  - `test_h3_baseline_is_better_mean`: uniform's mean exceeds base's, so the function returns `"uniform"`.
  - `test_escalation_flag`: H2 diffs [−0.6, 0.1, 0.9] (t interval spans 0, width > 1.0) set `escalate_to_5_seeds` True. Diffs [0.9, 1.0, 1.1] set it False.
  - `test_h4_uses_valid_cells_only`:
    - Setup: two utility rows. One cell is marked invalid in one row with `u = 1e9`. That cell's ū must equal the mean of its valid entries only.
    - ρ is computed against deltas over ≥ 3 cells and is finite.
  - `test_ablation_compares_validation_only`:
    - Setup: X-noharm run dirs have no `test_metrics.json`.
    - The ablation compare uses `best_val`, `boot is None`, and `summarize` does not raise.
  - `test_missing_lists_unevaluated_runs`: a complete C run without `test_metrics.json` appears in `missing`.

- [ ] **Step 2: Run them; they fail with ModuleNotFoundError.**
- [ ] **Step 3: Implement `cua/analysis/part_c.py` functions above** (no CLI yet).
- [ ] **Step 4: Run** `python -m pytest tests/cua/test_part_c_analysis.py -q -p no:cacheprovider`. Expected: pass.
- [ ] **Step 5: Commit** `feat(analysis): Part C statistics (H2, H3, H4, ablations, overhead)`.

### Task 4: Tables, figures and CLI

**Files:**
- Modify: `cua/analysis/part_c.py` (add `write_tables`, `write_figures`, `main`)
- Test: `tests/cua/test_part_c_analysis.py` (append)

**Interfaces:**
- Consumes: `summarize` output (Task 3).
- Produces:
  - `write_tables(results: dict, out: Path) -> None`, which writes:
    - `main_table.csv`: columns `dataset`, `detector`, then `<arm>_mean` and `<arm>_sd` for the arms in order base, uniform, measured, shared, cua.
    - `main_table.tex`:
      - An IEEE `table` with `booktabs`.
      - One row per dataset × detector and cells `mean ± sd` to 1 decimal.
      - The highest mean per row in `\textbf{}`.
      - The caption names the metric as test macro AP50 (%) and says it is the mean ± s.d. over 3 seeds.
    - `comparisons.csv`: one row per (`kind` in h2/h3/vs_measured/ablation, dataset, detector, a, b), with columns `n_seeds`, `t_mean`, `t_lo`, `t_hi`, `boot_lo`, `boot_hi` and `claim`.
  - `write_figures(results: dict, out: Path) -> None`, which writes:
    - `fig_differences.png`: a forest plot in black and white. It has one row per dataset × detector for each of H2 and H3. The point is the seed mean and the line is the bootstrap CI (the t interval when bootstrap is None), with a vertical dashed line at 0.
    - `fig_h4_<ds>_<det>.png`: a scatter of Part A ΔAP50 (x) against ū (y), one per available H4 entry, with ρ and CI in the title.
    - Use matplotlib only, with the `Agg` backend.
  - `main(argv) -> int`: CLI `python -m cua.analysis.part_c --runs-root DIR --data-root DIR --deltas CSV --out DIR [--n-boot 1000]`. It writes `results.json` plus tables and figures. It prints the `missing` count and returns 0.

- [ ] **Step 1: Write the failing tests.**
  - `test_cli_writes_tables_and_figures`: on the Task 3 synthetic runs root, `main` returns 0 and writes `results.json`, `main_table.csv`, `main_table.tex`, `comparisons.csv`, `fig_differences.png` and at least one `fig_h4_*.png`.
  - `test_latex_bolds_row_best`: the row's best arm cell is wrapped in `\textbf{` and appears exactly once per row.
  - `test_comparisons_csv_has_h2_h3_rows`: rows with `kind` `h2` and `h3` exist for the synthetic dataset × detector.
- [ ] **Step 2: Run them; they fail with AttributeError/ImportError.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `python -m pytest tests/cua/test_part_c_analysis.py -q -p no:cacheprovider`. Expected: pass.
- [ ] **Step 5: Commit** `feat(analysis): Part C tables, figures and CLI`.

### Task 5: Operator — run on CCAST after Part C completes

- [ ] **Step 1: Upload the code and submit the test evaluation (one login).**
  - Before submitting, check that every `C-` run dir has `COMPLETE.json`: expect 90.
  - Then upload `cua` and `tests` and `qsub cua_code/cua/ccast/test_eval.pbs`.
  - If fewer than 90 are complete, stop and report which are missing. The test set is opened only after all settings and runs are final.
- [ ] **Step 2: Wait for the job, then check it (one login).**
  - The log ends with every run `evaluated` or `already evaluated` and none `refused`.
  - Run `python -m cua.analysis.part_c --runs-root $P/cua_runs --data-root $P/cua_data --deltas <Part A deltas.csv> --out $P/cua_reports/part_c_<date>`.
  - Stream the output dir back with tar | base64, as for Part A.
- [ ] **Step 3: Copy the results into `reports/cua_part_c_report/` locally and commit them.**
