# Results behind "Realistic Evaluation of Class-Aware Augmentation for Object Detection"

Paper source: `research_paper/main_evaluation_paper.tex` (PDF next to it).

## What is here

`cua_runs/<run id>/` holds, for every training run in the paper:

| File | Content |
|---|---|
| `config.json` | Full training configuration and its digest (the run id resolves to it via `cua.experiments.specs.resolve`) |
| `history.jsonl` | One line per epoch: losses, validation per-class AP50, epoch time, and the policy's edit counts |
| `test_metrics.json` | Test macro and per-class AP50 of the best-validation checkpoint (runs scored on test only) |
| `COMPLETE.json` | Written when the run finished |

`reports_ccast/` holds the read-outs of the two pilots and the positive control as written on the cluster
(`cua_control/control.json`, `cua_strength/strength.json`, `cua_local/local.json`).

Not included: checkpoints, test predictions and the per-estimate utility logs (`utility.jsonl`, about 300 MB),
which are too large for Git, and the dataset images (RDD2022, dacl10k, PASCAL VOC), which are available from
their original sources under their own licences.

## Run families

| Prefix | Runs | Paper | Read-out |
|---|---|---|---|
| `A-` | 258 | Sec. 6.1, edit-wise study (3-epoch continuations from a warm start) | `python -m cua.analysis.part_a --runs-root results/cua_runs --out <dir>` → `reports/cua_part_a_report/` |
| `budget-`, `proxy-` | 8 | Sec. 3, budget check; Sec. 6.1, short vs full training | `cua.analysis.part_a --budget` |
| `C-` | 126 | Sec. 6.2–6.3, policy comparisons (uniform, measured, shared, CUA, v1 and v2) | `python -m cua.analysis.part_c ...` → `reports/cua_part_c_v2_report/` |
| `P-` | 21 | Sec. 6.2, copy-paste | test metrics; `reports/cua_paste_pilot/VAL_SUMMARY.md` |
| `X-` | 18 | Sec. 6.3, ablations (validation only) | `cua.analysis.part_c` |
| `T-` | 6 | Appendix B, estimator settings | validation AP50 in `history.jsonl` |
| `K-` | 15 | Sec. 6.4, positive control | `python -m cua.analysis.control ...` → `reports/cua_control/CONTROL.md` |
| `L-` | 24 | Sec. 6.1, object-local pilot | `python -m cua.analysis.local ...` → `reports/cua_local/LOCAL.md` |
| `S-` | 45 | Sec. 6.1, strength pilot | `python -m cua.analysis.strength ...` → `reports/cua_strength/STRENGTH.md` |

The per-class population-based search is in `reports/cua_pbt_pilot/`.

## Tables and figures

| Paper item | Script | Input |
|---|---|---|
| Interaction F-tests, mean effects, heatmaps | `cua/analysis/part_a.py` | `A-` runs |
| Cross-fitted selection gain | `reports/cua_stats/crossfit.py reports/cua_part_a_report/deltas.csv` | Part A deltas |
| Main test table, paired comparisons | `cua/analysis/part_c.py` | `C-` runs |
| Holm correction, minimum detectable effects | `reports/cua_stats/holm_mde.py` | `reports/cua_part_c_v2_report/results.json` and the `P-` test metrics |
| Forest plot, estimator agreement, teaser | `research_paper/figures/make_eval_figures.py` | `reports/cua_stats/holm_mde.json`, Part C results, Part A deltas |
| Augmentation gallery, copy-paste examples | `research_paper/figures/make_assets.py`, `build_real_figures.py` | dataset images (not included) |

The pre-registered protocols of the pilots and the positive control are `reports/cua_strength/PROTOCOL.md`,
`reports/cua_local/PROTOCOL.md` and `reports/cua_control/PROTOCOL.md`. The positive-control protocol records
one correction to the O2 statistic made after the runs; the paper reports it.
