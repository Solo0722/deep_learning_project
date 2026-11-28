# Realistic Evaluation of Class-Aware Augmentation for Object Detection

Code, protocols and per-run results for a controlled study of whether augmentation effects in object detection
depend on the class, and whether a class-aware policy can turn that dependence into accuracy.

- **Paper:** [`research_paper/main_evaluation_paper.tex`](research_paper/main_evaluation_paper.tex)
  ([PDF](research_paper/main_evaluation_paper.pdf))
- **Results of every run:** [`results/`](results/README.md), with a map from each table and figure to its script
- **Datasets:** RDD2022 (road damage), dacl10k (bridge damage), PASCAL VOC 2007+2012
- **Detectors:** Faster R-CNN and RetinaNet, ResNet-50 FPN, COCO-pretrained (torchvision)

## Findings

| Question | Answer |
|---|---|
| Does an edit's effect depend on the class? | Yes, in four of six dataset–detector pairs (edit × class interaction F-test), but the class-specific part is about 1–3 AP50 per class, close to seed noise. |
| Do per-class choices replicate on new seeds? | Mostly no. Choosing each class's edit on two seeds adds at most 0.09 AP50 on the third over one edit for all classes. |
| Does a class-aware policy beat its class-blind twin? | No method does after Holm correction: online class-utility augmentation (CUA, two estimator versions), a policy from measured effects, per-class population-based training, or class-balanced and class-adaptive copy-paste. |
| Do the online utility estimates track the measured effects? | No. They favour edits that add objects to the batch. |
| Stronger or object-local edits? | Stronger edits make the class differences replicate, but every strong edit lowers AP50 and per-class choice still loses to no edit. Object-local edits do not help either. |
| Can the machinery find a class-specific effect when one exists? | A planted harmful edit is detected by both estimators in every run, and CUA-v2 stops applying it to the affected class only. The planted harm was too small to show an accuracy gain. |

With three seeds, the policy comparisons rule out gains larger than 0.4–0.9 AP50 on five of the six pairs.
Uniform sampling from a moderate menu remains a strong default.

## Repository layout

| Path | Content |
|---|---|
| `cua/` | Training, augmentation menu, policies (uniform, measured, shared, CUA v1/v2, table/PBT), copy-paste, estimator, evaluation |
| `cua/experiments/specs.py` | Every run id and its configuration (`resolve(run_id)`); run ids encode family, dataset, detector, seed and arm |
| `cua/analysis/` | Part A interaction tests, Part C comparisons, positive control, object-local and strength pilots, AP50 and paired bootstrap |
| `cua/ccast/` | PBS job generation and scripts for the CCAST cluster |
| `reports/cua_*` | Read-outs and the pre-registered protocols (`PROTOCOL.md`) of the pilots and the positive control |
| `reports/cua_stats/` | Holm correction, minimum detectable effects, cross-fitted selection gain |
| `results/` | Config, training history and test metrics of all 521 runs |
| `research_paper/` | Paper source, figures and figure scripts |
| `tests/cua/` | Unit and integration tests (`pytest`) |

## Setup

Python 3.12, PyTorch 2.3 with torchvision 0.18 (CUDA 12.1 for training).

```bash
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r cua/requirements-test.txt
python -m pytest            # tests/cua; add -m "not slow" to skip the slow ones
```

## Reproduce

Prepare a dataset (the images are not in this repository; get them from the original sources):

```bash
python -m cua.prepare rdd --splits <split dir> --images <image root> --out <data root>/rdd
python -m cua.prepare voc --devkit <VOCdevkit> --out <data root>/voc
python -m cua.prepare dacl10k --root <dacl10k root> --out <data root>/dacl10k
```

Train one run by its id, for example the CUA-v2 arm on PASCAL VOC with RetinaNet, seed 0:

```bash
python -m cua.run --spec C-voc-retinanet-s0-cua2 --data-root <data root> --runs-root <runs root>
python -m cua.test_eval --runs-root <runs root> --data-root <data root> --prefix C-
```

Re-run the analyses from the released results without any images or GPU, for example:

```bash
python -m cua.analysis.strength --runs-root results/cua_runs --out <dir>
python reports/cua_stats/crossfit.py reports/cua_part_a_report/deltas.csv
```

`results/README.md` lists the command behind every table and figure.

## Licences of the data

RDD2022: Arya et al., Geoscience Data Journal, 2024, [doi:10.1002/gdj3.260](https://doi.org/10.1002/gdj3.260).
dacl10k: CC BY-NC 4.0. PASCAL VOC: Everingham et al. Images are not redistributed here.

## Earlier work in this repository

The first phase of the project trained a from-scratch CNN on RDD2022 defect crops with a class-adaptive
augmentation controller. Its scripts (`make_splits.py`, `materialize_crops.py`, `train_baseline.py`,
`summarize_baseline.py`), runbooks (`BASELINE_RUNBOOK.md`, `DATA_AUDIT.md`), frozen splits
(`dataset_splits_v1/`) and saved baseline (`reports/conventional_seed42_v1/`) are kept for reference. The
detection study above supersedes it.
