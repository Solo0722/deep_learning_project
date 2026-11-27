# Methodology brief for the chat (2026-11-16)

Use this to get the chat to write the methodology sections of the paper. Part 1 lists the files to
upload. Part 2 is the message to paste. Part 3 is what changed since `PAPER_CONTEXT.md` was written,
and it is already included in the message.

## 1. Files to upload

Required:

| File (repo path) | Why |
|---|---|
| `research_paper/PAPER_CONTEXT.md` | Rules, terminology, method and setup summary, citation map |
| `research_paper/main.tex` | Edit target; Sections III and V are outdated |
| `research_paper/references.bib` | The only citation keys allowed |
| `docs/superpowers/specs/2026-11-08-class-utility-augmentation-design.md` | Authoritative study design (Sec. 3 to 12) |

Helpful (the exact settings live in the code):

| File | What it pins down |
|---|---|
| `cua/augment/menu.py`, `cua/augment/base.py`, `cua/augment/mosaic.py` | Edit strengths, base recipe, mosaic rule |
| `cua/datasets/rdd.py`, `voc.py`, `dacl10k.py`, `cua/datasets/DACL10K_FORMAT.md` | How each dataset becomes boxes and splits |
| `cua/train.py`, `cua/models.py` | Optimiser, LR schedule, detector settings |
| `cua/analysis/part_a.py` | Interaction test and Holm correction exactly as computed |
| `reports/cua_part_a_report/deltas.csv`, `tests.json`, `heatmap_rdd_fasterrcnn.png` | First measurement results (RDD, Faster R-CNN) |

Do not upload: `METHOD_PROTOCOL.md`, `figures/methodology_framework.pdf`, anything `rdd_v2_*`. They describe
the old controller.

## 2. Message to paste

> Read `PAPER_CONTEXT.md` first and follow §0 on every reply. Task: write the methodology part of the
> paper as drop-in LaTeX for `main.tex`, replacing the current Section III (it describes the old
> overlap-group controller) and Section V, and adding the new Section IV:
>
> - **III. Class-Utility Augmentation**: problem setup and notation; base recipe and augmentation menu;
>   the class-utility estimator (equation for u(o,c), EMA and standard error, reference set, when it
>   starts); the image policy (score over the classes present, identity option, harm block, sampling);
>   the shared-utility variant; cost. Include Fig. 2 (framework diagram) as TikZ, or a `\todo` figure box plus
>   a matplotlib script.
> - **IV. Measuring Class-Specific Augmentation Effects**: the edit-wise measurement study (warm
>   checkpoint, short continuation, per-class ΔAP50, paired seeds), the interaction test and Holm-corrected
>   per-edit contrasts, the short-vs-full-run proxy check, and the class-targeted study (direct vs
>   spill-over, seam control).
> - **V. Experimental Setup**: datasets and splits (Table II), data gates and reference set, detectors and
>   training recipe, augmentation menu with validity per dataset (Table III), compared policies, metrics
>   and statistics, compute.
>
> Use the spec for every number. The spec and the updates below override `PAPER_CONTEXT.md` where they
> differ. Anything still undecided goes in as `\todo{}`, never a guess. Write in paper voice: no
> "Part A/B/C", "arm", "H1", "spec" or "CCAST". Results go in Section VI later; here, only describe how
> they are produced.
>
> **Updates since PAPER_CONTEXT.md (2026-11-09):**
>
> 1. *dacl10k source.* The official downloads are offline, so we use the public Hugging Face copy
>    `Voxel51/dacl10k` (FiftyOne export, 8,922 images, CC BY-NC 4.0). Its official test images have no
>    labels and are skipped (so they never count as damage-free images). Official splits are not used:
>    labelled images are grouped by near-duplicate hashing and re-split 70/15/15 by group. Each annotated
>    polygon becomes one box (min/max over all its rings), clipped to the image; degenerate polygons are
>    dropped. Only damage classes are kept (component classes such as bearings and joints are dropped).
>    The class gate (≥ 300 training boxes, not region-like) uses the training split only. Final class list
>    and counts: `\todo{}` (data preparation runs now). Cite the dataset as `flotzinger2024dacl10k`; a
>    citation for the mirror is `\todo{cite}`.
> 2. *RDD epoch budget.* The 20-epoch base-recipe run reached validation AP50 51.05 (curve 37.2 → 51.2,
>    peak at epoch 18). It did not meet the plateau rule (gain over the last 20% of epochs ≥ 0.3), so the
>    final budget is still `\todo{20 or 30 epochs}`.
> 3. *Edit-wise study status.* RDD × Faster R-CNN is finished for 3 seeds and 12 edits. Describe the
>    protocol only; do not write results here.
> 4. *Open decisions, keep as `\todo{}`:* the second detector family (spec says RetinaNet; a switch to
>    YOLOv8 is under discussion); an extra ablation that combines tolerance and benefit; the dacl10k
>    budget. The class-utility estimator and policy are designed but not yet run, so describe them as the
>    method without implying they have been tested.
>
> After the LaTeX, list the open `\todo`s and any claim you were unsure about (§0 rule 12).

## 3. Numbers available for the results section later (not for methodology)

Edit-wise study, RDD2022, Faster R-CNN, 3 seeds, mean per-class ΔAP50 (edit − base, validation):

| Edit | D00 | D10 | D20 | D40 |
|---|---|---|---|---|
| blur | -0.67 | -1.20 | -1.25 | -1.98 |
| brightness | -0.15 | -0.46 | -0.26 | +0.20 |
| contrast | -0.04 | -0.46 | -0.23 | -0.45 |
| erase | -0.24 | -0.36 | -0.20 | -0.19 |
| hue | -0.07 | -0.35 | -0.37 | -0.07 |
| jpeg | -1.03 | -0.66 | -0.49 | -0.51 |
| mosaic | -2.37 | -3.74 | -3.61 | -4.48 |
| noise | -0.83 | -0.66 | -0.35 | -0.22 |
| rotate | -0.58 | -1.63 | -0.27 | -0.52 |
| saturation | +0.14 | -0.17 | -0.05 | +0.07 |
| sharpen | +0.45 | +0.30 | -0.30 | +0.09 |
| translate | -0.18 | -1.11 | -0.48 | -0.00 |

Edit × class interaction: F(33, 88) = 3.73, p = 4.5e-7. No single edit's class contrast survives Holm
correction (smallest: blur p_Holm = 0.10, mosaic p_Holm = 0.18). Most edits are mildly negative in the
short run; the proxy check (short vs full runs) has not been read yet, so treat the size of these
effects, especially mosaic, as provisional.
