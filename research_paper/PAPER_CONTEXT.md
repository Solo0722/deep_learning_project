# PAPER CONTEXT — Class-Utility Augmentation (read this first)

This file is the single entry point for writing the paper. It summarises the study, says how the paper must be written, and points to the other files in this Project for detail. Files are referred to by **file name** (their repository path is given once in §12).

Last updated: 2026-11-09. Experiment status: **no results yet** (code built and tested; cluster runs not started).

---

## 0. Instructions for Claude (follow on every request)

You are co-writing a research paper with the author, a graduate student in CE 738 *Deep Learning for Engineers* (Drs. Aboah and Le). Your job is to produce text that reads as a **finished, peer-reviewable IEEE conference paper**, not as notes, a protocol, or a project report.

**Voice and style**

1. Write like a published computer-vision paper (CVPR/ICCV/WACV/IEEE Access style). Confident, precise, third person or "we". Short-to-medium sentences. One message per paragraph, stated in the paragraph's first sentence.
2. Lead with the *idea and why it works*; put mechanics (step counts, thresholds, sampling constants) in the Method and Experimental Setup sections, never in the Introduction.
3. Do **not** narrate the study design ("this paper first tests whether…", "the design makes either outcome informative", "we pre-declared…"). State what the paper does and what it finds.
4. Do **not** use internal labels in prose: no "H1–H4", "Part A/B/C", "arm", "measured-table", "uniform-menu", "v1/v2 controller", "spec", "ruling", "smoke run", "CCAST". Use paper names instead (see §5 naming table).
5. No parenthetical lists stuffed into sentences; no bullet lists in the Introduction except the final contribution list.
6. Use stable terminology (§5). Define a term once, then reuse it exactly.
7. Avoid hype words ("novel", "significantly" without a test, "state-of-the-art", "first"). Never claim to be the first class-adaptive, first online, first benefit-based, or first object-aware augmentation method — all have prior art (§10).

**Evidence rules**

8. Never invent numbers. Until a result is supplied, write it as `\todo{value}` or `\todo{describe result}` and phrase the surrounding sentence so it can be filled in without rewriting.
9. Every claim about another paper must match the "supports" column of the citation map (§10). Cite only keys that exist in `references.bib`. If a new reference is needed, write `\todo{cite: description}` and list it at the end of your reply — do not invent BibTeX.
10. Claims about our method's performance must be conditional or placeholder until results exist; once results arrive, claims must match them exactly, including confidence intervals.

**Output format**

11. Return complete, compilable LaTeX for the section(s) requested, as a drop-in replacement for that part of `main.tex` (IEEEtran conference class; packages already loaded: cite, amsmath, amssymb, graphicx, booktabs, tabularx, xcolor, hyperref; `\todo{}` macro exists).
12. After the LaTeX, give: (a) a 3–5 line list of open `\todo`s, (b) any claim you were unsure about.
13. When the author uploads new results (§11), update every affected section — abstract, contributions, results, discussion — not only the results section.
14. The paper must include the figures and tables listed in §4a, including a method architecture diagram (Fig. 2). When drafting a section, add its figures/tables at the same time.

---

## 1. The paper in one paragraph

Object detectors are trained with data augmentation, and almost every pipeline applies the same augmentation distribution to every image. But detection images usually contain several object classes, and a whole-image edit reaches all of them at once. Classes plausibly differ in which edits help them learn: colour jitter changes the very cue that defines rust while leaving a crack's shape intact; blur can erase a hairline crack but barely change a pothole. Existing adaptive methods either check whether the model still *recognises* an augmented example (tolerance: CTAugment, CUDA), or check whether training on augmented data lowers *one pooled* held-out loss (benefit: BDA, DDAS, MADAug); class-conditioned methods are built for single-label classification. We propose **Class-Utility Augmentation (CUA)**: during training, it estimates for every candidate edit and every class how much one training step on the edited image would reduce that class's loss on a held-out reference set (a first-order, gradient-alignment estimate), and for each training image picks one edit by combining these per-class benefits over the classes present — with "no edit" always available and edits reliably harmful to any present class blocked. We first measure whether classes actually respond differently to the same edits, across road damage (RDD2022), bridge damage (dacl10k) and everyday objects (PASCAL VOC) with two detector families, and then show whether CUA turns that difference into accuracy, against a control that is identical except that it pools benefit across classes.

## 2. Title

The title must say, in plain words, what the paper proposes: **choosing the augmentation for each training image according to which object classes it contains and how much each augmentation helps those classes learn.** Avoid jargon such as "utility" in the title; a reader should understand it without reading the paper.

Working title:

> **Class-Aware Augmentation Selection for Multi-Class Object Detection**

Alternatives the author may choose instead:
- "Choosing Data Augmentation per Object Class for Multi-Class Object Detection"
- "Not All Classes Need the Same Augmentation: Class-Aware Augmentation Selection for Object Detection"

The title in `main.tex` is outdated and should be replaced with the working title (or the author's choice). Inside the paper, the method is still called Class-Utility Augmentation (CUA), and "class utility" must be defined in plain words the first time it appears: *the estimated amount an augmentation helps a given class learn*.

## 3. Contributions (write them as claims; fill evidence when results arrive)

1. **Method — Class-Utility Augmentation.** An online augmentation policy for multi-class detection that scores each candidate edit by its estimated training benefit for *each* class and combines these scores over the classes present in an image, with an explicit no-edit option and a per-class harm block. It needs no policy network and no separate search phase.
2. **Measurement — class-specific augmentation effects in detection.** A controlled study of how individual augmentations change per-class detection accuracy across three domains and two detector families, with paired seeds and image-level bootstrap intervals, plus an experiment that applies an edit only to one class's objects to separate its direct effect from spill-over to other classes.
3. **Evaluation that isolates class information.** CUA is compared with a shared-utility counterpart that uses the same estimator and sampling but pools benefit across classes, so any difference is attributable to class information alone; the online estimates are also checked against the measured per-class effects.

**What makes the paper strong (if results support it):** (i) a clear example of an edit that helps one class and hurts another; (ii) CUA beats the shared-utility control on test macro AP50 with intervals excluding zero on at least two of three datasets; (iii) the online estimates rank-agree with measured effects, showing the method works for the stated reason; (iv) estimator overhead ≤ 20% of training time.

**Weak points a reviewer will raise — address them in the text:** every ingredient has precedent (state this openly in Related Work; novelty is the per-class combination for multi-class images); gains may be small (report intervals, not just means); no learned-policy baseline for detection (justify: the shared-utility control is the fair comparison; list as limitation); edit strength is fixed (limitation).

**If results are negative:** the paper becomes "classes do / do not need different augmentations in detection", with the measurement study as the main contribution. See outcome rules in §8.

## 4. Paper outline and status

| # | Section | Must contain | Status in `main.tex` |
|---|---|---|---|
| — | Title, abstract, keywords | Method-first title; abstract written last (problem → idea → evaluation → result) | Title outdated; abstract empty |
| I | Introduction | Multi-class images + whole-image edits; classes may need different edits (examples); tolerance ≠ benefit, pooled ≠ per-class; CUA idea in 3–4 sentences; evaluation in 1–2 sentences; contributions list | Drafted 2026-11-09 but reads like protocol notes — **rewrite in paper voice** |
| II | Related Work | Damage-detection augmentation; shared/detection policies; class-aware policies (incl. PSIS for detection); online tolerance vs benefit; gradient-alignment influence (TracIn); positioning + Table I | Drafted; content correct, prose needs paper voice; Table I is good |
| III | Class-Utility Augmentation (Method) | Problem setup and notation; benefit estimator (Eq.); image policy (score, harm block, sampling); shared-utility variant; cost; framework figure | **Outdated** (describes the old local overlap-group controller) — rewrite from §6 |
| IV | Measuring Class-Specific Augmentation Effects | One-edit-at-a-time protocol; one-class-at-a-time protocol; interaction test; proxy check (short vs full runs) | Not written |
| V | Experimental Setup | Datasets + gates; detectors and recipe; edit menu table; baselines; metrics and statistics; compute | **Outdated** — rewrite from §7 |
| VI | Results | Measurement results (heatmap of per-class ΔAP50); main comparison table; ablations; estimator agreement; overhead | Not written — placeholders until results |
| VII | Discussion and Limitations | When class information helps; cost; limitations (box labels, one strength per edit, no copy-paste, three domains) | Not written |
| VIII | Conclusion | 1 paragraph | Not written |

## 4a. Figures and tables (the paper must include these)

A paper without strong figures will not read as finished. Every figure must be referenced in the text, have a caption that explains what to look at (not just what it is), and be readable in black and white at single-column width (3.5 in) unless marked full width. `main.tex` does not yet load TikZ; add `\usepackage{tikz}` (and needed libraries) if you draw diagrams in TikZ.

| # | Figure | Where | What it shows | How to produce | Status |
|---|---|---|---|---|---|
| Fig. 1 | Teaser / motivation | Introduction, page 1 | One real multi-class image (e.g. RDD crack + pothole, or dacl10k rust + crack) shown unedited and under two edits (e.g. colour jitter, blur); beside it, the per-class effect of each edit, showing an edit that helps one class and hurts the other | Image panels from the datasets + small bar chart; values from the edit-wise measurement study | Needs results; draft layout now with `\todo` values |
| Fig. 2 | **Method architecture (framework diagram)** | Method section, full width (`figure*`) | The training loop: training batch → base recipe → per-image policy picks one edit → detector update. Side branch every K steps: estimation batch with each edit → gradients g_o, g₀; reference set split by class → class gradients h_c; their alignment → class-utility table ū(o, c) (edits × classes); table + the image's class set S → score, harm block, sampling → chosen edit. Show the identity ("no edit") option explicitly and contrast the shared-utility variant (one pooled row) in a small inset | Vector diagram: TikZ in LaTeX, or a Python/matplotlib script saved as `figures/cua_framework.pdf` (replaces the outdated `methodology_framework.pdf`) | Can be made now |
| Fig. 3 | Edit × class effect heatmap | Measurement results | Per-class ΔAP50 (edit − base) for every edit and class, one panel per dataset (Faster R-CNN; RetinaNet in supplement or second row); cells marked where the interval excludes zero; diverging colour map centred on 0 | `cua.analysis.part_a` already writes the heatmap; restyle for print | Needs results |
| Fig. 4 | Direct vs spill-over | Measurement results | For the class-targeted study: effect on the edited class vs on the other classes, per edit | Matrix or grouped bars | Needs results |
| Fig. 5 | Class utility during training | Method analysis / Results | ū(o, c) over training steps for a few edits and classes, showing that preferences differ by class and change over time | Line plot from estimator logs | Needs results |
| Fig. 6 | Estimator agreement | Results | Scatter of time-averaged ū(o, c) vs measured ΔAP50(o, c), one point per (edit, class), with Spearman ρ and CI | Scatter plot | Needs results |
| Fig. 7 | Edit choice by class set | Results | How often CUA picks each edit (including no edit) for images with different class sets, compared with the shared-utility policy | Stacked bars | Needs results |
| Fig. 8 | Qualitative detections | Results / Discussion | Same test images, base recipe vs CUA, with a class that improved and one failure case | Image grid | Needs results |

Tables:

| # | Table | Where | Status |
|---|---|---|---|
| Table I | Related methods by feedback signal and class conditioning | Related Work | Exists in `main.tex` |
| Table II | Datasets: images, classes, boxes, multi-class image rate, splits | Setup | Partly fillable now; dacl10k counts after data gates |
| Table III | Augmentation menu with strengths and validity per dataset | Setup | Can be made now (§7) |
| Table IV | Main comparison: test macro AP50 (mean ± s.d., CI vs shared-utility) for five policies × 3 datasets × 2 detectors | Results | Needs results |
| Table V | Ablations | Results | Needs results |
| Table VI | Cost: GPU hours and estimator overhead per policy | Results | Needs results |

When asked for a figure, give either (a) TikZ code that compiles in the IEEEtran document, or (b) a self-contained Python/matplotlib script that reads the named result file and writes a PDF into `figures/`, plus the LaTeX `figure` environment that includes it. Until data exist, include the figure environment with a `\todo{}` placeholder box so the layout and references are in place.

## 5. Terminology (use exactly these names in the paper)

| Concept | Paper name | Never write |
|---|---|---|
| The proposed method | Class-Utility Augmentation (CUA) | class-adaptive controller, Part C |
| Pooled-benefit control | shared-utility augmentation | shared arm |
| One edit sampled uniformly from {no edit} ∪ menu | uniform augmentation (TrivialAugment-style) | uniform-menu arm |
| Fixed policy from measured per-class effects | measured-effect policy | measured-table arm |
| Flip + scale jitter only | base recipe | base arm |
| A single transformation (blur, mosaic, …) | edit / augmentation operation | op |
| Set of candidate edits | augmentation menu | — |
| Estimated benefit of edit o for class c | class utility u(o, c) | benefit score, tolerance |
| Held-out images used only by the estimator | reference set | probe set |
| One-edit-at-a-time study | edit-wise measurement study | Part A |
| One-class-at-a-time study | class-targeted study | Part B |
| Old retention-gated controller (ablation) | tolerance-gated controller | v2 controller |

## 6. Method (source: spec §9 in `2026-11-08-class-utility-augmentation-design.md`)

**Setup.** Detector with parameters θ, classes 𝒞, augmentation menu 𝒪 (valid edits for the dataset) plus the identity edit o₀ ("no edit"). Every training image first receives the base recipe (horizontal flip p = 0.5, isotropic scale jitter 0.8–1.2 on a 640×640 letterbox canvas, rejecting draws that crop an annotated object). CUA then chooses at most one additional edit per image.

**Benefit estimator.** Let θ_s be a parameter subset (Faster R-CNN: ROI box head + top FPN level; RetinaNet: classification and regression subnets). Every K = 200 optimiser steps:
1. Draw an estimation batch B of 8 training images. Compute g₀ = ∇_{θ_s} L(B) and, for each edit o, g_o = ∇_{θ_s} L(T_o(B)), with the same base-geometry draw. L is the full detector training loss.
2. For each class c, compute h_c = ∇_{θ_s} L_c(R_c), where R_c is a 64-image subsample of the reference set containing c, and L_c is the classification + box-regression loss restricted to proposals/anchors matched to class-c ground truth.
3. Class utility (first-order predicted extra reduction in class-c reference loss from one SGD step on the edited instead of the unedited batch, normalised by current loss):

   u(o, c) = η · ⟨g_o − g₀, h_c⟩ / (L_c(R_c) + δ), δ = 0.01, η = current learning rate.

4. Smooth with an exponential moving average ū(o,c) (β = 0.9) and track a running standard error se(o,c).

Estimation starts after one epoch of the base recipe; before that the policy is uniform over menu ∪ {identity}. Overhead target ≤ 20% of base training time.

**Image policy.** For an image with class set S:
- Candidates: identity (utility 0 by definition) and all valid edits.
- Score: score(o) = (1/|S|) Σ_{c∈S} w_c ū(o,c), w_c = 1 (inverse-AP weights are an ablation).
- Harm block: remove o if for any c ∈ S, ū(o,c) + 1.64·se(o,c) < 0 (reliably harmful at one-sided 95%).
- Sampling: p(o) = (1 − ε)·softmax(score/τ) + ε/|allowed|, τ = std of current scores (floor 1e-6), ε = 0.1.
- Images without objects: identity or a uniformly sampled edit (p = 0.5), identical in every compared policy.
- Mosaic: chosen per anchor image using that image's S; the other three tiles are drawn uniformly.

**Shared-utility variant.** Same machinery, but ū_shared(o) = mean_c ū(o,c) and the harm block uses the pooled estimate; every image gets the same preferences. The *only* difference from CUA is class information.

**Tuned once, then frozen** (validation, RDD × Faster R-CNN, one seed): K ∈ {100, 200, 400}; θ_s ∈ {head, head + top FPN}; ε ∈ {0.05, 0.1, 0.2}.

**Intuition to convey in the paper:** ⟨g_o − g₀, h_c⟩ asks "does learning from the edited image push the weights further in the direction that lowers class c's held-out loss than learning from the plain image does?" Positive → the edit teaches c something useful now; negative → it pulls c's learning the wrong way. This is training benefit, not recognisability.

## 7. Experimental setup (source: spec §3–§12; implementation `cua/` and `README.md` in `cua/ccast`)

**Datasets.**

| Domain | Dataset | Classes | Splits |
|---|---|---|---|
| Road damage | RDD2022 | D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole | Frozen group-disjoint split: 26,863 train / 5,760 val / 5,760 test; 35% of damage-bearing train/val images contain ≥ 2 classes |
| Bridge damage | dacl10k (9,920 images, > 100 bridges) | Damage classes only, polygons → one box per connected component; final class list after data gates `\todo{}` | Official splits if test labels available, else group split 70/15/15 with near-duplicate screening |
| General objects | PASCAL VOC | 20 classes | 07+12 trainval → train/val (10% held out as val); VOC07 test → test |

Data gates (run before training): co-occurrence ≥ 20% of object-bearing training images contain ≥ 2 classes; dacl10k classes with < 300 training boxes, or > 50% of boxes covering > 50% of the image, are excluded and listed; near-duplicate leakage screen.

Reference set: ~1,000 group-disjoint training images per dataset, ≥ 100 objects per class where possible, removed from training in **every** compared policy.

Data roles: train (minus reference) → optimisation; reference → CUA/shared-utility estimates only; validation → measurement study, tuning, checkpoint selection; test → final comparison, opened once after all settings are frozen.

**Detectors and recipe.** Faster R-CNN ResNet-50-FPN and RetinaNet ResNet-50-FPN (torchvision, COCO-pretrained). 640×640 letterbox (pad 114); SGD momentum 0.9, weight decay 5e-4, base LR 0.0025, batch 4, linear warm-up over 1,000 iterations from 0.001×; LR ×0.1 at 60% and ×0.01 at 80% of epochs; mixed precision. Epoch budget fixed per dataset from a base-recipe run (candidates RDD 20, VOC 20, dacl10k 30; accepted if val AP50 gain over the last 20% of epochs < 0.3, else +50%). Seeds {0, 1, 2}, shared across compared policies.

**Augmentation menu (one moderate strength each).**

| Family | Edit | Strength |
|---|---|---|
| Geometric | Translation | ±10% of canvas |
| Geometric | Rotation | ±10°, boxes = rotated-corner hull |
| Geometric | Vertical flip | dacl10k only |
| Colour | Brightness / Contrast | ×[0.7, 1.3] |
| Colour | Saturation | ×[0.5, 1.5] |
| Colour | Hue | ±0.05 |
| Image quality | Gaussian blur | σ ∈ [0.5, 1.5] px |
| Image quality | Gaussian noise | σ = 0.03 |
| Image quality | JPEG | quality ∈ [30, 60] |
| Image quality | Sharpening | factor ∈ [1.5, 2.5] |
| Occlusion | Random erasing | 2–10% of canvas, ≤ 50% of any box |
| Composition | Mosaic | 2×2, objects < 30% visible dropped |

Vertical flip excluded for RDD (vehicle-camera view; rotation would turn D00 into D10) and VOC (upside-down scenes). Copy-paste out of scope.

**Edit-wise measurement study.** Shared warm checkpoint per dataset × detector × seed (base recipe, 3 epochs); then base and base + each valid edit (p = 0.5, whole image) continue 3 epochs (LR ×0.1 in the last). Output: per-class ΔAP50 (edit − base) on validation, per seed → edit × class effect table. 3 datasets × 2 detectors × 3 seeds (258 runs). Proxy check: RDD × Faster R-CNN, 4 edits spanning the ranking, full schedule, 1 seed; Spearman agreement short vs full run.

**Class-targeted study.** RDD and dacl10k, Faster R-CNN, 3 seeds; the 3 edits with the largest class disagreement; the edit is rendered only inside class-c boxes (union mask, 8-px feather) with p = 0.5; output: direct vs spill-over matrix. One seam-control run applies the edit to a random non-object region of equal area. Geometric edits excluded.

**Main comparison.** Base recipe; uniform augmentation; measured-effect policy (same scoring/harm/sampling rule with ū replaced by measured ΔAP50); shared-utility; CUA. 3 datasets × 2 detectors × 3 seeds, full schedule, checkpoint = best validation macro AP50, evaluated once on test.

**Ablations** (validation; RDD and dacl10k; Faster R-CNN; 3 seeds): no harm block; no identity option; inverse-AP class weights; tolerance-gated controller (the earlier retention-based design) as a "tolerance instead of benefit" baseline.

**Metrics and statistics.** Primary: test macro AP50 (COCO evaluation, IoU 0.5, mean over classes with ≥ 1 ground-truth object). Secondary: AP50:95, per-class AP50, AP small/medium/large, false-positive images on object-free RDD images, estimator overhead. Mean ± s.d. over paired seeds; 95% CI from a paired image-level bootstrap (1,000 resamples) and a paired-seed t interval — a claim requires both to exclude 0. Interaction test: two-way analysis of per-seed ΔAP50 with an edit × class term, blocking on seed × class; per-edit class contrasts with Holm correction. Estimator agreement: Spearman ρ between time-averaged ū(o,c) and measured ΔAP50(o,c), bootstrap CI over cells.

**Compute.** ~840 A10 GPU-hours planned (measurement ~160, proxy ~30, class-targeted ~60, tuning ~40, main comparison ~460, ablations ~60, gates/budget ~30). Report measured numbers once available.

## 8. Hypotheses and how to write each outcome

Internal labels (do not use in prose): H1 classes differ for at least one edit; H2 CUA > shared-utility on test macro AP50 (primary claim); H3 CUA > best of {base, uniform}; H4 estimates rank-agree with measured effects.

| Result | What the paper says |
|---|---|
| Classes differ, CUA > shared | Class-dependent benefit exists in detection and CUA exploits it. |
| Classes differ, CUA ≈ shared | Classes differ, but online estimation does not exploit it; use estimator agreement to say whether the estimator or the policy is the bottleneck. |
| Classes do not differ | In these domains, detection classes do not need different augmentations; CUA ≈ shared-utility is the expected consistency check; the measurement study is the main contribution. |
| CUA ≤ base or uniform | Report plainly; the method is not recommended over simple recipes. |

## 9. Pilot evidence (historical; one seed — use only as motivation or in Discussion)

An earlier controller (single seed, RDD2022, Faster R-CNN, 20 epochs) gated edit strength per class by detection retention ("tolerance") and applied edits locally to groups of overlapping boxes. It did not improve accuracy: validation AP50:95 22.56 vs 23.06 for its shared counterpart and 22.82 for no augmentation; AP50 50.18 vs 51.03 and 51.14. One seed cannot separate these differences from noise. This pilot motivated three design changes now in CUA: score benefit rather than tolerance, always allow "no edit", and include edits known to help detection. Source: `RESULTS.md` (from `reports/fasterrcnn_class_adaptive_v1/`). Present it as a pilot study, briefly, never as a main result.

## 10. Citation map (only these keys exist in `references.bib`)

| Key | Supports | Caution |
|---|---|---|
| arya2024rdd | RDD2022 dataset | — |
| flotzinger2024dacl10k | dacl10k bridge-damage dataset | — |
| everingham2010voc | PASCAL VOC | — |
| ren2017faster, lin2017focal | Faster R-CNN, RetinaNet | — |
| shang2024automated, zheng2024deep, vision2025inspection | Automated pavement/infrastructure inspection with deep learning (broad setting only) | No detailed findings |
| tsuchiya2019 | Road-damage augmentation methods compared by class-level effect | — |
| aghayan2024 | Box-level augmentation and GAN synthesis for road damage | — |
| rdd_detectron2 | Per-damage-type/country augmentation probabilities set before training | — |
| rdd_cutpaste | Content- and perspective-aware cut-and-paste for road damage | — |
| kirichenko2023 | Augmentation can help average ImageNet accuracy while hurting individual classes | Abstract-level only; no numbers |
| cubuk2019auto, cubuk2020rand, muller2021trivial | AutoAugment, RandAugment, TrivialAugment (shared policies) | — |
| zoph2020det | Learned detection policies incl. box-only operations | — |
| chen2021scale | Scale-aware image- and box-level policy search for detection | — |
| thopalli2023interaug | Tuning-free detection augmentation policy | No numbers |
| yoo2023 | Class-specific parameters from an initial classifier, scheduled decay in retraining | — |
| cheung2022adaaug | Class- and instance-adaptive learned policies | No fine algorithm claims |
| ahn2023cuda | Online per-class severity from recognition of augmented examples (long-tailed) | — |
| berthelot2020 | CTAugment (in ReMixMatch): online per-operation magnitude statistics pooled over classes | Component of ReMixMatch |
| suzuki2022teach | Adversarial augmentation kept recognisable to a teacher | — |
| lu2023bda | BDA: rewards augmentation by paired training gain vs unaugmented reference | — |
| liu2021ddas | DDAS: one-step-gradient meta-learning augmentation search; also detection | Abstract-level |
| hou2023madaug | MADAug: input-dependent policy learned jointly via bilevel validation objective | Do not call it less expressive |
| yang2025adaaugment | AdaAugment: actor–critic instance-level magnitudes from loss feedback | Not AdaAug |
| mehta2022rangeaugment | RangeAugment: learns magnitude ranges online with task loss (incl. detection) | arXiv |
| wang2019psis | PSIS: instance switching for detection, more switching for low-AP classes | arXiv |
| pruthi2020tracin | TracIn: training influence via gradient alignment with a test loss | — |

Unused entries in the bib (`pardeshi2025`, `tao2026multimodal`) may be ignored.

## 11. Results: where they will come from and how to update

None exist yet. As runs finish, the author will upload files and say which. Expected inputs and the sections they update:

| Upload | Contents | Update |
|---|---|---|
| `budget.json` | Base-recipe validation AP50 curve per dataset; plateau flag | Setup (epoch budgets) |
| Edit-wise study output (`cua.analysis.part_a`): per-class ΔAP50 table, interaction test, heatmap | Measurement results | Section IV results, Results, Abstract, Intro contributions |
| Proxy check (Spearman ρ) | Short vs full-run agreement | Section IV, Limitations |
| Class-targeted study matrix | Direct vs spill-over | Section IV, Discussion |
| Main comparison table (test macro AP50, CIs) | Five policies × datasets × detectors | Results, Abstract, Conclusion |
| Ablations, estimator agreement, overhead | — | Results, Discussion |
| Data-gate report | dacl10k classes kept/excluded, co-occurrence rates | Setup |

When results arrive: replace the matching `\todo{}`s, state each comparison with its mean, interval and whether it excludes zero, and re-check the abstract and contribution list against them.

## 12. Files in this Project

Upload set (repository paths in brackets):

| File | What it is | Use for |
|---|---|---|
| `PAPER_CONTEXT.md` [research_paper/] | This file | Always first |
| `main.tex` [research_paper/] | Current manuscript (IEEEtran) | Edit target |
| `references.bib` [research_paper/] | 33 verified entries | Citation keys |
| `SOURCE_AUDIT.md` [research_paper/] | How each reference was verified | Checking what a citation supports |
| `2026-11-08-class-utility-augmentation-design.md` [docs/superpowers/specs/] | Full study design (authoritative) | Method, setup, statistics details |
| `RESULTS.md` [reports/fasterrcnn_class_adaptive_v1/] | Pilot (old controller) results | §9 only |
| `README.md` [cua/ccast/] | How runs are executed on the cluster | Compute/reproducibility details only |
| Result files as they arrive | §11 | Results |

Historical / ignore unless asked: `METHOD_PROTOCOL.md`, `figures/build_methodology_framework.py` and `methodology_framework.pdf` (old controller figure — needs replacing), `main_manuscript.pdf`, `main_reviewed.pdf`, and anything named `rdd_v2_*`, `fasterrcnn_*_v1`, `PROJECT_PLAN.md`, `PILOT_RUNBOOK.md`, the October handoff documents, and the original crop-classification proposal. These describe earlier designs.

Authors, affiliations and funding are deliberately left blank; do not fill them in.
