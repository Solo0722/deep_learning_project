# Review: "Class-Aware Selective Supervision" proposal

Reviewed document: `Selective_Supervision_Architecture_Proposal.docx` (external review of CUA plus a redesign).
Checked against: the code on branch `study/class-utility-augmentation` (`cua/estimator/*`, `cua/policy.py`, `cua/augment/*`), the Part A deltas (`reports/cua_part_a_report/`), the Part C v1/v2 test results (`reports/cua_part_c_report/`, `reports/cua_part_c_v2_report/`), and the estimator diagnosis (`reports/cua_utility_check*/`).

## 1. Verdict

**Don't adopt the full architecture.** The online cosine estimator in particular is not worth building. Three reasons:

1. **On RDD, the problem it solves doesn't show up.** Selective supervision helps only when one edit helps class A and hurts class B in the same image. On RDD, almost all of the per-class edit effect is shared across classes: the edit main effect is 87% of the variance for Faster R-CNN. No edit is reliably good for one damage class and reliably bad for another. The proposal's worked example (blur keeps potholes and destroys cracks) is the reverse of our data: blur hurts D40 (pothole) the most, by −1.98 AP50, and D00 (crack) the least, by −0.67.
2. **Its online estimator would repeat the failure we already diagnosed.** The cosine `cos(g_{o,c}, h_c)` has no "compared with not editing" term, so it is positive for nearly every edit. The proposal admits this in its own Risk 1. Our offline check already scored a cosine variant (`cosdiff`) and the original variant. Neither carries a class signal within an edit (Spearman ρ ≈ 0). The 4-step lookahead (v2) didn't recover one either.
3. **Two of its five diagnoses are already contradicted by our data, and one rests on a wrong reading of the architecture.** See §2.

**The one part worth keeping** is the proposal's own Section 6 idea: a static, oracle ignore-mask test before any estimator is built. Our data already answers its step 1. Steps 2–3 are worth running only where class conflict actually exists: VOC and dacl10k with Faster R-CNN. The test also needs a control that the proposal leaves out (§6).

## 2. Claim-by-claim check of Section 1 ("why CUA fails")

| # | Proposal claim | Status | Evidence |
|---|---|---|---|
| 1.1 | Single-step utility systematically penalises regularising edits, so the policy collapses toward "no edit" | **Contradicted in practice** | The v1 estimator rated **mosaic, the most harmful edit in Part A, as by far the best edit**. On dacl10k Faster R-CNN its mean `dot` utility was 16.8, against values below 1 for every other edit. A systematic penalty would push it the other way. The `noid` ablation (no identity option) changed nothing (RDD +0.17 [−0.09, 0.43]; dacl10k +0.15 [−0.43, 0.74]). The multi-step lookahead in v2 removes the single-step issue and still finds no class signal (within-edit ρ −0.27 to +0.15). |
| 1.2 | Pooled batch gradient contaminates classes absent from B | **Correct as a description** | `Estimator.gradients` uses the pooled training loss on an 8-image batch. Every class is updated each round because `valid` comes from the stratified reference batch, not from B. For an image-level policy, though, cross-class transfer is part of what the policy should measure, so this is not clearly a bug. |
| 1.3 | θ_s (box head + top FPN level) is blind to small objects on P2/P3 | **Mostly wrong** | The Faster R-CNN box head (`TwoMLPHead` + predictor) is shared by RoIs from **all** FPN levels through `MultiScaleRoIAlign`. RetinaNet's heads are shared across P3–P7. The C5 lateral conv (`inner_blocks[-1]`) feeds every level through the FPN top-down sum. Only `layer_blocks[-1]` is P5-only. RetinaNet also has no P2, so the proposal's "P2–P5 for RetinaNet" is wrong. The valid part: photometric edits act mostly on the frozen backbone, which no head-level estimator sees. The proposal's own estimator keeps the backbone frozen too. |
| 1.4 | Unnormalised inner product is driven by gradient magnitude | **Already tested and rejected** (except mosaic) | `utility_check`: ‖g_o‖/‖g_0‖ ≈ 1 for every edit except mosaic (0.99–1.02 against 1.26 for mosaic on dacl10k Faster R-CNN; the handoff reports the same pattern on the other pairs). Norm-free variants (`normed`, `cosdiff`) correlate with Part A no better than `dot`. |
| 1.5 | Image-level harm blocking starves multi-class images | **Logically true, empirically not the cause** | The `noharm` ablation (CUA without the harm block) is null on both datasets: RDD −0.12 [−0.61, 0.38], dacl10k +0.07 [−0.40, 0.55] AP50, val, 3 seeds. |

The proposal's facts about Eq. (1), the 8-image batch, the moving average and the harm rule all match the code. Its inferences are where it goes wrong.

## 3. What our own data says about the proposal's premise

**Where is the class conflict?** I decomposed Part A per-cell mean ΔAP50 (3 seeds) into edit, class and residual parts. The residual includes seed noise, so it is an upper bound on the true interaction.

| Dataset / detector | Edit % | Class % | Residual (≤ interaction) % | Formal interaction test | Edits good for one class and bad for another (\|t\|>2) |
|---|---|---|---|---|---|
| RDD / Faster R-CNN | 87 | 2 | 11 | p = 4.5e-7 | **none** |
| RDD / RetinaNet | 59 | 19 | 22 | p = 0.77 | **none** |
| dacl10k / Faster R-CNN | 10 | 31 | 58 | p = 4e-11 | 7 edits |
| dacl10k / RetinaNet | 7 | 24 | 68 | p = 0.34 | 3 edits |
| VOC / Faster R-CNN | 6 | 36 | 58 | p = 2.4e-5 | 12 edits |
| VOC / RetinaNet | 30 | 30 | 40 | p < 1e-6 | 9 edits |

(With 3 seeds, t has 2 degrees of freedom. The sign-conflict column is a loose screen, not a significance test.)

RDD Faster R-CNN, mean ΔAP50 per class:

| edit | D00 crack | D10 | D20 | D40 pothole |
|---|---|---|---|---|
| blur | −0.67 | −1.20 | −1.25 | **−1.98** |
| jpeg | −1.03 | −0.66 | −0.49 | −0.51 |
| noise | −0.83 | −0.66 | −0.35 | −0.22 |
| mosaic | −2.37 | −3.74 | −3.61 | −4.48 |
| rotate | −0.58 | −1.63 | −0.27 | −0.52 |

On RDD every harmful edit hurts every class. Ignore-masking under such an edit masks all the image's objects, which is worse than simply not applying the edit. The right fix on RDD is to drop the edit from the menu, and the measured policy already roughly does that (test AP50: base 51.15, uniform 50.55, measured 51.47).

**Is Uniform a strong base to build on?** The proposal assumes TrivialAugment-style uniform sampling is the strong baseline. On our data it isn't, on three of six pairs:

| Test AP50 | base | uniform | measured |
|---|---|---|---|
| RDD / FRCNN | 51.15 | 50.55 | 51.47 |
| RDD / RetinaNet | 44.56 | 43.82 | 44.55 |
| dacl10k / FRCNN | 20.89 | **22.37** | 21.54 |
| dacl10k / RetinaNet | 9.26 | 7.70 | 9.80 |
| VOC / FRCNN | 85.60 | **86.54** | 86.30 |
| VOC / RetinaNet | 82.38 | 83.44 | 83.76 |

**Proposal §6 step 1 ("does the measured policy beat Uniform?") is already answered.** The measured policy beats the best baseline significantly on **no** pair. It is significantly *worse* than uniform on dacl10k Faster R-CNN (−0.84 [−1.27, −0.41]) and VOC Faster R-CNN (−0.24 [−0.45, −0.04]), the two pairs where uniform is best. By the proposal's own rule, that points to "not enough exploitable class conflict in this menu" at least as much as to a flaw in input-level selection.

**Edit severity (proposal Risk 1) is confirmed.** Apart from mosaic, no edit costs any RDD class more than about 2 AP50. The proposal is right that moderate edits rarely destroy class cues outright.

## 4. Problems in the proposed design itself

1. **The estimator has no counterfactual.** `cos(g_{o,c}, h_c)` asks whether training on class c in edited images still lowers clean class-c loss. For a mild edit and the same labels, the answer is almost always yes. Harm is a comparison with not editing, and that is exactly the `−g_0` term the proposal removes. With τ₀ = 0 and γ = 6, a cosine of 0.1 gives w ≈ 0.65, and masking (w < 0.25) needs a cosine below −0.18. That will rarely happen, and the method then collapses into Uniform, as the proposal itself predicts in Risk 1.
2. **Minibatch gradient cosines are small and noisy.** Per-example and per-batch gradient cosines in deep networks are typically near zero and shrink as training goes on (Fort et al., 2019). Du et al. (2018), the closest precedent for cosine-gated losses, note that cosine estimates need smoothing. Our own diagnosis found per-cell utilities mostly noise (|t| > 2 in only about 5–30% of cells), with signs that flip between checkpoints. Class isolation and a stratified buffer reduce contamination but do nothing about that variance.
3. **The ignore rule is mis-specified for our objects.** "Ignore anchors with IoU ≥ 0.5 to the ignored box" leaves small anchors inside a long crack box (IoU < 0.3) as **background negatives**. The claimed "zero false-negative penalty" does not hold. Standard practice uses intersection-over-foreground, as in mmdetection's `ignore_iof_thr` on `MaxIoUAssigner`. RDD and dacl10k cracks are long, thin boxes, so this matters here more than on VOC.
4. **It is not a drop-in change.** torchvision has no ignore-region support. Ignore handling would have to be added to RPN anchor matching, to RoI sampling (including `add_gt_proposals`) and to the RetinaNet loss. *Side finding:* our pipeline already marks VOC "difficult" boxes as `ignore` (`cua/datasets/voc.py`), but `cua/data.py` drops them from the targets and no loss uses them, so **they are currently trained as background**. That is a small issue that already exists in the VOC runs.
5. **Masking removes the label, not the input.** If an edit hurts a class through feature or distribution shift in the shared backbone rather than through a corrupted box label, masking won't recover it (the proposal's Risk 3). Part A deltas can't tell these mechanisms apart.
6. **Compute.** Class-isolated gradients need about C× as many backward passes over the estimation batch: about 4× on RDD, 12× on dacl10k and 20× on VOC, on top of the stratified buffer. The current estimator already costs about 15% of epoch time at K = 200.
7. **The figure's compatibility table is illustrative, not measured.** Its numbers (blur: D00 0.05, D40 0.94) don't come from any of our runs and contradict them. They should not appear in anything we write.
8. **The literature claim is overstated.** "Ignore-masking consistently outperforms" is too strong. The evidence (§5) is real but modest, comes from a different kind of noise (missing or uncertain labels, not cue-destroying augmentation), and includes soft reweighting beating hard ignoring.

## 5. Literature

All entries below were checked to exist. The notes describe only what the abstract or venue page says.

**Class-dependent harm from augmentation (the problem is real):**
- Balestriero, Bottou & LeCun (2022), *The effects of regularization and data augmentation are class dependent*, NeurIPS 35. Augmentation can sharply lower accuracy on individual ImageNet classes. They leave class-unbiased regularisation as an open problem.
- Kirichenko et al. (2023), *Understanding the detrimental class-level effects of data augmentation*, NeurIPS 36. Random-resized-crop hurts some classes by up to 20%, mostly ambiguous, co-occurring or fine-grained ones. Simple class-conditional augmentation helps those classes. This is classification, where each image has one label, so per-class input choice has no co-occurrence problem.
- Cheung & Yeung (2022), *AdaAug*, ICLR. Learns class- and instance-adaptive policies by minimising validation loss.

**Keep the augmentation and change the loss (closest prior art for the reframing):**
- Zhou et al. (2021), *MetaAugment*, AAAI 35. A policy network outputs a **weight on each augmented image's loss**, meta-learned against a validation set. Structurally this is the proposal's idea at the image level, in classification.
- Liu et al. (2023), *Soft Augmentation for Image Classification*, CVPR. Softens the target as augmentation severity grows instead of blocking strong edits.
- Ren et al. (2018), *Learning to Reweight Examples*, ICML. Weights come from a meta-gradient step against a clean validation set. This is the template for validation-aligned reweighting.

**Gradient-alignment gating:**
- Du et al. (2018), *Adapting Auxiliary Losses Using Gradient Similarity*, arXiv:1812.02224. An auxiliary loss is used only while its gradient's cosine with the main gradient is non-negative. This is the nearest precedent for the proposal's w(o, c). Cosine estimates are noisy.
- Yu et al. (2020), PCGrad, NeurIPS; Pruthi et al. (2020), TracIn, NeurIPS. Both are cited correctly in the proposal.
- Fort et al. (2019), *Stiffness*, arXiv:1901.09491. Per-example gradient cosine, and how it shrinks with training.

**Ignore regions and missing labels in detection:**
- Wu et al. (2019), *Soft Sampling for Robust Object Detection*, BMVC. Faster R-CNN loses only about 5% mAP with 30% of VOC annotations removed. **Reweighting** RoIs by overlap beats standard training. The false-negative penalty that ignore-masking avoids may therefore be small.
- Zhang et al. (2020), *Background Recalibration Loss*, ICASSP. Changes the negative branch of the focal loss for suspected missing labels.
- Zhang, Benenson & Schiele (2017), *CityPersons*, CVPR. Ignore-region handling is worth roughly 1 to 1.3 miss-rate points.
- mmdetection `MaxIoUAssigner` (`ignore_iof_thr`, `gt_bboxes_ignore`): the practical reference implementation, which uses IoF.

**Detection augmentation baselines:** Müller & Hutter (2021), TrivialAugment, ICCV. Zoph et al. (2020), *Learning Data Augmentation Strategies for Object Detection*, ECCV, which already has **box-only** operations, an input-level answer to per-object conflicts.

**Novelty.** I found no detection paper that gates each object's loss by an (edit, class) compatibility score. The proposal's "conditional label noise" reframing for detection is therefore plausibly new. It would still have to be positioned against MetaAugment, Soft Augmentation, L2RW and Du et al., and against box-only augmentation (Zoph et al.) as the input-level alternative. My search was targeted, not systematic, so "no prior detection work" is a finding of this search only.

## 6. If you want to test the idea anyway: one diagnostic

Run it only on **VOC and dacl10k with Faster R-CNN**, where the interaction is significant, uniform is the best arm, and sign-conflict edits exist. Skip RDD: §3 shows there is nothing there for masking to resolve.

Arms, 3 seeds each, judged on test:
1. **uniform**: already trained (Part C).
2. **uniform − pruned edits** (control the proposal leaves out): uniform without the edits that are harmful at the edit level, such as mosaic. Masking has to beat simple menu pruning to show any class-level value.
3. **uniform + static oracle ignore-mask**: mask (o, c) cells where all 3 Part A seeds are negative and the mean is below −1 AP50. The Part A deltas come from validation, so test stays held out. Use IoF ≥ 0.5 against the ignored box for both positives and negatives.

That is 12 Faster R-CNN runs plus about one to two days of matcher work with unit tests: RPN, RoI sampling, and VOC difficult boxes as a free test case.

**Go only if** arm 3 beats both arm 1 and arm 2 on macro AP50 and on the worst affected classes. Otherwise treat selective supervision as future work in the discussion. Raising edit strength, the proposal's other fix, is a different study. On RDD it would most likely just make every class worse first.

## 7. Limitations of this review

- Part A uses 3 seeds, so per-cell estimates are coarse. The RDD conclusion is robust because the edit main effect dominates so strongly (87%). The dacl10k and VOC cell lists are not.
- The proposal's exact estimator, a class-isolated cosine without the `−g_0` term, was not run. The closest offline variants (`cosdiff`, `normed`) were. Adding a plain `cos` variant to `cua/analysis/utility_check.py` would take one cluster job if a direct check is wanted.
- The literature search was targeted, not a systematic review.

AI disclosure: this review was prepared with AI assistance (Claude). Code and result files were read directly, and every cited work was checked to exist.
