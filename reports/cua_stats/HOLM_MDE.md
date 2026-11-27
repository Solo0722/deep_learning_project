# Multiplicity correction (Holm) and minimum detectable effects

Script: `reports/cua_stats/holm_mde.py`. Run it as `python holm_mde.py <repo root>`; the output is `holm_mde.json`.
Inputs: `reports/cua_part_c_v2_report/results.json` (per-seed test macro AP50 and stored image-bootstrap intervals), plus the copy-paste test results from CCAST job 817465.

## Method

- **Seed test.** Paired t on the 3 per-seed differences (df = 2). The two-sided p = 1 − |t|/√(t² + 2) is exact for df = 2.
- **Image bootstrap.** p is approximated from the stored 95% percentile interval, assuming normality: se = (hi − lo)/3.92.
- **Holm.** Applied within each family of 6 dataset × detector comparisons, separately to the seed p and the bootstrap p.
- **Claim.** A claim needs **both** Holm-adjusted p-values below 0.05 in the same direction. This is the pre-registered double rule with Holm added.
- **H4.** Spearman ρ; p from z = ρ√(n − 1), Holm across the 6 pairs. Cells share seeds and edits, so these p-values are optimistic.
- **Minimum detectable effect (MDE).** The true difference that would be detected with 80% power at a two-sided α of 0.05.
  - Seed t: δ·sd_diff/√3, where δ = 5.65 is the noncentrality of the t distribution with 2 df (simulated) and sd_diff is the median standard deviation of the per-seed differences over the pair's comparisons.
  - Image bootstrap: 2.80/1.96 × the median interval half-width.
  - Since a claim needs both tests, the effective MDE is the larger of the two.

## Results that survive Holm (both tests)

| Family | Pair | Difference (AP50) | Holm p, seed / bootstrap |
|---|---|---|---|
| H3 v1: cua − best baseline | dacl10k FRCNN | −1.87 vs uniform | 0.036 / < 0.001 |
| H3 v1 | RDD FRCNN | −1.02 vs base | 0.010 / < 0.001 |
| H3 v1 | RDD RetinaNet | −0.99 vs base | 0.018 / < 0.001 |
| H3 v2: cua2 − best baseline | RDD RetinaNet | −0.64 vs base | 0.015 / < 0.001 |
| H3 v2 | **VOC RetinaNet** | **+0.52 vs uniform** | **0.048** / < 0.001 |

**Nothing survives** in:
- **H2** (cua − shared and cua2 − shared2, in all 6 pairs);
- **measured − best baseline**;
- **copy-paste − uniform**.

**Cases that change under Holm:**
- **v1 VOC RetinaNet H3 (+0.28):** was a claim before correction; now lost (seed Holm p 0.13).
- **v2 RDD FRCNN H2 (+0.25):** survives on the seed test (Holm p 0.009), but the bootstrap fails (Holm p 0.076). No claim.
- **v2 VOC RetinaNet H3 (+0.52):** now the **only** positive result left, and it barely survives (seed Holm p 0.048). It is not class information: on that pair cua2 − shared2 is +0.05 (p 0.66), and shared (84.08) and measured (83.76) both score higher than uniform there as well.

**H4** (estimated utility vs measured effect):

| Pair | ρ | Holm p |
|---|---|---|
| RDD FRCNN | −0.67 | 2e-5 |
| VOC FRCNN | −0.20 | 0.012 |
| other four pairs | — | ≥ 0.56 |

Both significant correlations are **negative**: the estimator ranks edits in the opposite order to their measured effect.

**Copy-paste vs uniform:**

| Pair | Arm | Difference | Seed Holm p | Bootstrap Holm p |
|---|---|---|---|---|
| dacl10k | cpbal | +0.49 | 0.12 | 0.12 |
| dacl10k | cpada | +0.49 | 0.15 | 0.12 |
| RDD | cpbal | −0.55 | 0.15 | not run |
| RDD | cpada | −0.78 | 0.12 | not run |
| RDD | cprand | −0.52 | 0.15 | not run |

The image bootstrap was not run for the RDD paste arms.

## Minimum detectable effect per pair (test macro AP50, 80% power)

| Pair | Seed sd of differences | MDE, seed t | MDE, image bootstrap | **Effective MDE** |
|---|---|---|---|---|
| VOC FRCNN | 0.12 | 0.40 | 0.31 | **0.40** |
| VOC RetinaNet | 0.17 | 0.55 | 0.31 | **0.55** |
| RDD FRCNN | 0.20 | 0.65 | 0.38 | **0.65** |
| dacl10k FRCNN | 0.23 | 0.74 | 0.72 | **0.74** |
| RDD RetinaNet | 0.28 | 0.90 | 0.42 | **0.90** |
| dacl10k RetinaNet | 2.11 | 6.87 | 0.39 | **6.87** |

**How to phrase it in the paper.** With 3 seeds, the study can detect differences of about 0.4–0.9 AP50 on five pairs, and essentially nothing on dacl10k RetinaNet, where the seed-to-seed spread is about 2 AP50. So "no gain" means no gain larger than about 0.4–0.9 AP50, not that the effect is exactly zero. The seed test is the binding constraint everywhere: adding seeds would sharpen the results more than adding test images would.
