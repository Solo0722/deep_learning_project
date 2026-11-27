# Audit of the strict determinism diagnostic

Audit date: 2026-11-02. CCAST PBS job: 799047.bright04. This audit reads saved records and launches no experiment.

The strict diagnostic passes its declared exactness and resource gates. Plain repeated training, training after a controller probe, and training after a save/rebuild/restore all have exactly matching recorded states over the seven-update comparison. The first-update gate also passes. The earlier divergence is absent from this strict execution of the same materialized inputs and initial state.

Recommendation: close this specific repeatability investigation and prepare the previously proposed bounded, two-arm comparison through epoch 8, using strict execution for both geometry-only and class-adaptive local augmentation. Start both from the common original epoch-1 checkpoint in a new run. Keep the method fixed. This is a development experiment, not evidence that the proposed method will win.

## Integrity and protocol

Archive: `rdd_v2_strict_check_v1_records.tar.gz` (2,038,323 bytes).

SHA-256: `4912aec513e3f3b64f89234bdc65e2b3bc7d7d4f9b609bc486670af6bf0806ee`.

All 240 manifest entries in the 241-file archive match. Safe extraction rejects links, traversal and duplicate members; any existing extracted files must match. All 50 released package payloads match the source hashes recorded by the job. The 41 vendored reproduction files are unchanged from the prior release. The recorded protocol and effective configuration match the local releases, and the frozen training manifest and split configuration match their pinned hashes. The prior archive and its 167 checksummed extracted files were reverified for the cross-diagnostic comparison.

The run used PyTorch 2.3.0, TorchVision 0.18.0, CUDA 12.1 and an NVIDIA A10 on gpu0020. Strict algorithms were enabled with `warn_only=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic cuDNN, benchmarking off and TF32 off. The source checks strict flags at every recorded phase. The installed RoIAlign source is preserved and hashed; its strict CUDA dispatch was observed on every update. There are 120 recorded Python RoIAlign calls across the 30 updates and 128 during probing. No custom numerical kernel or compilation was introduced.

This behavior is consistent with the pinned [TorchVision 0.18 implementation](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/ops/roi_align.py), which dispatches strict CUDA RoIAlign to its differentiable tensor implementation. This observation does not isolate RoIAlign as the sole cause of the earlier divergence.

## What matched

The diagnostic uses 26 frozen training images: 17 with target objects and 9 without them. Batch sizes are 4/4/4/4/4/4/2, including a final entirely negative batch. Appearance is identity in every condition; the common geometry and labels are unchanged. Geometry/decision hashes were reconstructed, and the complete materialization record is identical to the earlier diagnostic. No raw training images were decoded again locally.

| Comparison | Updates compared | Recorded phases matched | Result |
|---|---:|---:|---|
| Independent first-update repeat | 1 | 4 | Exact |
| Plain seven-update repeat | 7 | 28 | Exact |
| Probe then seven updates | 7 | 28 | Exact |
| Save/resume during seven updates | 7 | 28 | Exact |

The audit reconciles all 120 phase records with the update ledger and recomputes 480 raw RNG component fingerprints, covering Python, NumPy, CPU Torch and CUDA state. All 88 comparison-phase records match their references, including inputs, pre-forward model/optimizer state, losses/buffers, gradients, optimizer updates and RNG. Thirty pre-forward state hashes and both probe/resume boundary hashes were independently reconstructed from the saved values. The gate and suite first-update records also match in eight cross-stage checks, confirming the restart from the original state. Distinct pre-forward model hashes across successive steps rule out a trivial no-update trajectory in the saved evidence.

The checkpoint restore occurs after update 1; all six subsequent updates match uninterrupted training. The probe's model, optimizer, RNG and module modes match before/after. The original observer also records unchanged gradients. Replaying the controller from saved probe observations reproduces its policy exactly. The two-image probe has detected-object/cluster support 0/0, 1/1, 1/1, 0/0 across the four classes and correctly returns identity. This deliberately tiny probe tests observation side effects; it does not test an active class-adaptive policy's accuracy.

Full model/gradient/optimizer/reference tensors remain on CCAST. Therefore the local audit verifies their returned fingerprints and the recorded GPU-side byte comparisons; it does not independently recompute those full tensor comparisons. The checkpoint hash is pinned by the executed input check, rather than by reloading the checkpoint locally.

## Resource measurements

| Measurement | Observed value |
|---|---:|
| Worker elapsed time | 164.68 seconds |
| Parent-supervised process elapsed time | 167.19 seconds |
| Optimizer updates observed | 30 |
| Median instrumented update | 2.658 seconds |
| Update time range | 2.143–7.030 seconds |
| Peak update allocated GPU memory | 7.267 GiB |
| Peak update reserved GPU memory | 12.730 GiB |
| Device memory reported by PyTorch | 22.058 GiB |
| Two-image probe elapsed time | 8.259 seconds |
| Reference snapshots retained on CCAST | 4.934 GiB |

All recorded measurements are below the declared limits. Allocated/reserved memory are PyTorch allocator measurements and exclude some external CUDA allocations. Timings include synchronization, CPU tensor copying, hashing and record writes; the overall figures exclude queue time and preceding unit tests. Neither these timings nor the earlier diagnostic's different duration establish a training speedup. The full 256-image controller panel has not yet been costed under strict execution.

## Interpretation and limits

Fact: the previous plain repeat first differed in backward at update 1; these strict repeats do not differ over the tested trajectory. The scientific inputs, training configuration and initial model/optimizer/RNG fingerprints agree between diagnostics. The runs used the same GPU model and software versions but different physical GPU UUIDs, and the diagnostic wrapper changed to enforce strictness and measure resources.

Inference: numerical execution was a credible source of the prior discrepancy, and strict execution provides a working remedy for this short comparison. This is not an operator-isolation experiment and does not prove all training code is correct. There is no evidence of a probe/resume state leak in the paths tested here.

Unknown: full-run repeatability, the cost of strict execution at all later training states, and whether class adaptation improves detection. There was no validation or test evaluation. The previous pilot's 0.2003 AP-point class-adaptive deficit against geometry-only remains an observed result; it cannot be retrospectively attributed to nondeterminism. [PyTorch's pinned reproducibility guidance](https://github.com/pytorch/pytorch/blob/v2.3.0/docs/source/notes/randomness.rst) also limits reproducibility guarantees across releases and platforms.

## Recommended next stage

1. Freeze a new comparison of geometry-only and class-adaptive local augmentation to the same epoch-8 endpoint. Restart both from the pinned epoch-1 identity checkpoint, so epochs 2–8 use the same strict settings. Preserve the earlier pilot as a separate experiment rather than mixing it with this execution regime.
2. Retain the detector, geometry, local appearance bank, sampling rule, support gates and retention threshold. Preselect any additional training probe panels and update epochs before launching. Keep data order, training steps and evaluation fixed across arms.
3. Put a short equality check through the actual training entry point at the beginning of that job, then measure the first full controller panel and early normal-training cost. This verifies integration of the strict settings and sets a practical runtime bound; it is not another open-ended diagnostic campaign. Stop on a strictness, integrity or resource failure.
4. Use clean validation AP50:95 at epoch 8 as the primary development comparison. Keep per-class AP, recall, false positives, controller support, augmentation exposure and extra compute as declared secondary diagnostics. Keep the test set closed and avoid choosing a favorable epoch after the run.
5. A favorable result would motivate paired independent seeds and the online-shared comparator before claiming a class-conditioning benefit. If the method remains worse without a declared compensating benefit, stop scaling this recipe and reassess the augmentation intervention. This short study is not a convergence test or proof of equivalence.

## Reproduction and evidence

Run `python reports/rdd_v2_strict_v1/audit_records.py` from the project directory. The program reads the verified archive, frozen source packages, training manifest and earlier diagnostic records. It performs no training or detector inference.

- `audit.json`: machine-readable findings and limitations.
- `archive_audit.json`: per-member hashes and source provenance.
- `exact_comparisons.csv`: all 22 update comparisons.
- `update_resources.csv`: all 30 update resource records.
- `returned_records/`: preserved downloaded evidence.
- `analysis_checksums.json`: hashes of the analysis code and outputs.
