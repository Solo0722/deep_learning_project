# Audit of the V2 reproducibility diagnostic

Audit date: 2026-10-29. CCAST PBS job: 797450.bright04. This report analyzes saved evidence and launches no GPU job.

The diagnostic reproduces divergence in a plain repeated training run, starting with small gradient differences during the first backward pass. Probing and checkpoint resumption are unnecessary for that divergence. No mismatch was found in the observed state around the probe or checkpoint restore. Numerical nondeterminism in backward computation is the leading explanation; a specific CUDA operation has not been isolated experimentally.

Recommendation: keep the augmentation method unchanged, preserve the existing pilot, and perform one bounded strict-determinism comparison before a longer experiment. Do not claim the earlier class-adaptive accuracy deficit was caused by numerical variation. This diagnostic did not evaluate accuracy.

## What was verified

The downloaded archive is `rdd_v2_repro_check_v2_records.tar.gz`, 4,152,302 bytes, with SHA-256:

`3ec5a81f35d667f71994f80956fa9ec691bbd2c9fe0b704173f6d2a97f9eedd9`

Its 168 regular files total 22,073,219 uncompressed bytes. All 167 checksummed content files match the returned manifest. The archive hash matches the CCAST value supplied by the user. Safe extraction rejects links, traversal, duplicate entries and unexpected checkpoint binaries. Existing extracted files are verified rather than replaced.

All 40 released package files match the source hashes recorded by the job. The 29 vendored pilot source files match both their original manifest and the original local pilot release. Protocol, source configuration, effective configuration, training manifest and split-configuration hashes agree. The effective diagnostic changes only the declared loader-worker and warmup settings. The pinned source checkpoint identity agrees across configuration records; its full tensors remain on CCAST and were not reloaded locally.

The job completed all 28 updates: four serial conditions with seven updates each. It used an NVIDIA A10, PyTorch 2.3.0, TorchVision 0.18.0 and CUDA 12.1. The recorded diagnostic elapsed time is 246.9 seconds, including preparation and instrumentation but excluding the preceding unit tests. The prior hashing failure prevented all real diagnostic updates in release v1; it is separate from this completed release-v2 result.

## Controlled inputs and state

All conditions use the same 26 training images, consisting of 17 positive images and nine images without target-class objects. The seven batch sizes are 4, 4, 4, 4, 4, 4 and 2. The final batch contains only negative images. Appearance is identity throughout; common geometry is applied according to the original smoke treatment. Nineteen geometry candidates were accepted and seven fell back because an annotation would be cropped.

Recomputing the image-order/geometry trace and complete decision hash reproduces the declared smoke pins. Recorded image IDs use the original zero-based training-manifest indices. Every per-step input hash agrees across all four conditions. Raw pixels were not included in the archive; this is a verification of the saved traces and executed input checks, not a new image decode or render.

Initial model, optimizer and RNG fork fingerprints agree across all four conditions. For update 1, the entire recorded pre-forward state agrees, followed by exactly equal loss components, buffers and RNG state. Internal activations and proposals were not separately recorded, so equality of every intermediate forward tensor is not asserted.

All 112 saved phase RNG fingerprints were reconstructed from their raw Python, NumPy, CPU Torch and CUDA state records. Every one of the 84 condition-versus-reference phase comparisons has identical RNG state. Thus a differing saved RNG stream is not the first observed discrepancy in this diagnostic.

## First difference and its magnitude

For `control_b`, `probe_only` and `resume_only`, the first observed difference is update 1, backward. There are respectively 69, 69 and 68 differing gradient tensors, all within the backbone/FPN. No first-update gradient difference was reported in the RPN or RoI box-head parameters. These counts are tensor entries in a comparison, not counts of bugs. The first-step losses are identical.

All quantities below compare with `control_a`. Loss difference is a percentage of the reference loss on that same batch, not an AP-point difference. The final column is the L2 norm of the complete gradient difference divided by the recorded reference gradient norm; it is not the maximum per-layer relative error.

| Condition | Update-1 largest gradient difference | Update-7 largest model-weight difference | Update-7 total-loss difference | Update-7 relative gradient L2 difference |
|---|---:|---:|---:|---:|
| control_b | 2.421e-08 | 1.163e-05 | -0.240% | 0.904% |
| probe_only | 2.328e-08 | 1.162e-05 | +0.181% | 0.847% |
| resume_only | 2.794e-08 | 4.422e-06 | -0.418% | 2.018% |

The initial maximum gradient differences are approximately 2.3e-8 to 2.8e-8. Their full-gradient relative L2 sizes are approximately 6.7e-7 to 6.9e-7. After the first optimizer update, the maximum parameter difference is 1.49e-8 in each comparison. These observations are consistent with small floating-point accumulation differences at the beginning.

Differences amplify over subsequent updates: model parameters, momentum, gradients and losses become different. At update 7, the maximum model-weight difference reaches approximately 1.16e-5. Some individual gradient tensors have larger relative differences than the full-gradient figures in the table; `resume_only` reaches a per-tensor relative L2 difference of about 8.29%. This does not establish a resume effect, because its trajectory already differed before the reload and there is only one execution of each condition.

The nine exported tensor pairs were independently recomputed with NumPy, including shape, dtype, changed-element counts, maximum absolute difference, L2 norms and recorded sample values. All agree with their comparison records. The remaining full-tensor magnitudes are GPU-side comparison results preserved in verified JSON; they were not independently recomputed locally because the full snapshots were intentionally excluded. Approximately 4.64 GB of reference snapshots remain on CCAST.

## Probe and checkpoint boundaries

The decoded before/after records match exactly around probing: model hash, optimizer hash, complete RNG state and every saved module mode. The probe's gradient-preservation check also reports success. Replaying the two-image probe's policy from its saved observations reproduces the saved policy exactly. Independent support recounts give 0, 1, 1 and 0 reference-detected objects for D00, D10, D20 and D40, respectively, with the same cluster counts. All routes correctly fall back to identity under the unchanged 16-object/eight-cluster support gate. No probe decision was applied to training in this diagnostic.

Checkpoint restore occurs after update 1. Its before/after model hash, optimizer hash, RNG and module-mode records are identical, and its saved metadata agree. Gradients are intentionally not restored: all conditions clear them before the next forward pass. The first divergence in `resume_only` precedes this checkpoint operation. These are positive checks of the observed boundaries; they do not prove that probing or rebuilding a model has no additional effect on CUDA scheduling, caches or later numerical behavior.

## Interpretation and limits

The configured cuDNN convolution setting was deterministic, with benchmarking and TF32 disabled. General PyTorch deterministic-algorithm mode was disabled, and `CUBLAS_WORKSPACE_CONFIG` was unset. The convolution flag alone does not make every GPU operation deterministic, as explained in the [PyTorch 2.3.0 reproducibility documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.0/docs/source/notes/randomness.rst).

TorchVision 0.18.0's [CUDA RoIAlign backward source](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/csrc/ops/cuda/roi_align_kernel.cu#L412) marks the operation nondeterministic and uses atomic gradient accumulation. The observed first-step pattern, with differing backbone/FPN gradients and equal head gradients, is consistent with this path. That makes RoIAlign a concrete candidate, not a proven unique cause. There are no operator-level traces or fixed-intermediate-tensor repetitions here. The cuDNN plan warnings alone do not identify the cause either.

The evidence supports the following bounded conclusions:

- The setup is not bitwise repeatable under its current GPU settings, even without probing or resumption.
- No recorded input, RNG or boundary-restoration mismatch was found in the tested path.
- The original smoke's trajectory discrepancy is plausible without an augmentation-controller state leak. This is not a blanket claim that all code is correct.
- Seven updates on one checkpoint, one machine and one serial execution per condition cannot estimate full-training variation or the additional effect of probe/resume.
- No validation/test images or metrics were evaluated in this diagnostic. No AP correction, equivalence margin or significance claim can be derived from it.

The existing four-arm pilot therefore remains a descriptive single-seed result. Its class-adaptive endpoint was about 0.20 AP points below geometry-only. This audit neither erases that deficit nor establishes that it is repeatable. The method has not yet demonstrated an advantage over the strongest simple control.

## Recommended next step

Run one narrowly bounded deterministic-backward check before extending training. Keep the same checkpoint, materialized inputs, optimizer and loss; enable strict deterministic algorithms, with the documented cuBLAS workspace configuration set before CUDA initialization. Do not use warning-only mode or silently disable determinism after an error.

Start by repeating the same first update. Record exact equality, runtime and peak GPU memory. If the path works within the declared resource cap, extend the repeats to the existing seven-update horizon; then verify the probe and resume paths under that same setting if they remain part of the intended training workflow. If an operation fails, report its identity and preserve the partial evidence. This is a new execution-setting intervention, not a change to the augmentation controller.

TorchVision 0.18.0's [RoIAlign dispatcher](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/ops/roi_align.py) selects an alternative Python tensor implementation for CUDA when deterministic algorithms are enabled. Its resource cost must be measured on CCAST; it must not be assumed to fit or run at the native-kernel speed. Agreement between repeated executions of the same new path is the relevant check. Bitwise agreement between native and alternative implementations is not a reasonable automatic requirement; any cross-path numerical differences must be recorded separately.

If the deterministic path is stable and affordable, freeze it for both sides of the next controlled comparison and return to the scientific question: class-adaptive versus geometry-only under matched training budgets, with independent paired seeds before claiming a repeatable advantage. A common checkpoint and horizon are required; do not compare a changed backend in one arm against an old-backend arm as if only augmentation changed.

If strict execution is unsupported or too expensive, do not start an open-ended kernel rewrite. Document the limitation, retain the matched execution configuration, and assess run-to-run variation with a bounded replication plan. Bitwise equality is useful for debugging, but is not itself a requirement for a valid empirical study. Numerical determinism also cannot replace independent seeds or establish augmentation efficacy.

## Audit artifacts

- `audit_records.py`: reproducible archive, source, state, tensor-pair and policy audit.
- `archive_audit.json`: complete archive inventory and provenance checks.
- `audit.json`: full numerical findings and explicit verification boundaries.
- `step_comparison.csv`: all 21 non-reference update comparisons.
- `analysis_checksums.json`: hashes of the audit code and derived outputs.
- `returned_records/`: preserved downloaded records, including the nine tensor-pair files.

Run locally with `python reports/rdd_v2_repro_v2/audit_records.py`. It reads only the existing training manifest and saved diagnostic evidence; it performs no detector inference and contacts no remote host. The source links above were inspected on 2026-10-29.
