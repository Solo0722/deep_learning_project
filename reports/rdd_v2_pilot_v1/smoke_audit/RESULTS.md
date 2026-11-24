# Audit of the saved V2 GPU smoke records

Date: 2026-10-29. PBS job: 797282.bright04. This is a retrospective software-evidence audit, not another training experiment.

The saved records confirm a training-reproducibility discrepancy under identical augmentation treatment. They do not identify its cause. The previous smoke gate checks execution, state restoration and input pairing, but does not test equality of training trajectories. That is a concrete test-coverage gap. It is not yet evidence of a faulty controller, an incorrect AP calculation, or an explanation for the full pilot's accuracy differences.

Recommendation: complete a short, controlled GPU diagnostic that separates ordinary repeat-run variation, probe execution and checkpoint resume before extending the experiment to epoch 8. A second full training run is not the appropriate debugging tool.

## Provenance and scope

The downloaded archive is `rdd_v2_pilot_smoke_797282.bright04_records_v1.tar.gz`, 1,670,919 bytes. Its SHA-256 is:

`12e8f65b129a55199f105c28bc7303fe9b1b852e950f3a062a76675aee5d0777`

This matches the CCAST `sha256sum` output supplied by the user. The missing local `.sha256` sidecar is therefore not an unresolved integrity issue. The archive contains 50 regular files, totaling 1,839,022 uncompressed bytes: the PASS summary, histories, validation predictions and metrics, checkpoint metadata, four raw controller observation files with attempt records, and the job log. Extraction rejects unsafe paths, links, duplicates and unexpected checkpoint binaries; existing extracted files are preserved and verified.

The PASS file agrees with its previously recorded hash. Its package-manifest hash matches the local release, and all 29 listed source files pass verification. The frozen train and validation manifest hashes also match the released protocol. File-level inventory and hashes are saved in `archive_audit.json`.

The smoke ran on the actual CCAST A10 with the pinned PyTorch 2.3.0 / TorchVision 0.18.0 environment. It used 26 training images and 24 validation images, with 42 validation target objects and seven validation images without target objects. Each arm trained seven updates in each of two smoke epochs. Epoch and step counters are smoke accounting: they are not a claim that the original warmup checkpoint was trained on this tiny subset. The full pilot restarted separately from the original epoch-1 checkpoint.

This audit used the saved observations and predictions. It did not read raw image pixels, reload checkpoint tensors, rerun the detector, or inspect new rendered previews. The local audit environment differs from CCAST; the versions are recorded in `audit.json`.

## What the records establish

1. **The online arms correctly fell back to identity appearance.** Each smoke probe contained only two training images. D10 and D20 each had one reference-detected object from one source cluster; D00 and D40 had none in the selected groups. None met the unchanged minimum of 16 detected objects and eight clusters. Recomputing all four policies from raw observations yields no admissible appearance actions on every route. Independent matching and affine-box checks pass for all 232 saved views. The original adopted policy objects were stored in excluded checkpoint binaries; the replay establishes what the recorded evidence implies, while the training traces establish what was applied.

2. **Geometry-only, online-shared and class-adaptive received identical recorded treatment.** All three made zero appearance edits in both epochs. Their per-epoch pairing hashes agree, covering image order, group opportunities, geometry, transformed boxes and reference-pixel hashes. With identity appearance, the renderer uses those reference pixels as the training pixels. The two online arms also have identical full decision hashes, which include input-pixel hashes. This is evidence from recorded hashes and verified code, not a new pixel rendering. Fixed-local made eight edits in epoch 2 and ten in epoch 3, so it is excluded from the identical-treatment comparison.

3. **The two online arms started their first probe with identical recorded state.** Their initial model, gradient and RNG hashes match, and all decoded initial probe records are exactly identical. The PASS file also attests that all four model/optimizer/RNG forks matched. The initial geometry-only fork was not retained as a separate tensor artifact in this archive, so its equality is an executed-check attestation rather than a fresh tensor comparison.

4. **The models had already diverged by the end of smoke epoch 2.** Model and gradient hashes differ across the three identical-treatment arms after their first seven training updates. Their final RNG hashes still agree at that point. Decoded validation predictions differ on all 24 images even though AP is exactly equal. Matching AP therefore did not demonstrate matching models. End-of-epoch averaged losses differ as well, particularly the classification component.

5. **By smoke epoch 3, AP differs and one RNG stream differs.** Geometry-only and class-adaptive still have matching final RNG hashes; online-shared differs. An end-of-epoch RNG mismatch does not establish a probe RNG leak or its direction of causation. A model difference can also change downstream sampling behavior. These records do not locate the first divergent operation.

6. **The saved metrics are internally consistent.** Official COCOeval, using independently assembled ground truth from the 24 selected frozen validation rows and their original manifest IDs, reproduces all eight evaluations exactly. Checked aggregate and per-class metrics have maximum absolute discrepancy zero. Prediction geometry, category IDs, negative-image counts, prediction-file checksums, checkpoint metadata and history references also pass. This rules out a replayable metric/record mismatch in these saved evaluations; it does not verify unseen checkpoint contents.

## Observed trajectories

AP50:95 is reported in percentage points. These are smoke-subset diagnostics, not method rankings.

| Arm | Appearance edits, epochs 2 / 3 | Epoch 2 AP50:95 | Epoch 3 AP50:95 |
|---|---:|---:|---:|
| Geometry-only | 0 / 0 | 12.828281 | 14.638331 |
| Fixed local | 8 / 10 | 11.145283 | 11.948369 |
| Online shared | 0 / 0 | 12.828281 | 14.203767 |
| Class-adaptive | 0 / 0 | 12.828281 | 14.801437 |

The three identical-treatment arms span 0.597669 AP points at epoch 3. This range comes from 24 images, one execution order and an intentionally different resume path for one arm. It is neither an uncertainty estimate nor a correction factor for the full pilot's 5,760-image validation results. It cannot establish that the full pilot's approximately 0.20-point class-adaptive deficit is merely numerical noise.

## What the code and tests do not establish

The smoke's `fork_state` checks equality of restored model/optimizer state and RNG. The observer wrapper records and verifies model state, gradients, optimizer state, RNG and module modes around probing and validation. Every saved observer audit reports success. These checks bound observable tensor/RNG changes at those boundaries; they do not audit CUDA workspace or kernel-selection state.

Only the class-adaptive smoke arm is intentionally interrupted after one training update. It rebuilds and restores the model, then checks inference before and after reload. The other arms continue uninterrupted. That is a valid checkpoint-reload check, but it confounds resume with arm identity when interpreting the subsequent training differences. The separate CPU resume test uses a toy detector, so it does not establish real Faster R-CNN CUDA trajectory equality.

The final `paired_summary` gate compares epoch coverage and augmentation-pairing hashes. It does not compare corresponding losses, gradients, parameters or optimizer updates. Consequently, `GPU FOUR-ARM SMOKE: PASS` means the implemented pipeline checks passed. It was never sufficient evidence that these same-treatment arms train identically. The archived PASS record explicitly declines a bitwise CUDA trajectory claim; the issue is the missing causal diagnostic, not a concealed promise in that file.

The package enables deterministic cuDNN convolution selection, disables benchmarking and TF32, and uses float32. It does not enable PyTorch's general deterministic-algorithm mode. TorchVision 0.18.0's CUDA RoIAlign backward implementation uses atomic accumulation and explicitly marks the operation nondeterministic. Thus GPU numerical nondeterminism is a concrete candidate cause. The archived run does not contain operator-level traces that establish it as the cause. [TorchVision 0.18.0 source](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/csrc/ops/cuda/roi_align_kernel.cu#L412), [PyTorch 2.3.0 reproducibility documentation](https://github.com/pytorch/pytorch/blob/v2.3.0/docs/source/notes/randomness.rst).

The cuDNN plan warnings in the log are not proof of the cause either. Execution continued and all saved losses/metrics are finite. No claim that the warnings explain the discrepancy follows from that observation.

## Smallest useful next check

Use the actual pinned CCAST environment and identical materialized batches from the same immutable checkpoint. Keep appearance identity throughout. Compare four conditions:

1. Uninterrupted control A.
2. An exact repeat of uninterrupted control A, to measure variation without probing or resume.
3. The same control with the probe call inserted, discarding its augmentation decision so treatment remains identity.
4. The same control with a save/rebuild/restore after one update, without probing.

Seven training updates per condition are enough to cover the interval in which the archived discrepancy is first observed. This is a debugging bound, not a statistical sample-size claim. At each update, record input and target hashes, learning rate, RNG state before/after each phase, loss components, model buffers, gradients, optimizer state and parameter updates. Keep the first differing tensor values so numerical magnitudes and locations can be inspected; hashes alone show difference, not severity. A small strict-determinism check can then identify unsupported operations or test an available deterministic path. Do not silently change the full experimental implementation to obtain a cleaner diagnostic.

A difference between the two plain control repeats would show that probing and resume are unnecessary for divergence, but would not exclude additional effects from either. If only the probe or resume condition differs, investigate that component. Inputs, state or RNG differing before the first training operation would identify a concrete preparation/restoration defect. A first difference inside backward with identical inputs and forward state would point toward the numerical kernel path. Report results as observed, without selecting tolerances after seeing favorable AP.

No new GPU package, training job or epoch-8 extension was started by this audit. The next action is a bounded reproducibility diagnostic, followed by a decision about whether a code fix or quantified numerical-variation limit is needed. The existing full pilot remains a descriptive single-seed result: class adaptation has not demonstrated an accuracy advantage over geometry-only.

## Reproducible audit artifacts

- `../audit_smoke.py`: extraction, provenance checks, independent COCO ground-truth assembly, validation replay, controller replay and cross-arm comparisons.
- `archive_audit.json`: archive and file inventory, source and manifest verification.
- `audit.json`: full numerical findings, original/local versions, per-epoch state hashes, evidence boundaries and controller support.
- `endpoint_results.csv`: eight smoke evaluations and appearance edit counts.
- `returned_records/`: preserved records extracted from the verified archive.

Run locally with `python reports/rdd_v2_pilot_v1/audit_smoke.py`. It reruns analysis of existing evidence and does not train or contact CCAST.
