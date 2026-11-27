# Diagnostic hash repair

CCAST job 797431.bright04 passed its 16 fixture tests, then failed while materializing training inputs. The four-condition GPU comparison had not begun. The failure archive SHA-256 reported by the user is `c44213140dcdf9e211d2c1b1848b1503b456b62f46965383e6f161814a1635fd`; the archive itself has not been downloaded or independently verified in this repair turn.

## Confirmed cause

The diagnostic's byte-hashing helper assumed that `contiguous().reshape(-1)` produced a final stride of one. That assumption is false for empty and certain singleton tensors. PyTorch can classify such tensors as contiguous while retaining a zero stride. Reinterpreting an empty NumPy-derived int64 label tensor as bytes then raises the exact reported error.

The error is reproduced locally both with a NumPy-derived empty label tensor and with an explicit zero-stride tensor. It also appears on local PyTorch 2.10.0, so a version difference is not needed to reproduce it. The production `torchvision_collate` function produces this empty label layout for a negative image. The previous diagnostic fixtures and actual-model check constructed targets manually using `torch.empty`, which missed it. This was a defect in the diagnostic and a gap in its tests, not evidence of a detector-training or augmentation-controller defect.

## Correction

The new `trace_state.raw_tensor` returns an empty byte payload for tensors with no elements. Dtype and shape remain part of the complete fingerprint. For nonempty tensors, it explicitly copies the flattened logical values into dense storage only if the final stride is not one. This preserves ordinary fingerprints and avoids the unsafe dtype reinterpretation. It does not change any training pixels, targets, model operations or optimizer updates.

Release `rdd_v2_repro_v2` preserves the original pilot source, checkpoint, selection, geometry, four conditions, seven updates and backend settings. The protocol differs only in release identification and a recorded amendment. New submission and output paths prevent overwriting `rdd_v2_repro_v1` or `runs/rdd_v2_repro_check_v1`.

## Verification performed

- 23 tests passed with zero failures or skips. The suite now obtains complete target dictionaries through the production collation function rather than handwritten substitutes.
- Regression cases cover the exact old failure, empty tensors with multiple shapes/dtypes, singleton zero strides, sliced/transposed/broadcast views and bfloat16 byte payloads.
- The suite exercises mixed and all-negative batches, complete on-disk fixture loading through geometry and materialization, and the existing fault-injection and checkpoint-continuation checks.
- New hashing agrees with independent NumPy logical-value bytes where NumPy supports the dtype. Ordinary tensors retain their original fingerprints.
- The actual 640-pixel Faster R-CNN CPU check uses production preparation and collation, including a negative image with labels of shape `[0]` and stride `[0]`. The original and instrumented updates match exactly in losses, gradient norm, parameters, gradients, optimizer state and RNG.
- All 29 original pilot source files are unchanged. The four new Bash scripts pass syntax checks.

These are local CPU checks, using a newer library stack than CCAST. The corrected A10 diagnostic is pending and is not presented as completed. There is still no result that identifies the cause of the earlier smoke-training divergence.

Evidence: `hash_reproduction.json`, `local_tests.txt`, `actual_model_check.json`, `metadata_preflight.json`, `shell_validation.json`, `failure_provenance.json`, and the sealed package's `LOCAL_VALIDATION.json`. Run the separately named corrected package using `CCAST_COMMANDS.md`.
