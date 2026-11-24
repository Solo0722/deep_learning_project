# Bounded V2 strict-determinism check

This package tests whether strict execution makes the existing small Faster R-CNN trajectory repeatable at a measured resource cost. It does not change the augmentation controller, evaluate AP, or start a longer training experiment.

The earlier diagnostic (PBS 797450.bright04, records SHA-256 `3ec5a81f35d667f71994f80956fa9ec691bbd2c9fe0b704173f6d2a97f9eedd9`) first differed during backward at update 1, including in the plain repeat. Saved inputs, first losses, RNG states and probe/restore boundaries matched. Strict mode is an execution-setting intervention; success would not uniquely identify the original nondeterministic operation.

## Frozen inputs and execution

The 41 files under `source_repro/` are an unchanged copy of the released reproduction package, including its manifest and unchanged original pilot sources. `strict_check.py` reuses its actual training update, materializer, checkpoint operations, observer and state comparator. Run the new entry point through `submit.sh`; the nested legacy launch scripts are preserved source evidence, not the commands for this check.

Use the existing identity epoch-1 checkpoint, the same 26 selected training images and seven materialized batches (4/4/4/4/4/4/2). Appearance remains identity. Geometry, target IDs, boxes, order, pixel hashes, optimizer and learning rate remain as in the diagnostic. The final batch is entirely negative. No validation/test manifest or pixels are read.

The job sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python. The worker requires fresh CUDA initialization, then enables `torch.use_deterministic_algorithms(True, warn_only=False)`, deterministic cuDNN, no benchmarking, no TF32 and deterministic uninitialized-memory filling. The backend flags are checked at every recorded phase. It does not compile or replace a numerical kernel.

TorchVision 0.18.0 dispatches strict CUDA RoIAlign to its Python tensor implementation. A transparent call counter verifies that this function executes and records feature/RoI shapes; its arguments and outputs are unchanged. The installed RoIAlign source is copied and hashed in the results. Strict-mode source references: [PyTorch 2.3.0](https://github.com/pytorch/pytorch/blob/v2.3.0/torch/__init__.py), [TorchVision 0.18.0](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/ops/roi_align.py), [cuBLAS workspace guidance](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility). Sources inspected 2026-10-29.

## Gated stages

1. Two independent single-update executions from the pinned state. They must be byte-exact in recorded inputs, state, losses/buffers, gradients, optimizer updates and RNG, and remain within the resource limits.
2. If the first stage passes, restart from the original checkpoint and run the same four conditions for seven updates each: uninterrupted reference, plain repeat, probe-only, and resume-only. The probe policy is discarded; resume happens after update 1. Stop on the first observed mismatch. An incomplete stage is not a pass.

There are at most 30 diagnostic optimizer updates (two gate updates plus 28 suite updates). These are not new scientific training endpoints. A deterministic path may differ numerically from the old native-kernel path; the primary check is exact repeatability within the new path, not exact agreement across implementations.

## Resource limits and failure records

- PBS: one A10, four CPU cores, 32 GB RAM, one-hour walltime. Two CPU compute threads retain the diagnostic configuration.
- Worker wall limit: 40 minutes, additionally bounded by a 50-minute job deadline. A parent outside CUDA can terminate a stalled worker and archive partial evidence.
- Instrumented update limit: 120 seconds; a still-running update has a hard supervisor limit of 180 seconds. Two-image probing is bounded at 300 seconds, with a 360-second hard limit.
- PyTorch allocator cap: 90% of the assigned GPU's total memory; peak allocated/reserved memory is recorded. This cap does not include all external CUDA-library/context allocations.
- Require 10 GiB free scratch space; retain at most 8 GiB of reference snapshots. Large `.pt` snapshots stay on CCAST, outside the download archive.

These are operational bounds for a short debugging allocation, not accuracy thresholds or promises of affordable full training. Phase and update timings include synchronization, tensor copying and record writing. They are not uninstrumented training throughput. Final performance/cost decisions require inspection of the actual measurements.

Failures are preserved as distinct statuses: strict mismatch, unsupported deterministic operation, CUDA OOM, resource limit, budget stop, ordinary failure or abrupt worker failure. There is no warning-only mode, tolerance relaxation, smaller replacement batch, retry, silent fallback, overwrite or automatic resubmission. A killed worker can leave an observed-update count below the exact number of optimizer calls that reached the device; no completion is inferred from an interrupted record.

## Use and outputs

Upload the supplied tarball from Windows PowerShell, verify/extract it on the CCAST login node, and run `bash submit.sh`. No interactive GPU request or manual environment activation is needed. The PBS script activates both existing environments and reruns the old regression tests and new strict-check tests before executing the GPU worker. A disconnect does not cancel the batch job.

Outputs: `../runs/rdd_v2_strict_check_v1/` and `../runs/rdd_v2_strict_check_v1_records.tar.gz`, with an archive checksum sidecar. Live logs: `../runs/strict_job_logs/<PBS_JOBID>.log`. `bash status.sh` shows scheduler state and recent log lines.

Inspect `status.json`, `summary.json`, `execution.json`, per-step comparison/resource records, probe/restore boundaries, `worker.log`, and any `failure.json`. `COMPLETE.json` is written only when both stages completed exactly with all 30 observed updates. A pass is limited to this small run. It does not establish generalization, accuracy improvement, multi-seed stability, all-machine determinism or approval to extend training.

`LOCAL_VALIDATION.json` states what was actually tested locally and what remains pending on the pinned CCAST CUDA stack. CPU tests are implementation checks, not a substitute for this GPU diagnostic.
