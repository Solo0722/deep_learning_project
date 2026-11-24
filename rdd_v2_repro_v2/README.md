# V2 training reproducibility diagnostic

This package investigates the unexplained training differences in the saved GPU smoke. It is not a new augmentation experiment and produces no accuracy estimate.

Release 2 repairs a diagnostic-only hashing failure in job 797431.bright04. Empty NumPy-derived int64 labels can be zero-stride tensors that PyTorch considers contiguous; directly reinterpreting them as bytes raised an exception. The hasher now returns an empty payload for empty tensors and explicitly packs nonempty tensors when their final stride is not one. Shape and dtype remain part of the fingerprint. Ordinary tensor fingerprints and the scientific comparison are unchanged.

The tests now use the actual production collation function with every target field, and cover negative images, all-negative batches, empty and nonempty zero-stride layouts, and complete file loading/materialization. The real Faster R-CNN CPU update check also uses production collation. The failed run and package remain under their original v1 paths; this release writes separate v2 results. No GPU diagnostic condition ran before the v1 failure.

The source pilot is preserved byte-for-byte under `source_pilot/`. Its model construction, optimizer recipe, checkpoint restoration, geometry, input loader and observer are reused. The new `repro.py` instruments the original optimizer update. `trace_state.py` compares tensor state exactly and records numerical difference sizes. The local tests deliberately inject errors and check equivalence with the original update. See `LOCAL_VALIDATION.json` for executed checks and their environment limits.

## Fixed design

All conditions begin at the pinned original identity epoch-1 checkpoint. They use the same 26 training images and seven batches as smoke epoch 2. Geometry-only inputs are materialized once; both recorded geometry decision and input-pairing hashes must match the archived smoke. There are zero appearance edits throughout.

| Condition | Intervention |
|---|---|
| control_a | Seven uninterrupted updates; reference trajectory |
| control_b | Repeat the same seven updates |
| probe_only | Run the original two-image frozen observer before training; record and discard its policy; train the same seven batches |
| resume_only | Save/rebuild/restore after the first update, using the original checkpoint functions; finish the other six updates; no observer call |

The model, optimizer and RNG fork is checked for every condition. Inputs/targets, learning rates, model modes, losses, buffers, gradients, parameter updates, optimizer state and component RNG states are checked or recorded at every update. Probe and resume boundaries must preserve observable model/optimizer/RNG/mode state. Gradients are intentionally not checkpointed and are cleared before every next update, matching the original training implementation.

The four phases are: before forward, after forward, after backward and after the optimizer step. Comparisons use exact state fingerprints and numerical differences, without a tuned acceptance tolerance. The first differing phase is identified for each condition. Bounded numeric examples and small full tensor pairs are exported; reference state snapshots remain on CCAST for further inspection.

## Interpretation and limits

- Different plain controls mean that probing and resume are unnecessary for divergence. They do not exclude additional effects from those paths or prove a particular CUDA kernel caused it.
- A probe/resume difference with exact plain controls narrows the next investigation to that path. One serial run per condition is not a statistical estimate of its effect.
- Exact results mean that the discrepancy was not reproduced in this instrumented diagnostic. They do not establish the cause of the earlier discrepancy or guarantee bitwise training generally.
- CPU copies and synchronization can change GPU scheduling. Inputs are materialized rather than reloaded during each condition. These are deliberate debugging controls, and limit conclusions about the previous execution.
- GPU backend settings match the source recipe; general deterministic-algorithm mode is not silently enabled. No change to the experimental method is introduced by this diagnostic.

No validation or test images are read. There is no AP evaluation, longer training stage or automatic resubmission. The seven updates cover the interval in which the saved smoke had already diverged; they do not establish a learning or convergence budget.

## Execution

Upload the verified archive from Windows, then run `bash submit.sh` in this package on a CCAST login node. Do not run `repro.py` on the login node. The batch script requests one A10, four CPUs and 32 GB RAM; it activates the existing environments and runs the fixture tests before the actual CUDA diagnostic.

The PBS ceiling is one hour. The diagnostic has a 40-minute in-process limit, within a 50-minute job-script deadline. Checkpoints/reference snapshots may require approximately 5 GiB of scratch storage; the script requires 10 GiB free and caps reference snapshots at 8 GiB. Full tensors remain on CCAST and are excluded from the downloadable archive. Each exported full tensor pair is capped at 8 MiB; smaller numeric samples remain available when that cap applies.

Monitor with `bash status.sh`. SSH disconnection does not terminate the submitted batch job. The fixed output directory refuses overwrite and duplicate submission. A failed or incomplete attempt must be inspected, not bypassed or automatically resubmitted.

## Returned evidence

Output: `runs/rdd_v2_repro_check_v2/`.

- `context.json`: source hashes, environment, effective configuration and backend flags.
- `materialized_inputs.json`: image order, target counts, input hashes and augmentation traces.
- `suite/comparison.json`: completed updates, first differences and interpretation limits.
- Each condition's `steps/`: phase observations, complete RNG records and comparisons against control A.
- `probe_only/`: raw observer evidence, policy and boundary checks.
- `resume_only/`: exact restoration checks and checkpoint metadata.
- `tensor_evidence/`: bounded first-difference values.
- `status.json`, and `failure.json` when appropriate.
- `checksums.json`: manifest for returned records.

The job prints the archive path and SHA-256. `complete_with_divergence` is an observed diagnostic result, not a failed program. `complete_exact_on_this_diagnostic` is a bounded equality result, not approval for scientific claims. `failed` and `budget_stopped` preserve partial evidence. Inspect the records before deciding whether a code repair or a numerical-variation investigation is warranted.
