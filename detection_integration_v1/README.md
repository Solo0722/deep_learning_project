# Real-image detector integration check v1

This package connects the verified full-image loader to the detector paths that
already passed synthetic CCAST checks. It tests software integration on the
same 15 selected training images, containing 16 target boxes. It is not a
detector accuracy comparison, a training baseline, or a controller experiment.

## Fixed scope

Run Faster R-CNN ResNet50-FPN and YOLOv8n sequentially on one allocated GPU.
Both start with random weights and process one pass through the same selection
in the same order: eight batches at batch size two, including the final single
image. The sixth batch contains two images with no target boxes. Every batch
performs a forward pass, backward pass, one SGD update at learning rate 0.001,
and inference. There is no momentum, weight decay, mixed precision, augmentation,
or pretrained download. This is a bounded integration check, not a selected
training recipe. Seed 42 is reset before each model; exact CUDA repeatability
is not claimed.

The four copied loader files are byte-identical to the verified loader release.
The training manifest and selection remain pinned. Each raw image/XML pair,
decoded RGB hash, dimensions, and stored EXIF orientation are checked by that
loader. Validation and test manifests/images are not read. Existing datasets,
checkpoints, release packages, and results remain unchanged.

The complete image is fit into a 640-pixel square with bilinear resizing and
centered padding, as in the passed CPU check. Both model entry points receive
identical float32 RGB pixels in [0, 1], verified by equality and input hashes.
Faster R-CNN retains its own internal mean/std normalization; its resize is
fixed at 640, with a runtime hook checking image sizes and unchanged target
boxes. YOLO receives the prepared tensor directly without another preprocessing
or augmentation pipeline. The final study resolution is still undecided.

Targets include all retained fractional boxes and difficult flags. Faster
R-CNN uses foreground IDs 1 through 4; YOLO uses IDs 0 through 3. Empty-target
images and entirely empty batches are preserved. No raw images or previews
are written to the result archive.

## Checks and outputs

Each step requires finite losses, finite gradients, a nonzero classifier
gradient, an actual classifier weight change, finite updated parameters, and
valid inference shapes, coordinates, class IDs, and scores. Named loss values
are diagnostic only and must not be ranked across the two architectures.

YOLO NMS uses zero confidence and at most 100 detections to exercise inference
with random weights. This is not an evaluation threshold. Its decoded boxes
may extend beyond the image; the check does not silently clip them or invent
an evaluation policy. Prediction counts and tensor hashes are recorded, not
accuracy metrics. No checkpoint is saved; these short random-weight updates
are not a training run to resume.

The output contains `run.json`, `steps.jsonl`, a runtime package-version list,
and the exact package sources. Step records preserve image IDs, input tensor
hashes, geometry, target counts, canonical class labels, loss components,
update checks, and inference counts/hashes. Corresponding batches must have
the same signature for both models. Diagnostic elapsed time and peak CUDA
memory are recorded for resource troubleshooting. Timing includes validation
and transfer overhead, so it is not a throughput benchmark.

A failure records its traceback and partial progress and cannot print the
success marker or create a success archive. A hard process termination may
leave a report marked running; that is incomplete, not a passed run. Existing
output directories/archives are refused, and there is no automatic retry.

## CCAST execution

Verify the transferred archive and its `SHA256SUMS`, then use an interactive
allocation with four CPU cores, one GPU, 16 GB RAM, and one hour. Activate the
shared `pytorch` Conda environment and the project's `rdd_detection_v1` virtual
environment. Commands are supplied one at a time in the chat; the examples
below document the package and are not a request to run ahead.

From this package directory, run the focused tests:

```bash
python -m unittest discover -s tests -p test_integration.py -v
```

Then run the integration check:

```bash
python -u integration.py \
  --root /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/raw \
  --splits /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_splits_v1 \
  --out /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_integration_check_v1
```

Only a completed run prints `REAL-IMAGE DETECTOR INTEGRATION: PASS` and the
archive path/checksum. Inspect the returned records before beginning a longer
feasibility experiment. The architecture choice, useful probe signals, learned
detection quality, full training/evaluation pipeline, and controller remain
unresolved by this check.

## Evidence and implementation references

The prior CPU check archive is pinned in `protocol.json`; it passed all 15
selected images in CCAST job 794025. `LOCAL_VALIDATION.json` describes local
tests and their limitations. Local unit tests use synthetic inputs and mocked
CCAST/CUDA orchestration. They cannot establish real GPU execution.

Model interfaces were checked against the
[TorchVision 0.18 builder documentation](https://docs.pytorch.org/vision/0.18/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
and the pinned Ultralytics 8.4.182
[model](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.182/ultralytics/nn/tasks.py)
and [loss source](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.182/ultralytics/utils/loss.py).
The two model paths also passed the earlier synthetic GPU checks in CCAST job
794021. Those results and this software check are not detector efficacy evidence.
