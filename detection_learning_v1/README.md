# Training-only detector learning check v1

This package checks whether the pretrained Faster R-CNN and YOLOv8n paths can
learn from the 15 training images already used for integration. It adds a tested
COCO evaluator and real checkpoint restoration. It does not estimate validation
or test performance, select an architecture, or test the augmentation controller.

The prior integration check passed eight random-weight updates per model. Those
updates established execution, not learning. This is a separate run with fresh
official COCO weights and a fixed budget. All previous releases and results are
preserved. The source, settings, and decision rule are saved before CCAST execution.

## Fixed run

Both models receive the same stored-orientation full-image pixels and target boxes.
The immutable loader, reference, selector, and selection files are copied from the
passed integration release. The set contains 15 images from seven sources, 16
target boxes across four classes, and eight images with no target annotation.
These cases were selected for interface coverage, not representativeness.

| Setting | Value |
| --- | --- |
| Image preparation | Full image fit to 640 square, bilinear resize, centered padding 114 |
| Batch order | Same eight fixed batches repeated 25 times; final batch has one image |
| Budget | 200 updates per detector; 375 image presentations per detector |
| Optimizer | SGD, learning rate 0.001, momentum 0.9, weight decay 0.0005 |
| Precision | FP32, no AMP or TF32 |
| Augmentation / EMA / LR schedule | None |
| Seed | 42 before constructing each model; exact CUDA reproducibility is not claimed |
| Evaluation | All 15 training images at steps 0, 50, 100, 200 |
| Checkpoints | Save and reload at steps 100 and 200 |

The fixed order retains the mixed, all-empty, and final single-image cases.
Validation and test files are never opened. There is no early stopping on a
favorable score, automatic extension, or hyperparameter search. Each model starts
from its own official checkpoint, not the preceding model or the integration run.

## Pretrained transfer

Both weight files must match the recorded sizes and full SHA-256 values in
`protocol.json` and the `SHA256SUMS` saved beside them. No network download or
automatic dependency installation occurs during a run.

Faster R-CNN uses the COCO_V1 ResNet50-FPN architecture with FrozenBatchNorm2d
and epsilon zero, matching the official pretrained builder. The complete
91-class source state is loaded strictly before replacing its classification
and class-specific regression predictors with five-class predictors: background plus
D00, D10, D20, D40. All five backbone stages are trainable; FrozenBatchNorm
statistics and affine buffers remain fixed. Internal RGB mean/std normalization
is retained, and a hook verifies that the already prepared geometry is unchanged.

YOLOv8n is first constructed with the COCO 80-class hidden classifier widths and
loaded strictly. Only the three final classification convolutions are replaced
with four-output layers. Their new biases use the usual YOLO initial class prior;
pretrained class-agnostic box-regression layers and biases are retained. The hidden
classifier widths are therefore preserved from COCO, which differs from building
`nc=4` directly in the earlier random-weight smoke. BatchNorm running statistics
remain active. The effective configuration records all transferred state keys.
Checkpoint reconstruction follows the identical architecture without reloading
the original COCO weights.

The models differ in size, normalization, source training recipe, and transferred
heads. Equal images and update counts do not establish a fair model-quality
ranking. A future scratch comparison must explicitly match architecture and
normalization; switching a TorchVision builder's weights argument alone can
change BatchNorm behavior. These settings are a learning diagnostic, not a tuned
baseline recipe.

## Metrics and interpretation

Both detectors use score threshold 0.001, class-aware NMS IoU 0.5, and at most
100 detections per image. YOLO permits multiple class candidates per box, as does
the Faster R-CNN output path. Predictions are mapped through the inverse resize
and padding into the original stored-image coordinates, clipped to image bounds,
and discarded only if clipping collapses the box. Clipping counts are recorded.
All retained targets, including difficult annotations, are ordinary ground truth
under the frozen study convention.

The official `pycocotools` bbox evaluator computes AP over IoU 0.50:0.05:0.95,
AP50, AP75, AR100, and per-class AP. Every selected image remains in the evaluation,
including images with no target annotation. Their false predictions can lower AP.
The package also counts predicted target boxes on these images at confidence 0.25.
No-target images are not asserted to be healthy or exhaustively annotated.
Undefined metrics for categories without ground truth are null, not zero; all
four categories are present in this fixed real selection.

Raw COCO ground truth and all prediction lists are saved, allowing metrics to be
recomputed without access to images. Focused tests include perfect, empty, wrongly
classified, missed, poorly localized, and false-positive predictions; they also
cover fractional inverse coordinates and padding-only boxes.

The project engineering review rule requires final training AP50 at least 0.50
and a gain of at least 0.10 over step zero to label a model's result
`learning_signal_observed`. These are chosen diagnostic cutoffs, not a benchmark
target, statistical significance test, or expected generalization accuracy. A
completed run below either cutoff is `needs_review` and still archives all results.
Inspect per-class curves and predictions before deciding the next experiment.
Do not automatically lengthen the run or rank architectures from this small set.

## Checkpoints, failures, and outputs

At steps 100 and 200, save FP32 model parameters and buffers, SGD momentum/state,
Python/NumPy/Torch/CUDA RNG states, completed step, and protocol/data/weight/code
identity. Reconstruct a fresh identical model and optimizer, load the checkpoint,
and require exact equality of their saved states and restored RNG states.
Training steps 101 through 200 execute after the step-100 reload. At step 200,
repeat inference and compare ordered predictions with absolute tolerances of
0.001 original-image coordinate units and 0.000001 score units, both with
relative tolerance 0.00001. CUDA differences beyond these tolerances stop the
check; they are not silently accepted.

The focused CPU test also compares an uninterrupted stochastic training trajectory
with a restored trajectory, including momentum and all three CPU random sources.
That test does not establish bit-identical CUDA reruns. This package exercises
an internal scheduled reload, not a general command-line recovery system for
arbitrarily interrupted jobs. Failed or partial runs require inspection.

Each training step checks finite losses, gradients, parameters, and an actual
classifier update. Input signatures must match the fixed reference batches for
both models. Any failed preflight, corruption, numerical error, changed input,
missing target or checkpoint discrepancy stops execution. Partial outputs and
the traceback remain available; no completed-result archive is created. Existing
output folders, archives and checkpoints are never overwritten.

The result directory contains `run.json`, `steps.jsonl`, the ground truth,
scheduled metrics and predictions, paired input records, package versions, exact
sources, and four full checkpoints. Full checkpoints remain on CCAST and occupy
roughly 0.7 GB combined. The small `.tar.gz` transfer archive excludes their binary
contents but includes their hashes, sizes and saved-step records. These tiny-set
checkpoints are diagnostic artifacts; a later baseline should start afresh from
the prescribed initialization.

## CCAST sequence

Transfer and verify the new archive and package checksum list before requesting
a GPU. Use four CPU cores, one GPU, 16 GB RAM, and one hour. Activate `pytorch`,
then the existing project `rdd_detection_v1` environment. Commands are supplied
one at a time in the chat; these examples document the sequence.

```bash
python -m unittest discover -s tests -p test_learning.py -v
```

```bash
python -u learning.py \
  --root /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/raw \
  --splits /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_splits_v1 \
  --weights /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/weights/detection_v1 \
  --out /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_learning_check_v1
```

Completion prints `TRAINING-ONLY LEARNING CHECK: COMPLETE`, the records archive,
and its checksum. The `learning_review` field is separate from execution status.
Download and inspect the records before committing to a larger training run.

## References and local evidence

Architecture handling follows the pinned
[TorchVision backbone builder](https://raw.githubusercontent.com/pytorch/vision/v0.18.0/torchvision/models/detection/backbone_utils.py),
[Faster R-CNN builder](https://raw.githubusercontent.com/pytorch/vision/v0.18.0/torchvision/models/detection/faster_rcnn.py),
and [Ultralytics model source](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.182/ultralytics/nn/tasks.py).
Metrics use the [official COCO evaluator](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py).
`LOCAL_VALIDATION.json` describes local checks and exact versions. Local CPU tests
use synthetic images; the actual CCAST GPU run and RDD learning results remain
pending at package release.
