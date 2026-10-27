# YOLO evaluation-only diagnostic v1

The 200-update learning check completed, but YOLOv8n's final training AP50 was
zero. Independent local evaluation reproduced every saved AP value. None of
the 1,307 final retained boxes overlaps any target at IoU 0.5, even if class
labels are ignored. The records also show a classification-loss and classifier
update spike on the no-target batch at step 182. These observations do not yet
identify the cause or establish that the YOLO architecture cannot learn RDD.

This new package tests one specific question: does the choice of BatchNorm
statistics contribute to the gap between training and inference behavior?
It performs **zero optimizer steps**. Existing data, predictions, checkpoint
files and the released learning code are never overwritten.

For each existing YOLO checkpoint at steps 100 and 200, the package performs:

1. Normal inference with saved running statistics. Predictions must reproduce
   the original saved prediction list within the original tolerances.
2. Inference using each current batch's statistics in BatchNorm2d only. All
   other modules, including the detection head, stay in evaluation mode.
   Running-stat tracking is disabled for this pass, so buffers are not updated.
3. Normal inference again. It must reproduce the first pass after the
   intervention's attributes are restored.

Each pass verifies that model parameters, persistent buffers and RNG states
remain exactly unchanged. The full source checkpoint hash is checked before
loading and after evaluation. Checkpoints use the previously verified
weights-only loading path. No optimizer is constructed or invoked.

The same 15 training images, fixed two-image batches (last batch one image),
640-pixel preparation, labels, NMS, score threshold and COCO evaluator are
retained. The whole diagnostic uses 48 model forwards: two checkpoints, three
normalization modes and eight batches. It never opens validation or test data.
Input signatures and ground truth must match the previous run exactly.

Alongside AP and predictions, each pass records raw candidate overlap with
targets before NMS, class-score ranges, candidate counts above the fixed 0.001
threshold, foreground assignments, soft target-score sums, and loss components
computed without backpropagation. These measurements help distinguish poor
raw localization, score filtering, assignment behavior and normalization effects.
The soft assignment diagnostics describe the inference weights in each mode;
they do not reconstruct historical training gradients.

This is an exploratory diagnostic chosen after examining the failed learning
check. A change in scores under batch statistics isolates sensitivity to that
normalization intervention on this fixed set. It does not establish a general
remedy, justify deployment with batch statistics, or measure generalization.
Unchanged poor results leave optimization, initialization and other causes
unresolved. The observed gradient-update spike is a separate clue: this
diagnostic cannot prove what caused it or whether gradient clipping would help.
Do not automatically train longer, select a detector or alter the protocol.

`protocol.json` pins the previous records, checkpoint hashes, frozen split
configuration and sibling `detection_learning_v1` release. Both package
directories must sit under the same CCAST project directory. That existing
release provides the unchanged architecture, loader, evaluator and checkpoint
utilities. This package is an additional diagnostic, not a replacement release.

Run on an allocated GPU job with the existing `pytorch` and `rdd_detection_v1`
environments. The chat supplies the transfer, checksum, test and execution
commands one at a time. The focused unit-test command is:

```bash
python -m unittest discover -s tests -p test_diagnose.py -v
```

The eventual diagnostic command is:

```bash
python -u diagnose.py \
  --root /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/raw \
  --splits /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_splits_v1 \
  --previous /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_learning_check_v1 \
  --out /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_diagnostic_check_v1
```

Completion produces a new records archive with code, protocol, metrics,
predictions and raw-output/assignment summaries. Model checkpoint binaries are
not copied. Any source mismatch, prediction-replay discrepancy, mutation or
numerical error stops the diagnostic and preserves the error report.

The implementation follows the pinned [PyTorch 2.3 BatchNorm forward logic](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.0/torch/nn/modules/batchnorm.py)
and [Ultralytics 8.4.182 assignment/loss implementation](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.182/ultralytics/utils/loss.py).
Local validation uses synthetic inputs and a newer CPU PyTorch stack; the
actual checkpoint diagnostic on CCAST remains pending until its results return.
