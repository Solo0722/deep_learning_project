# Full-image training data loader v1

This package checks the data interface before detector feasibility experiments.
It does not train or evaluate a detector. The existing synthetic Faster R-CNN
and YOLOv8n checks establish execution of those model paths, not detection
accuracy, model ranking, or the augmentation controller's effectiveness.

## What is preserved

The input is the pinned `detection_splits_v1/train.jsonl`, with 26,863 images
and 38,530 target objects. The check never reads validation or test manifests
or their images. Image bytes, XML bytes, decoded RGB pixels, stored dimensions,
and EXIF orientation must match the frozen training records. No EXIF rotation
is applied. Raw images, XML, split manifests, and previous results are unchanged.

Every retained target is used, including fractional coordinates and difficult
flags. There is no crop-size threshold, target-size filter, class remapping of
other annotations, or automatic crowd/ignore treatment. Empty target lists
remain empty; they are not a verified healthy-road class. Non-target annotation
metadata remains in each loaded example.

## Shared geometry and model adapters

For this loader check, the complete RGB image is resized with Pillow bilinear
resampling to fit inside 640 x 640, then centered on padding with RGB value 114.
Boxes use the actual rounded resized width and height as separate scale factors;
they are not rounded to integer pixels. The inverse mapping is retained.
This is a documented input convention for engineering checks, not a selected
final detector resolution or an evaluation of preprocessing choices.

Both adapters receive exactly the same float32 RGB tensor in [0, 1]:

- Faster R-CNN: continuous XYXY pixel boxes; foreground IDs 1, 2, 3, 4;
  background remains zero. Every target has `iscrowd=0`.
- YOLO: normalized center-x, center-y, width, height boxes; foreground IDs
  0, 1, 2, 3; explicit image indices for targets within a batch.

Class order is D00, D10, D20, D40. Mixed positive/empty and entirely empty
batches are supported. Float32 conversion must not collapse a positive box.
Recovered boxes must agree within 0.002 original-image pixels in the check.
For subsequent Faster R-CNN runs using these prepared squares, configure the
detector's resize to the same size; do not let its default 800-pixel resize
silently change this convention. Configure later YOLO runs without a second
letterbox/augmentation stage when using this adapter.

`TrainingDataset` is a map-style dataset. Its items are prepared examples.
`torchvision_collate` returns images, targets, and inverse-geometry metadata.
`yolo_collate` returns the model batch dictionary and inverse-geometry metadata.
The first verification uses a CPU DataLoader with batch size 2 and zero worker
processes to check deterministic I/O and mapping; worker throughput is untested.

## Bounded real-image check

`selection.json` is produced by `select_training_cases.py` using only training
metadata. A deterministic greedy coverage rule selects source/annotation cases,
all foreground classes, mixed-class and overlapping boxes, fractional boxes,
difficult targets, and the available EXIF orientations; the largest training
image is included as a memory/geometry case. The saved selection contains
15 images covering 26 available features. This is an engineering coverage set,
not a random or representative performance sample.

Run tests in the project environment first:

```bash
python -m unittest discover -s tests -p test_rdd_data.py -v
```

Run the real-image check in a CPU compute allocation. Its CLI refuses login
nodes, a different environment/version stack, changed package/input hashes,
and an existing output directory or archive. From the extracted package:

```bash
python -u verify_training_loader.py \
  --root /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/raw \
  --splits /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_splits_v1 \
  --out /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/detection_loader_check_v1
```

The output records package sources, selection, versions, per-image checks,
elapsed time, and status. A failure during image checking records its error
and cannot produce a success archive. A complete run prints
`TRAINING DATA LOADER CHECK: PASS` and the archive checksum. It saves no image
previews and performs no detector forward passes. Commands are supplied to the
user one at a time; these instructions describe the package, not extra commands
to run ahead of the current step.

## Evidence boundary

`LOCAL_VALIDATION.json` records local synthetic tests and metadata-only selection.
The workstation uses a different PyTorch/Pillow stack from CCAST; the same tests
must pass in the pinned CCAST environment before processing the selected real
images. No raw RDD images were read locally for this package. Passing this check
will validate these data paths and cases, not annotation truth, full-dataset
loader coverage, controller probes, or detection accuracy. Detector choice and
the bounded feasibility experiment remain open.
