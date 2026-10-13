# RDD crop dataset, version 1

Prepared October 12, 2026. These are split and crop manifests; the PNG files have not yet been generated on CCAST.

| Partition | Source images | Crops | D00 | D10 | D20 | D40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 16,624 | 38,513 | 18,216 | 8,282 | 7,435 | 4,580 |
| Validation | 3,573 | 8,242 | 3,898 | 1,771 | 1,592 | 981 |
| Test | 3,569 | 8,247 | 3,899 | 1,776 | 1,589 | 983 |
| Total | 23,766 | 55,002 | 26,013 | 11,829 | 10,616 | 6,544 |

Seed 42; approximate 70/15/15 allocation. The allocator balances source-image counts, class crop counts, and class-bearing image counts within each source. Every source/class combination has support in every partition. Some combinations remain small: India D10 has 10 crops from 9 images in each held-out partition, so per-source metrics require care.

All 557 candidate similarity edges remain within partitions. They connect 733 images into 281 nontrivial components; other eligible images are singleton groups. All candidates are conservatively grouped, including those not visually reviewed. The 36 highest-ranked pair previews were inspected and include repeated scenes. No assertion is made that all 557 pairs are duplicates. Similarity screening can miss related views, and capture-sequence or geographic identifiers were unavailable. This is a mixed-source grouped split, not a test of generalization to unseen countries or entirely unseen roads.

One exact copy, United_States_003917, is excluded in favor of United_States_001623. Their object classes match but two box coordinates differ. The one D20 object with coordinates (198,474,198,475), index 3 in Japan_001265, fails the minimum two-pixel endpoint-span rule. Both EXIF-flagged India images have no objects and yield no crops. Raw files are preserved.

## Coordinate and pixel policy

Interpret the stored XML coordinates as continuous XYXY edges, round minima downward and maxima upward, and clip to the image boundary. Do not shift the origin or add contextual padding. Reject any target box whose original horizontal or vertical endpoint span is below two pixels. This removes one target box in this inventory. Fractional values are preserved until rounding.

This is an explicit extraction convention, not proof of a common annotation origin across countries. The authors describe a Pascal VOC XML structure. Conventional VOC loaders sometimes assume one-based inclusive integer coordinates and subtract one from the minima; see [Detectron2's documented assumption](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/pascal_voc.py). The actual RDD files here include zero-valued minima and 10,965 target boxes with fractional coordinates, all in Norway. We therefore record the numeric-edge convention explicitly and use it consistently for every experimental condition. See the [RDD authors' repository](https://github.com/sekilab/RoadDamageDetector) for the dataset structure.

Crop from stored RGB pixels without EXIF transposition. Resize with LANCZOS while preserving aspect ratio and center on a 224 x 224 RGB(127,127,127) canvas. Save lossless PNG files. This is deterministic preprocessing; training augmentation is a later step. Annotation-provided cropping means results evaluate crop classification, not full-scene detection. Bounding-box shape and context can influence this classification task and should be acknowledged in the manuscript.

## Run on a CCAST compute node

From `/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research`:

```bash
/mmfs1/apps/pyenvs/anaconda3-2022.05/bin/python -u dataset_splits_v1/materialize_crops.py dataset/raw dataset_splits_v1/crops.csv dataset/processed_v1
```

Outputs must be new. The script verifies the manifest hash, source image/XML hashes, dimensions, and partition consistency. Source files remain unchanged. `dataset/processed_v1/run.json` is marked complete only when all crops and previews are written. An interrupted run remains incomplete; do not train from it.

The output includes train/val/test class folders, `crops.csv`, `config.json`, per-crop checksums, a summary, and training-only QA sheets. A separate `dataset/processed_v1_qa.tar.gz` contains the sheets and run metadata for transfer to the local workspace. Inspect these before starting baseline training. Validation is for development; test evaluation is reserved until models and augmentation policies are fixed.

## Reproducibility and checks

`config.json` records the seed, allocation and preprocessing rules, source inventory hashes, and the crop-manifest SHA256. `make_splits.py` generates the manifests; `materialize_crops.py` writes the images. `images.csv` records original-image assignments; exclusion files document omitted records.

Independent checks verified 55,002 unique crop IDs, 23,766 disjoint original images, valid crop bounds, no duplicate source hashes across partitions, no related groups split across partitions, all 557 similarity edges within partitions, class/source coverage, and exclusion of the reviewed records. Four additional tests cover coordinate handling, reproducible allocation, crop pixels/aspect ratio, source integrity, and rejection of cross-partition source images or changed XML files. Remote crop generation and visual QA remain pending.
