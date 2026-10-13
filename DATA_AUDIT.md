# RDD2022 acquisition and initial audit

Recorded October 11, 2026 from terminal output supplied by the student. These are remote CCAST observations, not an independent inspection of the remote dataset by Codex. The original detailed report remains on CCAST at `/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/dataset_audit.json`.

## Acquisition

- Source: `https://ndownloader.figshare.com/files/38030910`.
- Archive: `RDD2022_figshare.zip`, completed at 13,264,172,619 bytes.
- Outer ZIP integrity test: all seven embedded archives reported OK; no compressed-data errors detected.
- All seven sources' `train` folders were extracted under `dataset/raw`. Public test directories were not extracted.
- No publisher checksum comparison or locally computed archive checksum has been recorded.

## Metadata counts

Counts under D00/D10/D20/D40 are annotation objects, not distinct images or final usable crops. They include flagged boxes.

| Source | Images | XMLs | D00 | D10 | D20 | D40 | Empty XMLs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| China_Drone | 2,401 | 2,401 | 1,426 | 1,263 | 293 | 86 | 5 |
| China_MotorBike | 1,977 | 1,977 | 2,678 | 1,096 | 641 | 235 | 0 |
| Czech | 2,829 | 2,829 | 988 | 399 | 161 | 197 | 1,757 |
| India | 7,706 | 7,706 | 1,555 | 68 | 2,021 | 3,187 | 3,921 |
| Japan | 10,506 | 10,506 | 4,049 | 3,979 | 6,199 | 2,243 | 794 |
| Norway | 8,161 | 8,161 | 8,570 | 1,730 | 468 | 461 | 5,247 |
| United_States | 4,805 | 4,805 | 6,750 | 3,295 | 834 | 135 | 0 |
| Total | 38,385 | 38,385 | 26,016 | 11,830 | 10,617 | 6,544 | 11,724 |

There are 55,007 target-class objects before filtering. The D00:D40 count ratio is approximately 3.98:1. Class composition also varies substantially by source; country coverage needs checking when constructing splits.

Other labels: `Block crack`: 3; `D01`: 179; `D0w0`: 1; `D11`: 45; `D43`: 793; `D44`: 5,057; `D50`: 3,581; `Repair`: 1,046. Preserve these in the raw annotations. For the approved four-class task, exclude them from the target crop manifest; do not silently map labels to a target class.

## Checks and limitations

The script reported no missing image/XML partners, duplicate stems within a source, XML parsing failures, invalid XML dimensions, missing object labels, or box errors under its implemented bounds checks.

The student inspected the flagged object in `raw/Japan/train/annotations/xmls/Japan_001265.xml`: class D20, XML image size 600 x 600, bounding box `(xmin=198, ymin=474, xmax=198, ymax=475)`, difficult=0, truncated=0. Its width by endpoint subtraction is zero; inclusive coordinates would make it one pixel wide and two pixels high. Either interpretation makes it unsuitable as a standalone defect-classification crop. Recommendation: exclude this object from the derived crop dataset, retaining the original image/XML and any other usable objects in that image. Finalize and apply a consistent minimum-dimension rule across all classes after establishing coordinate conventions. No remote files have been changed and no filtered crop count has yet been verified.

The 11,724 empty XMLs contain no objects and therefore yield no labeled defect crops. They are not evidence of corruption and do not establish a verified healthy-road class. Images containing only non-target labels are a separate category not yet counted.

Image decoding, dimension checks, exact-duplicate review, and near-duplicate screening are complete. Both orientation-flagged India images have zero annotated objects and produce no crops. The US duplicate annotations differ slightly, as documented below. On October 12, grouped split manifests and explicit crop rules were created locally; remote crop generation and training-only visual QA remain pending. Capture-sequence identifiers are unavailable and perceptual screening is not exhaustive; geographic independence is not claimed.

## Class scope discussion

The student proposed including all observed labels, then accepted retaining the four approved classes for the main study. Additional verified categories can be considered as a later extension. The [dataset authors' repository](https://github.com/sekilab/RoadDamageDetector#damage-categories-to-be-considered) explicitly lists D00, D10, D20, and D40 for RDD2022. Observed annotation strings alone do not establish distinct, consistently defined categories: `D0w0` has one object and `Block crack` has three. Their intended meanings have not been established. Such counts cannot support credible independent training, validation, and test evaluation; augmented variants of the same example are not independent samples. Any extension needs verified label definitions, image-level class support, source coverage, and adequate held-out evaluation.

## Image audit completed on CCAST

`validate_images.py` checks complete Pillow decoding, actual versus XML dimensions, byte-identical files, and identical decoded RGB images across all source folders. It also flags non-RGB modes and EXIF orientations requiring review. It writes a timestamped `image_audit_*` report directory next to `raw`, containing `inventory.jsonl`, `summary.json`, and `run.json`. It does not modify raw data or perform splitting. Near-duplicate photographs are outside this script's checks. Byte and RGB duplicate group counts overlap and must not be added.

Three local fixture tests passed, covering corrupt inputs, dimension mismatches, cross-source duplicate detection including metadata differences, unchanged raw files, and rejection of invalid or existing output directories. The remote result below comes from the student's terminal output, separately from those synthetic tests.

The student's environment check reports Python at `/mmfs1/apps/pyenvs/anaconda3-2022.05/bin/python` with Pillow 9.0.1. The local fixture tests used Python 3.14.2 and Pillow 12.1.1. The remote `interact -h` confirms `-c` CPU cores, `-m` memory in GB, and `-t` walltime in hours. Planned interactive request: `interact -c 1 -m 4 -t 2`; run the audit only after the compute-node session starts. Do not infer the scheduler type from this wrapper's options.

The student subsequently reported `AUDIT COMPLETE` from a `gpu0020` prompt:

- Images checked and successfully decoded: 38,385 of 38,385.
- Flagged images: 2, both with `exif_orientation` flags: `India/train/images/India_000430.jpg` (720 x 720, orientation 8) and `India/train/images/India_004810.jpg` (720 x 720, orientation 6).
- No dimension mismatches or other issue types reported.
- Byte-identical groups: 1; RGB-identical groups: 1. The student printed the memberships: both contain exactly the same pair, `United_States/train/images/United_States_001623.jpg` and `United_States/train/images/United_States_003917.jpg`. The transferred files confirm identical image bytes and slightly different XML boxes; see the review below.
- Detailed report: `/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/dataset/image_audit_20261012T035733_790780Z/summary.json`.

The report-directory timestamp is UTC, corresponding to the evening of October 11 in America/Chicago. This audit does not establish annotation accuracy or absence of near duplicates. Do not automatically rotate flagged images without checking how their bounding boxes align, and do not delete duplicate raw files. Exact duplicates must be grouped within a single split; conflicting annotations should be reviewed before deciding whether to retain one or multiple derived samples.

## Direct review of transferred files

The student copied `annotation_review.tar.gz` into the shared project folder. Its eight image/XML files and saved reports were extracted under `tmp/annotation_review`. Direct XML inspection established that `India_000430.xml` and `India_004810.xml` contain no objects. They yield no labeled crops, so their EXIF flags do not affect this crop-classification study. No image rotation or annotation correction is required for them.

The US image bytes match. Both XMLs contain three D00 objects and one D10 object, with the same class sequence. Two boxes are identical; these two differ:

| Object index | Class | United_States_001623 | United_States_003917 |
| --- | --- | --- | --- |
| 2 | D10 | (236,449,603,479) | (253,454,626,477) |
| 3 | D00 | (429,456,488,557) | (431,465,482,557) |

Both annotation sets were viewed on the shared image in `reports/annotation_review/duplicate_comparison.png`. Decision: retain `United_States_001623` and its annotations as the deterministic, lexicographically first representative; exclude `United_States_003917` from derived samples. Preserve both raw files and do not merge the two annotation sets, which would create duplicate object samples. This removes three duplicate D00 records and one duplicate D10 record from the planned crop inventory, before any other filtering. It is not a verified final crop count. Hashes, records, and decisions are in `reports/annotation_review/decisions.json`.

## Near-duplicate candidates and annotation inventory

`screen_near_duplicates.py` is prepared for CCAST. It reads all XML records, retains object indices and coordinates in an annotation inventory, skips images without target objects, and retains the first image per exact RGB hash for screening. It checks eligible image file hashes against the completed audit, then uses a 128-bit horizontal/vertical difference hash with an initial Hamming radius of 8. The radius is a screening heuristic, not an established guarantee or a tuned result. Candidate images may be unrelated, and overlapping scenes with substantial viewpoint changes may be missed. Candidate matches are not automatically removed or assigned splits.

The script uses Pillow JPEG draft decoding for efficient thumbnail processing; see [Pillow documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.draft). Difference hashing is a perceptual similarity technique; see the [ImageHash project](https://github.com/JohannesBuchner/imagehash). This script implements its own horizontal/vertical combination and uses no new dependencies. It writes reports and up to 36 pair previews to `dataset/preparation_review`, and packages them as `dataset/preparation_review.tar.gz` for transfer. A 20,000-pair limit stops excessive candidate generation and explicitly reports an incomplete screen. Near-duplicate and capture-sequence leakage remain limitations until reviewed.

Four local tests passed: radius queries against exhaustive comparisons, exact-copy/negative filtering with deliberately colliding hashes, incomplete status when capped, and rejection of images modified since their initial audit. The subsequent remote screening completed and the student transferred `preparation_review.tar.gz` for direct inspection.

## October 12 screening results and split manifests

The complete report and annotation inventory are extracted under `tmp/preparation_review_extract/preparation_review`. The screen took 282.97 seconds and processed all 23,766 unique target-bearing images. Before exact deduplication there were 23,767 target-bearing images; 14,618 images had no target objects, including but not limited to the 11,724 empty XMLs.

There are 557 candidate pairs, all within their respective sources. They connect 733 images into 281 components, with a largest component of 14 images. All nine preview sheets, covering the 36 highest-ranked pairs, were viewed. They include repeated road scenes in China, Japan, India, and the US. This supports grouping related images but does not prove that every candidate is a duplicate. All 557 edges are conservatively enforced as grouping constraints; no near-duplicate candidate image is deleted. Remaining related views may be missed by the hash screen.

The annotated inventory includes 10,965 target boxes with fractional coordinates, all from Norway. An explicit continuous-edge crop convention was chosen: floor minima, ceil maxima, clip at image boundaries, no origin shift or context padding. Reject original horizontal or vertical spans below two pixels. This excludes only the previously identified Japan D20 object. The origin convention of every source is not established; the chosen extraction policy is recorded consistently for all experimental conditions rather than silently treating fractional coordinates as integer indices.

`make_splits.py` generated the versioned manifests in `dataset_splits_v1` using seed 42 and approximately 70/15/15 grouped allocation within each source, balancing images, class crops, and class-bearing images. Exact duplicate removal and the minimum-span rule leave 55,002 crop records from 23,766 images in 23,314 groups.

| Partition | Images | Crops | D00 | D10 | D20 | D40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 16,624 | 38,513 | 18,216 | 8,282 | 7,435 | 4,580 |
| Validation | 3,573 | 8,242 | 3,898 | 1,771 | 1,592 | 981 |
| Test | 3,569 | 8,247 | 3,899 | 1,776 | 1,589 | 983 |

Independent checks confirmed unique crop IDs, valid bounds, all source/class cells represented, no original image or source-file hash crossing partitions, no related group crossing partitions, and all 557 candidate edges kept within partitions. These are checks against identified relationships, not a guarantee that every repeated scene has been found.

`materialize_crops.py` is prepared and tested to verify source hashes, crop in stored orientation, preserve aspect ratio with LANCZOS resizing, and center each crop on a 224 x 224 neutral-gray canvas saved as PNG. Four tests passed for fractional/border coordinates, deterministic allocation, actual crop pixels/aspect ratio, raw-file integrity, and rejection of changed sources or cross-partition source images. The script makes training-only preview sheets for visual review; no random augmentation is performed at this stage.

The transfer package is `dataset_splits_v1.tar.gz`, containing manifests, configuration, exclusion records, both scripts, and methodological notes. Its SHA256 is `6e1050bc1bb3a47d0beb2733a95578194be13994a11ebe155e4ef13a4313e97a`. The crop-manifest SHA256 is `6c319939a0b913693935c5e6e83acb8e12b442c84e13fef3d79b740423a01e78`. This package is retained unchanged as the reproducible preparation record.

## Crop generation and visual QA complete

The transferred `processed_v1_qa.tar.gz` reports successful remote generation of all 55,002 crops from 23,766 images in 659.01 seconds using Pillow 9.0.1, finishing at 2026-10-12 05:31:54 UTC. Its manifest hash matches the versioned local manifest. Split/class counts exactly match the table above. Full crop files remain on CCAST; local confirmation is based on the completed run report and transferred preview bundle, not a second full-file audit.

All seven source-specific sheets were viewed, covering 112 training crops. The first column intentionally contains the most extreme aspect ratio for each source/class, so these are not a random quality sample. Extraction and letterboxing appear consistent; some defects are blurred, faint, or represented by very thin strips. Individual difficult examples are not evidence for changing their labels. No further samples were dropped or relabeled. See `reports/crop_review/REVIEW.md` for observations.

Training metadata show median width/height ratios of 0.680 (D00), 5.356 (D10), 1.382 (D20), and 1.652 (D40). Some 73.68% of training D10 crops have a long/short-side ratio of at least four. These are measured training-set facts; the inference is that annotation geometry exposed through letterbox padding could supply a classification shortcut. Add a training-fitted geometry-only comparator evaluated on validation data before interpreting CNN performance. This is not evidence of train/test leakage, and does not establish how much a CNN will rely on geometry. Model training and validation remain pending.
