# Review of the full-pool audit and detection split v1

The complete CCAST audit covered 38,385 images and annotations. The downloaded
archive matches the checksum printed by the remote run, and all ten internal
hashes match. `audit_run.json` and `audit_summary.json` are exact copies from
that verified archive. The run took 588.61 seconds on gpu0020, using Python
3.12.3 and Pillow 10.3.0. The user's terminal recorded a CPU-only allocation;
this audit did not run a detector.

The screen produced 1,107 candidate edges: 558 target/target, 441 no-target/
no-target, and 108 target/no-target. All 23,766 comparable prior fingerprints
were unchanged. The graph contains 37,517 components before exclusions; its
largest component has 36 images. These are conservative similarity groups,
not a determination that all pairs depict the same road or are duplicates.

## Decisions made before full-image detector training

The India component rooted at India_000129 contains 36 images: six previously
in training, three in validation, and 27 previously omitted no-target images.
The Norway component rooted at Norway_000115 contains seven images: one
previously in training, three in validation, and three previously omitted
no-target images. No component mixes the previous test partition with
development data.

Both conflicting components are assigned wholly to training in the new
manifest. This moves six previous validation images, containing 17 target
boxes, into training. Moving the training members into validation would make
the validation population contain previously trained examples. Splitting the
components would retain a screened similarity connection across partitions.
Conservative grouping can overgroup unrelated images; that limitation is
preferable here to selecting edges using model outcomes. No image previews or
model scores were used to make these decisions.

The earlier exact-copy exclusion United_States_003917 is retained in favor of
United_States_001623. Their pixels match, but two box annotations differ; no
annotation merging or selection based on performance is performed.

India_006389 has one annotation, `D0w0`. Its intended class cannot be established
from this record. The full image is excluded from this version, preventing
either an unsupported label correction or treating this potentially mislabeled
target as background. This is a new, documented exclusion; no raw data change
is made.

Japan_001265's object index 3 has zero width. That object is omitted while the
image and its other three positive-area targets remain. The old crop cutoff
is not used for detection. All six difficult target boxes and both images with
EXIF orientation flags are retained. Flags are preserved; targets marked
difficult are ordinary GT under the declared study convention. Readers must
use stored orientation and continuous XYXY coordinates.

The remaining historical assignments are fixed. New no-target groups with
historical anchors follow those anchors. Unanchored no-target groups are
allocated within source toward 70/15/15, balancing total image counts and
empty-XML versus other-label-only counts. Seed 42 breaks allocation-order ties.
This creates a mixed-source detection split; it does not evaluate unseen
countries or certify unseen geographic routes. Validation remains development
data. The old crop split and earlier crop results are separate artifacts.

## Saved manifest counts

| Partition | Images | Target-bearing images | No-target images | Target boxes |
| --- | ---: | ---: | ---: | ---: |
| Training | 26,863 | 16,630 | 10,233 | 38,530 |
| Validation | 5,760 | 3,567 | 2,193 | 8,225 |
| Test | 5,760 | 3,569 | 2,191 | 8,247 |
| Total | 38,383 | 23,766 | 14,617 | 55,002 |

Every source/class combination retains positive support in every partition.
Some source/class cells are small; this check is coverage, not a sample-size
justification. No-target images comprise 11,724 empty-XML images and 2,893
other-label-only images. Neither category establishes healthy pavement or
complete target annotation.

## Checks and current status

Eight local synthetic tests passed for `make_detection_splits.py`. They cover
prior test/development conflicts, allowed and disallowed historical moves,
no-target bridges, small positive boxes, exclusions, label/index preservation,
changed inputs, graph mismatches, determinism, and overwrite protection.

`verify_manifests.py` separately read the generated JSONL/CSV files and checked
them against the returned audit and old memberships. It verified exact retained
image coverage; all 55,002 targets and annotation flags; every previous test
assignment; all 1,107 fresh and 557 historical candidate edges; group and exact
file/RGB hash separation; all 14 transfer members; and LF-only checksum text.
It found no cross-partition screened groups, retained candidate edges, or exact
hashes. These checks are recorded in `verification.json` and do not establish
annotation semantics, complete geographic independence, or detector efficacy.

The manifest bundle has SHA-256
`89d1c16954187ef0126f69c670b83875c8b4ff8863aa5a1864c264df91b0cce4`
and size 6,776,871 bytes. The user's pasted CCAST terminal output confirms the
same archive hash, successful extraction, and all 13 listed file checks passing
under `dataset/detection_splits_v1`. Deployment is recorded in `deployment.json`;
step 2 is complete. No full-image detector training has been executed by this step.

## Reproduction

From the project root:

```bash
python -m unittest discover -s tests -p test_make_detection_splits.py -v
```

To create a new output from the pinned audit and original crop membership:

```bash
python make_detection_splits.py \
  --audit-bundle detection_audit_v1.tar.gz \
  --previous-images dataset_splits_v1/images.csv \
  --policy reports/detection_split_review_v1/policy.json \
  --out detection_splits_reproduction
```

The manifest assignment and data are deterministic. A newly compressed archive
can have a different hash because tar/gzip record creation metadata. Never
overwrite the delivered v1 archive when reproducing it. To recheck the delivered
files and archive, run `python reports/detection_split_review_v1/verify_manifests.py`.
