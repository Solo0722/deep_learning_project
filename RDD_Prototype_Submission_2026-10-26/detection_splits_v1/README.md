# Full-image detection split v1

This manifest is for complete images and four target classes. `train.jsonl`,
`val.jsonl`, and `test.jsonl` contain relative raw-image/XML paths, original
byte hashes, continuous XYXY boxes, source groups, and original annotation flags.
`images.csv` is the membership table. Raw images remain in the audited root.

The foreground class_index is D00=0, D10=1, D20=2, D40=3. A detector adapter
must translate this explicitly when its API reserves index zero for background.
Use stored pixel orientation; suppress automatic EXIF rotation. Keep empty
target lists. Non-target labels are preserved in other_annotations without
remapping. Difficult and truncated target boxes are retained as ordinary GT;
this is an explicit study convention, not an assertion about official scoring.

Every screened component stays within one partition, including connections
through excluded images. Prior test assignments are preserved. The six reviewed
validation images in groups with training exposure move to training. New
unanchored no-target groups are allocated by source toward 70/15/15 proportions.
Seed 42 determines group-order ties. Observed proportions are in summary.json.
An exact-copy exclusion and one ambiguous-label exclusion are documented.
The one zero-area target is omitted; its image retains its other valid targets.

These are heuristic grouping constraints, not proof that all similar pairs are
duplicates or all roads/capture sequences are independent. Empty annotations
are not proof of healthy pavement. This is a mixed-source evaluation, not an
unseen-country benchmark. Model effectiveness has not been evaluated here.

The archive and member hashes in config.json identify the full CCAST audit.
reviewed_policy.json records the choices made after that audit, before detector
training. Reproduce with make_detection_splits.py --audit-bundle <audit.tar.gz>
--previous-images <frozen-images.csv> --policy reviewed_policy.json --out <new-dir>.
The output must be new. The generator needs only the Python standard library.
Verify this package from its directory with: sha256sum -c SHA256SUMS
