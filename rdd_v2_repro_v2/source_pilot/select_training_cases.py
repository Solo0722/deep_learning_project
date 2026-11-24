"""Select engineering coverage cases from frozen training metadata only."""
import argparse
import json
from pathlib import Path

from rdd_data import read_manifest


def features(row):
    result = {f"source:{row['source']}:{'positive' if row['targets'] else 'no_target'}",
              f"category:{row['annotation_category']}"}
    labels = {t["label"] for t in row["targets"]}
    result.update(f"class:{label}" for label in labels)
    if len(labels) > 1:
        result.add("mixed_class")
    if row["exif_orientation"] != 1:
        result.add(f"orientation:{row['exif_orientation']}")
    for i, target in enumerate(row["targets"]):
        box = target["box_xyxy"]
        if any(float(v) != int(v) for v in box):
            result.add("fractional_coordinates")
        if min(box[2] - box[0], box[3] - box[1]) < 2:
            result.add("positive_side_below_two_pixels")
        if str(target["difficult"]) == "1":
            result.add("difficult_target_retained")
        for other in row["targets"][i + 1:]:
            b = other["box_xyxy"]
            if target["label"] != other["label"] and min(box[2], b[2]) > max(box[0], b[0]) and min(box[3], b[3]) > max(box[1], b[1]):
                result.add("cross_class_overlap")
    return result


def make_selection(rows, manifest_sha256):
    tagged = [(row, features(row)) for row in sorted(rows, key=lambda r: r["image"])]
    uncovered = set().union(*(tags for _, tags in tagged))
    required = sorted(uncovered)
    selected = []
    while uncovered:
        row, tags = min(tagged, key=lambda item: (-len(item[1] & uncovered), item[0]["image"]))
        if not tags & uncovered:
            raise ValueError("Coverage selection made no progress")
        selected.append(row)
        uncovered -= tags
    largest = min(rows, key=lambda r: (-r["width"] * r["height"], r["image"]))
    if largest["image"] not in {r["image"] for r in selected}:
        selected.append(largest)
    return {
        "purpose": "Deterministic training-only software coverage; not a performance sample",
        "manifest_sha256": manifest_sha256,
        "required_features": required,
        "largest_training_image": largest["image"],
        "selected": [{"image": row["image"], "image_sha256": row["image_sha256"],
                      "features": sorted(features(row))} for row in sorted(selected, key=lambda r: r["image"])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(Path(__file__).with_name("reference.json").read_text(encoding="utf-8"))
    rows = read_manifest(args.manifest, reference["manifest_sha256"])
    plan = make_selection(rows, reference["manifest_sha256"])
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"Selected {len(plan['selected'])} training images covering {len(plan['required_features'])} features")


if __name__ == "__main__":
    main()
