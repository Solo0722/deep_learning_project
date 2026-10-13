"""Build reproducible image-grouped RDD crop manifests from audited records.

No image pixels are required here. All screened similarity edges become
conservative grouping constraints, not deletion decisions. No test scores or
model outputs are used to allocate groups.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("D00", "D10", "D20", "D40")
SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def crop_bounds(box, size):
    """Explicit continuous XYXY edge convention; no hidden origin shift.

    Keep fractional XML coordinates, round outward and clip at image edges.
    Reject endpoint spans <2 pixels before rounding. This is a chosen crop
    convention, not a claim that every source's annotation origin is known.
    """
    x1, y1, x2, y2 = box
    width, height = size
    if not all(math.isfinite(v) for v in box):
        raise ValueError("Nonfinite box")
    if min(x1, y1) < 0 or x2 > width or y2 > height or x2 < x1 or y2 < y1:
        raise ValueError("Box outside image or reversed")
    if min(x2 - x1, y2 - y1) < 2:
        return None
    return (max(0, math.floor(x1)), max(0, math.floor(y1)),
            min(width, math.ceil(x2)), min(height, math.ceil(y2)))


class Groups:
    def __init__(self, names):
        self.parent = {name: name for name in names}

    def find(self, name):
        node = name
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def allocate(vectors, seed):
    """Balance image counts, per-class crop counts and class-bearing images.

    Allocate the groups with the largest relative class contribution first;
    minimize squared deviations scaled by target counts. A deterministic local
    move pass improves balance. The seed only breaks allocation-order ties.
    """
    total = [sum(v[k] for v in vectors.values()) for k in range(9)]
    targets = [[ratio * n for n in total] for ratio in RATIOS]
    current = [[0] * 9 for _ in SPLITS]
    assigned = {}
    tie = lambda key: hashlib.sha256((str(seed) + ":" + key).encode()).hexdigest()
    order = sorted(vectors, key=lambda key: (
        -max(vectors[key][k] / max(total[k], 1) for k in range(1, 9)),
        -vectors[key][0], tie(key)))

    def change_cost(split, vector, sign):
        return sum(((current[split][k] + sign * vector[k] - targets[split][k]) ** 2
                    - (current[split][k] - targets[split][k]) ** 2)
                   / max(targets[split][k], 1) for k in range(9))

    for key in order:
        vector = vectors[key]
        destination = min(range(3), key=lambda s: (change_cost(s, vector, 1), s))
        assigned[key] = destination
        current[destination] = [a + b for a, b in zip(current[destination], vector)]
    for _ in range(3):
        moved = False
        for key in sorted(order, key=tie):
            source, vector = assigned[key], vectors[key]
            # Never remove the final image or final class-bearing image from a split.
            if any(current[source][k] - vector[k] <= 0 for k in (0, 5, 6, 7, 8)):
                continue
            candidates = [(change_cost(source, vector, -1) + change_cost(dest, vector, 1), dest)
                          for dest in range(3) if dest != source]
            improvement, destination = min(candidates)
            if improvement < -1e-10:
                current[source] = [a - b for a, b in zip(current[source], vector)]
                current[destination] = [a + b for a, b in zip(current[destination], vector)]
                assigned[key], moved = destination, True
        if not moved:
            break
    if any(current[s][k] == 0 for s in range(3) for k in (0, 5, 6, 7, 8)):
        raise ValueError("Insufficient source/class coverage in at least one split")
    return {key: SPLITS[value] for key, value in assigned.items()}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(review, inventory, output, seed=42):
    summary = json.loads((review / "summary.json").read_text(encoding="utf-8"))
    run = json.loads((review / "run.json").read_text(encoding="utf-8"))
    if not summary["screen_complete"] or run["status"] != "complete":
        raise ValueError("Near-duplicate screening is incomplete")
    if digest(inventory) != run["inventory_sha256"]:
        raise ValueError("Image audit differs from the near-duplicate screen input")
    records = read_jsonl(review / "annotations.jsonl")
    audited = {r["image"]: r for r in read_jsonl(inventory)}
    if len(records) != len(audited) or len({r["image"] for r in records}) != len(records):
        raise ValueError("Annotation inventory is incomplete or repeats paths")
    pairs = read_jsonl(review / "candidate_pairs.jsonl")
    if len(pairs) != summary["candidate_pairs"]:
        raise ValueError("Candidate pair list is incomplete")
    images, excluded_images, excluded_objects = {}, [], []
    target_images = []
    for record in records:
        image = record["image"]
        prior = audited[image]
        if record["size"] != prior["size"] or record["rgb_sha256"] != prior["rgb_sha256"]:
            raise ValueError("Audit metadata mismatch: " + image)
        if record.get("duplicate_of"):
            duplicate = record["duplicate_of"]
            if prior["rgb_sha256"] != audited[duplicate]["rgb_sha256"]:
                raise ValueError("Invalid exact duplicate alias")
            excluded_images.append({"image": image, "reason": "exact_duplicate", "canonical": duplicate})
            continue
        if record["target_counts"]:
            target_images.append(image)
        crops = []
        for obj in record["objects"]:
            if obj["label"] not in LABELS:
                continue
            bounds = crop_bounds(obj["box"], record["size"])
            if bounds is None:
                excluded_objects.append({"image": image, "object_index": obj["index"],
                                         "label": obj["label"], "reason": "endpoint_span_below_2_pixels"})
            else:
                crops.append((obj, bounds))
        if not crops:
            excluded_images.append({"image": image, "reason": "no_usable_target_crops", "canonical": ""})
        else:
            images[image] = dict(record, crops=crops)
    groups = Groups(target_images)
    for pair in pairs:
        groups.join(pair["a"], pair["b"])
    group_images = defaultdict(list)
    for image in images:
        group_images[groups.find(image)].append(image)
    sources = defaultdict(dict)
    for key, members in group_images.items():
        countries = {images[name]["source"] for name in members}
        if len(countries) != 1:
            raise ValueError("Cross-source similarity component requires a joint allocator")
        vector = [len(members)] + [0] * 8
        for name in members:
            counts = Counter(obj["label"] for obj, _ in images[name]["crops"])
            for k, label in enumerate(LABELS):
                vector[1 + k] += counts[label]
                vector[5 + k] += int(counts[label] > 0)
        sources[next(iter(countries))][key] = vector
    assignments = {}
    for source, vectors in sorted(sources.items()):
        assignments.update(allocate(vectors, seed))
    rows, image_rows, statistics = [], [], {s: {"images": 0, "groups": 0, "crops": 0,
        "class_crops": Counter(), "source_images": Counter(),
        "source_class_crops": defaultdict(Counter), "source_class_images": defaultdict(Counter)} for s in SPLITS}
    for key in assignments:
        statistics[assignments[key]]["groups"] += 1
    for image, record in sorted(images.items()):
        group_id = groups.find(image)
        split = assignments[group_id]
        stats = statistics[split]
        source = record["source"]
        counts = Counter(obj["label"] for obj, _ in record["crops"])
        stats["images"] += 1
        stats["source_images"][source] += 1
        stats["source_class_images"][source].update(counts.keys())
        image_rows.append({"image": image, "source": source, "split": split,
                           "group_id": group_id, "crops": len(record["crops"])})
        for obj, bounds in record["crops"]:
            label = obj["label"]
            sample_id = source + "__" + Path(image).stem + "__o%04d" % obj["index"]
            rows.append({"sample_id": sample_id, "split": split, "source": source,
                         "image": image, "xml": record["xml"], "object_index": obj["index"],
                         "label": label, "label_id": LABELS.index(label), "group_id": group_id,
                         "xmin": obj["box"][0], "ymin": obj["box"][1],
                         "xmax": obj["box"][2], "ymax": obj["box"][3],
                         "left": bounds[0], "top": bounds[1], "right": bounds[2], "bottom": bounds[3],
                         "original_width": record["size"][0], "original_height": record["size"][1],
                         "file_sha256": audited[image]["file_sha256"], "xml_sha256": record["xml_sha256"],
                         "crop_path": split + "/" + label + "/" + sample_id + ".png"})
            stats["crops"] += 1
            stats["class_crops"][label] += 1
            stats["source_class_crops"][source][label] += 1
    image_split = {row["image"]: row["split"] for row in image_rows}
    violations = [p for p in pairs if p["a"] in image_split and p["b"] in image_split
                  and image_split[p["a"]] != image_split[p["b"]]]
    if violations or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Split leakage or sample ID collision detected")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "crops.csv", rows, list(rows[0]))
    write_csv(output / "images.csv", image_rows, ["image", "source", "split", "group_id", "crops"])
    write_csv(output / "excluded_images.csv", excluded_images, ["image", "reason", "canonical"])
    write_csv(output / "excluded_objects.csv", excluded_objects, ["image", "object_index", "label", "reason"])
    config = {"version": "dataset_splits_v1", "seed": seed, "ratios": dict(zip(SPLITS, RATIOS)),
              "labels": dict(enumerate(LABELS)), "crop_size": 224,
              "bbox_convention": "XML values as continuous XYXY edges; floor minima, ceil maxima; clip; no origin shift or context padding",
              "minimum_endpoint_span": 2,
              "pixel_preprocessing": "Stored orientation; RGB; preserve aspect ratio; LANCZOS resize; centered 224x224 canvas with RGB(127,127,127); PNG",
              "grouping": "All connected dHash candidates at radius 8 remain within one split; candidates are retained, not deleted",
              "split_balance": "Per source; balance images, per-class crops, and class-bearing images; deterministic greedy allocation and up to three improvement passes",
              "limitations": ["Not a geographic or capture-sequence holdout; such IDs are unavailable",
                              "Perceptual screening may miss related views; no claim of eliminating all leakage",
                              "Mixed-source coordinate origins are not proven; extraction convention is explicit and fixed",
                              "Classification uses annotation-provided crops, not full-scene detection"],
              "source_hashes": {"annotations.jsonl": digest(review / "annotations.jsonl"),
                                "candidate_pairs.jsonl": digest(review / "candidate_pairs.jsonl"),
                                "inventory.jsonl": digest(inventory)},
              "manifest_sha256": digest(output / "crops.csv"),
              "generator_sha256": digest(Path(__file__))}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"images": len(images), "groups": len(group_images), "crops": len(rows),
              "candidate_edges": len(pairs), "cross_split_candidate_edges": len(violations),
              "excluded_images": dict(Counter(r["reason"] for r in excluded_images)),
              "excluded_objects": excluded_objects, "splits": statistics,
              "group_size_histogram": dict(sorted(Counter(map(len, group_images.values())).items()))}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split in SPLITS:
        print(split, json.dumps(statistics[split], default=dict))
    print("Images:", len(images), "Crops:", len(rows), "Cross-split candidate edges:", len(violations))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.review, args.inventory, args.output, args.seed)
