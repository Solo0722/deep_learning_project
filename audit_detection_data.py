"""Audit the full RDD image pool before making a detection split.

Requires Pillow and the unchanged screen_near_duplicates.py helper. Run on a
compute node. Inputs and old splits are never modified. All images are checked
against the saved byte hashes, and all perceptual hashes are computed in the
same current runtime. This includes images with no target-class annotations.
No model evaluation, crop generation, split assignment, or preview is performed.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import platform
import socket
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image

from screen_near_duplicates import HashIndex, TARGETS, dhash128, hamming, inside, read_objects


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Components:
    def __init__(self, names):
        self.parent = {name: name for name in names}

    def find(self, name):
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class PairLimitReached(RuntimeError):
    pass


def input_paths(inventory, review, previous_images):
    return {
        "inventory.jsonl": inventory,
        "image_audit_run.json": inventory.with_name("run.json"),
        "annotations.jsonl": review / "annotations.jsonl",
        "fingerprints.jsonl": review / "fingerprints.jsonl",
        "candidate_pairs.jsonl": review / "candidate_pairs.jsonl",
        "screen_run.json": review / "run.json",
        "screen_summary.json": review / "summary.json",
        "previous_images.csv": previous_images,
    }


def load_inputs(inventory, review, previous_images, reference):
    paths = input_paths(inventory, review, previous_images)
    expected = read_json(reference)["sha256"]
    if set(expected) != set(paths):
        raise ValueError("Reference hash roles do not match the required inputs")
    hashes = {name: digest(path) for name, path in paths.items()}
    for name, actual in hashes.items():
        if actual != expected[name]:
            raise ValueError("Reference checksum mismatch: " + name)
    audit_run, screen_run = read_json(paths["image_audit_run.json"]), read_json(paths["screen_run.json"])
    screen_summary = read_json(paths["screen_summary.json"])
    if audit_run.get("status") != "complete" or screen_run.get("status") != "complete":
        raise ValueError("Previous audit and screening must be complete")
    if not screen_summary.get("screen_complete"):
        raise ValueError("Previous screening summary is incomplete")
    if screen_run["inventory_sha256"] != hashes["inventory.jsonl"]:
        raise ValueError("Previous screening used a different image inventory")

    records = read_jsonl(inventory)
    annotated = read_jsonl(paths["annotations.jsonl"])
    images = {r["image"]: r for r in records}
    annotations = {r["image"]: r for r in annotated}
    if len(images) != len(records) or len(annotations) != len(annotated):
        raise ValueError("Repeated image path in input inventories")
    if set(images) != set(annotations) or len(images) != audit_run["images_expected"]:
        raise ValueError("Input inventories do not cover the same complete image pool")
    for name, r in images.items():
        a = annotations[name]
        if not r.get("decoded") or not r.get("rgb_sha256"):
            raise ValueError("Image did not pass the prior decoding audit: " + name)
        if any(issue["type"] not in {"exif_orientation", "non_rgb_mode"}
               for issue in r.get("issues", [])):
            raise ValueError("Unresolved image-audit issue: " + name)
        if any(r[k] != a[k] for k in ("source", "xml", "size", "rgb_sha256")):
            raise ValueError("Image/annotation inventory disagreement: " + name)
        counts = Counter(o["label"] for o in a["objects"] if o["label"] in TARGETS)
        if dict(counts) != a["target_counts"]:
            raise ValueError("Cached target counts disagree with object records: " + name)
        if a.get("duplicate_of"):
            other = a["duplicate_of"]
            if other not in images or images[other]["rgb_sha256"] != r["rgb_sha256"]:
                raise ValueError("Invalid historical exact-duplicate link: " + name)

    fingerprints = read_jsonl(paths["fingerprints.jsonl"])
    cached = {}
    for row in fingerprints:
        name, value = row["image"], row["dhash128"]
        if name in cached or len(value) != 32:
            raise ValueError("Invalid or duplicate previous fingerprint")
        cached[name] = int(value, 16)
    eligible = {name for name, a in annotations.items() if a["target_counts"] and not a.get("duplicate_of")}
    if set(cached) != eligible:
        raise ValueError("Previous fingerprints do not cover the expected target-bearing images")

    with previous_images.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    legacy = {row["image"]: row for row in rows}
    if len(legacy) != len(rows) or not set(legacy).issubset(eligible):
        raise ValueError("Invalid previous split membership")
    old_groups = defaultdict(set)
    for name, row in legacy.items():
        if row["split"] not in {"train", "val", "test"} or row["source"] != images[name]["source"]:
            raise ValueError("Invalid previous split row: " + name)
        old_groups[row["group_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in old_groups.values()):
        raise ValueError("The previous split already divides a recorded source cluster")
    old_pairs = read_jsonl(paths["candidate_pairs.jsonl"])
    if len(old_pairs) != screen_summary["candidate_pairs"]:
        raise ValueError("Previous candidate pair count disagrees with its summary")
    for pair in old_pairs:
        if pair["a"] not in cached or pair["b"] not in cached:
            raise ValueError("Previous candidate references an unknown fingerprint")
        if hamming(cached[pair["a"]], cached[pair["b"]]) != pair["distance"]:
            raise ValueError("Previous candidate distance disagrees with its fingerprints")
    return images, annotations, cached, legacy, old_pairs, hashes


def check_file_membership(root, images):
    actual_images, actual_xmls = set(), set()
    for folder in sorted(root.glob("*/train")):
        for path in (folder / "images").iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                actual_images.add(path.relative_to(root).as_posix())
        for path in (folder / "annotations/xmls").iterdir():
            if path.is_file() and path.suffix.lower() == ".xml":
                actual_xmls.add(path.relative_to(root).as_posix())
    if actual_images != set(images):
        raise ValueError("Raw image file membership differs from the pinned inventory")
    expected_xmls = {r["xml"] for r in images.values()}
    if len(expected_xmls) != len(images) or actual_xmls != expected_xmls:
        raise ValueError("Raw XML file membership differs from the pinned inventory")


def target_boxes(record):
    """Retain positive-area targets; report clipping and invalid targets explicitly.

    Preserve continuous coordinates for detector supervision. Rasterization for
    augmentation masks is a later operation. No crop-size cutoff is inherited.
    """
    width, height = record["size"]
    targets, issues = [], []
    for obj in record["objects"]:
        if obj["label"] not in TARGETS:
            continue
        box = obj["box"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Nonfinite or malformed target box: " + record["image"])
        clipped = [max(0.0, min(float(width), box[0])), max(0.0, min(float(height), box[1])),
                   max(0.0, min(float(width), box[2])), max(0.0, min(float(height), box[3]))]
        detail = {"image": record["image"], "object_index": obj["index"], "label": obj["label"],
                  "original_box": box, "clipped_box": clipped}
        if box[2] <= box[0] or box[3] <= box[1] or clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            issues.append(dict(detail, issue="nonpositive_target_area"))
            continue
        if clipped != box:
            issues.append(dict(detail, issue="target_clipped_to_image"))
        targets.append({"object_index": obj["index"], "label": obj["label"], "box_xyxy": clipped,
                        "difficult": obj.get("difficult"), "truncated": obj.get("truncated")})
    return targets, issues


def write_components(output, groups, images, annotations, legacy, counts_by_image):
    members = defaultdict(list)
    for name in sorted(images):
        members[groups.find(name)].append(name)
    conflicts, affected = [], Counter()
    with (output / "source_clusters.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["image", "source", "source_cluster_id", "previous_split", "previous_group_id",
                  "valid_targets", "annotation_category", "historical_duplicate_of"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, names in sorted(members.items()):
            assignments = Counter(legacy[n]["split"] for n in names if n in legacy)
            if len(assignments) > 1:
                conflicts.append({"source_cluster_id": key, "members": names,
                                  "previous_split_counts": dict(assignments),
                                  "newly_considered_images": sum(n not in legacy for n in names)})
                affected.update(assignments)
            for name in names:
                previous = legacy.get(name, {})
                writer.writerow({"image": name, "source": images[name]["source"],
                                 "source_cluster_id": key, "previous_split": previous.get("split", ""),
                                 "previous_group_id": previous.get("group_id", ""),
                                 "valid_targets": counts_by_image[name][0],
                                 "annotation_category": counts_by_image[name][1],
                                 "historical_duplicate_of": annotations[name].get("duplicate_of", "")})
    with (output / "previous_split_conflicts.jsonl").open("w", encoding="utf-8") as stream:
        for row in conflicts:
            stream.write(json.dumps(row) + "\n")
    return {"source_clusters": len(members),
            "largest_source_cluster": max(map(len, members.values()), default=0),
            "clusters_crossing_previous_splits": len(conflicts),
            "previously_assigned_images_in_conflicts": dict(affected),
            "cross_source_clusters": sum(len({images[n]["source"] for n in names}) > 1
                                         for names in members.values())}


def audit(root, inventory, review, previous_images, reference, output, radius=8, max_pairs=200000):
    root, inventory, review = Path(root).resolve(), Path(inventory).resolve(), Path(review).resolve()
    previous_images, reference, output = (Path(p).resolve() for p in (previous_images, reference, output))
    if not 0 <= radius <= 16 or max_pairs < 1:
        raise ValueError("Radius must be 0..16 and max-pairs must be positive")
    if not root.is_dir():
        raise ValueError("Raw image root does not exist")
    protected = (root, review, inventory.parent, previous_images.parent, reference.parent)
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError("Output must be separate from raw inputs and previous artifact directories")
    if output.exists() or output.with_suffix(".tar.gz").exists():
        raise FileExistsError("Output or bundle already exists; use a new output directory")
    images, annotations, cached, legacy, old_pairs, hashes = load_inputs(
        inventory, review, previous_images, reference)
    check_file_membership(root, images)
    output.mkdir(parents=True)
    started = time.monotonic()
    meta = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(), "python": platform.python_version(), "pillow": PIL.__version__,
            "root": str(root), "input_sha256": hashes, "reference_sha256": digest(reference),
            "code_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("screen_near_duplicates.py"))},
            "images_expected": len(images), "radius": radius, "max_pairs": max_pairs,
            "fingerprint_policy": "Recompute all 128-bit dHashes using JPEG draft at 256px in one runtime",
            "pixel_policy": "Stored orientation; no EXIF transpose; no image writes",
            "target_policy": "Four exact label strings; continuous XYXY edges clipped to bounds; positive area; no crop-size cutoff",
            "partition_policy": "Audit only; keep historical groups and candidate edges as constraints; assign no new splits",
            "limitations": ["Perceptual matches are grouping candidates, not verified duplicates",
                            "This screen cannot establish complete geographic or sequence independence",
                            "No target annotation does not establish a verified healthy-road image",
                            "Historical train/val exposure remains relevant when resolving split conflicts",
                            "No model evaluation or visual inspection of held-out images is performed"]}
    run_path = output / "run.json"
    write_json(run_path, meta)
    print("Output:", output, flush=True)
    print("Images to verify and screen:", len(images), flush=True)
    print("All old splits remain unchanged. New split assignment is deferred.", flush=True)
    groups = Components(images)
    for pair in old_pairs:
        groups.join(pair["a"], pair["b"])
    prior_groups = {}
    for name, row in legacy.items():
        key = row["group_id"]
        if key in prior_groups:
            groups.join(prior_groups[key], name)
        else:
            prior_groups[key] = name
    identical = defaultdict(list)
    for name, row in images.items():
        identical[row["rgb_sha256"]].append(name)
    exact_groups = []
    for names in identical.values():
        if len(names) > 1:
            names.sort()
            for name in names[1:]:
                groups.join(names[0], name)
            signatures = {json.dumps(annotations[n]["objects"], sort_keys=True) for n in names}
            exact_groups.append({"images": names, "annotation_records_identical": len(signatures) == 1,
                                 "historical_duplicate_links": {n: annotations[n]["duplicate_of"]
                                                                for n in names if annotations[n].get("duplicate_of")}})
    write_json(output / "exact_duplicate_groups.json", exact_groups)
    counts, labels, other_labels, annotation_flags = Counter(), Counter(), Counter(), Counter()
    by_source, counts_by_image = defaultdict(Counter), {}
    pair_counts, distance_counts = Counter(), Counter()
    index, changed_cached, processed, pair_total = HashIndex(), [], 0, 0
    try:
        with (output / "fingerprints.jsonl").open("w", encoding="utf-8") as fp, \
                (output / "candidate_pairs.jsonl").open("w", encoding="utf-8") as pairs, \
                (output / "validated_annotations.jsonl").open("w", encoding="utf-8") as ann, \
                (output / "annotation_issues.jsonl").open("w", encoding="utf-8") as issue_file:
            for number, name in enumerate(sorted(images), 1):
                record, original = images[name], annotations[name]
                objects, xml_hash = read_objects(inside(root, record["xml"]))
                if xml_hash != original["xml_sha256"] or objects != original["objects"]:
                    raise ValueError("XML changed or cached objects disagree: " + record["xml"])
                data = inside(root, name).read_bytes()
                if hashlib.sha256(data).hexdigest() != record["file_sha256"]:
                    raise ValueError("Image changed since image audit: " + name)
                with Image.open(io.BytesIO(data)) as picture:
                    if list(picture.size) != record["size"]:
                        raise ValueError("Image dimensions disagree with inventory: " + name)
                    picture.draft("RGB", (256, 256))
                    value = dhash128(picture)
                fp.write(json.dumps({"image": name, "dhash128": "%032x" % value}) + "\n")
                if name in cached and value != cached[name]:
                    changed_cached.append(name)
                targets, issues = target_boxes(original)
                for issue in issues:
                    issue_file.write(json.dumps(issue) + "\n")
                    counts[issue["issue"]] += 1
                category = ("target_bearing" if targets else "invalid_target_only" if original["target_counts"]
                            else "only_other_labels" if objects else "empty_xml")
                if name in legacy and not targets:
                    raise ValueError("Previously split image has no valid detection targets: " + name)
                counts_by_image[name] = (len(targets), category)
                counts[category] += 1
                by_source[record["source"]][category] += 1
                by_source[record["source"]]["images"] += 1
                counts["valid_target_objects_before_exact_dedup"] += len(targets)
                counts["historical_exact_copy_images"] += bool(original.get("duplicate_of"))
                counts["images_with_exif_orientation_flag"] += record.get("exif_orientation", 1) != 1
                labels.update(t["label"] for t in targets)
                other_labels.update(o["label"] for o in objects if o["label"] not in TARGETS)
                for target in targets:
                    annotation_flags["difficult=%s,truncated=%s" % (target["difficult"], target["truncated"])] += 1
                enriched = dict(original, image_sha256=record["file_sha256"], targets=targets,
                                annotation_category=category, exif_orientation=record.get("exif_orientation", 1))
                ann.write(json.dumps(enriched) + "\n")
                processed = number
                for distance, other in index.query(value, radius):
                    if pair_total >= max_pairs:
                        raise PairLimitReached("Candidate limit reached; screening is incomplete")
                    left_target, right_target = bool(annotations[other]["target_counts"]), bool(original["target_counts"])
                    kind = "target_target" if left_target and right_target else "negative_negative" if not left_target and not right_target else "target_negative"
                    pairs.write(json.dumps({"a": other, "b": name, "distance": distance, "category": kind}) + "\n")
                    groups.join(other, name)
                    pair_counts[kind] += 1
                    distance_counts[distance] += 1
                    pair_total += 1
                index.add(value, name)
                if number % 500 == 0 or number == len(images):
                    for stream in (fp, pairs, ann, issue_file):
                        stream.flush()
                    print("Checked %d/%d; candidate pairs %d; %.1f min" %
                          (number, len(images), pair_total, (time.monotonic() - started) / 60), flush=True)
        grouping = write_components(output, groups, images, annotations, legacy, counts_by_image)
        summary = {"screen_complete": True, "new_split_created": False, "images_checked": processed,
                   "counts": dict(counts), "by_source": dict(by_source),
                   "valid_target_labels_before_exact_dedup": dict(labels), "other_annotation_labels": dict(other_labels),
                   "target_annotation_flags": dict(annotation_flags), "exact_duplicate_groups": len(exact_groups),
                   "candidate_pairs": pair_total, "candidate_pair_categories": dict(pair_counts),
                   "distance_histogram": dict(distance_counts), "retained_historical_candidate_edges": len(old_pairs),
                   "previous_fingerprints_compared": len(cached), "previous_fingerprints_changed": len(changed_cached),
                   "component_statistics": grouping, "limitations": meta["limitations"]}
        write_json(output / "previous_fingerprint_changes.json", changed_cached)
        write_json(output / "summary.json", summary)
        meta.update(status="complete", images_checked=processed)
    except Exception as error:
        meta.update(status="incomplete_candidate_limit" if isinstance(error, PairLimitReached) else "failed",
                    images_checked=processed, candidate_pairs=pair_total, error=str(error),
                    finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(run_path, meta)
        raise
    meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic() - started, 2))
    write_json(run_path, meta)
    output_hashes = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "checksums.json", output_hashes)
    bundle = output.with_suffix(".tar.gz")
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    print("\nDETECTION AUDIT COMPLETE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Review bundle:", bundle, flush=True)
    print("Bundle SHA256:", digest(bundle), flush=True)
    print("No new split assigned. Review flags and historical split conflicts before creating manifests.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--previous-images", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--distance", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=200000)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower().startswith("login"):
        parser.error("Run image verification and screening on a compute node, not a login node")
    try:
        audit(args.root, args.inventory, args.review, args.previous_images, args.reference,
              args.out, args.distance, args.max_pairs)
    except PairLimitReached as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
