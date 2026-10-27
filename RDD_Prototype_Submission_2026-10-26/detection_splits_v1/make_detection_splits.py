"""Create full-image detection manifests from a checksum-pinned audit bundle.

This metadata-only operation preserves historical evaluation exposure. All
screened groups remain indivisible, including links through excluded images.
It neither opens raw images nor changes previous crop manifests.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

LABELS = ("D00", "D10", "D20", "D40")
SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def jsonl(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(value):
    p = PurePosixPath(value)
    require(bool(value) and not p.is_absolute() and ".." not in p.parts
            and "\\" not in value and ":" not in value, "Unsafe dataset path: " + value)


def read_bundle(path, expected_sha):
    data = path.read_bytes()
    require(sha(data) == expected_sha, "Audit archive checksum mismatch")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        require(sum(m.size for m in members) < 150 * 1024**2, "Unexpected audit archive size")
        roots = set()
        for member in members:
            relative_path(member.name)
            roots.add(PurePosixPath(member.name).parts[0])
            if member.isdir():
                continue
            require(member.isfile(), "Audit archive contains a link or special file")
            parts = PurePosixPath(member.name).parts
            require(len(parts) == 2 and parts[1] not in files, "Unexpected/repeated audit member")
            files[parts[1]] = archive.extractfile(member).read()
        require(len(roots) == 1, "Multiple audit archive roots")
    checks = json.loads(files["checksums.json"])
    require(set(files) == set(checks) | {"checksums.json"}, "Audit checksum coverage mismatch")
    for name, expected in checks.items():
        require(sha(files[name]) == expected, "Audit member checksum mismatch: " + name)
    return files


def load_inputs(bundle, previous, policy):
    files = read_bundle(bundle, policy["audit_bundle_sha256"])
    run, summary = (json.loads(files[name]) for name in ("run.json", "summary.json"))
    require(run["status"] == "complete" and summary["screen_complete"], "Incomplete audit")
    prior_sha = sha(previous.read_bytes())
    require(prior_sha == policy["previous_images_sha256"]
            == run["input_sha256"]["previous_images.csv"], "Historical split checksum mismatch")
    require(run["code_sha256"] == policy["audit_code_sha256"], "Unexpected audit implementation")
    with previous.open(encoding="utf-8", newline="") as stream:
        prior_rows = list(csv.DictReader(stream))
    prior = {r["image"]: r for r in prior_rows}
    require(len(prior) == len(prior_rows), "Repeated historical image")
    records = jsonl(files["validated_annotations.jsonl"])
    by_image = {r["image"]: r for r in records}
    require(len(by_image) == len(records) == run["images_checked"] == run["images_expected"]
            == summary["images_checked"], "Incomplete/repeated annotation inventory")
    require(set(prior) <= set(by_image), "Historical image missing from audit")
    cluster_rows = list(csv.DictReader(io.StringIO(files["source_clusters.csv"].decode("utf-8"))))
    cluster = {r["image"]: r for r in cluster_rows}
    require(len(cluster) == len(cluster_rows) and set(cluster) == set(by_image), "Cluster coverage mismatch")
    exclusions = {r["image"]: r for r in policy["excluded_images"]}
    require(len(exclusions) == len(policy["excluded_images"])
            and set(exclusions) <= set(by_image), "Invalid image exclusions")
    dropped = {(r["image"], r["object_index"]): r for r in policy["excluded_objects"]}
    require(len(dropped) == len(policy["excluded_objects"]), "Repeated object exclusion")
    observed_dropped = set()
    groups, old_groups, rgb_groups = defaultdict(list), defaultdict(set), defaultdict(set)
    allowed_other = set(policy["retained_non_target_labels"])
    require(not allowed_other.intersection(LABELS), "Target class listed as non-target")
    for image, r in sorted(by_image.items()):
        relative_path(image)
        relative_path(r["xml"])
        require(r["source"] == image.split("/")[0], "Source/path mismatch")
        width, height = r["size"]
        require(width > 0 and height > 0, "Invalid dimensions")
        c = cluster[image]
        require(c["source"] == r["source"] and c["annotation_category"] == r["annotation_category"]
                and int(c["valid_targets"]) == len(r["targets"]), "Cluster/annotation mismatch")
        old = prior.get(image, {})
        require(c["previous_split"] == old.get("split", "")
                and c["previous_group_id"] == old.get("group_id", ""), "Historical assignment mismatch")
        require(not old or old["split"] in SPLITS, "Unknown historical split")
        require(c["historical_duplicate_of"] == r.get("duplicate_of", ""), "Duplicate metadata mismatch")
        groups[c["source_cluster_id"]].append(image)
        rgb_groups[r["rgb_sha256"]].add(c["source_cluster_id"])
        if old:
            old_groups[old["group_id"]].add(c["source_cluster_id"])
        targets = {t["object_index"]: t for t in r["targets"]}
        require(len(targets) == len(r["targets"]), "Repeated target index")
        objects = {o["index"]: o for o in r["objects"]}
        require(len(objects) == len(r["objects"]), "Repeated annotation index")
        expected_targets = set()
        for index, obj in objects.items():
            if obj["label"] not in LABELS:
                require(image in exclusions or obj["label"] in allowed_other,
                        "Unreviewed annotation label: " + obj["label"])
                continue
            box = obj["box"]
            require(len(box) == 4 and all(math.isfinite(x) for x in box), "Invalid target coordinates")
            clipped = [min(max(box[0], 0), width), min(max(box[1], 0), height),
                       min(max(box[2], 0), width), min(max(box[3], 0), height)]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                require((image, index) in dropped, "Unreviewed nonpositive target box")
                observed_dropped.add((image, index))
                continue
            expected_targets.add(index)
            require(index in targets, "Positive-area target omitted")
            target = targets[index]
            require(target["label"] == obj["label"] and target["box_xyxy"] == clipped,
                    "Target geometry or label mismatch")
            require(all(target.get(k) == obj.get(k) for k in ("difficult", "truncated")), "Lost annotation flags")
        require(set(targets) == expected_targets, "Unexpected target index")
        category = ("target_bearing" if targets else "invalid_target_only" if any(
            o["label"] in LABELS for o in objects.values()) else "only_other_labels" if objects else "empty_xml")
        require(category == r["annotation_category"], "Annotation category mismatch")
        require(category != "invalid_target_only" or image in exclusions,
                "Invalid-only targets must not become background images")
        if r.get("duplicate_of"):
            canonical = r["duplicate_of"]
            require(image in exclusions and exclusions[image].get("canonical") == canonical
                    and exclusions[image]["reason"] == "historical_exact_duplicate", "Unreviewed exact duplicate")
            require(canonical in by_image and canonical not in exclusions
                    and r["rgb_sha256"] == by_image[canonical]["rgb_sha256"], "Invalid duplicate canonical")
        if image in exclusions:
            e = exclusions[image]
            require(image not in prior, "Exclusion would remove a historical split member")
            if e["reason"] == "ambiguous_annotation_label":
                require(not targets and {o["label"] for o in objects.values()} == set(e["expected_labels"]),
                        "Ambiguous-label exclusion no longer matches annotation")
            else:
                require(e["reason"] == "historical_exact_duplicate" and r.get("duplicate_of") == e["canonical"],
                        "Unsupported image exclusion")
    require(observed_dropped == set(dropped), "Object exclusion no longer matches audit")
    require(all(len(v) == 1 for v in old_groups.values()), "Historical group was separated")
    require(all(len(v) == 1 for v in rgb_groups.values()), "Exact pixels cross audit groups")
    for key, members in groups.items():
        require(key == min(members), "Noncanonical audit group identifier")
        require(len({by_image[m]["source"] for m in members}) == 1, "Cross-source group needs a joint allocator")
    pairs = jsonl(files["candidate_pairs.jsonl"])
    require(len(pairs) == summary["candidate_pairs"], "Incomplete candidate list")
    for pair in pairs:
        require(pair["a"] in cluster and pair["b"] in cluster
                and cluster[pair["a"]]["source_cluster_id"] == cluster[pair["b"]]["source_cluster_id"],
                "Candidate edge crosses audit groups")
    return by_image, cluster, groups, prior, exclusions, files, run, summary


def allocate_free_groups(vectors, fixed, seed):
    """Balance per-source image and two no-target category counts, with anchors fixed."""
    total = [sum(v[k] for v in vectors.values()) for k in range(3)]
    target = [[ratio * n for n in total] for ratio in RATIOS]
    current = [[0] * 3 for _ in SPLITS]
    assignments = dict(fixed)
    for key, split in fixed.items():
        for k, value in enumerate(vectors[key]):
            current[SPLITS.index(split)][k] += value
    tie = lambda key: sha((str(seed) + ":" + key).encode("utf-8"))
    order = sorted(set(vectors) - set(fixed), key=lambda key: (-vectors[key][0], tie(key)))

    def delta(s, vector, sign):
        return sum(((current[s][k] + sign * vector[k] - target[s][k])**2
                    - (current[s][k] - target[s][k])**2) / max(target[s][k], 1) for k in range(3))

    def adjust(s, vector, sign):
        for k, value in enumerate(vector):
            current[s][k] += sign * value

    for key in order:
        s = min(range(3), key=lambda s: (delta(s, vectors[key], 1), s))
        assignments[key] = SPLITS[s]
        adjust(s, vectors[key], 1)
    for _ in range(3):
        moved = False
        for key in sorted(order, key=tie):
            source, vector = SPLITS.index(assignments[key]), vectors[key]
            change, dest = min((delta(source, vector, -1) + delta(s, vector, 1), s)
                               for s in range(3) if s != source)
            if change < -1e-9:
                adjust(source, vector, -1)
                adjust(dest, vector, 1)
                assignments[key], moved = SPLITS[dest], True
        if not moved:
            break
    return assignments


def make_assignments(records, cluster, groups, prior, exclusions, policy):
    assigned, conflicts, sources = {}, [], defaultdict(dict)
    for group, members in sorted(groups.items()):
        active = [m for m in members if m not in exclusions]
        if not active:
            continue
        old = {prior[m]["split"] for m in members if m in prior}
        if len(old) > 1:
            require(old == {"train", "val"}, "Group links historical test and development images")
            assigned[group] = "train"
            conflicts.append({"source_cluster_id": group, "destination": "train", "members": members,
                              "previous_split_counts": dict(Counter(prior[m]["split"] for m in members if m in prior))})
        elif old:
            assigned[group] = next(iter(old))
        else:
            require(not any(records[m]["targets"] for m in active), "New target-bearing group needs a reviewed allocation")
        counts = Counter(records[m]["annotation_category"] for m in active)
        sources[records[active[0]]["source"]][group] = [len(active), counts["empty_xml"], counts["only_other_labels"]]
    for source, vectors in sorted(sources.items()):
        fixed = {g: assigned[g] for g in vectors if g in assigned}
        assigned.update(allocate_free_groups(vectors, fixed, policy["seed"]))
    moves = []
    for image, row in sorted(prior.items()):
        new = assigned[cluster[image]["source_cluster_id"]]
        if row["split"] != new:
            require(row["split"] == "val" and new == "train", "Disallowed historical reassignment")
            moves.append({"image": image, "previous_split": row["split"], "split": new,
                          "source_cluster_id": cluster[image]["source_cluster_id"],
                          "reason": "Full-pool similarity group contains prior training exposure"})
    require({r["image"] for r in moves} == set(policy["expected_val_to_train_images"]), "Unreviewed historical reassignment")
    return assigned, moves, conflicts


def build(bundle, previous, policy_path, output):
    output = output.resolve()
    archive_path = output.with_suffix(".tar.gz")
    require(not output.exists() and not archive_path.exists(), "Output or archive already exists")
    for protected in (bundle.resolve(), previous.resolve(), policy_path.resolve(), Path(__file__).resolve()):
        require(output not in protected.parents and output != protected, "Output overlaps an input")
    require(previous.resolve().parent not in output.parents, "Output is inside the frozen crop split")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    records, cluster, groups, prior, exclusions, files, run, audit_summary = load_inputs(bundle, previous, policy)
    assigned, moves, conflicts = make_assignments(records, cluster, groups, prior, exclusions, policy)
    reported = {r["source_cluster_id"] for r in jsonl(files["previous_split_conflicts.jsonl"])}
    require(reported == {r["source_cluster_id"] for r in conflicts}, "Audit conflict report mismatch")
    partitions = {s: [] for s in SPLITS}
    image_rows = []
    stats = {s: {"images": 0, "groups": 0, "objects": 0, "class_objects": Counter(),
                 "categories": Counter(), "source_images": Counter(), "source_class_objects": defaultdict(Counter),
                 "source_categories": defaultdict(Counter)} for s in SPLITS}
    for image, r in sorted(records.items()):
        if image in exclusions:
            continue
        group = cluster[image]["source_cluster_id"]
        split = assigned[group]
        targets = [dict(t, class_index=LABELS.index(t["label"])) for t in r["targets"]]
        record = {"image": image, "xml": r["xml"], "source": r["source"], "split": split, "source_cluster_id": group,
                  "previous_split": prior.get(image, {}).get("split", ""), "width": r["size"][0], "height": r["size"][1],
                  "image_sha256": r["image_sha256"], "xml_sha256": r["xml_sha256"], "rgb_sha256": r["rgb_sha256"],
                  "exif_orientation": r.get("exif_orientation"), "annotation_category": r["annotation_category"],
                  "targets": targets, "other_annotations": [o for o in r["objects"] if o["label"] not in LABELS]}
        partitions[split].append(record)
        counts = Counter(t["label"] for t in targets)
        st = stats[split]
        st["images"] += 1
        st["objects"] += len(targets)
        st["class_objects"].update(counts)
        st["categories"][r["annotation_category"]] += 1
        st["source_images"][r["source"]] += 1
        st["source_class_objects"][r["source"]].update(counts)
        st["source_categories"][r["source"]][r["annotation_category"]] += 1
        image_rows.append({"image": image, "source": r["source"], "split": split, "source_cluster_id": group,
                           "previous_split": record["previous_split"], "annotation_category": r["annotation_category"],
                           "targets": len(targets)})
    for split in assigned.values():
        stats[split]["groups"] += 1
    observed_classes = {(r["source"], t["label"]) for image, r in records.items() if image not in exclusions for t in r["targets"]}
    for source, label in observed_classes:
        require(all(stats[s]["source_class_objects"][source][label] > 0 for s in SPLITS),
                "Lost source/class coverage: " + source + "/" + label)
    summary = {"status": "complete", "raw_audited_images": len(records), "included_images": len(image_rows),
               "included_target_objects": sum(s["objects"] for s in stats.values()),
               "excluded_images": len(exclusions), "excluded_target_objects": len(policy["excluded_objects"]),
               "historical_reassignments": moves, "resolved_train_val_components": len(conflicts),
               "historical_test_images_preserved": sum(r["split"] == "test" for r in prior.values()),
               "group_crossings": 0, "candidate_edge_crossings": 0, "statistics": stats,
               "limitations": run["limitations"] + ["The ambiguous D0w0-only image is excluded without label repair",
                   "Historical validation remains development data; this is a new detection split, not the old crop split"]}
    config = {"version": "detection_splits_v1", "class_indices": {label: i for i, label in enumerate(LABELS)},
              "class_index_convention": "Zero-based foreground index; a model adapter must reserve background separately if required",
              "seed": policy["seed"], "target_ratios": dict(zip(SPLITS, RATIOS)),
              "allocation": "Preserve historical anchors; train absorbs reviewed train/val conflicts; balance remaining no-target groups within source",
              "coordinates": "Continuous XYXY numeric edges, clipped to stored image bounds; no origin shift or pixel rounding",
              "pixel_orientation": "Stored image orientation; do not apply EXIF transpose",
              "difficult_truncated": "Preserve flags; all valid target boxes included, including difficult=1; no automatic ignore or crowd mapping",
              "background_images": "Empty target list means no retained annotation for the four classes, not verified healthy pavement",
              "audit_bundle_sha256": policy["audit_bundle_sha256"], "audit_member_sha256": json.loads(files["checksums.json"]),
              "previous_images_sha256": policy["previous_images_sha256"], "policy_sha256": sha(policy_path.read_bytes()),
              "generator_sha256": sha(Path(__file__).read_bytes()), "audit_source_root": run["root"], "raw_pixels_read_locally": False}
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in partitions.items():
        with (output / (split + ".jsonl")).open("w", encoding="utf-8", newline="\n") as stream:
            for record in rows:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    with (output / "images.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(image_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(image_rows)
    for name, value in (("summary.json", summary), ("config.json", config), ("conflict_resolutions.json", conflicts),
                        ("excluded_images.json", policy["excluded_images"]), ("excluded_objects.json", policy["excluded_objects"])):
        (output / name).write_bytes(json_bytes(value))
    (output / "reviewed_policy.json").write_bytes(policy_path.read_bytes())
    (output / "make_detection_splits.py").write_bytes(Path(__file__).read_bytes())
    readme = """# Full-image detection split v1

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
"""
    (output / "README.md").write_bytes(readme.encode("utf-8"))
    hashes = {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "checksums.json").write_bytes(json_bytes(hashes))
    hashes["checksums.json"] = sha((output / "checksums.json").read_bytes())
    (output / "SHA256SUMS").write_bytes("".join(h + "  " + n + "\n" for n, h in sorted(hashes.items())).encode("utf-8"))
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Manifest bundle:", archive_path)
    print("Bundle SHA256:", sha(archive_path.read_bytes()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-bundle", type=Path, required=True)
    parser.add_argument("--previous-images", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.audit_bundle, args.previous_images, args.policy, args.out)


if __name__ == "__main__":
    main()
