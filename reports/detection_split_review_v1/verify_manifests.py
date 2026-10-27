"""Independently verify saved detection manifests against the returned CCAST audit."""
import csv
import hashlib
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "detection_splits_v1"
REVIEW = Path(__file__).resolve().parent
AUDIT_HASH = "07963cfdc3e14d2032dd1a80a1240a19c415598ef449bde737fe422902dd4f7b"
PACKAGE_HASH = "89d1c16954187ef0126f69c670b83875c8b4ff8863aa5a1864c264df91b0cce4"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_lines(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    audit = ROOT / "detection_audit_v1.tar.gz"
    check(digest(audit.read_bytes()) == AUDIT_HASH, "Audit archive hash changed")
    with tarfile.open(audit, "r:gz") as archive:
        blobs = {Path(m.name).name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    audit_checks = json.loads(blobs["checksums.json"])
    check(set(blobs) == set(audit_checks) | {"checksums.json"}, "Audit checksum coverage")
    for name, expected in audit_checks.items():
        check(digest(blobs[name]) == expected, "Audit member changed: " + name)
    run = json.loads(blobs["run.json"])
    check(run["status"] == "complete" and run["images_checked"] == 38385, "Audit incomplete")
    raw = {r["image"]: r for r in read_lines(blobs["validated_annotations.jsonl"])}
    with (ROOT / "dataset_splits_v1/images.csv").open(encoding="utf-8", newline="") as stream:
        prior = {r["image"]: r for r in csv.DictReader(stream)}
    check(digest((ROOT / "dataset_splits_v1/images.csv").read_bytes()) == run["input_sha256"]["previous_images.csv"],
          "Frozen crop split changed")
    policy = json.loads((REVIEW / "policy.json").read_bytes())
    exclusions = {r["image"] for r in policy["excluded_images"]}
    manifest = []
    summary = json.loads((OUT / "summary.json").read_bytes())
    for partition in ("train", "val", "test"):
        rows = read_lines((OUT / (partition + ".jsonl")).read_bytes())
        check(all(r["split"] == partition for r in rows), "Wrong partition file")
        manifest.extend(rows)
        stats = summary["statistics"][partition]
        check(len(rows) == stats["images"], "Image count mismatch")
        check(sum(len(r["targets"]) for r in rows) == stats["objects"], "Object count mismatch")
        check(Counter(t["label"] for r in rows for t in r["targets"]) == stats["class_objects"], "Class count mismatch")
        check(Counter(r["annotation_category"] for r in rows) == stats["categories"], "Category count mismatch")
        check(Counter(r["source"] for r in rows) == stats["source_images"], "Source count mismatch")
    records = {r["image"]: r for r in manifest}
    check(len(records) == len(manifest) == 38383, "Missing or repeated output image")
    check(set(records) == set(raw) - exclusions, "Included population mismatch")
    with (OUT / "images.csv").open(encoding="utf-8", newline="") as stream:
        image_rows = list(csv.DictReader(stream))
    check(len(image_rows) == len(records) and {r["image"] for r in image_rows} == set(records), "CSV coverage mismatch")
    for r in image_rows:
        check(r["split"] == records[r["image"]]["split"]
              and r["source_cluster_id"] == records[r["image"]]["source_cluster_id"], "CSV membership mismatch")
    groups, file_hashes, rgb_hashes, old_groups = (defaultdict(set) for _ in range(4))
    moved, exif, difficult = [], [], 0
    for image, r in records.items():
        original = raw[image]
        for key in ("image_sha256", "xml_sha256", "rgb_sha256", "source", "annotation_category", "xml", "exif_orientation"):
            check(r[key] == original[key], "Metadata mismatch: " + image + "/" + key)
        check([r["width"], r["height"]] == original["size"], "Dimensions changed")
        unindexed = [{k: v for k, v in t.items() if k != "class_index"} for t in r["targets"]]
        check(unindexed == original["targets"], "Audited target changed: " + image)
        check(all(t["class_index"] == ("D00", "D10", "D20", "D40").index(t["label"]) for t in r["targets"]),
              "Class index mismatch")
        check(r["other_annotations"] == [o for o in original["objects"] if o["label"] not in ("D00", "D10", "D20", "D40")],
              "Non-target annotation changed")
        groups[r["source_cluster_id"]].add(r["split"])
        file_hashes[r["image_sha256"]].add(r["split"])
        rgb_hashes[r["rgb_sha256"]].add(r["split"])
        if image in prior:
            old_groups[prior[image]["group_id"]].add(r["split"])
            if r["split"] != prior[image]["split"]:
                check(prior[image]["split"] == "val" and r["split"] == "train", "Historical exposure reassigned")
                moved.append(image)
        if r["exif_orientation"] not in (None, 1):
            exif.append(image)
        difficult += sum(t.get("difficult") == "1" for t in r["targets"])
    for description, mapping in (("full-pool groups", groups), ("file hashes", file_hashes),
                                 ("RGB hashes", rgb_hashes), ("historical groups", old_groups)):
        check(all(len(s) == 1 for s in mapping.values()), "Partition crossing in " + description)
    check(set(moved) == set(policy["expected_val_to_train_images"]), "Historical move list mismatch")
    old_test = {i for i, r in prior.items() if r["split"] == "test"}
    check(len(old_test) == 3569 and all(records[i]["split"] == "test" for i in old_test), "Previous test changed")
    edges = read_lines(blobs["candidate_pairs.jsonl"])
    old_edges_path = ROOT / "tmp/preparation_review_extract/preparation_review/candidate_pairs.jsonl"
    check(digest(old_edges_path.read_bytes()) == run["input_sha256"]["candidate_pairs.jsonl"], "Historical edge input changed")
    old_edges = read_lines(old_edges_path.read_bytes())
    for edge in edges + old_edges:
        if edge["a"] in records and edge["b"] in records:
            check(records[edge["a"]]["split"] == records[edge["b"]]["split"], "Candidate edge crosses saved partitions")
    checks = json.loads((OUT / "checksums.json").read_bytes())
    for name, expected in checks.items():
        check(digest((OUT / name).read_bytes()) == expected, "Manifest member checksum: " + name)
    sha_text = (OUT / "SHA256SUMS").read_bytes()
    check(b"\r" not in sha_text and not sha_text.startswith(b"\xef\xbb\xbf"), "Checksum text must be LF without BOM")
    listed = {}
    for line in sha_text.decode("utf-8").splitlines():
        expected, name = line.split("  ", 1)
        check(digest((OUT / name).read_bytes()) == expected, "SHA256SUMS entry failed")
        listed[name] = expected
    check(set(listed) == {p.name for p in OUT.iterdir() if p.is_file()} - {"SHA256SUMS"}, "SHA256SUMS coverage")
    package = ROOT / "detection_splits_v1.tar.gz"
    check(digest(package.read_bytes()) == PACKAGE_HASH, "Transfer archive changed")
    with tarfile.open(package, "r:gz") as archive:
        packed = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    check(set(packed) == {"detection_splits_v1/" + p.name for p in OUT.iterdir() if p.is_file()}, "Transfer member coverage")
    for name, data in packed.items():
        check(data == (OUT / Path(name).name).read_bytes(), "Transfer member differs: " + name)
    deployment_path = REVIEW / "deployment.json"
    deployment_status = "pending"
    if deployment_path.exists():
        deployment = json.loads(deployment_path.read_bytes())
        check(deployment["archive_sha256"] == PACKAGE_HASH, "Deployment record refers to a different archive")
        check(set(deployment["member_verification"]["verified_files"]) == set(listed), "Deployment file list differs")
        deployment_status = deployment["status"]
    report = {"status": "passed", "python": sys.version, "audit_bundle_sha256": AUDIT_HASH,
              "audit_members_verified": len(audit_checks), "manifest_bundle_sha256": PACKAGE_HASH,
              "manifest_bundle_bytes": package.stat().st_size, "manifest_members_verified": len(packed),
              "images": len(records), "target_objects": sum(len(r["targets"]) for r in records.values()),
              "full_pool_candidate_edges": len(edges), "historical_candidate_edges": len(old_edges),
              "cross_partition_groups": 0, "cross_partition_file_or_rgb_hashes": 0, "cross_partition_retained_candidate_edges": 0,
              "previous_test_images_preserved": len(old_test), "val_to_train_images": sorted(moved),
              "exif_flagged_images_preserved": exif, "difficult_target_boxes_preserved": difficult,
              "raw_image_pixels_opened_locally": False, "ccast_deployment": deployment_status,
              "unit_test_evidence": {"command": "python -m unittest discover -s tests -p test_make_detection_splits.py -v",
                                     "tests_passed": 8, "exit_code": 0,
                                     "scope": "Local synthetic software checks; separate from this saved-manifest audit"}}
    (REVIEW / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for original, name in (("run.json", "audit_run.json"), ("summary.json", "audit_summary.json"),
                           ("previous_split_conflicts.jsonl", "audit_conflicts.jsonl"),
                           ("annotation_issues.jsonl", "audit_annotation_issues.jsonl"),
                           ("exact_duplicate_groups.json", "audit_exact_duplicates.json")):
        target = REVIEW / name
        if target.exists():
            check(target.read_bytes() == blobs[original], "Saved audit excerpt changed")
        else:
            target.write_bytes(blobs[original])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
