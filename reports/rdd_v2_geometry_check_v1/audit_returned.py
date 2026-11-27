"""Independent audit of the CCAST geometry preflight; no detector execution."""
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ARCHIVE = ROOT/"rdd_v2_geometry_check_v1.tar.gz"
EXPECTED = "a6465d65ec2a0756b7574678184b8b088fcd73eff880b5b8796194258c20413f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    if path.exists():
        assert path.read_bytes() == data, f"Preserve earlier audit {path}"
    else:
        path.write_bytes(data)


def evidence():
    assert file_sha(ARCHIVE) == EXPECTED, "Archive hash"
    prefix = "rdd_v2_geometry_check_v1/"
    blobs = {}
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        members = tf.getmembers()
        assert len({m.name for m in members}) == len(members), "Duplicate archive names"
        for member in members:
            assert member.isdir() or member.isfile(), "Links/devices forbidden"
            parts = PurePosixPath(member.name).parts
            assert not PurePosixPath(member.name).is_absolute() and ".." not in parts and "\\" not in member.name
            if member.isdir():
                continue
            assert member.name.startswith(prefix), "Unexpected archive root"
            assert member.size < 40_000_000, "Unbounded member"
            blobs[member.name[len(prefix):]] = tf.extractfile(member).read()
    checks = {}
    for line in blobs["SHA256SUMS"].decode().splitlines():
        digest, name = line.split("  ", 1)
        assert name not in checks and len(digest) == 64
        checks[name] = digest
    assert set(checks) == set(blobs)-{"SHA256SUMS"}, "Checksum coverage"
    for name, digest in checks.items():
        assert sha(blobs[name]) == digest, name
    folder = OUT/"returned_records"
    folder.mkdir(exist_ok=True)
    for name, data in blobs.items():
        path = (folder/name).resolve()
        assert path.is_relative_to(folder.resolve()), "Unsafe output"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data, "Preserve previously extracted evidence"
        else:
            with path.open("xb") as stream:
                stream.write(data)
    return folder, checks


def prepare(row, n):
    """Independent pixel-edge matrix, retaining every original annotation."""
    w, h = row["width"], row["height"]
    rw, rh = max(1, round(w*min(n/w, n/h))), max(1, round(h*min(n/w, n/h)))
    resize = np.array([[rw/w, 0, (n-rw)//2], [0, rh/h, (n-rh)//2], [0, 0, 1]], float)
    original = []
    for kind, field, coord, index in (("target", "targets", "box_xyxy", "object_index"), ("other", "other_annotations", "box", "index")):
        for ann in row[field]:
            a, b, c, d = ann[coord]
            assert np.isfinite([a,b,c,d]).all() and 0 <= a < c <= w and 0 <= b < d <= h
            original.append({"kind": kind, "id": ann[index], "original": ann})
    raw = np.array([a["original"]["box_xyxy" if a["kind"] == "target" else "box"] for a in original], float).reshape(-1, 4)
    return original, corners(raw) @ resize.T


def corners(boxes):
    if len(boxes) == 0:
        return np.empty((0, 4, 3))
    x1,y1,x2,y2 = boxes.T
    return np.stack([np.stack([x1,y1,np.ones(len(boxes))], axis=-1),
                     np.stack([x1,y2,np.ones(len(boxes))], axis=-1),
                     np.stack([x2,y1,np.ones(len(boxes))], axis=-1),
                     np.stack([x2,y2,np.ones(len(boxes))], axis=-1)], axis=1)


def rectangles(points):
    return np.concatenate([points[..., :2].min(axis=-2), points[..., :2].max(axis=-2)], axis=-1)


def main():
    folder, checks = evidence()
    result = json.loads((folder/"result.json").read_text())
    returned = json.loads((folder/"annotation_report.json").read_text())
    pixels = json.loads((folder/"pixel_records.json").read_text())
    package = ROOT/"rdd_v2_preflight_v1"
    cfg = json.loads((package/"protocol.json").read_text())
    assert result["protocol_sha256"] == file_sha(package/"protocol.json")
    for name, digest in result["source_sha256"].items():
        assert file_sha(package/name) == digest, f"Source mismatch: {name}"
    assert result["numpy"] == "1.26.4" and result["Pillow"] == "10.3.0"
    assert result["status"] == "numerical_complete_visual_review_pending"
    assert result["annotation_failures"] == result["pixel_check"]["pixel_failures"] == 0
    train_path = ROOT/"detection_splits_v1"/"train.jsonl"
    assert file_sha(train_path) == cfg["train_manifest_sha256"] == result["train_sha256"]
    rows = [json.loads(line) for line in train_path.read_text().splitlines() if line]
    assert len(rows) == 26863 and all(r["split"] == "train" for r in rows)
    index = {r["image"]: r for r in rows}
    assert len(index) == len(rows)
    g = cfg["geometry"]; n = g["size"]
    settings = list(product(g["horizontal_flip"], g["resized_canvas_sizes"], g["translations_pixels"], g["translations_pixels"]))
    matrices = []
    for flip, m, tx, ty in settings:
        reflection = np.array([[-1,0,n],[0,1,0],[0,0,1]], float) if flip else np.eye(3)
        transform = np.array([[m/n,0,(n-m)//2+tx],[0,m/n,(n-m)//2+ty],[0,0,1]], float)
        matrices.append(transform @ reflection)
    matrices = np.array(matrices)
    identity = settings.index((False, n, 0, 0))
    counts, sources, classes, all_candidates = Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
    tiny_before = 0; new_tiny_candidates = 0
    for row in rows:
        originals, prepared = prepare(row, n)
        target_mask = np.array([a["kind"] == "target" for a in originals], bool)
        before = rectangles(prepared)
        small_before = ((before[:, 2:]-before[:, :2]) < 1).any(axis=1) & target_mask
        tiny_before += int(small_before.sum())
        mapped = np.einsum("kij,bcj->kbci", matrices, prepared)
        boxes = rectangles(mapped)
        bad = (boxes[..., :2] < 0).any(axis=-1) | (boxes[..., 2:] > n).any(axis=-1)
        accepted = ~bad.any(axis=1)
        sources[row["source"]]["images"] += 1
        labels = {a["label"] for a in row["targets"]}
        for label in labels:
            classes[label]["images"] += 1
        for j, yes in enumerate(accepted):
            state = "accepted_candidates" if yes else "rejected_candidates"
            counts[state] += 1
            sources[row["source"]][state] += 1
            all_candidates[f"{j}/{state}"] += 1
            for label in labels:
                classes[label][state] += 1
            if yes:
                assert (boxes[j,:,2:] > boxes[j,:,:2]).all()
                new_tiny_candidates += int((((boxes[j,:,2:]-boxes[j,:,:2]) < 1).any(axis=1) & target_mask & ~small_before).sum())
    assert dict(sources) == returned["by_source"], "Independent census source counts differ"
    assert dict(classes) == returned["by_class_multilabel_images"], "Independent class counts differ"
    assert dict(counts) == {k: returned[k] for k in counts}, "Independent census totals differ"
    assert returned["disabled_images"] == []
    assert len(pixels) == 56 and len({r["image"] for r in pixels}) == 56
    assert [r["image"] for r in pixels] == returned["pixel_selection"]
    per_source, sample_labels = Counter(), Counter()
    maximum_error = 0.; accepted_pixel, rejected_pixel, boxes_checked, other_boxes, negative_images = 0,0,0,0,0
    details = []
    for record in pixels:
        row = index[record["image"]]
        per_source[row["source"]] += 1
        for t in row["targets"]: sample_labels[t["label"]] += 1
        negative_images += not bool(row["targets"])
        originals, prepared = prepare(row, n)
        before = rectangles(prepared)
        assert record["status"] == "numerical_pass_visual_review_pending"
        assert [r["candidate"] for r in record["candidates"]] == list(range(54))
        identity_record = record["candidates"][identity]
        assert identity_record["accepted"] and identity_record["changed_pixels"] == 0
        for j, candidate in enumerate(record["candidates"]):
            proposed = rectangles(prepared @ matrices[j].T)
            bad = (proposed[:,:2] < 0).any(axis=1) | (proposed[:,2:] > n).any(axis=1)
            accepted = not bad.any()
            assert candidate["accepted"] == accepted
            assert candidate["cropped_ids"] == [a["id"] for a,b in zip(originals,bad) if b]
            assert candidate["reason"] == ("accepted" if accepted else "cropped_annotation")
            assert type(candidate["changed_pixels"]) is int and 0 <= candidate["changed_pixels"] <= n*n
            assert len(candidate["rgb_sha256"]) == 64 and all(v in "0123456789abcdef" for v in candidate["rgb_sha256"])
            if not accepted:
                rejected_pixel += 1
                assert candidate["changed_pixels"] == 0
                assert candidate["rgb_sha256"] == identity_record["rgb_sha256"]
                expected = before
            else:
                accepted_pixel += 1
                expected = proposed
            assert len(candidate["annotations"]) == len(originals)
            for k, (actual, original) in enumerate(zip(candidate["annotations"], originals)):
                assert {key: actual[key] for key in ("kind", "id", "original")} == original
                error = max(abs(np.asarray(actual["box_xyxy"])-expected[k]))
                maximum_error = max(maximum_error, float(error))
                assert error < 1e-9, "Mapped annotation differs"
                boxes_checked += 1; other_boxes += original["kind"] == "other"
        overlay = folder/"overlays"/record["overlay"]
        with Image.open(overlay) as im:
            im.verify()
        with Image.open(overlay) as im:
            assert im.size == (960,730) and im.mode == "RGB"
        details.append({"image": record["image"], "source": row["source"], "targets": len(row["targets"]),
                        "target_labels": [t["label"] for t in row["targets"]],
                        "other_annotations": len(row["other_annotations"]),
                        "rejected_candidates": sum(not r["accepted"] for r in record["candidates"]),
                        "overlay": record["overlay"]})
    assert set(per_source.values()) == {8}
    assert result["pixel_check"]["rendered_candidates"] == accepted_pixel+rejected_pixel == 3024
    save(OUT/"numerical_audit.json", {
        "status": "independent_numerical_audit_passed_visual_review_separate", "archive_sha256": EXPECTED,
        "archive_bytes": ARCHIVE.stat().st_size, "verified_file_hashes": len(checks),
        "protocol_and_source_hashes_match_release": True, "target_numpy_pillow_versions_match": True,
        "training_images": len(rows), "geometry_candidates_per_image": len(settings),
        "independent_annotation_census_matches_ccast": True, "census": dict(counts),
        "source_census": {k: dict(v) for k,v in sources.items()}, "class_census": {k: dict(v) for k,v in classes.items()},
        "all_candidate_counts": dict(all_candidates), "selected_images": len(pixels),
        "selected_sources": dict(per_source), "selected_target_objects_by_class": dict(sample_labels),
        "selected_no_target_images": negative_images, "pixel_accepted_candidates": accepted_pixel,
        "pixel_rejected_candidates": rejected_pixel, "returned_annotation_instances_checked": boxes_checked,
        "returned_non_target_annotation_instances_checked": other_boxes,
        "maximum_independent_coordinate_error": maximum_error,
        "targets_with_a_prepared_side_under_one_pixel_before_geometry": tiny_before,
        "accepted_target_transform_pairs_newly_under_one_pixel": new_tiny_candidates,
        "ccast_elapsed_seconds": result["seconds"], "detector_training_or_inference": False,
        "limits": ["Original full-resolution RGB arrays remain on CCAST; local pixels cannot be independently re-rendered from this archive",
                   "336 saved grayscale 320x320 display panels do not resolve every thin crack or all 3024 rendered candidates",
                   "No controller fresh-panel or downstream detector efficacy check occurred in this CPU job"],
        "source_sha256": file_sha(Path(__file__)), "returned_files_sha256": checks})
    save(OUT/"image_inventory.json", details)
    print(json.dumps({"hashes": len(checks), "census": dict(counts), "sample_sources": dict(per_source),
                      "sample_objects": dict(sample_labels), "no_target_images": negative_images,
                      "accepted_pixel_candidates": accepted_pixel, "rejected_pixel_candidates": rejected_pixel,
                      "annotation_instances": boxes_checked, "other_annotation_instances": other_boxes,
                      "coordinate_error": maximum_error, "tiny_before": tiny_before,
                      "new_tiny_pairs": new_tiny_candidates}, indent=2))


if __name__ == "__main__":
    main()
