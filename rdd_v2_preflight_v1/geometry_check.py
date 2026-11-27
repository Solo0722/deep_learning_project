"""Train-only annotation census and bounded real-pixel geometry inspection."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from controller import digest
import geometry as g
from rdd_data import read_manifest, load_example, prepare_example
from replay import save, file_hash


def selection(rows, count_per_source=8):
    selected = []
    for source in sorted({r["source"] for r in rows}):
        pool = sorted((r for r in rows if r["source"] == source), key=lambda r: digest(["geometry_selection", 42, r["image"]]))
        chosen = []
        predicates = [lambda r, c=c: any(t["class_index"] == c for t in r["targets"]) for c in range(4)]
        predicates += [lambda r: not r["targets"], lambda r: bool(r["other_annotations"]),
                       lambda r: any(t["truncated"] == "1" for t in r["targets"]),
                       lambda r: any(float(v) != int(v) for t in r["targets"] for v in t["box_xyxy"])]
        for predicate in predicates:
            candidate = next((r for r in pool if r not in chosen and predicate(r)), None)
            if candidate is not None:
                chosen.append(candidate)
        for row in pool:
            if len(chosen) == count_per_source:
                break
            if row not in chosen:
                chosen.append(row)
        if len(chosen) != count_per_source:
            raise ValueError("Insufficient source coverage")
        selected.extend(chosen)
    return selected


def census(rows, settings):
    candidates = g.bank(settings)
    by_source, by_class = defaultdict(Counter), defaultdict(Counter)
    blocked, preview_selection = [], selection(rows)
    maximum_inverse_error, accepted, rejected = 0., 0, 0
    for k, row in enumerate(rows, 1):
        try:
            annotations = g.prepared_annotations(row, settings["size"])
        except g.InvalidAnnotation as error:
            blocked.append({"image": row["image"], "reason": str(error)})
            by_source[row["source"]]["images_geometry_disabled_invalid_annotation"] += 1
            for label in {t["label"] for t in row["targets"]}:
                by_class[label]["images_geometry_disabled_invalid_annotation"] += 1
            continue
        boxes = np.array([a["box_xyxy"] for a in annotations], dtype=np.float64).reshape(-1, 4)
        by_source[row["source"]]["images"] += 1
        for label in {t["label"] for t in row["targets"]}:
            by_class[label]["images"] += 1
        for candidate in candidates:
            verdict = g.evaluate(annotations, candidate, settings["size"])
            state = "accepted_candidates" if verdict["accepted"] else "rejected_candidates"
            by_source[row["source"]][state] += 1
            for label in {t["label"] for t in row["targets"]}:
                by_class[label][state] += 1
            if verdict["accepted"]:
                accepted += 1
                if len(boxes):
                    inverted = g.inverse_boxes(verdict["boxes"], candidate, settings["size"])
                    error = float(np.max(np.abs(inverted-boxes)))
                    maximum_inverse_error = max(maximum_inverse_error, error)
                    if error > 1e-9 or not (verdict["boxes"][:, 2:] > verdict["boxes"][:, :2]).all():
                        raise ValueError("Geometry annotation invariant failed")
            else:
                rejected += 1
        if k % 4000 == 0:
            print(f"ANNOTATIONS {k}/{len(rows)}", flush=True)
    return {"images": len(rows), "candidates_per_image": len(candidates),
            "accepted_candidates": accepted, "rejected_candidates": rejected,
            "disabled_images": blocked, "maximum_inverse_error": maximum_inverse_error,
            "by_source": {k: dict(v) for k, v in by_source.items()},
            "by_class_multilabel_images": {k: dict(v) for k, v in by_class.items()},
            "pixel_selection": [r["image"] for r in preview_selection],
            "annotation_failures": 0,
            "interpretation": "Enumeration over the declared bank, not measured training exposure; disabled images would use identity."}


def overlay(rgb, annotations, title):
    # Grayscale preview and black/white labels keep the diagnostic printable.
    with Image.fromarray(rgb) as im:
        tile = im.convert("L").convert("RGB")
    draw = ImageDraw.Draw(tile)
    for ann in annotations:
        coords = tuple(ann["box_xyxy"])
        draw.rectangle(coords, outline="white", width=4)
        draw.rectangle(coords, outline="black", width=2)
        label = f"{ann['id']}:{ann['original']['label']}"
        x, y = coords[:2]
        w = draw.textlength(label)+4
        draw.rectangle((x, y, x+w, y+14), fill="white")
        draw.text((x+2, y), label, fill="black")
    tile = tile.resize((320, 320))
    output = Image.new("RGB", (320, 350), "white")
    output.paste(tile, (0, 30))
    ImageDraw.Draw(output).text((4, 5), title, fill="black")
    return output


def pixel_check(root, rows, settings, out):
    candidates, records = g.bank(settings), []
    folder = out/"overlays"
    folder.mkdir()
    for k, row in enumerate(selection(rows), 1):
        before = digest(row)
        example = prepare_example(load_example(root, row), settings["size"])
        original_hash = hashlib.sha256(example.rgb.tobytes()).hexdigest()
        try:
            annotations = g.prepared_annotations(row, settings["size"])
        except g.InvalidAnnotation as error:
            records.append({"image": row["image"], "status": "geometry_disabled", "reason": str(error)})
            continue
        target_boxes = np.array([a["box_xyxy"] for a in annotations if a["kind"] == "target"]).reshape(-1, 4)
        if not np.array_equal(target_boxes, example.boxes):
            raise ValueError("Prepared annotation reader mismatch")
        unchanged = digest(annotations)
        evidence, tiles = [], [overlay(example.rgb, annotations, "Original prepared image")]
        # Fixed representative views, including shrink, expand, flips and translation.
        indices = [i for i, v in enumerate(candidates)
                   if ((not v["flip"] and v["resized_size"] in (608, 672) and v["tx"] == v["ty"] == 0)
                       or (v["flip"] and v["resized_size"] == 640 and v["tx"] == v["ty"] == 0)
                       or (not v["flip"] and v["resized_size"] == 640 and v["tx"] == 16 and v["ty"] == 0)
                       or (v["flip"] and v["resized_size"] == 608 and v["tx"] == -16 and v["ty"] == 16))]
        for i, candidate in enumerate(candidates):
            pixels, transformed, verdict = g.apply(example.rgb, annotations, candidate)
            if digest(row) != before or digest(annotations) != unchanged or hashlib.sha256(example.rgb.tobytes()).hexdigest() != original_hash:
                raise ValueError("Input mutated by renderer")
            if len(transformed) != len(annotations) or any(a["original"] != b["original"] or a["id"] != b["id"] or a["kind"] != b["kind"] for a, b in zip(annotations, transformed)):
                raise ValueError("Annotation identity, class or flags changed")
            if not verdict["accepted"] and (not np.array_equal(pixels, example.rgb) or transformed != annotations):
                raise ValueError("Rejected transform did not fall back to exact identity")
            # Exact known pixel cases independently detect image/box flip misalignment.
            if candidate["resized_size"] == settings["size"] and candidate["tx"] == candidate["ty"] == 0:
                expected = example.rgb[:, ::-1] if candidate["flip"] else example.rgb
                if not np.array_equal(pixels, expected):
                    raise ValueError("Pure flip/identity pixel invariant failed")
            evidence.append({"candidate": i, "accepted": verdict["accepted"], "reason": verdict["reason"],
                             "cropped_ids": verdict["cropped_ids"], "changed_pixels": int(np.any(pixels != example.rgb, axis=2).sum()),
                             "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                             "annotations": transformed})
            if i in indices:
                title = f"flip={int(candidate['flip'])} s={candidate['scale']:.2f} xy={candidate['tx']},{candidate['ty']}"
                if not verdict["accepted"]:
                    title = "REJECT -> identity: " + str(i)
                tiles.append(overlay(pixels, transformed, title))
        sheet = Image.new("RGB", (960, 730), "white")
        ImageDraw.Draw(sheet).text((6, 5), row["image"], fill="black")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 3)*320, 30+(i//3)*350))
        name = f"{k:02d}_{Path(row['image']).stem}.png"
        sheet.save(folder/name)
        records.append({"image": row["image"], "status": "numerical_pass_visual_review_pending", "overlay": name, "candidates": evidence})
        print(f"PIXELS {k}/56 {row['image']}", flush=True)
    save(out/"pixel_records.json", records)
    return {"selected_images": len(records), "rendered_images": sum("candidates" in r for r in records),
            "rendered_candidates": sum(len(r.get("candidates", [])) for r in records),
            "visual_review": "pending", "pixel_failures": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--require-target-versions", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(__file__).with_name("protocol.json").read_text())
    out = args.out.resolve()
    inputs = [args.splits.resolve(), Path(__file__).parent.resolve()]
    if args.root:
        inputs.append(args.root.resolve())
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError("Output overlaps inputs")
    archive = out.with_suffix(".tar.gz")
    if out.exists() or archive.exists():
        raise ValueError("Preserve existing results/archive")
    if not args.manifest_only and not args.root:
        raise ValueError("Raw root required for pixel checks")
    if args.require_target_versions and (np.__version__ != "1.26.4" or pillow_version != "10.3.0"):
        raise ValueError("Pinned CCAST NumPy/Pillow versions differ")
    rows = read_manifest(args.splits/"train.jsonl", cfg["train_manifest_sha256"])
    if len(rows) != cfg["expected_train_images"]:
        raise ValueError("Wrong training row count")
    out.mkdir(parents=True)
    started = time.monotonic()
    try:
        annotation_report = census(rows, cfg["geometry"])
        save(out/"annotation_report.json", annotation_report)
        pixels = None if args.manifest_only else pixel_check(args.root, rows, cfg["geometry"], out)
        result = {"status": "annotations_complete_pixels_not_checked" if args.manifest_only else "numerical_complete_visual_review_pending",
                  "annotation_failures": 0, "pixel_check": pixels,
                  "seconds": time.monotonic()-started, "python": sys.version, "platform": platform.platform(),
                  "numpy": np.__version__, "Pillow": pillow_version,
                  "protocol_sha256": file_hash(Path(__file__).with_name("protocol.json")),
                  "train_sha256": cfg["train_manifest_sha256"],
                  "source_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
                  "held_out_data_used": False, "training_or_gpu_inference": False}
        save(out/"result.json", result)
        with (out/"SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
            for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
                handle.write(f"{file_hash(path)}  {path.relative_to(out).as_posix()}\n")
        with tarfile.open(archive, "x:gz") as tf:
            tf.add(out, arcname=out.name)
        print(f"GEOMETRY PREFLIGHT: {result['status']}\nArchive: {archive}\nArchive SHA-256: {file_hash(archive)}", flush=True)
    except BaseException as error:
        save(out/"failure.json", {"status": "failed", "error": str(error), "seconds": time.monotonic()-started})
        raise


if __name__ == "__main__":
    main()
