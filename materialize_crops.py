"""Create fixed RDD crop images from an already grouped and checked manifest.

Pillow only; Python 3.9 compatible. No random augmentation is applied. All
source hashes are checked; output must be a new directory outside raw.
Training-only preview sheets are bundled separately for visual review.
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import socket
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFile

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
ImageFile.LOAD_TRUNCATED_IMAGES = False


def safe_path(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def letterbox(picture, size=224):
    scale = min(size / picture.width, size / picture.height)
    new_size = (max(1, min(size, round(picture.width * scale))),
                max(1, min(size, round(picture.height * scale))))
    resized = picture.resize(new_size, RESAMPLE)
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    resized.close()
    return canvas


def select_previews(rows):
    pools = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            pools[(row["source"], row["label"])].append(row)
    selected = defaultdict(dict)
    for (source, label), pool in sorted(pools.items()):
        # Include a thin crop to make geometry problems visible, plus up to
        # three reproducible varied examples. Selection uses no model scores.
        def ratio(row):
            width, height = int(row["right"])-int(row["left"]), int(row["bottom"])-int(row["top"])
            return max(width / height, height / width)
        extreme = max(pool, key=lambda r: (ratio(r), r["sample_id"]))
        others = sorted((r for r in pool if r is not extreme), key=lambda r:
                        hashlib.sha256(("qa:" + r["sample_id"]).encode()).hexdigest())
        selected[source][label] = [extreme] + others[:3]
    return selected


def previews(rows, output, labels):
    directory = output / "qa"
    directory.mkdir()
    selected = select_previews(rows)
    chosen = []
    for source, categories in sorted(selected.items()):
        canvas = Image.new("RGB", (960, 1080), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), source + " | TRAIN ONLY | column 1: thinnest crop; others: sampled", fill="black")
        for row, label in enumerate(labels):
            y = 30 + row * 260
            draw.text((12, y), label, fill="black")
            for column, record in enumerate(categories.get(label, [])):
                x = column * 240 + 8
                with Image.open(safe_path(output, record["crop_path"])) as picture:
                    canvas.paste(picture, (x, y + 17))
                caption = Path(record["image"]).stem + " o" + record["object_index"]
                draw.text((x, y + 242), caption, fill="black")
                chosen.append(record)
        canvas.save(directory / (source + ".jpg"), quality=94)
        canvas.close()
    with (directory / "selected_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(chosen)
    return len(chosen)


def materialize(root, manifest, output):
    root, manifest, output = root.resolve(), manifest.resolve(), output.resolve()
    config = json.loads(manifest.with_name("config.json").read_text(encoding="utf-8"))
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Manifest differs from the versioned split configuration")
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the raw dataset")
    review_bundle = output.parent / (output.name + "_qa.tar.gz")
    if output.exists() or review_bundle.exists():
        raise FileExistsError("Output or QA bundle exists; use a new output directory")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["crop_path"] for r in rows}) != len(rows):
        raise ValueError("Empty manifest or duplicate crop paths")
    grouped = defaultdict(list)
    for row in rows:
        if config["labels"][row["label_id"]] != row["label"]:
            raise ValueError("Class mapping mismatch")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        safe_path(output, row["crop_path"])
        grouped[row["image"]].append(row)
    if any(len({r["split"] for r in group}) != 1 for group in grouped.values()):
        raise ValueError("A source image spans multiple splits")
    group_splits = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    if any(len(value) != 1 for value in group_splits.values()):
        raise ValueError("A related-image group spans multiple splits")
    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        for label in config["labels"].values():
            (output / split / label).mkdir(parents=True)
    start = time.monotonic()
    run = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "manifest_sha256": config["manifest_sha256"], "pillow": PIL.__version__,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "preprocessing": config["pixel_preprocessing"], "root": str(root),
           "expected_images": len(grouped), "expected_crops": len(rows)}
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    counts = {split: Counter() for split in ("train", "val", "test")}
    written = 0
    with (output / "crop_checksums.csv").open("w", newline="", encoding="utf-8") as checksums:
        writer = csv.writer(checksums)
        writer.writerow(("crop_path", "sha256", "bytes"))
        for number, (name, members) in enumerate(sorted(grouped.items()), 1):
            first = members[0]
            if any(r["file_sha256"] != first["file_sha256"] or r["xml_sha256"] != first["xml_sha256"]
                   or r["xml"] != first["xml"] for r in members):
                raise ValueError("Inconsistent source metadata: " + name)
            data = safe_path(root, name).read_bytes()
            if hashlib.sha256(data).hexdigest() != first["file_sha256"]:
                raise ValueError("Source image changed: " + name)
            if hashlib.sha256(safe_path(root, first["xml"]).read_bytes()).hexdigest() != first["xml_sha256"]:
                raise ValueError("Source XML changed: " + first["xml"])
            with Image.open(io.BytesIO(data)) as original:
                if original.getexif().get(274, 1) != 1:
                    raise ValueError("Unresolved EXIF orientation: " + name)
                if original.size != (int(first["original_width"]), int(first["original_height"])):
                    raise ValueError("Source image size changed: " + name)
                with original.convert("RGB") as image:
                    for record in members:
                        box = tuple(int(record[key]) for key in ("left", "top", "right", "bottom"))
                        if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                            raise ValueError("Invalid crop bounds")
                        with image.crop(box) as crop:
                            with letterbox(crop, config["crop_size"]) as canvas:
                                path = safe_path(output, record["crop_path"])
                                canvas.save(path, format="PNG", compress_level=3)
                        content = path.read_bytes()
                        writer.writerow((record["crop_path"], hashlib.sha256(content).hexdigest(), len(content)))
                        counts[record["split"]][record["label"]] += 1
                        written += 1
            if number % 250 == 0 or number == len(grouped):
                checksums.flush()
                print("Images %d/%d | crops %d/%d | %.1f min" %
                      (number, len(grouped), written, len(rows), (time.monotonic()-start)/60), flush=True)
    selected = previews(rows, output, [config["labels"][str(i)] for i in range(4)])
    if written != len(rows):
        raise ValueError("Crop count mismatch")
    shutil.copyfile(manifest, output / "crops.csv")
    shutil.copyfile(manifest.with_name("config.json"), output / "config.json")
    summary = {"status": "complete", "images_processed": len(grouped), "crops_written": written,
               "crop_size": config["crop_size"], "counts": counts,
               "training_preview_samples": selected,
               "qa_note": "Preview sheets use training crops only; inspect before model training",
               "test_policy": "Use test crops only after model and policy selection are fixed"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run.update(status="complete", elapsed_seconds=round(time.monotonic()-start, 2),
               finished_utc=datetime.now(timezone.utc).isoformat())
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    with tarfile.open(review_bundle, "w:gz") as archive:
        archive.add(output / "qa", arcname="crop_review/qa")
        for name in ("summary.json", "run.json", "config.json"):
            archive.add(output / name, arcname="crop_review/" + name)
    print("\nCROP PREPARATION COMPLETE")
    print("Images:", len(grouped), "Crops:", written)
    for split, values in counts.items():
        print(split + ":", dict(values))
    print("Review bundle:", review_bundle, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower().startswith("login"):
        parser.error("Run crop preparation on a compute node")
    materialize(args.root, args.manifest, args.output)
