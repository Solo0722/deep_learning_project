"""Screen target-bearing RDD images for near duplicates before splitting.

Uses only Pillow and Python's standard library (Python 3.9 compatible).
Writes annotation records, candidate pairs and contact sheets; does not change
raw files or create splits. A perceptual-hash match is a review candidate,
not proof of duplication. This screen cannot rule out overlapping road scenes.
"""
import argparse
import hashlib
import io
import json
import math
import platform
import socket
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFile

TARGETS = ("D00", "D10", "D20", "D40")
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
ImageFile.LOAD_TRUNCATED_IMAGES = False


def hamming(left, right):
    value = left ^ right
    return value.bit_count() if hasattr(value, "bit_count") else bin(value).count("1")


class HashIndex:
    """BK-tree: exact radius queries in Hamming space, with repeated hashes."""
    def __init__(self):
        self.root = None

    def add(self, value, key):
        node = self.root
        if node is None:
            self.root = [value, [key], {}]
            return
        while True:
            distance = hamming(value, node[0])
            if distance == 0:
                node[1].append(key)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [key], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = hamming(value, node[0])
            if distance <= radius:
                for key in node[1]:
                    yield distance, key
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)


def dhash128(picture):
    """Concatenate 64 horizontal and 64 vertical luminance comparisons."""
    gray = picture.convert("L")
    horizontal = gray.resize((9, 8), RESAMPLE).tobytes()
    vertical = gray.resize((8, 9), RESAMPLE).tobytes()
    gray.close()
    bits = [horizontal[y * 9 + x] > horizontal[y * 9 + x + 1]
            for y in range(8) for x in range(8)]
    bits.extend(vertical[y * 8 + x] > vertical[(y + 1) * 8 + x]
                for y in range(8) for x in range(8))
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def read_objects(path):
    data = path.read_bytes()
    tree = ET.fromstring(data)
    objects = []
    for index, obj in enumerate(tree.findall("object")):
        box = [float(obj.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if not all(math.isfinite(v) for v in box):
            raise ValueError("Nonfinite box: " + str(path))
        objects.append({"index": index, "label": (obj.findtext("name") or "").strip(),
                        "box": box, "difficult": obj.findtext("difficult"),
                        "truncated": obj.findtext("truncated")})
    return objects, hashlib.sha256(data).hexdigest()


def contact_sheets(root, pairs, output, limit=36):
    selected = sorted(pairs, key=lambda p: (p["distance"], p["a"], p["b"]))[:limit]
    files = []
    for start in range(0, len(selected), 4):
        page = selected[start:start + 4]
        sheet = Image.new("RGB", (660, len(page) * 282 + 25), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 5), "Near-duplicate candidates: inspect scene overlap; hash matches may be unrelated.", fill="black")
        for row, pair in enumerate(page):
            y = 25 + row * 282
            draw.text((8, y), "Distance %d / 128" % pair["distance"], fill="black")
            for col, key in enumerate(("a", "b")):
                x = 8 + col * 330
                draw.text((x, y + 16), Path(pair[key]).name, fill="black")
                with Image.open(inside(root, pair[key])) as image:
                    image.draft("RGB", (320, 240))
                    thumb = image.convert("RGB")
                    thumb.thumbnail((320, 240), RESAMPLE)
                    sheet.paste(thumb, (x, y + 34))
                    thumb.close()
        path = output / ("candidates_%02d.jpg" % (start // 4 + 1))
        sheet.save(path, quality=92)
        sheet.close()
        files.append(path.name)
    return files


def screen(root, inventory, output, radius=8, max_pairs=20000):
    root, inventory, output = root.resolve(), inventory.resolve(), output.resolve()
    if not 0 <= radius <= 16 or max_pairs < 1:
        raise ValueError("Radius must be 0..16 and max-pairs must be positive")
    if output == root or root in output.parents:
        raise ValueError("Reports must be outside raw")
    prior_run = json.loads(inventory.with_name("run.json").read_text(encoding="utf-8"))
    if prior_run.get("status") != "complete":
        raise ValueError("The source image audit is incomplete")
    records = [json.loads(line) for line in inventory.read_text(encoding="utf-8").splitlines()]
    if len(records) != prior_run["images_expected"]:
        raise ValueError("Inventory row count does not match the completed audit")
    if len({r["image"] for r in records}) != len(records):
        raise ValueError("Repeated image paths in inventory")
    records.sort(key=lambda r: r["image"])
    if output.exists() or output.with_suffix(".tar.gz").exists():
        raise FileExistsError("Report or bundle already exists; choose a new --out directory")
    output.mkdir(parents=True)
    start = time.monotonic()
    run = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "python": platform.python_version(), "pillow": PIL.__version__,
           "root": str(root), "inventory": str(inventory),
           "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "radius": radius, "max_pairs": max_pairs,
           "method": "128-bit horizontal+vertical dHash; Pillow JPEG draft at 256px; stored orientation",
           "limitations": ["Heuristic screening; no guarantee of finding all similar images",
                           "Matches are candidates, not verified duplicates",
                           "Capture sequence IDs are unavailable in this inventory",
                           "Only images with at least one target annotation are screened"]}
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print("Reading all XML annotations...", flush=True)
    eligible, skipped, canonical = [], [], {}
    counts, by_source = Counter(), defaultdict(Counter)
    coord_stats = defaultdict(Counter)
    with (output / "annotations.jsonl").open("w", encoding="utf-8") as log:
        for record in records:
            objects, xml_hash = read_objects(inside(root, record["xml"]))
            targets = Counter(o["label"] for o in objects if o["label"] in TARGETS)
            entry = {"image": record["image"], "source": record["source"],
                     "xml": record["xml"], "xml_sha256": xml_hash,
                     "size": record["size"], "rgb_sha256": record.get("rgb_sha256"),
                     "target_counts": dict(targets), "objects": objects}
            if not targets:
                counts["images_without_target_objects"] += 1
            else:
                if not record.get("decoded") or not record.get("rgb_sha256"):
                    raise ValueError("Target image did not pass image audit: " + record["image"])
                if record.get("exif_orientation", 1) != 1 or record.get("issues"):
                    raise ValueError("Unresolved target image flag: " + record["image"])
                counts["target_bearing_images_before_exact_dedup"] += 1
                rgb_hash = record["rgb_sha256"]
                if rgb_hash in canonical:
                    entry["duplicate_of"] = canonical[rgb_hash]
                    skipped.append({"image": record["image"], "duplicate_of": canonical[rgb_hash],
                                    "target_counts": dict(targets)})
                else:
                    canonical[rgb_hash] = record["image"]
                    eligible.append(record)
                    by_source[record["source"]].update(targets)
                    for obj in objects:
                        if obj["label"] not in TARGETS:
                            continue
                        x1, y1, x2, y2 = obj["box"]
                        stat = coord_stats[record["source"]]
                        stat["target_objects_before_box_filtering"] += 1
                        stat["objects_with_zero_xmin_or_ymin"] += int(x1 == 0 or y1 == 0)
                        stat["objects_touching_xml_width_or_height"] += int(x2 == record["size"][0] or y2 == record["size"][1])
                        for threshold in (1, 2, 4, 8, 16):
                            stat["objects_min_endpoint_span_below_%d" % threshold] += int(min(x2-x1, y2-y1) < threshold)
            log.write(json.dumps(entry) + "\n")
    counts["inventory_images"] = len(records)
    counts["exact_duplicate_copies_excluded"] = len(skipped)
    counts["eligible_unique_images"] = len(eligible)
    print("Target-bearing unique images to screen:", len(eligible), flush=True)
    print("Exact duplicate copies omitted from screening:", len(skipped), flush=True)
    index = HashIndex()
    pairs, seen = [], 0
    capped = False
    with (output / "fingerprints.jsonl").open("w", encoding="utf-8") as fingerprints, \
            (output / "candidate_pairs.jsonl").open("w", encoding="utf-8") as candidates:
        for number, record in enumerate(eligible, 1):
            data = inside(root, record["image"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != record["file_sha256"]:
                raise ValueError("Image changed since image audit: " + record["image"])
            with Image.open(io.BytesIO(data)) as picture:
                picture.draft("RGB", (256, 256))
                value = dhash128(picture)
            fingerprints.write(json.dumps({"image": record["image"], "dhash128": "%032x" % value}) + "\n")
            for distance, other in index.query(value, radius):
                if len(pairs) >= max_pairs:
                    capped = True
                    break
                pair = {"a": other, "b": record["image"], "distance": distance}
                pairs.append(pair)
                candidates.write(json.dumps(pair) + "\n")
            index.add(value, record["image"])
            seen = number
            if number % 500 == 0 or number == len(eligible) or capped:
                fingerprints.flush()
                candidates.flush()
                print("Screened %d/%d; candidate pairs %d; %.1f min" %
                      (number, len(eligible), len(pairs), (time.monotonic()-start)/60), flush=True)
            if capped:
                break
    previews = contact_sheets(root, pairs, output)
    summary = {"screen_complete": not capped, "counts": dict(counts),
               "images_screened": seen, "candidate_pairs": len(pairs),
               "distance_histogram": dict(sorted(Counter(p["distance"] for p in pairs).items())),
               "candidate_limit_reached": capped, "radius": radius,
               "excluded_exact_copies": skipped,
               "target_objects_by_source_before_box_filtering": dict(by_source),
               "coordinate_stats_before_box_filtering": dict(coord_stats),
               "preview_files": previews, "preview_limit_pairs": 36,
               "limitations": run["limitations"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run.update(status="incomplete_candidate_limit" if capped else "complete",
               elapsed_seconds=round(time.monotonic()-start, 2),
               finished_utc=datetime.now(timezone.utc).isoformat())
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    bundle = output.with_suffix(".tar.gz")
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    print("\nNEAR-DUPLICATE SCREEN " + ("INCOMPLETE (candidate limit)" if capped else "COMPLETE"))
    for key, value in counts.items():
        print(key + ":", value)
    print("Images screened:", seen)
    print("Candidate pairs:", len(pairs))
    print("Distance histogram:", summary["distance_histogram"])
    print("Review bundle:", bundle)
    print("Candidates require review; raw images and annotations are unchanged.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--distance", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=20000)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower().startswith("login"):
        parser.error("Run this screening on a compute node, not a login node")
    output = args.out or args.root.resolve().parent / "preparation_review"
    result = screen(args.root, args.inventory, output, args.distance, args.max_pairs)
    if not result["screen_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
