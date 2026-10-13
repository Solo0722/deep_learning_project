"""Read-only RDD image audit. Requires Python 3 and Pillow; uses one CPU.

Run on a compute node:
    python -u validate_images.py /path/to/dataset/raw

Writes a new timestamped report directory beside raw. Does not crop, delete,
repair, split, or alter input images/XML. Exact duplicate groups are evidence
for later split grouping, not instructions to delete files.
"""
import argparse
import hashlib
import io
import json
import math
import platform
import socket
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import PIL
    from PIL import Image, ImageFile
except ImportError:
    raise SystemExit("Pillow is missing from this Python environment. No images checked.")

ImageFile.LOAD_TRUNCATED_IMAGES = False
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_image(path, root, xml_paths):
    record = {
        "image": path.relative_to(root).as_posix(),
        "source": path.parent.parent.parent.name,
        "issues": [],
    }
    xml_size = None
    if len(xml_paths) != 1:
        record["issues"].append({"type": "xml_partner_count", "count": len(xml_paths)})
    else:
        record["xml"] = xml_paths[0].relative_to(root).as_posix()
        try:
            tree = ET.parse(xml_paths[0]).getroot()
            values = [float(tree.findtext("size/" + key)) for key in ("width", "height")]
            if not all(math.isfinite(x) and x > 0 and x.is_integer() for x in values):
                raise ValueError("XML dimensions must be positive integers")
            xml_size = [int(x) for x in values]
            record["xml_size"] = xml_size
        except (ET.ParseError, OSError, TypeError, ValueError) as error:
            record["issues"].append({"type": "xml_error", "detail": str(error)})

    try:
        data = path.read_bytes()
        record["file_bytes"] = len(data)
        record["file_sha256"] = hashlib.sha256(data).hexdigest()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with Image.open(io.BytesIO(data)) as picture:
                record.update(size=list(picture.size), mode=picture.mode, format=picture.format)
                picture.load()  # Force pixel decoding; opening the header alone is insufficient.
                record["decoded"] = True
                orientation = picture.getexif().get(274, 1)
                record["exif_orientation"] = orientation
                if orientation != 1:
                    record["issues"].append({"type": "exif_orientation", "value": orientation})
                if picture.mode != "RGB":
                    record["issues"].append({"type": "non_rgb_mode", "value": picture.mode})
                if getattr(picture, "n_frames", 1) != 1:
                    record["issues"].append({"type": "multiple_frames"})
                if xml_size is not None and list(picture.size) != xml_size:
                    record["issues"].append({"type": "dimension_mismatch"})
                # Do not transpose EXIF orientation: retain the stored coordinate frame.
                # Include dimensions so equal byte streams with different shapes cannot match.
                with picture.convert("RGB") as rgb:
                    digest = hashlib.sha256()
                    digest.update(("RGB:%d:%d:" % rgb.size).encode("ascii"))
                    digest.update(rgb.tobytes())
                    record["rgb_sha256"] = digest.hexdigest()
            for warning in observed:
                record["issues"].append({"type": "image_warning", "detail": str(warning.message)})
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        record["issues"].append({"type": "image_error", "detail": str(error)})
    return record


def audit(root, output, progress_every=250):
    root = root.resolve()
    folders = sorted(root.glob("*/train"))
    if not folders:
        raise ValueError("No source/train folders found under " + str(root))
    inputs = []
    for folder in folders:
        images = sorted(p for p in (folder / "images").iterdir()
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
        xml_index = defaultdict(list)
        for path in (folder / "annotations" / "xmls").iterdir():
            if path.is_file() and path.suffix.lower() == ".xml":
                xml_index[path.stem].append(path)
        inputs.extend((path, xml_index.get(path.stem, [])) for path in images)
    if not inputs:
        raise ValueError("No supported images found")
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Reports must be outside the raw dataset directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = {
        "status": "running", "root": str(root), "images_expected": len(inputs),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "pillow": PIL.__version__,
        "host": socket.gethostname(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pixel_hash_definition": "SHA256(ASCII RGB:width:height: + RGB pixels, stored orientation)",
        "limitations": ["Exact duplicates only; no near-duplicate or sequence audit",
                        "No coordinate-convention or crop-quality decision",
                        "Warnings and nonstandard modes require review"],
    }
    metadata_path = output / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Report directory:", output, flush=True)
    print("Images to check:", len(inputs), flush=True)
    file_groups, pixel_groups = defaultdict(list), defaultdict(list)
    issue_counts, by_source = Counter(), {}
    flagged = []
    with (output / "inventory.jsonl").open("w", encoding="utf-8") as inventory:
        for index, (path, xml_paths) in enumerate(inputs, 1):
            record = inspect_image(path, root, xml_paths)
            inventory.write(json.dumps(record) + "\n")
            for key, groups in (("file_sha256", file_groups), ("rgb_sha256", pixel_groups)):
                if key in record:
                    groups[record[key]].append(record["image"])
            counts = by_source.setdefault(record["source"], Counter())
            counts["images"] += 1
            counts["decoded"] += bool(record.get("decoded"))
            counts["flagged_images"] += bool(record["issues"])
            if record["issues"]:
                # Count images per issue type, not repeated warnings on one image.
                issue_counts.update({issue["type"] for issue in record["issues"]})
                flagged.append(record)
            if index % progress_every == 0 or index == len(inputs):
                inventory.flush()
                print("Checked %d/%d images (%.1f minutes)" %
                      (index, len(inputs), (time.monotonic() - started) / 60), flush=True)
    summary = {
        "images_checked": len(inputs),
        "decoded_images": sum(c["decoded"] for c in by_source.values()),
        "flagged_images": len(flagged), "issue_counts_images": dict(issue_counts),
        "by_source": by_source, "flagged_records": flagged,
        "byte_identical_groups": [v for v in file_groups.values() if len(v) > 1],
        "rgb_identical_groups": [v for v in pixel_groups.values() if len(v) > 1],
        "limitations": metadata["limitations"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=round(time.monotonic() - started, 2))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nAUDIT COMPLETE", flush=True)
    print("Images checked:", summary["images_checked"])
    print("Decoded successfully:", summary["decoded_images"])
    print("Flagged images:", summary["flagged_images"])
    print("Issue counts (images):", dict(issue_counts))
    print("Byte-identical groups:", len(summary["byte_identical_groups"]))
    print("RGB-identical groups:", len(summary["rgb_identical_groups"]))
    print("These two duplicate-group counts overlap; do not add them.")
    print("Detailed report:", output / "summary.json")
    print("Near-duplicate images and capture sequences still need checking.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="RDD dataset/raw directory")
    parser.add_argument("--out", type=Path, help="New report directory outside raw")
    args = parser.parse_args()
    if socket.gethostname().split(".")[0].lower().startswith("login"):
        parser.error("Run this image audit on a compute node, not a login node.")
    root = args.root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.out or root.parent / ("image_audit_" + stamp)
    try:
        audit(root, output)
    except (OSError, ValueError) as error:
        raise SystemExit("Audit stopped: " + str(error))


if __name__ == "__main__":
    main()
