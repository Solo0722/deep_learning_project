"""Read RDD2022 file/annotation metadata; do not modify images or XML files."""
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def audit(root):
    folders = sorted(root.glob("*/train"))
    if not folders:
        raise SystemExit("No source/train folders found under " + str(root))
    report = {}
    for folder in folders:
        source = folder.parent.name
        images = [p for p in (folder / "images").iterdir()
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        xmls = sorted(p for p in (folder / "annotations" / "xmls").iterdir()
                      if p.is_file() and p.suffix.lower() == ".xml")
        image_ids = Counter(p.stem for p in images)
        xml_ids = Counter(p.stem for p in xmls)
        issues = {
            "images_without_xml": sorted(set(image_ids) - set(xml_ids)),
            "xml_without_images": sorted(set(xml_ids) - set(image_ids)),
            "duplicate_image_stems": sorted(k for k, v in image_ids.items() if v > 1),
            "duplicate_xml_stems": sorted(k for k, v in xml_ids.items() if v > 1),
            "xml_errors": [], "size_errors": [], "box_errors": [],
            "zero_span_boxes": [], "missing_labels": [],
        }
        labels = Counter()
        empty_xmls = 0
        for path in xmls:
            try:
                tree = ET.parse(path).getroot()
            except (ET.ParseError, OSError) as error:
                issues["xml_errors"].append([path.name, str(error)])
                continue
            width = height = None
            try:
                width, height = [float(tree.findtext("size/" + key))
                                 for key in ("width", "height")]
                if not all(math.isfinite(v) and v > 0 for v in (width, height)):
                    raise ValueError("Nonpositive or nonfinite dimensions")
            except (TypeError, ValueError) as error:
                issues["size_errors"].append([path.name, str(error)])
                width = height = None
            objects = tree.findall("object")
            empty_xmls += int(not objects)
            for index, obj in enumerate(objects):
                label = (obj.findtext("name") or "").strip()
                labels[label or "<missing>"] += 1
                if not label:
                    issues["missing_labels"].append([path.name, index])
                try:
                    x1, y1, x2, y2 = [float(obj.findtext("bndbox/" + key))
                                      for key in ("xmin", "ymin", "xmax", "ymax")]
                    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                        raise ValueError("Nonfinite coordinates")
                    if min(x1, y1) < 0 or x2 < x1 or y2 < y1:
                        raise ValueError("Negative or reversed coordinates")
                    if width is not None and (x2 > width or y2 > height):
                        raise ValueError("Box exceeds XML image dimensions")
                    if x1 == x2 or y1 == y2:
                        issues["zero_span_boxes"].append([path.name, index])
                except (TypeError, ValueError) as error:
                    issues["box_errors"].append([path.name, index, str(error)])
        report[source] = {
            "images": len(images), "xml_files": len(xmls),
            "empty_xml_files": empty_xmls,
            "labels_all_parsed_objects": dict(sorted(labels.items())),
            "issues": issues,
        }
    return report


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
    result = audit(root)
    output = root.parent / "dataset_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SOURCE             IMAGES     XML     D00     D10     D20     D40")
    totals = Counter()
    for source, data in result.items():
        counts = data["labels_all_parsed_objects"]
        totals.update(counts)
        print("{:<18} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
            source, data["images"], data["xml_files"],
            *(counts.get(key, 0) for key in ("D00", "D10", "D20", "D40"))))
        flags = {key: len(value) for key, value in data["issues"].items() if value}
        print("  Flags:", flags or "none", "| Empty XMLs:", data["empty_xml_files"])
    print("All label counts (objects, including flagged boxes):", dict(sorted(totals.items())))
    print("Detailed report:", output)
    print("Metadata audit only: image decoding, duplicates, and coordinate conventions remain to check.")
