"""PASCAL VOC adapter: loads VOC2007/VOC2012 devkit annotations into common Records."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cua.records import Record

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
)

_CLASS_INDEX = {name: i for i, name in enumerate(VOC_CLASSES)}


def _required(elem, tag: str) -> str:
    """Return the stripped text of a required child element."""
    child = elem.find(tag) if elem is not None else None
    if child is None or child.text is None or not child.text.strip():
        raise ValueError(f"missing or empty <{tag}>")
    return child.text.strip()


def _flag(obj, tag: str) -> bool:
    """Read an optional 0/1 flag; a missing or empty tag counts as 0."""
    text = obj.findtext(tag)
    if text is None or not text.strip():
        return False
    return int(text.strip()) == 1


def parse_voc_xml(path: Path, year: str, *, counts: dict | None = None) -> Record:
    """Parse one VOC annotation file into a Record.

    VOC boxes are 1-based inclusive pixel coordinates; they are converted to
    zero-based continuous XYXY (xmin-1, ymin-1, xmax, ymax). Boxes that reach
    outside the image are clamped to [0, width] x [0, height]; a box that is
    degenerate after clamping is dropped. If `counts` is given, it accumulates
    "clamped_boxes" (kept boxes whose coordinates changed) and "dropped_boxes".
    Only direct <object> children are read, so nested <part> elements are ignored.
    """
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"{path}: malformed XML: {e}") from e
    image_id = path.stem
    try:
        size = root.find("size")
        width = int(_required(size, "width"))
        height = int(_required(size, "height"))
    except ValueError as e:
        raise ValueError(f"{path}: bad <size>: {e}") from e

    boxes, labels, ignore = [], [], []
    clamped = dropped = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in _CLASS_INDEX:
            raise ValueError(f"{path}: unknown VOC class {name!r}")
        try:
            bb = obj.find("bndbox")
            x1 = float(_required(bb, "xmin")) - 1.0
            y1 = float(_required(bb, "ymin")) - 1.0
            x2 = float(_required(bb, "xmax"))
            y2 = float(_required(bb, "ymax"))
            difficult = _flag(obj, "difficult")
        except ValueError as e:
            raise ValueError(f"{path}: bad <object> {name!r}: {e}") from e
        cx1 = min(max(x1, 0.0), float(width))
        cy1 = min(max(y1, 0.0), float(height))
        cx2 = min(max(x2, 0.0), float(width))
        cy2 = min(max(y2, 0.0), float(height))
        if cx2 <= cx1 or cy2 <= cy1:
            dropped += 1
            continue
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            clamped += 1
        boxes.append([cx1, cy1, cx2, cy2])
        labels.append(_CLASS_INDEX[name])
        ignore.append(difficult)

    if counts is not None:
        counts["clamped_boxes"] = counts.get("clamped_boxes", 0) + clamped
        counts["dropped_boxes"] = counts.get("dropped_boxes", 0) + dropped

    return Record(
        image=f"VOC{year}/JPEGImages/{image_id}.jpg",
        width=width,
        height=height,
        boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        labels=np.asarray(labels, dtype=np.int64),
        ignore=np.asarray(ignore, dtype=bool),
        group=f"{year}/{image_id}",
        source=f"VOC{year}",
    )


def split_trainval(ids, fraction: float, seed: int) -> tuple[list, list]:
    """Deterministically split ids into (train, val) by a keyed SHA-256 hash.

    An id goes to val when the first 8 bytes of SHA-256(f"{seed}:{id}"), read as a
    big-endian integer, fall below fraction * 2**64. Input order is preserved.
    """
    threshold = fraction * 2**64
    train, val = [], []
    for i in ids:
        digest = hashlib.sha256(f"{seed}:{i}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        (val if value < threshold else train).append(i)
    return train, val


def _read_ids(path: Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_ids(devkit: Path, year: str, split: str, counts: dict | None) -> list[Record]:
    base = devkit / f"VOC{year}"
    ids = _read_ids(base / "ImageSets" / "Main" / f"{split}.txt")
    return [
        parse_voc_xml(base / "Annotations" / f"{i}.xml", year=year, counts=counts) for i in ids
    ]


def load_voc(
    devkit: Path,
    val_fraction: float = 0.10,
    seed: int = 0,
    *,
    stats: dict | None = None,
) -> dict[str, list[Record]]:
    """Load VOC2007+VOC2012 trainval (split into train/val) and VOC2007 test.

    Val membership is decided per group ("{year}/{image_id}") by split_trainval.
    If `stats` is given, it receives the totals "clamped_boxes" and "dropped_boxes"
    across all loaded images (see parse_voc_xml).
    """
    devkit = Path(devkit)
    if stats is None:
        stats = {}
    stats.setdefault("clamped_boxes", 0)
    stats.setdefault("dropped_boxes", 0)
    trainval = _load_ids(devkit, "2007", "trainval", stats) + _load_ids(
        devkit, "2012", "trainval", stats
    )
    by_group = {r.group: r for r in trainval}
    train_ids, val_ids = split_trainval(list(by_group), val_fraction, seed)
    return {
        "train": [by_group[g] for g in train_ids],
        "val": [by_group[g] for g in val_ids],
        "test": _load_ids(devkit, "2007", "test", stats),
    }
