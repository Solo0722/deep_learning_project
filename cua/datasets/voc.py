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


def parse_voc_xml(path: Path, year: str) -> Record:
    """Parse one VOC annotation file into a Record.

    VOC boxes are 1-based inclusive pixel coordinates; they are converted to
    zero-based continuous XYXY (xmin-1, ymin-1, xmax, ymax).
    """
    root = ET.parse(Path(path)).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    image_id = Path(path).stem

    boxes, labels, ignore = [], [], []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in _CLASS_INDEX:
            raise ValueError(f"unknown VOC class {name!r} in {path}")
        bb = obj.find("bndbox")
        boxes.append([
            float(bb.findtext("xmin")) - 1.0,
            float(bb.findtext("ymin")) - 1.0,
            float(bb.findtext("xmax")),
            float(bb.findtext("ymax")),
        ])
        labels.append(_CLASS_INDEX[name])
        ignore.append(int(obj.findtext("difficult", "0").strip()) == 1)

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


def _load_ids(devkit: Path, year: str, split: str) -> list[Record]:
    base = devkit / f"VOC{year}"
    ids = _read_ids(base / "ImageSets" / "Main" / f"{split}.txt")
    return [parse_voc_xml(base / "Annotations" / f"{i}.xml", year=year) for i in ids]


def load_voc(devkit: Path, val_fraction: float = 0.10, seed: int = 0) -> dict[str, list[Record]]:
    """Load VOC2007+VOC2012 trainval (split into train/val) and VOC2007 test.

    Val membership is decided per group ("{year}/{image_id}") by split_trainval.
    """
    devkit = Path(devkit)
    trainval = _load_ids(devkit, "2007", "trainval") + _load_ids(devkit, "2012", "trainval")
    by_group = {r.group: r for r in trainval}
    train_ids, val_ids = split_trainval(list(by_group), val_fraction, seed)
    return {
        "train": [by_group[g] for g in train_ids],
        "val": [by_group[g] for g in val_ids],
        "test": _load_ids(devkit, "2007", "test"),
    }
