"""Full-image RDD data adapters. No augmentation, sampling, or model training."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

CLASSES = ("D00", "D10", "D20", "D40")
CATEGORIES = {"target_bearing", "empty_xml", "only_other_labels"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def inside(root: Path, name: str) -> Path:
    root = Path(root).resolve()
    path = root.joinpath(*relative_path(name).parts).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path escapes raw root: {name}")
    return path


def validate_record(row: dict, expected_split: str = "train") -> None:
    if row["split"] != expected_split:
        raise ValueError(f"Unexpected split: {row['split']}")
    for key in ("image", "xml"):
        if relative_path(row[key]).parts[0] != row["source"]:
            raise ValueError(f"Source/path mismatch: {row[key]}")
    for key in ("image_sha256", "xml_sha256", "rgb_sha256"):
        value = row[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid {key}")
    width, height = row["width"], row["height"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Invalid stored dimensions")
    if row["annotation_category"] not in CATEGORIES:
        raise ValueError("Unknown annotation category")
    if bool(row["targets"]) != (row["annotation_category"] == "target_bearing"):
        raise ValueError("Target/category mismatch")
    object_ids = set()
    for target in row["targets"]:
        index = target["class_index"]
        if type(index) is not int or not 0 <= index < len(CLASSES) or target["label"] != CLASSES[index]:
            raise ValueError("Foreground class index/label mismatch")
        values = target["box_xyxy"]
        if len(values) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("Invalid box coordinates")
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Box is outside stored bounds or has nonpositive area")
        if target["object_index"] in object_ids:
            raise ValueError("Duplicate target object index")
        object_ids.add(target["object_index"])


def read_manifest(path: Path, expected_sha256: str, expected_split: str = "train") -> list[dict]:
    content = Path(path).read_bytes()
    if sha256(content) != expected_sha256:
        raise ValueError("Manifest checksum mismatch")
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    names = set()
    for row in rows:
        validate_record(row, expected_split)
        if row["image"] in names:
            raise ValueError("Duplicate image in manifest")
        names.add(row["image"])
    if not rows:
        raise ValueError("Empty manifest")
    return rows


@dataclass
class Example:
    record: dict
    image_id: int
    rgb: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    geometry: dict | None = None


def load_example(root: Path, row: dict, image_id: int = 0) -> Example:
    validate_record(row)
    data = inside(root, row["image"]).read_bytes()
    if sha256(data) != row["image_sha256"]:
        raise ValueError(f"Image checksum mismatch: {row['image']}")
    if sha256(inside(root, row["xml"]).read_bytes()) != row["xml_sha256"]:
        raise ValueError(f"XML checksum mismatch: {row['xml']}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != (row["width"], row["height"]):
            raise ValueError(f"Stored image dimensions changed: {row['image']}")
        if image.getexif().get(274, 1) != row["exif_orientation"]:
            raise ValueError(f"EXIF orientation changed: {row['image']}")
        # Deliberately do not call exif_transpose: annotation coordinates use stored pixels.
        with image.convert("RGB") as rgb:
            digest = sha256(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii") + rgb.tobytes())
            if digest != row["rgb_sha256"]:
                raise ValueError(f"Decoded RGB checksum mismatch: {row['image']}")
            pixels = np.array(rgb, dtype=np.uint8, copy=True)
    boxes = np.array([t["box_xyxy"] for t in row["targets"]], dtype=np.float64).reshape(-1, 4)
    labels = np.array([t["class_index"] for t in row["targets"]], dtype=np.int64)
    return Example(copy.deepcopy(row), image_id, pixels, boxes, labels)


def prepare_example(example: Example, size: int = 640) -> Example:
    """Fit the complete image into a square; transform edges using actual rounded sizes."""
    if type(size) is not int or size <= 0 or size % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    if example.geometry is not None:
        raise ValueError("Example is already resized/padded")
    height, width = example.rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - resized_w) // 2, (size - resized_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    with Image.fromarray(example.rgb) as image:
        with image.resize((resized_w, resized_h), Image.Resampling.BILINEAR) as resized:
            canvas[top:top + resized_h, left:left + resized_w] = np.asarray(resized)
    boxes = example.boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (resized_w / width) + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (resized_h / height) + top
    geometry = {
        "original_width": width, "original_height": height,
        "resized_width": resized_w, "resized_height": resized_h,
        "pad_left": left, "pad_top": top, "size": size,
    }
    return Example(copy.deepcopy(example.record), example.image_id, canvas, boxes, example.labels.copy(), geometry)


def restore_boxes(boxes, geometry: dict) -> np.ndarray:
    restored = np.asarray(boxes, dtype=np.float64).copy()
    if restored.ndim != 2 or restored.shape[1] != 4:
        raise ValueError("Expected boxes with shape (N, 4)")
    restored[:, [0, 2]] = ((restored[:, [0, 2]] - geometry["pad_left"])
                          * geometry["original_width"] / geometry["resized_width"])
    restored[:, [1, 3]] = ((restored[:, [1, 3]] - geometry["pad_top"])
                          * geometry["original_height"] / geometry["resized_height"])
    return restored


def metadata(example: Example) -> dict:
    return {"image": example.record["image"], "image_id": example.image_id,
            "geometry": copy.deepcopy(example.geometry)}


def image_tensor(example: Example):
    import torch
    if example.geometry is None:
        raise ValueError("Prepare the full image before model collation")
    return torch.from_numpy(example.rgb.copy()).permute(2, 0, 1).contiguous().float().div_(255)


def torchvision_collate(examples: list[Example]):
    import torch
    images, targets = [], []
    for example in examples:
        images.append(image_tensor(example))
        boxes = torch.tensor(example.boxes, dtype=torch.float32)
        if not torch.all(boxes[:, 2:] > boxes[:, :2]).item():
            raise ValueError("A positive box collapsed during float32 conversion")
        targets.append({
            "boxes": boxes,
            "labels": torch.tensor(example.labels + 1, dtype=torch.int64),
            "image_id": torch.tensor([example.image_id], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        })
    return images, targets, [metadata(e) for e in examples]


def yolo_collate(examples: list[Example]):
    import torch
    if not examples or len({e.rgb.shape for e in examples}) != 1:
        raise ValueError("YOLO batches need at least one image and equal prepared dimensions")
    boxes, classes, indices = [], [], []
    for batch_index, example in enumerate(examples):
        xyxy = example.boxes
        xywh = np.concatenate(((xyxy[:, :2] + xyxy[:, 2:]) / 2,
                               xyxy[:, 2:] - xyxy[:, :2]), axis=1) / example.geometry["size"]
        normalized = torch.tensor(xywh, dtype=torch.float32)
        if not torch.all(normalized[:, 2:] > 0).item():
            raise ValueError("A positive YOLO box collapsed during float32 conversion")
        boxes.append(normalized)
        classes.append(torch.tensor(example.labels, dtype=torch.float32).reshape(-1, 1))
        indices.append(torch.full((len(xyxy),), batch_index, dtype=torch.int64))
    batch = {
        "img": torch.stack([image_tensor(e) for e in examples]),
        "bboxes": torch.cat(boxes, dim=0), "cls": torch.cat(classes, dim=0),
        "batch_idx": torch.cat(indices, dim=0),
    }
    return batch, [metadata(e) for e in examples]


class TrainingDataset:
    """Map-style PyTorch dataset; images are read only when __getitem__ is called."""
    def __init__(self, root: Path, manifest: Path, expected_sha256: str,
                 selected_images: list[str] | None = None, size: int = 640):
        self.root, self.size = Path(root), size
        all_rows = read_manifest(manifest, expected_sha256, "train")
        by_name = {r["image"]: (i, r) for i, r in enumerate(all_rows)}
        names = list(by_name) if selected_images is None else selected_images
        if not names or len(set(names)) != len(names):
            raise ValueError("Selection is empty or contains duplicates")
        if any(name not in by_name for name in names):
            raise ValueError("Selection contains an image outside the frozen training manifest")
        self.rows = [by_name[name] for name in names]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_id, row = self.rows[index]
        return prepare_example(load_example(self.root, row, image_id), self.size)
