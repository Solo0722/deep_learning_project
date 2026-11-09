"""Image loading and letterbox resizing for CUA records. No augmentation or sampling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from cua.records import Record


@dataclass
class Sample:
    """A letterboxed image with boxes in canvas coordinates."""

    rgb: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    ignore: np.ndarray


def load_rgb(root: Path, record: Record) -> np.ndarray:
    """Decode the image in stored orientation as RGB uint8 (no EXIF transpose)."""
    path = Path(root).joinpath(*PurePosixPath(record.image).parts)
    with Image.open(path) as image:
        image.load()
        if image.size != (record.width, record.height):
            raise ValueError(
                f"Decoded size {image.size} does not match record {(record.width, record.height)}: {record.image}"
            )
        with image.convert("RGB") as rgb:
            return np.array(rgb, dtype=np.uint8, copy=True)


def letterbox(rgb: np.ndarray, record: Record, size: int = 640) -> tuple[Sample, dict]:
    """Fit the complete image into a square canvas padded with 114; transform boxes to match."""
    if type(size) is not int or size <= 0 or size % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    rgb = np.asarray(rgb)
    if rgb.shape != (record.height, record.width, 3):
        raise ValueError(
            f"Image shape {rgb.shape} does not match record {(record.height, record.width, 3)}: {record.image}"
        )
    width, height = record.width, record.height
    scale = min(size / width, size / height)
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - resized_w) // 2, (size - resized_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    with Image.fromarray(rgb) as image:
        with image.resize((resized_w, resized_h), Image.Resampling.BILINEAR) as resized:
            canvas[top:top + resized_h, left:left + resized_w] = np.asarray(resized)
    boxes = record.boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (resized_w / width) + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (resized_h / height) + top
    geometry = {
        "original_width": width, "original_height": height,
        "resized_width": resized_w, "resized_height": resized_h,
        "pad_left": left, "pad_top": top, "size": size,
    }
    sample = Sample(canvas, boxes, record.labels.copy(), record.ignore.copy())
    return sample, geometry


def restore_boxes(boxes, geometry: dict) -> np.ndarray:
    """Map canvas XYXY boxes back to original-image pixels."""
    restored = np.asarray(boxes, dtype=np.float64).copy()
    if restored.ndim != 2 or restored.shape[1] != 4:
        raise ValueError("Expected boxes with shape (N, 4)")
    restored[:, [0, 2]] = ((restored[:, [0, 2]] - geometry["pad_left"])
                          * geometry["original_width"] / geometry["resized_width"])
    restored[:, [1, 3]] = ((restored[:, [1, 3]] - geometry["pad_top"])
                          * geometry["original_height"] / geometry["resized_height"])
    return restored
