"""Geometric edits (translate, rotate, vertical flip) and random erasing.

All randomness comes from the passed rng. Returned samples never alias the input arrays.
"""
from __future__ import annotations

import numpy as np

from cua.augment.boxes import affine_sample, rotate_matrix, translate_matrix, vflip_matrix
from cua.augment.pixel import Edit, _with_rgb
from cua.image import Sample

SIZE = 640
MAX_TRIES = 10
TRANSLATE_PX = 0.10 * SIZE
ROTATE_DEG = 10.0
ERASE_AREA = (0.02, 0.10)
ERASE_ASPECT = (0.3, 3.3)
ERASE_MAX_COVER = 0.5


def _copy(sample: Sample) -> Sample:
    return _with_rgb(sample, sample.rgb.copy())  # ascontiguousarray alone would alias the input pixels


def translate(sample: Sample, rng: np.random.Generator) -> Sample:
    for _ in range(MAX_TRIES):
        dx = float(rng.uniform(-TRANSLATE_PX, TRANSLATE_PX))
        dy = float(rng.uniform(-TRANSLATE_PX, TRANSLATE_PX))
        out = affine_sample(sample, translate_matrix(dx, dy))
        if out is not None:
            return out
    return _copy(sample)


def sampled_rotation_angle(rng: np.random.Generator) -> float:
    """The draw `rotate` makes for each attempt (degrees)."""
    return float(rng.uniform(-ROTATE_DEG, ROTATE_DEG))


def rotate(sample: Sample, rng: np.random.Generator) -> Sample:
    for _ in range(MAX_TRIES):
        out = affine_sample(sample, rotate_matrix(sampled_rotation_angle(rng), SIZE))
        if out is not None:
            return out
    return _copy(sample)


def vflip(sample: Sample, rng: np.random.Generator) -> Sample:
    """Deterministic; rng is ignored. A vertical flip maps the canvas onto itself, so it never rejects a valid box."""
    out = affine_sample(sample, vflip_matrix(SIZE))
    return out if out is not None else _copy(sample)


def sampled_erase_rect(sample: Sample, rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the erase rectangle, or None after MAX_TRIES draws that cover > 50% of a non-ignored box."""
    boxes = np.asarray(sample.boxes, dtype=np.float64).reshape(-1, 4)
    ignore = np.asarray(sample.ignore, dtype=bool)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    check = ~ignore & (areas > 0)
    for _ in range(MAX_TRIES):
        area = float(rng.uniform(*ERASE_AREA)) * SIZE * SIZE
        aspect = float(rng.uniform(*ERASE_ASPECT))
        h = int(np.clip(round(np.sqrt(area * aspect)), 1, SIZE))
        w = int(np.clip(round(np.sqrt(area / aspect)), 1, SIZE))
        x0 = int(rng.integers(0, SIZE - w + 1))
        y0 = int(rng.integers(0, SIZE - h + 1))
        x1, y1 = x0 + w, y0 + h
        iw = np.clip(np.minimum(x1, boxes[:, 2]) - np.maximum(x0, boxes[:, 0]), 0.0, None)
        ih = np.clip(np.minimum(y1, boxes[:, 3]) - np.maximum(y0, boxes[:, 1]), 0.0, None)
        if not (check & (iw * ih > ERASE_MAX_COVER * areas)).any():
            return x0, y0, x1, y1
    return None


def erase(sample: Sample, rng: np.random.Generator) -> Sample:
    rect = sampled_erase_rect(sample, rng)
    if rect is None:
        return _copy(sample)
    x0, y0, x1, y1 = rect
    rgb = sample.rgb.copy()
    rgb[y0:y1, x0:x1] = rng.integers(0, 256, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
    return _with_rgb(sample, rgb)


GEOMETRIC_EDITS: dict[str, Edit] = {
    "translate": translate,
    "rotate": rotate,
    "vflip": vflip,
    "erase": erase,
}
