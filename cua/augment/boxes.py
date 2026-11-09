"""Affine matrices and the shared geometric warp for samples.

Matrices are 3x3 forward maps in continuous pixel coordinates (pixel edges at
integers, so a 640 canvas spans [0, 640]).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from cua.image import Sample

FILL = (114, 114, 114)
_EPS = 1e-6


def hflip_matrix(size: int) -> np.ndarray:
    return np.array([[-1.0, 0.0, float(size)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def scale_matrix(s: float, size: int) -> np.ndarray:
    c = size / 2.0
    return np.array([[s, 0.0, c - s * c], [0.0, s, c - s * c], [0.0, 0.0, 1.0]])


def compose(*matrices: np.ndarray) -> np.ndarray:
    """Matrices applied in argument order: compose(a, b) applies a first, then b."""
    out = np.eye(3)
    for m in matrices:
        out = np.asarray(m, dtype=np.float64) @ out
    return out


def _hull(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack([
        np.stack([x0, y0], -1), np.stack([x1, y0], -1),
        np.stack([x1, y1], -1), np.stack([x0, y1], -1),
    ], axis=1)  # (N, 4, 2)
    pts = corners @ matrix[:2, :2].T + matrix[:2, 2]
    return np.concatenate([pts.min(axis=1), pts.max(axis=1)], axis=1)


def affine_sample(sample: Sample, matrix: np.ndarray) -> Sample | None:
    """Warp image (bilinear, pad 114) and boxes (corner hull). None if a kept box leaves the canvas or collapses."""
    matrix = np.asarray(matrix, dtype=np.float64)
    height, width = sample.rgb.shape[:2]
    boxes = np.asarray(sample.boxes, dtype=np.float64).reshape(-1, 4)
    hull = _hull(boxes, matrix) if len(boxes) else boxes.copy()
    ignore = np.asarray(sample.ignore, dtype=bool)
    limit = np.array([width, height, width, height], dtype=np.float64)

    outside = ((hull < -_EPS) | (hull > limit + _EPS)).any(axis=1)
    if (outside & ~ignore).any():
        return None
    hull = np.clip(hull, 0.0, limit)
    small = ((hull[:, 2] - hull[:, 0]) < 1.0) | ((hull[:, 3] - hull[:, 1]) < 1.0)
    if (small & ~ignore).any():
        return None
    keep = ~small  # only ignored boxes can be small here
    inverse = np.linalg.inv(matrix)
    with Image.fromarray(sample.rgb) as image:
        with image.transform((width, height), Image.Transform.AFFINE,
                             tuple(inverse[:2].flatten()),
                             resample=Image.Resampling.BILINEAR, fillcolor=FILL) as warped:
            rgb = np.asarray(warped, dtype=np.uint8).copy()
    return Sample(rgb, hull[keep], np.asarray(sample.labels)[keep].copy(), ignore[keep].copy())
