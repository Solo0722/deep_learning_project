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


def vflip_matrix(size: int) -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, float(size)], [0.0, 0.0, 1.0]])


def translate_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]])


def rotate_matrix(deg: float, size: int) -> np.ndarray:
    """Rotation about the canvas centre; positive angle is counter-clockwise on screen (y axis points down)."""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    cx = size / 2.0
    return np.array([
        [c, s, cx - c * cx - s * cx],
        [-s, c, cx + s * cx - c * cx],
        [0.0, 0.0, 1.0],
    ])


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


def _valid(boxes: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Inside the canvas (within eps) and at least 1 px wide and tall."""
    inside = ((boxes >= -_EPS) & (boxes <= limit + _EPS)).all(axis=1)
    big = ((boxes[:, 2] - boxes[:, 0]) >= 1.0 - _EPS) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0 - _EPS)
    return inside & big


def _clipped_area(boxes: np.ndarray, limit: np.ndarray) -> np.ndarray:
    c = np.clip(boxes, 0.0, limit)
    return np.clip(c[:, 2] - c[:, 0], 0.0, None) * np.clip(c[:, 3] - c[:, 1], 0.0, None)


def affine_sample(sample: Sample, matrix: np.ndarray) -> Sample | None:
    """Warp image (bilinear, pad 114) and boxes (corner hull). None if a valid non-ignored box becomes invalid; already-invalid boxes are clipped and kept."""
    matrix = np.asarray(matrix, dtype=np.float64)
    height, width = sample.rgb.shape[:2]
    boxes = np.asarray(sample.boxes, dtype=np.float64).reshape(-1, 4)
    hull = _hull(boxes, matrix) if len(boxes) else boxes.copy()
    ignore = np.asarray(sample.ignore, dtype=bool)
    limit = np.array([width, height, width, height], dtype=np.float64)

    in_valid = _valid(boxes, limit)
    out_valid = _valid(hull, limit)
    if (in_valid & ~out_valid & ~ignore).any():
        return None  # a valid box was pushed out of the canvas or collapsed by this transform
    in_area = _clipped_area(boxes, limit)
    hull = np.clip(hull, 0.0, limit)
    out_area = _clipped_area(hull, limit)
    if (~in_valid & ~ignore & (in_area > _EPS) & (out_area <= _EPS)).any():
        return None  # an invalid box that still overlapped the canvas lost all its visible area
    small = ((hull[:, 2] - hull[:, 0]) < 1.0 - _EPS) | ((hull[:, 3] - hull[:, 1]) < 1.0 - _EPS)
    keep = ~(small & ignore)  # ignored degenerate boxes are dropped; non-ignored are always kept
    inverse = np.linalg.inv(matrix)
    with Image.fromarray(sample.rgb) as image:
        with image.transform((width, height), Image.Transform.AFFINE,
                             tuple(inverse[:2].flatten()),
                             resample=Image.Resampling.BILINEAR, fillcolor=FILL) as warped:
            rgb = np.asarray(warped, dtype=np.uint8).copy()
    return Sample(rgb, hull[keep], np.asarray(sample.labels)[keep].copy(), ignore[keep].copy())
