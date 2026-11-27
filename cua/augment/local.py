"""Object-local pixel edits: an edit that changes only the pixels of annotated objects.

The usual whole-image pixel edit is rendered with the same rng (so the strength draw is identical to the
whole-image arm) and composited into the base image inside the objects' polygons. The background keeps
its base pixels, and boxes, labels and ignore flags are unchanged.

Polygons are given per record, per box, as rings in original-image pixels (cua.datasets.dacl10k_masks).
They are mapped to the canvas through the letterbox and then through the base-geometry matrix, so the
mask follows the image exactly.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from cua.image import Sample
from cua.records import Record

SCOPES = ("image", "object")
FEATHER_SIGMA = 1.0  # canvas pixels; softens the mask edge


def canvas_matrix(rec: Record, size: int) -> np.ndarray:
    """Original-image pixels -> letterboxed canvas (same rounding as cua.image.letterbox)."""
    scale = min(size / rec.width, size / rec.height)
    rw, rh = max(1, round(rec.width * scale)), max(1, round(rec.height * scale))
    left, top = (size - rw) // 2, (size - rh) // 2
    return np.array([[rw / rec.width, 0.0, left], [0.0, rh / rec.height, top], [0.0, 0.0, 1.0]])


def object_mask(polygons: list[list[np.ndarray]], rec: Record, matrix: np.ndarray, size: int,
                feather: float = FEATHER_SIGMA) -> np.ndarray:
    """(size, size) float mask in [0, 1]: 1 inside any object polygon after letterbox then `matrix`."""
    m = np.asarray(matrix, dtype=np.float64) @ canvas_matrix(rec, size)
    im = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(im)
    for rings in polygons:
        for ring in rings:
            pts = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
            if len(pts) < 3:
                continue
            q = pts @ m[:2, :2].T + m[:2, 2]
            draw.polygon([tuple(p) for p in q.tolist()], fill=255)
    if feather > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=feather))
    return np.asarray(im, dtype=np.float32) / 255.0


def composite(edited: Sample, base: Sample, mask: np.ndarray) -> Sample:
    """Edited pixels where mask is 1, base pixels where it is 0; annotations from `base`."""
    a = mask[..., None]
    rgb = np.rint(edited.rgb.astype(np.float32) * a + base.rgb.astype(np.float32) * (1.0 - a))
    return Sample(np.clip(rgb, 0, 255).astype(np.uint8), np.copy(base.boxes), np.copy(base.labels),
                  np.copy(base.ignore))
