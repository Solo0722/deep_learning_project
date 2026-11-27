"""Four-image mosaic with a visibility filter on boxes."""
from __future__ import annotations

import numpy as np
from PIL import Image

from cua.image import Sample

CANVAS = 640
TILE = 320
MIN_VISIBLE = 0.30
FILL = 114
CENTRE_RANGE = (160.0, 480.0)


def clip_and_filter(boxes: np.ndarray, areas: np.ndarray, canvas: int = CANVAS, min_visible: float = MIN_VISIBLE) -> np.ndarray:
    """Keep mask: areas > 0 and visible (canvas-clipped) area / areas >= min_visible. Inputs are not modified."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)
    c = np.clip(boxes, 0.0, float(canvas))
    visible = np.clip(c[:, 2] - c[:, 0], 0.0, None) * np.clip(c[:, 3] - c[:, 1], 0.0, None)
    safe = np.where(areas > 0, areas, 1.0)
    return (areas > 0) & (visible / safe >= min_visible - 1e-9)


def mosaic(sample: Sample, others: list[Sample], rng) -> Sample:
    """Place sample + 3 others as 320px tiles meeting at a random centre (cx, cy) ~ U(160, 480)."""
    if len(others) != 3:
        raise ValueError("mosaic needs exactly 3 other samples")
    cx = int(round(float(rng.uniform(*CENTRE_RANGE))))
    cy = int(round(float(rng.uniform(*CENTRE_RANGE))))
    origins = [(cx - TILE, cy - TILE), (cx, cy - TILE), (cx - TILE, cy), (cx, cy)]
    canvas = np.full((CANVAS, CANVAS, 3), FILL, dtype=np.uint8)
    all_boxes, all_labels, all_ignore, all_areas = [], [], [], []
    for tile, (ox, oy) in zip([sample, *others], origins):
        with Image.fromarray(tile.rgb) as im:
            with im.resize((TILE, TILE), Image.Resampling.BILINEAR) as small:
                arr = np.asarray(small, dtype=np.uint8)
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + TILE, CANVAS), min(oy + TILE, CANVAS)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = arr[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        b = np.asarray(tile.boxes, dtype=np.float64).reshape(-1, 4) * 0.5
        all_areas.append(np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None))
        all_boxes.append(b + np.array([ox, oy, ox, oy], dtype=np.float64))
        all_labels.append(np.asarray(tile.labels))
        all_ignore.append(np.asarray(tile.ignore, dtype=bool))
    boxes = np.concatenate(all_boxes)
    areas = np.concatenate(all_areas)
    keep = clip_and_filter(boxes, areas)
    boxes = np.clip(boxes, 0.0, float(CANVAS))[keep]
    return Sample(canvas, boxes, np.concatenate(all_labels)[keep].copy(), np.concatenate(all_ignore)[keep].copy())
