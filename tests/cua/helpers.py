import numpy as np


def toy_sample(boxes=None, fill: int | None = None):
    """Deterministic 640x640 Sample; labels 0..N-1, no ignored boxes. `fill` makes the image constant."""
    from cua.image import Sample

    if boxes is None:
        boxes = [[100.0, 100.0, 220.0, 260.0], [300.0, 350.0, 480.0, 500.0]]
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if fill is not None:
        rgb = np.full((640, 640, 3), fill, dtype=np.uint8)
    else:
        yy, xx = np.mgrid[0:640, 0:640]
        rgb = np.stack([xx % 256, yy % 256, (xx * 3 + yy * 5) % 256], axis=-1).astype(np.uint8)
    n = len(boxes)
    return Sample(rgb, boxes, np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool))


def covered_fraction(out, s, box: int) -> float:
    """Fraction of box `box`'s integer pixel region where out.rgb differs from s.rgb in any channel."""
    x0, y0, x1, y1 = (int(round(v)) for v in s.boxes[box])
    a = out.rgb[y0:y1, x0:x1]
    b = s.rgb[y0:y1, x0:x1]
    changed = (a != b).any(axis=-1)
    return float(changed.mean()) if changed.size else 0.0
