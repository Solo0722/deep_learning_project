import numpy as np

def toy_sample(boxes=None):
    """Deterministic non-uniform 640x640 Sample; labels 0..N-1, no ignored boxes."""
    from cua.image import Sample

    if boxes is None:
        boxes = [[100.0, 100.0, 220.0, 260.0], [300.0, 350.0, 480.0, 500.0]]
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    yy, xx = np.mgrid[0:640, 0:640]
    rgb = np.stack([xx % 256, yy % 256, (xx * 3 + yy * 5) % 256], axis=-1).astype(np.uint8)
    n = len(boxes)
    return Sample(rgb, boxes, np.arange(n, dtype=np.int64), np.zeros(n, dtype=bool))
