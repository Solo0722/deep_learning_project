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


def toy_fetch(j: int):
    """Deterministic toy sample that varies with j (different fill and box)."""
    off = float(j % 7) * 10.0
    return toy_sample(boxes=[[50.0 + off, 60.0 + off, 200.0 + off, 220.0 + off]], fill=(40 + 13 * j) % 256)


def _pred(i, boxes, scores, labels):
    return {
        "image_index": int(i),
        "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        "scores": np.asarray(scores, dtype=np.float64).reshape(-1),
        "labels": np.asarray(labels, dtype=np.int64).reshape(-1),
    }


def _jitter(rng, box, rel):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    d = rng.normal(0.0, rel, 4) * np.array([bw, bh, bw, bh])
    nx1, ny1, nx2, ny2 = x1 + d[0], y1 + d[1], x2 + d[2], y2 + d[3]
    return [min(nx1, nx2 - 1.0), min(ny1, ny2 - 1.0), max(nx2, nx1 + 1.0), max(ny2, ny1 + 1.0)]


def random_detection_problem(n_images, n_classes, seed, with_ignored=False):
    """Random GT records plus predictions (0-based labels, original coords).

    Includes images without GT, near-tied and tied scores, low-IoU detections, detections on ignored
    objects (when with_ignored), small/medium/large boxes, and 130 class-0 detections on image 0
    (exercises maxDet=100 truncation).
    """
    from cua.records import Record

    rng = np.random.default_rng(seed)
    records, preds = [], []
    for i in range(n_images):
        w, h = int(rng.integers(160, 641)), int(rng.integers(160, 641))
        n_gt = 0 if (i > 0 and rng.random() < 0.15) else int(rng.integers(1, 6))
        gb, gl, gi = [], [], []
        for g in range(n_gt):
            bw, bh = rng.uniform(4.0, w * 0.6), rng.uniform(4.0, h * 0.6)
            x1, y1 = rng.uniform(0.0, w - bw), rng.uniform(0.0, h - bh)
            gb.append([x1, y1, x1 + bw, y1 + bh])
            gl.append(0 if (i == 0 and g == 0) else int(rng.integers(0, n_classes)))
            gi.append(bool(with_ignored and g > 0 and rng.random() < 0.25))
        records.append(Record(f"img/{i}.jpg", w, h, np.asarray(gb, dtype=np.float64).reshape(-1, 4),
                              np.asarray(gl, dtype=np.int64), np.asarray(gi, dtype=bool), "g", "s"))
        db, ds, dl = [], [], []
        for g in range(n_gt):
            for _ in range(int(rng.integers(0, 4))):
                db.append(_jitter(rng, gb[g], float(rng.choice([0.03, 0.15, 0.4]))))
                dl.append(gl[g] if rng.random() < 0.85 else int(rng.integers(0, n_classes)))
                ds.append(rng.uniform(0.05, 1.0))
        for _ in range(int(rng.integers(0, 4))):
            bw, bh = rng.uniform(3.0, w * 0.5), rng.uniform(3.0, h * 0.5)
            x1, y1 = rng.uniform(-5.0, w - bw), rng.uniform(-5.0, h - bh)
            db.append([x1, y1, x1 + bw, y1 + bh])
            dl.append(int(rng.integers(0, n_classes)))
            ds.append(rng.uniform(0.0, 0.8))
        if i == 0:
            for _ in range(130):
                db.append(_jitter(rng, gb[0], float(rng.choice([0.02, 0.3]))))
                dl.append(0)
                ds.append(rng.uniform(0.0, 1.0))
        ds = np.round(np.asarray(ds, dtype=np.float64), 2)  # exact ties
        ds = ds + rng.choice([0.0, 1e-12, -1e-12], size=len(ds))  # near ties
        preds.append(_pred(i, db, ds, dl))
    return records, preds


def problem_where_class_2_has_no_gt():
    """Three-class problem: GT only for classes 0 and 1; class-2 detections exist (false positives)."""
    records, preds = random_detection_problem(30, 2, seed=3)
    rng = np.random.default_rng(99)
    out = []
    for p in preds:
        extra = int(rng.integers(0, 3))
        boxes = np.concatenate([p["boxes"], np.tile(rng.uniform(0, 100, (extra, 2)), 2) + [0, 0, 20, 30]])
        out.append(_pred(p["image_index"], boxes, np.concatenate([p["scores"], rng.uniform(0, 1, extra)]),
                         np.concatenate([p["labels"], np.full(extra, 2)])))
    return records, out


def synthetic_deltas(edits, classes, seeds, conflict_edit, effect, noise, seed,
                     dataset="rdd", detector="fasterrcnn", base_noise=0.0, missing_frac=0.0):
    """Delta rows (arm - base) for one dataset/detector pair.

    The conflict edit helps half the classes (+effect) and hurts the other half (-effect); every delta gets
    Gaussian `noise`. With base_noise > 0 one base-run error b is drawn per (seed, cls) and subtracted from every
    edit's delta in that cell, as in real arm - base data. `missing_frac` drops that fraction of rows at random.
    """
    from cua.augment.menu import MENU_ORDER

    rng = np.random.default_rng(seed)
    rng_b = np.random.default_rng([seed, 1])
    rng_m = np.random.default_rng([seed, 2])
    b = {(s, c): rng_b.normal(0.0, base_noise) if base_noise > 0 else 0.0
         for s in range(seeds) for c in range(classes)}
    rows = []
    for s in range(seeds):
        for ei, e in enumerate(MENU_ORDER[:edits]):
            for c in range(classes):
                mu = 0.0
                if conflict_edit is not None and ei == conflict_edit:
                    mu = effect if c < classes // 2 else -effect
                delta = float(mu + rng.normal(0.0, noise) - b[(s, c)])
                if missing_frac and rng_m.random() < missing_frac:
                    continue
                rows.append({"dataset": dataset, "detector": detector, "seed": s, "edit": e,
                             "cls": f"c{c}", "delta": delta})
    return rows
