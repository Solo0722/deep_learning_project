"""dacl10k adapter (ASSUMED labelme-style JSON; see DACL10K_FORMAT.md): polygons become boxes."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from cua.dedup import dhash, duplicate_groups
from cua.gates import class_box_stats, retained_classes
from cua.image import load_rgb
from cua.records import Record

DACL_DAMAGE = (
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket", "Hollowareas",
    "Cavity", "Spalling", "Graffiti", "Weathering", "Restformwork", "ExposedRebars",
)

SPLIT_FRACTIONS = (0.70, 0.15)  # train, val; the remainder is test


def parse_dacl_json(path: Path, classes: tuple[str, ...], *, counts: dict | None = None) -> Record:
    """Parse one annotation JSON. One box per damage polygon, clipped to the image.

    Non-damage labels are dropped. Degenerate polygons (<3 distinct points, collinear points, or
    an empty box after clipping) are dropped. If `counts` is given it accumulates "clipped_boxes"
    and "dropped_degenerate".
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        width = int(data["imageWidth"])
        height = int(data["imageHeight"])
        image = str(data["imagePath"])
        shapes = data["shapes"]
        if not isinstance(shapes, list) or width <= 0 or height <= 0:
            raise ValueError("bad shapes or image size")
        index = {name: i for i, name in enumerate(classes)}
        boxes, labels = [], []
        clipped = dropped = 0
        for shape in shapes:
            label = shape["label"]
            if label not in index:
                continue
            pts = np.asarray(shape["points"], dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
                raise ValueError(f"bad points for label {label!r}")
            if _degenerate(pts):
                dropped += 1
                continue
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            cx1, cx2 = (min(max(v, 0.0), float(width)) for v in (x1, x2))
            cy1, cy2 = (min(max(v, 0.0), float(height)) for v in (y1, y2))
            if cx2 <= cx1 or cy2 <= cy1:
                dropped += 1
                continue
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                clipped += 1
            boxes.append([cx1, cy1, cx2, cy2])
            labels.append(index[label])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path}: bad dacl10k annotation: {e!r}") from e

    if counts is not None:
        counts["clipped_boxes"] = counts.get("clipped_boxes", 0) + clipped
        counts["dropped_degenerate"] = counts.get("dropped_degenerate", 0) + dropped
    return Record(
        image=image,
        width=width,
        height=height,
        boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        labels=np.asarray(labels, dtype=np.int64),
        ignore=np.zeros(len(labels), dtype=bool),
        group=path.stem,
        source="dacl10k",
    )


def _degenerate(pts: np.ndarray) -> bool:
    """True for fewer than 3 distinct points or zero polygon extent (all collinear)."""
    if len(np.unique(pts, axis=0)) < 3:
        return True
    d = pts - pts[0]
    # Collinear iff every point lies on the line through pts[0] and the farthest point.
    far = d[np.argmax(np.abs(d).sum(axis=1))]
    cross = d[:, 0] * far[1] - d[:, 1] * far[0]
    scale = max(float(np.abs(d).max()), 1.0)
    return bool(np.all(np.abs(cross) <= 1e-9 * scale * scale))


def _split_of(group: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < SPLIT_FRACTIONS[0]:
        return "train"
    return "val" if u < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1] else "test"


def load_dacl10k(
    root: Path, seed: int = 0, *, stats: dict | None = None
) -> tuple[dict[str, list[Record]], tuple[str, ...]]:
    """Pool every labelled image under root, group near-duplicates, split groups 70/15/15.

    The class-exclusion gate is computed on the TRAIN split only; excluded classes' boxes are
    dropped from all splits and labels re-indexed to the retained tuple (images left with no
    boxes stay as zero-box records). If `stats` is given it receives "class_stats" (pre-exclusion
    train stats over DACL_DAMAGE), "excluded" and the parse counters.
    """
    root = Path(root)
    counts: dict = {"clipped_boxes": 0, "dropped_degenerate": 0}
    records: list[Record] = []
    for ann in sorted(root.glob("annotations/*/*.json")):
        rec = parse_dacl_json(ann, DACL_DAMAGE, counts=counts)
        image = f"images/{ann.parent.name}/{Path(rec.image).name}"
        records.append(dataclasses.replace(rec, image=image))
    hashes = [dhash(load_rgb(root, r)) for r in records]
    gids = duplicate_groups(hashes)
    records = [dataclasses.replace(r, group=f"dup-{g}") for r, g in zip(records, gids)]

    splits: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    for r in records:
        splits[_split_of(r.group, seed)].append(r)

    train_stats = class_box_stats(splits["train"], len(DACL_DAMAGE))
    retained = retained_classes(DACL_DAMAGE, train_stats)
    remap = {DACL_DAMAGE.index(c): i for i, c in enumerate(retained)}
    for name, recs in splits.items():
        out = []
        for r in recs:
            keep = np.array([int(c) in remap for c in r.labels], dtype=bool)
            out.append(dataclasses.replace(
                r,
                boxes=r.boxes[keep],
                labels=np.array([remap[int(c)] for c in r.labels[keep]], dtype=np.int64),
                ignore=r.ignore[keep],
            ))
        splits[name] = out

    if stats is not None:
        stats.update(counts)
        stats["class_stats"] = train_stats
        stats["excluded"] = {
            c: train_stats[i] for i, c in enumerate(DACL_DAMAGE) if c not in retained
        }
    return splits, retained
