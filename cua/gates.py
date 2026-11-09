"""Data gates and the reference split."""
from __future__ import annotations

import numpy as np

from cua.records import Record


class GateFailure(RuntimeError):
    """Raised when a dataset fails a pre-registered data gate."""


def cooccurrence_rate(records: list[Record]) -> float:
    """Share of records with >=1 non-ignored box that have >=2 distinct classes."""
    with_target = [r for r in records if len(r.class_set()) >= 1]
    if not with_target:
        return 0.0
    return sum(len(r.class_set()) >= 2 for r in with_target) / len(with_target)


def class_box_stats(records: list[Record], n_classes: int) -> list[dict]:
    """Per-class box count and share of boxes covering > 50% of the image area."""
    count = [0] * n_classes
    large = [0] * n_classes
    for r in records:
        keep = ~r.ignore
        area_img = float(r.width) * float(r.height)
        for box, c in zip(r.boxes[keep], r.labels[keep]):
            c = int(c)
            count[c] += 1
            if (box[2] - box[0]) * (box[3] - box[1]) > 0.5 * area_img:
                large[c] += 1
    return [
        {"count": count[c], "large_fraction": (large[c] / count[c]) if count[c] else 0.0}
        for c in range(n_classes)
    ]


def _class_counts(records: list[Record], n_classes: int) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    for r in records:
        keep = ~r.ignore
        counts += np.bincount(r.labels[keep], minlength=n_classes)[:n_classes]
    return counts


def reference_split(
    train: list[Record],
    n_classes: int,
    n_target: int = 1000,
    min_per_class: int = 100,
    seed: int = 0,
) -> tuple[list[Record], list[Record]]:
    """Group-disjoint split of `train` into (train_rest, reference); original order kept."""
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(train):
        by_group.setdefault(r.group, []).append(i)
    groups = list(by_group)
    order = np.random.default_rng(seed).permutation(len(groups))
    shuffled = [groups[i] for i in order]

    chosen: set[str] = set()
    counts = np.zeros(n_classes, dtype=np.int64)
    n_images = 0

    def add(g: str) -> None:
        nonlocal n_images
        chosen.add(g)
        n_images += len(by_group[g])
        counts[:] += _class_counts([train[i] for i in by_group[g]], n_classes)

    group_counts = {g: _class_counts([train[i] for i in by_group[g]], n_classes) for g in groups}
    # Pass 1: groups that contain a class still below quota.
    for g in shuffled:
        if (counts >= min_per_class).all():
            break
        short = counts < min_per_class
        if (group_counts[g][short] > 0).any():
            add(g)
    # Pass 2: fill up to n_target images.
    for g in shuffled:
        if n_images >= n_target:
            break
        if g not in chosen:
            add(g)

    ref = [r for r in train if r.group in chosen]
    rest = [r for r in train if r.group not in chosen]
    return rest, ref
