"""Augmentation menu registry, per-dataset validity and the arm transform."""
from __future__ import annotations

from typing import Callable

import numpy as np

from cua.augment.base import base_geometry, base_geometry_matrix
from cua.augment.geometric import GEOMETRIC_EDITS
from cua.augment.local import SCOPES, composite, object_mask
from cua.augment.mosaic import mosaic
from cua.augment.pixel import PIXEL_EDITS, SCALABLE_EDITS, STRENGTH_LEVELS
from cua.augment.rng import keyed_rng
from cua.image import Sample
from cua.records import Record

MENU_ORDER = ("translate", "rotate", "vflip", "brightness", "contrast", "saturation", "hue",
              "blur", "noise", "jpeg", "sharpen", "erase", "mosaic")
INVALID = {"rdd": {"vflip"}, "voc": {"vflip"}, "dacl10k": set()}
EDITS = {**GEOMETRIC_EDITS, **PIXEL_EDITS}
# Positive control only: a planted edit with a known, class-specific harm. It deletes the annotations of
# one target class (pixels unchanged), so that class's objects are trained as background. On an image
# without the target class it is the identity. It is never part of the real menu.
PLANT = "plant"


def drop_class(sample: Sample, target: int) -> Sample:
    """Remove every box of class `target` (and its ignore flag); the image is unchanged."""
    keep = np.asarray(sample.labels).reshape(-1) != int(target)
    return Sample(np.copy(sample.rgb), np.asarray(sample.boxes).reshape(-1, 4)[keep].copy(),
                  np.asarray(sample.labels).reshape(-1)[keep].copy(), np.asarray(sample.ignore).reshape(-1)[keep].copy())


def valid_menu(dataset: str) -> tuple[str, ...]:
    if dataset not in INVALID:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return tuple(e for e in MENU_ORDER if e not in INVALID[dataset])


class ArmTransform:
    """Base geometry for every arm, plus one optional edit applied with probability p.

    scope="object" (cua.augment.local) applies a pixel edit only inside the annotated objects' polygons:
    `polygons` (per record, per box, rings in original pixels) and `records` (same order as the dataset)
    are then required, and only pixel edits are allowed.
    strength (cua.augment.pixel.STRENGTH_LEVELS) widens the range of a scalable pixel edit; 1 is the menu.
    """

    def __init__(self, dataset: str, edit: str | None, seed: int, p: float = 0.5, plant_class: int | None = None,
                 scope: str = "image", polygons: list | None = None, records: list[Record] | None = None,
                 strength: int = 1):
        self.menu = valid_menu(dataset) + ((PLANT,) if plant_class is not None else ())
        if edit is not None and edit not in self.menu:
            raise ValueError(f"Edit {edit!r} is not valid for dataset {dataset!r}")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        if scope == "object":
            if polygons is None or records is None or len(polygons) != len(records):
                raise ValueError("scope='object' needs polygons and records of the same length")
            if edit is not None and edit not in PIXEL_EDITS:
                raise ValueError(f"scope='object' supports pixel edits only, got {edit!r}")
        if strength not in STRENGTH_LEVELS:
            raise ValueError(f"strength must be one of {STRENGTH_LEVELS}, got {strength!r}")
        if strength != 1 and (edit is None or edit not in SCALABLE_EDITS):
            raise ValueError(f"strength {strength} needs one of {SCALABLE_EDITS}, got {edit!r}")
        self.strength = strength
        self.dataset, self.edit, self.seed, self.p = dataset, edit, seed, p
        self.plant_class = plant_class
        self.scope, self.polygons, self.records = scope, polygons, records

    def base(self, sample: Sample, epoch: int, index: int) -> Sample:
        """Base geometry for item `index` (the same call __call__ and mosaic fetches use)."""
        return base_geometry(sample, keyed_rng(self.seed, epoch, index, "base"))

    def __call__(self, sample: Sample, epoch: int, index: int,
                 fetch: Callable[[int], Sample], n_items: int) -> Sample:
        base, matrix = base_geometry_matrix(sample, keyed_rng(self.seed, epoch, index, "base"))
        er = keyed_rng(self.seed, epoch, index, "edit")
        applied = er.random() < self.p  # always drawn so the coin is identical across arms
        if not applied or self.edit is None:
            return base
        out = self._render(base, self.edit, er, epoch, index, fetch, n_items)
        return self._localize(out, base, matrix, index, self.edit)

    def apply(self, sample: Sample, epoch: int, index: int, fetch: Callable[[int], Sample], n_items: int,
              edit: str | None) -> Sample:
        """Base geometry plus `edit` (None = identity), with no coin: the policy already chose."""
        if edit is not None and edit not in self.menu:
            raise ValueError(f"Edit {edit!r} is not valid for dataset {self.dataset!r}")
        base, matrix = base_geometry_matrix(sample, keyed_rng(self.seed, epoch, index, "base"))
        if edit is None:
            return base
        out = self._render(base, edit, keyed_rng(self.seed, epoch, index, "edit"), epoch, index, fetch, n_items)
        return self._localize(out, base, matrix, index, edit)

    def _localize(self, out: Sample, base: Sample, matrix: np.ndarray, index: int, edit: str) -> Sample:
        """scope='object': keep the edit only inside the objects of item `index` (pixel edits only)."""
        if self.scope != "object" or edit not in PIXEL_EDITS:
            return out
        mask = object_mask(self.polygons[index], self.records[index], matrix, base.rgb.shape[0])
        return composite(out, base, mask)

    def _render(self, base: Sample, edit: str, er, epoch: int, index: int,
                fetch: Callable[[int], Sample], n_items: int) -> Sample:
        if edit == "mosaic":
            if n_items < 1:
                raise ValueError("mosaic needs n_items >= 1")
            mr = keyed_rng(self.seed, epoch, index, "mosaic")
            others = []
            for _ in range(3):
                j = int(mr.integers(0, n_items))
                while n_items > 1 and j == index:
                    j = int(mr.integers(0, n_items))
                others.append(fetch(j))
            return mosaic(base, others, er)
        if edit == PLANT:
            return drop_class(base, self.plant_class)
        if self.strength != 1 and edit in SCALABLE_EDITS:
            return PIXEL_EDITS[edit](base, er, level=self.strength)
        return EDITS[edit](base, er)
