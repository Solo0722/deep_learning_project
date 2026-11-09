"""Augmentation menu registry, per-dataset validity and the arm transform."""
from __future__ import annotations

from typing import Callable

from cua.augment.base import base_geometry
from cua.augment.geometric import GEOMETRIC_EDITS
from cua.augment.mosaic import mosaic
from cua.augment.pixel import PIXEL_EDITS
from cua.augment.rng import keyed_rng
from cua.image import Sample

MENU_ORDER = ("translate", "rotate", "vflip", "brightness", "contrast", "saturation", "hue",
              "blur", "noise", "jpeg", "sharpen", "erase", "mosaic")
INVALID = {"rdd": {"vflip"}, "voc": {"vflip"}, "dacl10k": set()}
EDITS = {**GEOMETRIC_EDITS, **PIXEL_EDITS}


def valid_menu(dataset: str) -> tuple[str, ...]:
    if dataset not in INVALID:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return tuple(e for e in MENU_ORDER if e not in INVALID[dataset])


class ArmTransform:
    """Base geometry for every arm, plus one optional edit applied with probability p."""

    def __init__(self, dataset: str, edit: str | None, seed: int, p: float = 0.5):
        menu = valid_menu(dataset)
        if edit is not None and edit not in menu:
            raise ValueError(f"Edit {edit!r} is not valid for dataset {dataset!r}")
        self.dataset, self.edit, self.seed, self.p = dataset, edit, seed, p

    def base(self, sample: Sample, epoch: int, index: int) -> Sample:
        """Base geometry for item `index` (the same call __call__ and mosaic fetches use)."""
        return base_geometry(sample, keyed_rng(self.seed, epoch, index, "base"))

    def __call__(self, sample: Sample, epoch: int, index: int,
                 fetch: Callable[[int], Sample], n_items: int) -> Sample:
        base = self.base(sample, epoch, index)
        er = keyed_rng(self.seed, epoch, index, "edit")
        applied = er.random() < self.p  # always drawn so the coin is identical across arms
        if not applied or self.edit is None:
            return base
        if self.edit == "mosaic":
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
        return EDITS[self.edit](base, er)
