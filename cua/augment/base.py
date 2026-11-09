"""Base geometry augmentation: horizontal flip then scale jitter."""
from __future__ import annotations

from cua.augment.boxes import affine_sample, compose, hflip_matrix, scale_matrix
from cua.image import Sample

SCALE_RANGE = (0.8, 1.2)
MAX_SCALE_TRIES = 10


def base_geometry(sample: Sample, rng) -> Sample:
    """hflip (p=0.5) then scale ~ U(0.8, 1.2) about the centre; never returns None."""
    size = sample.rgb.shape[0]
    flip = rng.random() < 0.5
    flip_m = hflip_matrix(size) if flip else None
    for _ in range(MAX_SCALE_TRIES):
        s = float(rng.uniform(*SCALE_RANGE))
        parts = [scale_matrix(s, size)] if flip_m is None else [flip_m, scale_matrix(s, size)]
        out = affine_sample(sample, compose(*parts))
        if out is not None:
            return out
    out = affine_sample(sample, flip_m if flip else scale_matrix(1.0, size))
    assert out is not None  # hflip / identity never reject valid boxes
    return out
