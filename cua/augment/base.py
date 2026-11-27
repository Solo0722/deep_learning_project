"""Base geometry augmentation: horizontal flip then scale jitter."""
from __future__ import annotations

import numpy as np

from cua.augment.boxes import affine_sample, compose, hflip_matrix, scale_matrix
from cua.image import Sample

SCALE_RANGE = (0.8, 1.2)
MAX_SCALE_TRIES = 10


def base_geometry(sample: Sample, rng) -> Sample:
    """hflip (p=0.5) then scale ~ U(0.8, 1.2) about the centre; never returns None."""
    return base_geometry_matrix(sample, rng)[0]


def base_geometry_matrix(sample: Sample, rng) -> tuple[Sample, np.ndarray]:
    """base_geometry plus the 3x3 canvas matrix it applied (same rng draws, same output)."""
    size = sample.rgb.shape[0]
    flip = rng.random() < 0.5
    flip_m = hflip_matrix(size) if flip else None
    for _ in range(MAX_SCALE_TRIES):
        s = float(rng.uniform(*SCALE_RANGE))
        parts = [scale_matrix(s, size)] if flip_m is None else [flip_m, scale_matrix(s, size)]
        m = compose(*parts)
        out = affine_sample(sample, m)
        if out is not None:
            return out, m
    m = flip_m if flip else scale_matrix(1.0, size)
    out = affine_sample(sample, m)
    if out is None:
        m = np.eye(3)
        out = affine_sample(sample, m)
    if out is None:
        raise RuntimeError("base_geometry fallback (flip/identity) rejected a sample; affine_sample must never reject these")
    return out, m
