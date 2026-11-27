"""Colour and image-quality edits. Each edit draws its strength from the passed rng only.

Boxes, labels and ignore are copied unchanged; only the image pixels change.
"""
from __future__ import annotations

import io
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from cua.image import Sample

Edit = Callable[[Sample, np.random.Generator], Sample]

NOISE_SIGMA = 0.03 * 255
# Strength levels (reports/cua_strength/PROTOCOL.md). Level 1 is the menu strength; levels 2 and 3 widen the
# range (factors around the same centre, sigmas upward) and consume the same single draw, so a given image gets
# the same quantile at every level. The widest ranges keep objects visible: a brightness or contrast factor of
# 0.1 blanks the image, so the lowest factor is 0.3. Explicit literals keep level 1 bit-identical.
STRENGTH_LEVELS = (1, 2, 3)
SCALABLE_EDITS = ("brightness", "contrast", "saturation", "blur", "noise")
_RANGES = {
    "brightness": {1: (0.7, 1.3), 2: (0.5, 1.5), 3: (0.3, 1.7)},
    "contrast": {1: (0.7, 1.3), 2: (0.5, 1.5), 3: (0.3, 1.7)},
    "saturation": {1: (0.5, 1.5), 2: (0.0, 2.0), 3: (-0.5, 2.5)},  # factor clipped at 0 (greyscale)
    "blur": {1: (0.5, 1.5), 2: (1.0, 3.0), 3: (1.5, 4.5)},
}
_NOISE_SCALE = {1: 1.0, 2: 2.0, 3: 10.0 / 3.0}  # sigma 0.03, 0.06, 0.10 of the intensity range


def _range(edit: str, level: int) -> tuple[float, float]:
    if level not in STRENGTH_LEVELS:
        raise ValueError(f"strength level must be one of {STRENGTH_LEVELS}, got {level!r}")
    return _RANGES[edit][level]


def _with_rgb(sample: Sample, rgb: np.ndarray) -> Sample:
    return Sample(
        np.ascontiguousarray(rgb, dtype=np.uint8),
        np.copy(sample.boxes),
        np.copy(sample.labels),
        np.copy(sample.ignore),
    )


def _pil(sample: Sample, transform: Callable[[Image.Image], Image.Image]) -> Sample:
    with Image.fromarray(sample.rgb) as image:
        with transform(image) as out:
            return _with_rgb(sample, np.asarray(out.convert("RGB"), dtype=np.uint8).copy())


def brightness(sample: Sample, rng: np.random.Generator, level: int = 1) -> Sample:
    factor = float(rng.uniform(*_range("brightness", level)))
    return _pil(sample, lambda im: ImageEnhance.Brightness(im).enhance(factor))


def contrast(sample: Sample, rng: np.random.Generator, level: int = 1) -> Sample:
    factor = float(rng.uniform(*_range("contrast", level)))
    return _pil(sample, lambda im: ImageEnhance.Contrast(im).enhance(factor))


def saturation(sample: Sample, rng: np.random.Generator, level: int = 1) -> Sample:
    factor = max(0.0, float(rng.uniform(*_range("saturation", level))))
    return _pil(sample, lambda im: ImageEnhance.Color(im).enhance(factor))


def hue(sample: Sample, rng: np.random.Generator) -> Sample:
    """Shift hue by up to +/-0.05 of the full circle; pixels with zero saturation are restored exactly."""
    shift = round(float(rng.uniform(-0.05, 0.05)) * 255)
    with Image.fromarray(sample.rgb) as image:
        with image.convert("HSV") as hsv:
            arr = np.asarray(hsv, dtype=np.uint8).copy()
        grey = arr[..., 1] == 0
        arr[..., 0] = ((arr[..., 0].astype(np.int64) + shift) % 256).astype(np.uint8)
        height, width = arr.shape[:2]
        with Image.frombytes("HSV", (width, height), arr.tobytes()) as shifted:
            out = np.asarray(shifted.convert("RGB"), dtype=np.uint8).copy()
    out[grey] = sample.rgb[grey]
    return _with_rgb(sample, out)


def blur(sample: Sample, rng: np.random.Generator, level: int = 1) -> Sample:
    sigma = float(rng.uniform(*_range("blur", level)))
    return _pil(sample, lambda im: im.filter(ImageFilter.GaussianBlur(radius=sigma)))


def noise(sample: Sample, rng: np.random.Generator, level: int = 1) -> Sample:
    if level not in STRENGTH_LEVELS:
        raise ValueError(f"strength level must be one of {STRENGTH_LEVELS}, got {level!r}")
    noisy = sample.rgb.astype(np.float64) + rng.normal(0.0, NOISE_SIGMA * _NOISE_SCALE[level], size=sample.rgb.shape)
    return _with_rgb(sample, np.clip(np.rint(noisy), 0, 255).astype(np.uint8))


def jpeg(sample: Sample, rng: np.random.Generator) -> Sample:
    quality = int(rng.integers(30, 61))  # inclusive [30, 60]
    buffer = io.BytesIO()
    with Image.fromarray(sample.rgb) as image:
        image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return _with_rgb(sample, np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy())


def sharpen(sample: Sample, rng: np.random.Generator) -> Sample:
    factor = float(rng.uniform(1.5, 2.5))
    return _pil(sample, lambda im: ImageEnhance.Sharpness(im).enhance(factor))


PIXEL_EDITS = {
    "brightness": brightness,
    "contrast": contrast,
    "saturation": saturation,
    "hue": hue,
    "blur": blur,
    "noise": noise,
    "jpeg": jpeg,
    "sharpen": sharpen,
}
