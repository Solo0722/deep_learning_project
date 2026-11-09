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


def brightness(sample: Sample, rng: np.random.Generator) -> Sample:
    factor = float(rng.uniform(0.7, 1.3))
    return _pil(sample, lambda im: ImageEnhance.Brightness(im).enhance(factor))


def contrast(sample: Sample, rng: np.random.Generator) -> Sample:
    factor = float(rng.uniform(0.7, 1.3))
    return _pil(sample, lambda im: ImageEnhance.Contrast(im).enhance(factor))


def saturation(sample: Sample, rng: np.random.Generator) -> Sample:
    factor = float(rng.uniform(0.5, 1.5))
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


def blur(sample: Sample, rng: np.random.Generator) -> Sample:
    sigma = float(rng.uniform(0.5, 1.5))
    return _pil(sample, lambda im: im.filter(ImageFilter.GaussianBlur(radius=sigma)))


def noise(sample: Sample, rng: np.random.Generator) -> Sample:
    noisy = sample.rgb.astype(np.float64) + rng.normal(0.0, NOISE_SIGMA, size=sample.rgb.shape)
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
