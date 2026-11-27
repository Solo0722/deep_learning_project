import numpy as np
import pytest
from PIL import Image

from cua.augment.pixel import (
    PIXEL_EDITS, blur, brightness, contrast, hue, jpeg, noise, saturation, sharpen,
)
from helpers import toy_sample

ALL_EDITS = [brightness, contrast, saturation, hue, blur, noise, jpeg, sharpen]


@pytest.mark.parametrize("edit", ALL_EDITS)
def test_pixel_edits_keep_boxes_and_dtype(edit):
    s = toy_sample(boxes=[[10., 10., 50., 50.]])
    out = edit(s, np.random.default_rng(0))
    assert out.rgb.dtype == np.uint8 and out.rgb.shape == s.rgb.shape
    assert np.array_equal(out.boxes, s.boxes) and not np.shares_memory(out.boxes, s.boxes)
    assert np.array_equal(out.labels, s.labels) and not np.shares_memory(out.labels, s.labels)
    assert np.array_equal(out.ignore, s.ignore) and not np.shares_memory(out.ignore, s.ignore)


def test_hue_leaves_grey_unchanged():
    grey = toy_sample(fill=128)
    assert np.array_equal(hue(grey, np.random.default_rng(0)).rgb, grey.rgb)


def test_brightness_range():
    s = toy_sample(fill=100)
    vals = {int(brightness(s, np.random.default_rng(i)).rgb[0, 0, 0]) for i in range(200)}
    assert min(vals) >= 70 and max(vals) <= 130


@pytest.mark.parametrize("edit", ALL_EDITS)
def test_same_seed_gives_identical_output(edit):
    s = toy_sample()
    a = edit(s, np.random.default_rng(42))
    b = edit(s, np.random.default_rng(42))
    assert np.array_equal(a.rgb, b.rgb)
    assert np.array_equal(a.boxes, b.boxes)


def test_noise_changes_pixels():
    s = toy_sample(fill=128)
    out = noise(s, np.random.default_rng(0))
    assert not np.array_equal(out.rgb, s.rgb)
    # sigma is 0.03 * 255 ~= 7.65 levels, so most values stay near 128
    assert abs(float(out.rgb.astype(np.float64).mean()) - 128.0) < 1.0


@pytest.mark.parametrize("edit", [jpeg, blur])
def test_jpeg_and_blur_change_non_constant_image(edit):
    s = toy_sample()
    out = edit(s, np.random.default_rng(3))
    assert not np.array_equal(out.rgb, s.rgb)


@pytest.mark.parametrize("edit", [contrast, saturation, sharpen, blur])
def test_constant_image_is_unchanged_by_contrast_saturation_sharpen_blur(edit):
    s = toy_sample(fill=100)
    out = edit(s, np.random.default_rng(5))
    assert np.array_equal(out.rgb, s.rgb)


class SpyRng:
    """Records each draw and returns the low or high endpoint of the requested range."""

    def __init__(self, end):
        self.end = end
        self.calls = []

    def uniform(self, lo, hi):
        self.calls.append(("uniform", lo, hi))
        return lo if self.end == "low" else hi

    def integers(self, lo, hi):
        self.calls.append(("integers", lo, hi))
        return lo if self.end == "low" else hi - 1

    def normal(self, loc, scale, size=None):
        self.calls.append(("normal", loc, scale))
        return np.zeros(size)


SPY_CASES = [
    (brightness, ("uniform", 0.7, 1.3)),
    (contrast, ("uniform", 0.7, 1.3)),
    (saturation, ("uniform", 0.5, 1.5)),
    (hue, ("uniform", -0.05, 0.05)),
    (blur, ("uniform", 0.5, 1.5)),
    (sharpen, ("uniform", 1.5, 2.5)),
    (jpeg, ("integers", 30, 61)),
    (noise, ("normal", 0.0, 0.03 * 255)),
]


@pytest.mark.parametrize("end", ["low", "high"])
@pytest.mark.parametrize("edit,expected", SPY_CASES, ids=[c[0].__name__ for c in SPY_CASES])
def test_strength_draws_match_global_ranges(edit, expected, end):
    rng = SpyRng(end)
    out = edit(toy_sample(), rng)  # both endpoints must run without error
    assert out.rgb.shape == (640, 640, 3)
    assert len(rng.calls) == 1
    kind, *args = rng.calls[0]
    assert kind == expected[0]
    assert args == pytest.approx(list(expected[1:]))


def test_brightness_endpoints_apply_drawn_factor():
    s = toy_sample(fill=100)
    low = brightness(s, SpyRng("low")).rgb[0, 0, 0]
    high = brightness(s, SpyRng("high")).rgb[0, 0, 0]
    assert int(low) == 70 and int(high) == 130


def test_jpeg_quality_stays_in_range_and_covers_endpoints(monkeypatch):
    captured = []
    real_save = Image.Image.save

    def spy_save(self, fp, format=None, **params):
        if format == "JPEG":
            captured.append(params["quality"])
        return real_save(self, fp, format=format, **params)

    monkeypatch.setattr(Image.Image, "save", spy_save)
    s = toy_sample(fill=100)
    for seed in range(500):
        jpeg(s, np.random.default_rng(seed))
    assert len(captured) == 500
    assert min(captured) >= 30 and max(captured) <= 60
    assert 30 in captured and 60 in captured


def test_noise_std_matches_sigma():
    out = noise(toy_sample(fill=128), np.random.default_rng(0))
    deviation = out.rgb.astype(np.float64) - 128.0
    assert abs(float(deviation.std()) - 0.03 * 255) < 0.15


def test_hue_endpoint_wraps_pure_red_to_expected_hsv():
    red = toy_sample(fill=0)
    red.rgb[...] = np.array([255, 0, 0], dtype=np.uint8)
    out = hue(red, SpyRng("low"))  # shift = round(-0.05 * 255) = -13; H 0 -> 243
    expected = np.asarray(Image.new("HSV", (1, 1), (243, 255, 255)).convert("RGB"), dtype=np.uint8)[0, 0]
    assert np.all(out.rgb == expected)


def test_hue_mixed_image_keeps_grey_and_changes_colour():
    s = toy_sample(fill=128)
    s.rgb[:, 320:] = np.array([200, 50, 80], dtype=np.uint8)
    grey_half = s.rgb[:, :320].copy()
    colour_half = s.rgb[:, 320:].copy()
    for seed in range(10):
        out = hue(s, np.random.default_rng(seed))
        assert np.array_equal(out.rgb[:, :320], grey_half)
        assert (out.rgb[:, 320:] != colour_half).any()


def test_pixel_edits_registry():
    assert PIXEL_EDITS == {
        "brightness": brightness, "contrast": contrast, "saturation": saturation,
        "hue": hue, "blur": blur, "noise": noise, "jpeg": jpeg, "sharpen": sharpen,
    }
