import numpy as np
import pytest

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


def test_jpeg_quality_draw_in_range():
    s = toy_sample(fill=100)
    # Constant images compress losslessly-ish; check the output stays close to the input.
    for i in range(20):
        out = jpeg(s, np.random.default_rng(i))
        assert np.abs(out.rgb.astype(int) - 100).max() <= 3


def test_pixel_edits_registry():
    assert PIXEL_EDITS == {
        "brightness": brightness, "contrast": contrast, "saturation": saturation,
        "hue": hue, "blur": blur, "noise": noise, "jpeg": jpeg, "sharpen": sharpen,
    }
