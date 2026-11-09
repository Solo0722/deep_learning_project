import numpy as np
import pytest

from cua.augment import geometric
from cua.augment.boxes import rotate_matrix, translate_matrix, vflip_matrix
from cua.augment.geometric import (
    GEOMETRIC_EDITS, erase, rotate, sampled_erase_rect, sampled_rotation_angle, translate, vflip,
)
from cua.image import Sample
from helpers import covered_fraction, toy_sample

ALL = [translate, rotate, vflip, erase]


def test_rotate_boxes_stay_inside_or_identity():
    s = toy_sample(boxes=[[0., 0., 640., 640.]])
    out = rotate(s, np.random.default_rng(0))
    assert np.array_equal(out.rgb, s.rgb) and np.array_equal(out.boxes, s.boxes)   # any rotation would leave canvas
    near = toy_sample(boxes=[[5., 5., 635., 635.]])   # angles under ~0.9 deg still fit, so the result is either identity or inside
    for i in range(50):
        b = rotate(near, np.random.default_rng(i)).boxes
        assert (b >= 0).all() and (b <= 640).all()


def test_rotate_angle_within_10_degrees():
    angles = [sampled_rotation_angle(np.random.default_rng(i)) for i in range(500)]
    assert max(abs(a) for a in angles) <= 10.0


def test_erase_never_covers_more_than_half_a_box():
    s = toy_sample(boxes=[[300., 300., 340., 340.]])
    for i in range(200):
        out = erase(s, np.random.default_rng(i))
        assert covered_fraction(out, s, box=0) <= 0.5


def test_geometric_edits_on_empty_sample():
    for edit in (translate, rotate, vflip, erase):
        assert edit(toy_sample(boxes=[]), np.random.default_rng(0)).boxes.shape == (0, 4)


def test_registry():
    assert GEOMETRIC_EDITS == {"translate": translate, "rotate": rotate, "vflip": vflip, "erase": erase}


def test_translate_matrix_moves_box():
    m = translate_matrix(12.5, -7.0)
    assert np.allclose(m @ [10., 20., 1.], [22.5, 13.0, 1.])


def test_rotate_matrix_90_sanity():
    # positive angle = counter-clockwise on screen (y down): a point right of centre moves up
    m = rotate_matrix(90.0, 640)
    assert np.allclose(m @ [420., 320., 1.], [320., 220., 1.])
    assert np.allclose(m @ [320., 320., 1.], [320., 320., 1.])


def test_vflip_matrix_and_edit_formula():
    assert np.allclose(vflip_matrix(640) @ [30., 100., 1.], [30., 540., 1.])
    s = toy_sample(boxes=[[10., 20., 50., 90.]])
    out = vflip(s, np.random.default_rng(0))
    assert np.allclose(out.boxes, [[10., 550., 50., 620.]])
    assert np.array_equal(out.rgb, s.rgb[::-1])


def test_translate_applies_exact_shift():
    s = toy_sample(boxes=[[300., 300., 340., 340.]])
    rng = np.random.default_rng(3)
    ref = np.random.default_rng(3)
    dx, dy = ref.uniform(-64, 64), ref.uniform(-64, 64)
    out = translate(s, rng)
    assert np.allclose(out.boxes, [[300 + dx, 300 + dy, 340 + dx, 340 + dy]])


class SpyRng:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.calls = []

    def uniform(self, low=0.0, high=1.0, size=None):
        self.calls.append((low, high))
        return self.rng.uniform(low, high, size)

    def __getattr__(self, name):
        return getattr(self.rng, name)


def test_translate_draw_ranges():
    spy = SpyRng()
    translate(toy_sample(boxes=[[300., 300., 340., 340.]]), spy)
    assert spy.calls[:2] == [(-64.0, 64.0), (-64.0, 64.0)]


def test_translate_retries_then_identity():
    s = toy_sample(boxes=[[0., 0., 640., 640.]])   # any shift rejects
    spy = SpyRng()
    out = translate(s, spy)
    assert len(spy.calls) == 20
    assert np.array_equal(out.rgb, s.rgb) and np.array_equal(out.boxes, s.boxes)
    assert out.rgb is not s.rgb and out.boxes is not s.boxes


def test_rotate_first_draw_matches_sampled_angle_and_retries():
    spy = SpyRng(5)
    s = toy_sample(boxes=[[0., 0., 640., 640.]])
    rotate(s, spy)
    assert len(spy.calls) == 10 and all(c == (-10.0, 10.0) for c in spy.calls)
    assert sampled_rotation_angle(np.random.default_rng(5)) == np.random.default_rng(5).uniform(-10, 10)


def test_erase_area_and_aspect_ranges():
    s = toy_sample(boxes=[])
    for i in range(500):
        x0, y0, x1, y1 = sampled_erase_rect(s, np.random.default_rng(i))
        assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 640
        frac = (x1 - x0) * (y1 - y0) / 640 ** 2
        assert 0.02 - 0.003 <= frac <= 0.10 + 0.003
        assert 0.3 * 0.9 <= (y1 - y0) / (x1 - x0) <= 3.3 * 1.1


def test_sampled_rect_exact_fraction_at_most_half():
    s = toy_sample(boxes=[[300., 300., 340., 340.], [310.5, 200.2, 360.7, 240.9]])
    for i in range(500):
        r = sampled_erase_rect(s, np.random.default_rng(i))
        if r is None:
            continue
        x0, y0, x1, y1 = r
        for b in s.boxes:
            iw = max(0.0, min(x1, b[2]) - max(x0, b[0]))
            ih = max(0.0, min(y1, b[3]) - max(y0, b[1]))
            assert iw * ih / ((b[2] - b[0]) * (b[3] - b[1])) <= 0.5


def test_erase_fills_noise_and_keeps_annotations():
    s = toy_sample(boxes=[[10., 10., 50., 50.]], fill=0)
    out = erase(s, np.random.default_rng(1))
    changed = (out.rgb != s.rgb).any(axis=-1)
    assert changed.sum() > 0.015 * 640 * 640
    assert np.array_equal(out.boxes, s.boxes) and not np.shares_memory(out.boxes, s.boxes)
    assert np.array_equal(out.labels, s.labels) and np.array_equal(out.ignore, s.ignore)


def test_erase_all_ignored_always_applies():
    s = toy_sample(boxes=[[0., 0., 640., 640.]], fill=0)
    s = Sample(s.rgb, s.boxes, s.labels, np.ones(1, dtype=bool))
    for i in range(50):
        assert sampled_erase_rect(s, np.random.default_rng(i)) is not None
        assert (erase(s, np.random.default_rng(i)).rgb != 0).any()


def test_erase_identity_when_impossible():
    tiles = [[x, y, x + 20., y + 20.] for x in range(0, 640, 32) for y in range(0, 640, 32)]
    s = toy_sample(boxes=tiles)   # any rectangle fully covers some tile
    assert sampled_erase_rect(s, np.random.default_rng(0)) is None
    out = erase(s, np.random.default_rng(0))
    assert np.array_equal(out.rgb, s.rgb) and out.rgb is not s.rgb


@pytest.mark.parametrize("edit", ALL)
def test_determinism(edit):
    s = toy_sample(boxes=[[300., 300., 340., 340.]])
    a = edit(s, np.random.default_rng(7))
    b = edit(s, np.random.default_rng(7))
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.boxes, b.boxes)
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.ignore, b.ignore)


def _square_sample():
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    rgb[270:370, 270:370] = 255
    return Sample(rgb, np.array([[270., 270., 370., 370.]]), np.zeros(1, dtype=np.int64), np.zeros(1, dtype=bool))


def _check_agreement(src, out, matrix):
    assert not np.array_equal(out.boxes, src.boxes), "transform must be non-identity"
    ys, xs = np.nonzero(out.rgb.min(axis=-1) > 128)
    box = out.boxes[0]
    assert xs.min() >= box[0] - 1.5 and ys.min() >= box[1] - 1.5
    assert xs.max() + 1 <= box[2] + 1.5 and ys.max() + 1 <= box[3] + 1.5
    # bright extent also not much smaller than the box hull minus rotation slack: centroid check
    expected = (matrix @ np.array([320., 320., 1.]))[:2]
    got = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
    assert np.abs(got - expected).max() <= 1.0


@pytest.mark.parametrize("deg", [-8.0, 9.0])
def test_rotate_matrix_image_and_boxes_agree(deg):
    from cua.augment.boxes import affine_sample
    s = _square_sample()
    m = rotate_matrix(deg, 640)
    _check_agreement(s, affine_sample(s, m), m)


@pytest.mark.parametrize("seed", [0, 2, 3, 4, 5])
def test_rotate_edit_image_and_boxes_agree(seed):
    s = _square_sample()
    angle = sampled_rotation_angle(np.random.default_rng(seed))
    assert abs(angle) > 0.5
    _check_agreement(s, rotate(s, np.random.default_rng(seed)), rotate_matrix(angle, 640))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_translate_edit_image_and_boxes_agree(seed):
    s = _square_sample()
    ref = np.random.default_rng(seed)
    dx, dy = ref.uniform(-64, 64), ref.uniform(-64, 64)
    _check_agreement(s, translate(s, np.random.default_rng(seed)), translate_matrix(dx, dy))


@pytest.mark.parametrize("edit", ALL)
def test_edits_do_not_mutate_input(edit):
    s = toy_sample(boxes=[[300., 300., 340., 340.]])
    rgb, boxes, labels, ign = s.rgb.copy(), s.boxes.copy(), s.labels.copy(), s.ignore.copy()
    edit(s, np.random.default_rng(2))
    assert np.array_equal(s.rgb, rgb) and np.array_equal(s.boxes, boxes)
    assert np.array_equal(s.labels, labels) and np.array_equal(s.ignore, ign)
