import numpy as np
import pytest

from cua.augment.base import base_geometry
from cua.augment.menu import EDITS, MENU_ORDER, ArmTransform, valid_menu
from cua.augment.mosaic import clip_and_filter, mosaic
from cua.augment.rng import keyed_rng
from helpers import toy_fetch, toy_sample


def test_invalid_edit_rejected():
    with pytest.raises(ValueError):
        ArmTransform("rdd", "vflip", seed=0)
    with pytest.raises(ValueError):
        ArmTransform("nope", None, seed=0)
    ArmTransform("dacl10k", "vflip", seed=0)
    ArmTransform("rdd", None, seed=0)


def test_base_identical_across_arms_when_edit_not_applied():
    outs = [ArmTransform("rdd", e, p=0.0, seed=0)(toy_sample(), 3, 11, toy_fetch, 100) for e in (None, "blur", "mosaic")]
    assert all(np.array_equal(o.rgb, outs[0].rgb) for o in outs)


def test_clip_and_filter_visibility_rule():
    boxes = np.array([[-100., 0., 20., 10.], [0., 0., 100., 100.]]); areas = np.array([1200., 10000.])
    before = boxes.copy()
    assert clip_and_filter(boxes, areas).tolist() == [False, True]
    assert np.array_equal(boxes, before)


def test_mosaic_on_empty_samples():
    t = ArmTransform("rdd", "mosaic", seed=0, p=1.0)
    assert t(toy_sample(boxes=[]), 4, 0, lambda j: toy_sample(boxes=[]), 10).boxes.shape == (0, 4)


def test_valid_menu_sizes():
    assert len(valid_menu("rdd")) == 12 and len(valid_menu("dacl10k")) == 13
    assert set(MENU_ORDER) - {"mosaic"} == set(EDITS)


def _applied(edit, p, indices):
    t = ArmTransform("rdd", edit, seed=1, p=p)
    base = ArmTransform("rdd", None, seed=1, p=p)
    return [not np.array_equal(t(toy_sample(), 0, i, toy_fetch, 100).rgb, base(toy_sample(), 0, i, toy_fetch, 100).rgb)
            for i in indices]


def test_coin_identical_across_arms():
    assert _applied("blur", 0.5, range(50)) == _applied("noise", 0.5, range(50))


def test_applied_fraction():
    frac = np.mean(_applied("noise", 0.5, range(200)))
    assert 0.4 <= frac <= 0.6


def test_p_one_always_changes_noise():
    assert all(_applied("noise", 1.0, range(20)))


class _Stub:
    def __init__(self, v):
        self.v = v

    def uniform(self, lo, hi):
        return self.v


def test_mosaic_geometry_stubbed():
    s = toy_sample(boxes=[[100., 100., 220., 260.]], fill=10)
    others = [toy_sample(boxes=[[100., 100., 220., 260.]], fill=f) for f in (20, 30, 40)]
    out = mosaic(s, others, _Stub(320.0))
    expected = [[50, 50, 110, 130], [370, 50, 430, 130], [50, 370, 110, 450], [370, 370, 430, 450]]
    assert np.allclose(out.boxes, expected)
    assert out.labels.tolist() == [0, 0, 0, 0]
    assert out.rgb[10, 10, 0] == 10 and out.rgb[10, 600, 0] == 20
    assert out.rgb[600, 10, 0] == 30 and out.rgb[600, 600, 0] == 40


def test_mosaic_visibility_threshold():
    # centre (160,160): tile 0 origin (-160,-160). Box x in [x0, x0+200] scales to width 100, left edge x0/2-160.
    def run(x0):
        s = toy_sample(boxes=[[x0, 400., x0 + 200., 500.]], fill=10)
        return mosaic(s, [toy_sample(boxes=[]) for _ in range(3)], _Stub(160.0)).boxes
    assert len(run(160.0)) == 0   # right edge at x=20: 20% visible, dropped
    assert len(run(180.0)) == 1   # right edge at x=30: exactly 30% visible, kept
    assert len(run(200.0)) == 1   # 40% visible, kept
    assert np.allclose(run(200.0), [[0, 40, 40, 90]])


def test_mosaic_never_fetches_self():
    calls = []

    def fetch(j):
        calls.append(j)
        return toy_fetch(j)
    t = ArmTransform("rdd", "mosaic", seed=0, p=1.0)
    idx = []
    for i in range(30):
        t(toy_sample(), 0, i % 3, fetch, 3)
        idx += [i % 3] * 3
    assert len(calls) == len(idx) and all(c != i for c, i in zip(calls, idx))
    t(toy_sample(), 0, 0, fetch, 1)
    assert calls[-3:] == [0, 0, 0]


def test_determinism_and_no_mutation():
    s = toy_sample()
    rgb, boxes = s.rgb.copy(), s.boxes.copy()
    for e in (None, "mosaic", "noise", "rotate"):
        t = ArmTransform("rdd", e, seed=5, p=1.0)
        a = t(s, 2, 9, toy_fetch, 50)
        b = t(s, 2, 9, toy_fetch, 50)
        assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.boxes, b.boxes)
    assert np.array_equal(s.rgb, rgb) and np.array_equal(s.boxes, boxes)


def test_base_unaffected_by_edit():
    s = toy_sample()
    b = base_geometry(s, keyed_rng(0, 1, 2, "base"))
    out = ArmTransform("rdd", "mosaic", seed=0, p=0.0)(s, 1, 2, toy_fetch, 10)
    assert np.array_equal(out.rgb, b.rgb) and np.array_equal(out.boxes, b.boxes)
