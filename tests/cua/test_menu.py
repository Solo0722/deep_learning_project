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


class _SeqStub:
    """uniform() returns the given values in order (cx, then cy)."""

    def __init__(self, *vals):
        self.vals = list(vals)

    def uniform(self, lo, hi):
        return self.vals.pop(0)


def _tile(fill, box, label=0, ignore=False):
    from cua.image import Sample
    s = toy_sample(boxes=[box], fill=fill)
    return Sample(s.rgb, s.boxes, np.array([label], dtype=np.int64), np.array([ignore]))


def test_mosaic_geometry_distinct_centre_crops_both_axes():
    # cx=200, cy=400: origins (-120,80), (200,80), (-120,400), (200,400); bottom tiles cropped at y=640
    tiles = [_tile(10, [400., 300., 560., 500.]), _tile(20, [100., 100., 300., 200.]),
             _tile(30, [400., 100., 500., 200.]), _tile(40, [200., 400., 400., 600.])]
    out = mosaic(tiles[0], tiles[1:], _SeqStub(200.0, 400.0))
    expected = [[80, 230, 160, 330], [250, 130, 350, 180], [80, 450, 130, 500], [300, 600, 400, 640]]
    assert np.allclose(out.boxes, expected)
    assert out.rgb.shape == (640, 640, 3)
    px = lambda y, x: int(out.rgb[y, x, 0])
    assert (px(100, 50), px(100, 300), px(500, 50), px(500, 300)) == (10, 20, 30, 40)
    assert px(639, 300) == 40 and px(639, 199) == 30 and px(79, 100) == 114 and px(79, 300) == 114
    assert px(10, 10) == 114 and px(100, 600) == 114 and px(500, 600) == 114
    assert px(399, 100) == 10 and px(400, 100) == 30 and px(100, 199) == 10 and px(100, 200) == 20


def test_mosaic_label_ignore_alignment_after_filter():
    from cua.image import Sample
    s0 = toy_sample(boxes=[[400., 300., 560., 500.], [40., 300., 280., 500.], [420., 400., 500., 460.]], fill=10)
    tile0 = Sample(s0.rgb, s0.boxes, np.array([5, 6, 7], dtype=np.int64), np.array([False, True, True]))
    others = [_tile(20, [100., 100., 300., 200.], 8, False), _tile(30, [400., 100., 500., 200.], 9, True),
              _tile(40, [200., 100., 400., 300.], 10, False)]
    out = mosaic(tile0, others, _SeqStub(200.0, 400.0))
    triples = [(tuple(float(v) for v in np.round(b, 6)), int(l), bool(i)) for b, l, i in zip(out.boxes, out.labels, out.ignore)]
    assert triples == [
        ((80., 230., 160., 330.), 5, False),      # tile 0 box A
        ((90., 280., 130., 310.), 7, True),       # tile 0 box B (box X, label 6 ignored, 17% visible, dropped)
        ((250., 130., 350., 180.), 8, False),
        ((80., 450., 130., 500.), 9, True),
        ((300., 450., 400., 550.), 10, False),
    ]
    assert out.labels.dtype == s0.labels.dtype and out.ignore.dtype == bool


def test_armtransform_uses_fetched_samples_unmodified():
    from PIL import Image
    seed, epoch, index = 3, 2, 7
    calls = {}

    def fetch(j):
        s = toy_sample(boxes=[])
        s.rgb = ((s.rgb.astype(np.int32) + 17 * j) % 256).astype(np.uint8)
        calls[len(calls)] = s.rgb.copy()
        return s
    out = ArmTransform("rdd", "mosaic", seed=seed, p=1.0)(toy_sample(), epoch, index, fetch, 100)
    er = keyed_rng(seed, epoch, index, "edit")
    assert er.random() < 1.0
    cx, cy = int(round(float(er.uniform(160, 480)))), int(round(float(er.uniform(160, 480))))
    assert len(calls) == 3
    for k, (ox, oy) in zip(range(3), [(cx, cy - 320), (cx - 320, cy), (cx, cy)]):
        with Image.fromarray(calls[k]) as im, im.resize((320, 320), Image.Resampling.BILINEAR) as r:
            tile = np.asarray(r)
        x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + 320, 640), min(oy + 320, 640)
        assert np.array_equal(out.rgb[y0:y1, x0:x1], tile[y0 - oy:y1 - oy, x0 - ox:x1 - ox])


def test_coin_is_first_draw_on_edit_stream_and_base_unchanged(monkeypatch):
    from cua.augment import menu
    seen = {}

    def spy(sample, rng):
        seen["state"] = rng.bit_generator.state
        seen["rgb"] = sample.rgb.copy()
        return sample
    monkeypatch.setitem(menu.EDITS, "noise", spy)
    p1 = ArmTransform("rdd", "noise", seed=4, p=1.0)(toy_sample(), 6, 9, toy_fetch, 20)
    ref = keyed_rng(4, 6, 9, "edit")
    ref.random()
    assert seen["state"] == ref.bit_generator.state
    p0 = ArmTransform("rdd", "noise", seed=4, p=0.0)(toy_sample(), 6, 9, toy_fetch, 20)
    assert np.array_equal(seen["rgb"], p0.rgb) and np.array_equal(p1.rgb, p0.rgb)


def test_voc_menu():
    assert len(valid_menu("voc")) == 12 and "vflip" not in valid_menu("voc")


def test_mosaic_rejects_empty_dataset():
    with pytest.raises(ValueError):
        ArmTransform("rdd", "mosaic", seed=0, p=1.0)(toy_sample(), 0, 0, toy_fetch, 0)
