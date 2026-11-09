import numpy as np

from cua.augment.base import base_geometry
from cua.augment.boxes import affine_sample, compose, hflip_matrix, scale_matrix
from cua.augment.rng import keyed_rng
from helpers import toy_sample


def test_keyed_rng_pure_and_streams_differ():
    a = base_geometry(toy_sample(), keyed_rng(0, 4, 17, "base"))
    b = base_geometry(toy_sample(), keyed_rng(0, 4, 17, "base"))
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.boxes, b.boxes)
    assert np.array_equal(keyed_rng(0, 4, 17, "base").random(5), keyed_rng(0, 4, 17, "base").random(5))
    draws = [keyed_rng(0, 4, 17, n).random(3).tolist() for n in ("base", "edit", "mosaic")]
    assert len({tuple(d) for d in draws}) == 3


def test_keys_differ():
    assert keyed_rng(0, 1, 2, "base").random() != keyed_rng(0, 1, 2, "edit").random()
    assert keyed_rng(0, 1, 2, "base").random() != keyed_rng(0, 2, 1, "base").random()


def test_hflip_box_formula():
    s = toy_sample(boxes=[[100., 50., 200., 150.]])
    out = affine_sample(s, hflip_matrix(640))
    assert out.boxes[0].tolist() == [440.0, 50.0, 540.0, 150.0]


def test_box_touching_edge_rejects_upscale():
    s = toy_sample(boxes=[[0., 0., 640., 640.]])
    assert affine_sample(s, scale_matrix(1.2, 640)) is None


def test_empty_sample_passes_base():
    out = base_geometry(toy_sample(boxes=[]), keyed_rng(0, 0, 0, "base"))
    assert out.boxes.shape == (0, 4)
    assert out.labels.shape == (0,) and out.ignore.shape == (0,)
    assert out.rgb.shape == (640, 640, 3) and out.rgb.dtype == np.uint8


def test_identity_leaves_rgb_unchanged():
    s = toy_sample()
    out = affine_sample(s, np.eye(3))
    assert np.array_equal(out.rgb, s.rgb)
    assert np.allclose(out.boxes, s.boxes)


def test_hflip_rgb_exactly_fliplr():
    s = toy_sample()
    out = affine_sample(s, hflip_matrix(640))
    assert np.array_equal(out.rgb, np.fliplr(s.rgb))


def test_edge_box_survives_hflip():
    s = toy_sample(boxes=[[0., 0., 640., 640.]])
    out = affine_sample(s, hflip_matrix(640))
    assert out is not None and out.boxes[0].tolist() == [0.0, 0.0, 640.0, 640.0]


def test_ignored_box_beyond_canvas_is_clipped():
    s = toy_sample(boxes=[[0., 0., 640., 640.], [100., 100., 200., 200.]])
    s.ignore[0] = True
    out = affine_sample(s, scale_matrix(1.2, 640))
    assert out is not None
    assert out.boxes[0].tolist() == [0.0, 0.0, 640.0, 640.0]
    assert out.labels.tolist() == [0, 1] and out.ignore.tolist() == [True, False]


def test_degenerate_ignored_box_dropped_and_aligned():
    s = toy_sample(boxes=[[630., 100., 640., 200.], [100., 100., 200., 200.]])
    s.ignore[0] = True
    out = affine_sample(s, np.array([[1, 0, 20.0], [0, 1, 0], [0, 0, 1]]))
    assert out.boxes.shape == (1, 4)
    assert out.labels.tolist() == [1] and out.ignore.tolist() == [False]


def test_small_nonignored_box_rejected():
    s = toy_sample(boxes=[[100., 100., 101.5, 200.]])
    assert affine_sample(s, scale_matrix(0.5, 640)) is None


def test_compose_order():
    m = compose(hflip_matrix(640), scale_matrix(1.0, 640))
    assert np.allclose(m, hflip_matrix(640))
    t = np.array([[1, 0, 10.0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(compose(t, hflip_matrix(640)) @ [0, 0, 1], [630, 0, 1])  # shift first, then flip


class _UpOnly:
    def __init__(self, flip):
        self.flip = flip

    def random(self):
        return 0.0 if self.flip else 0.9

    def uniform(self, lo, hi):
        return 1.1


def test_base_falls_back_to_scale_one_keeps_flip():
    s = toy_sample(boxes=[[0., 0., 640., 640.]])
    out = base_geometry(s, _UpOnly(flip=True))
    assert out is not None
    assert np.array_equal(out.rgb, np.fliplr(s.rgb))
    assert out.boxes[0].tolist() == [0.0, 0.0, 640.0, 640.0]
    out2 = base_geometry(s, _UpOnly(flip=False))
    assert np.array_equal(out2.rgb, s.rgb)


def test_tiny_box_does_not_raise_and_is_kept():
    s = toy_sample(boxes=[[100., 100., 100.5, 100.5]])
    for k in range(20):
        out = base_geometry(s, keyed_rng(0, 0, k, "base"))
        assert out.boxes.shape == (1, 4)
        w = out.boxes[0, 2] - out.boxes[0, 0]
        h = out.boxes[0, 3] - out.boxes[0, 1]
        assert 0.35 < w < 0.65 and 0.35 < h < 0.65


def test_slightly_outside_box_clipped_not_raised():
    s = toy_sample(boxes=[[-3., 10., 100., 200.]])
    for k in range(20):
        out = base_geometry(s, keyed_rng(0, 0, k, "base"))
        assert out.boxes.shape == (1, 4)
        assert (out.boxes >= 0).all() and (out.boxes <= 640).all()


def test_base_geometry_property_boxes_valid():
    s = toy_sample(boxes=[[0., 0., 640., 640.], [300., 300., 330., 320.], [5., 600., 80., 640.]])
    for k in range(200):
        out = base_geometry(s, keyed_rng(1, 2, k, "base"))
        assert out.rgb.shape == (640, 640, 3)
        b = out.boxes
        assert len(b) == 3
        assert (b >= 0).all() and (b <= 640 + 1e-6).all()
        assert ((b[:, 2] - b[:, 0]) >= 1).all() and ((b[:, 3] - b[:, 1]) >= 1).all()


class _Flip:
    def random(self):
        return 0.0

    def uniform(self, lo, hi):
        return 1.0


def test_exact_one_px_box_survives_flip():
    s = toy_sample(boxes=[[127.02641775, 100., 128.02641775, 300.]])
    out = base_geometry(s, _Flip())
    assert out.boxes.shape == (1, 4)
    assert abs((out.boxes[0, 2] - out.boxes[0, 0]) - 1.0) < 1e-6
    assert affine_sample(s, hflip_matrix(640)) is not None


def test_pre_invalid_box_pushed_fully_out_rejects():
    s = toy_sample(boxes=[[630., 100., 630.5, 200.]])
    assert affine_sample(s, scale_matrix(1.2, 640)) is None
    out = base_geometry(s, keyed_rng(0, 0, 0, "base"))
    assert out.boxes.shape == (1, 4)


def test_fully_outside_input_box_passes_through_clipped():
    s = toy_sample(boxes=[[700., 100., 750., 200.]])
    out = affine_sample(s, hflip_matrix(640))
    assert out is not None and out.boxes.shape == (1, 4)
    assert (out.boxes >= 0).all() and (out.boxes <= 640).all()


def test_property_mixed_valid_and_subpixel_inputs():
    s = toy_sample(boxes=[[100., 100., 200., 200.], [300., 300., 300.4, 400.],
                          [-5., 50., 20., 90.], [10., 10., 10.5, 10.5]])
    for k in range(200):
        out = base_geometry(s, keyed_rng(3, 1, k, "base"))
        b = out.boxes
        assert len(b) == 4
        assert (b >= 0).all() and (b <= 640 + 1e-6).all()
        assert ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) > 0).all()
