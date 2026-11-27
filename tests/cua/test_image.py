import numpy as np
import pytest
from PIL import Image

from cua.image import Sample, letterbox, load_rgb, restore_boxes
from cua.records import Record


def _rec(width=1000, height=500, boxes=None, labels=None, ignore=None):
    if boxes is None:
        boxes = np.array([[100.0, 50.0, 300.0, 250.0]])
        labels = np.array([0])
        ignore = np.array([False])
    return Record("x.jpg", width, height, boxes, labels, ignore, "g", "s")


def test_letterbox_then_restore_is_identity():
    rec = _rec()
    sample, geo = letterbox(np.zeros((500, 1000, 3), np.uint8), rec)
    assert isinstance(sample, Sample)
    assert sample.rgb.shape == (640, 640, 3) and geo["pad_top"] == 160
    assert np.allclose(restore_boxes(sample.boxes, geo), rec.boxes)


def test_padding_value_is_114():
    empty = _rec(boxes=np.zeros((0, 4)), labels=np.zeros(0, np.int64), ignore=np.zeros(0, bool))
    sample, _ = letterbox(np.zeros((500, 1000, 3), np.uint8), empty)
    assert (sample.rgb[0, 0] == 114).all()


def test_geometry_keys_and_values():
    _, geo = letterbox(np.zeros((500, 1000, 3), np.uint8), _rec())
    assert set(geo) == {"original_width", "original_height", "resized_width",
                        "resized_height", "pad_left", "pad_top", "size"}
    assert geo == {"original_width": 1000, "original_height": 500, "resized_width": 640,
                   "resized_height": 320, "pad_left": 0, "pad_top": 160, "size": 640}


def test_letterbox_carries_labels_and_ignore_from_record():
    rec = _rec(boxes=np.array([[10.0, 10.0, 90.0, 90.0], [1.0, 1.0, 5.0, 5.0]]),
               labels=np.array([2, 3]), ignore=np.array([False, True]))
    sample, _ = letterbox(np.zeros((100, 100, 3), np.uint8),
                          Record("x.jpg", 100, 100, rec.boxes, rec.labels, rec.ignore, "g", "s"))
    assert np.array_equal(sample.labels, [2, 3])
    assert np.array_equal(sample.ignore, [False, True])


def test_letterbox_rejects_bad_size():
    rgb = np.zeros((500, 1000, 3), np.uint8)
    for bad in (0, -32, 600, 640.0):
        with pytest.raises(ValueError):
            letterbox(rgb, _rec(), size=bad)


def test_letterbox_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        letterbox(np.zeros((400, 1000, 3), np.uint8), _rec())


def test_load_rgb_correct_size_passes(tmp_path):
    Image.new("RGB", (20, 10), (255, 0, 0)).save(tmp_path / "ok.png")
    rec = Record("ok.png", 20, 10, np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool), "g", "s")
    rgb = load_rgb(tmp_path, rec)
    assert rgb.dtype == np.uint8 and rgb.shape == (10, 20, 3)
    assert (rgb[..., 0] == 255).all()


def test_load_rgb_wrong_size_raises(tmp_path):
    Image.new("RGB", (20, 10)).save(tmp_path / "bad.png")
    rec = Record("bad.png", 30, 10, np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool), "g", "s")
    with pytest.raises(ValueError):
        load_rgb(tmp_path, rec)
