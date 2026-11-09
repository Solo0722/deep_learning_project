import numpy as np
import pytest

from cua.records import Record, read_classes, read_records, write_classes, write_records


def test_roundtrip_preserves_arrays(tmp_path):
    r = Record("a/b.jpg", 600, 400, np.array([[1., 2., 30., 40.]]), np.array([2]), np.array([False]), "g1", "Japan")
    write_records(tmp_path / "x.jsonl", [r]); (back,) = read_records(tmp_path / "x.jsonl")
    assert back.boxes.dtype == np.float64 and back.boxes.shape == (1, 4)
    assert np.array_equal(back.labels, [2]) and back.group == "g1"


def test_empty_record_has_shape_0x4(tmp_path):
    r = Record("n.jpg", 10, 10, np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool), "g", "s")
    write_records(tmp_path / "e.jsonl", [r])
    assert read_records(tmp_path / "e.jsonl")[0].boxes.shape == (0, 4)


def test_invalid_box_rejected():
    with pytest.raises(ValueError):
        Record("a.jpg", 10, 10, np.array([[5., 5., 5., 9.]]), np.array([0]), np.array([False]), "g", "s")


def test_class_set_ignores_ignored():
    r = Record("a.jpg", 50, 50, np.array([[0,0,9,9],[1,1,8,8.]]), np.array([0, 3]), np.array([False, True]), "g", "s")
    assert r.class_set() == frozenset({0})


def test_box_outside_image_rejected():
    with pytest.raises(ValueError):
        Record("a.jpg", 10, 10, np.array([[0., 0., 11., 5.]]), np.array([0]), np.array([False]), "g", "s")


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        Record("a.jpg", 10, 10, np.array([[0., 0., 5., 5.]]), np.array([0, 1]), np.array([False]), "g", "s")


@pytest.mark.parametrize("path", ["/abs/a.jpg", "../a.jpg", "C:/a.jpg"])
def test_non_relative_image_path_rejected(path):
    with pytest.raises(ValueError):
        Record(path, 10, 10, np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool), "g", "s")


def test_classes_roundtrip(tmp_path):
    write_classes(tmp_path / "c.json", ("cat", "dog"))
    assert read_classes(tmp_path / "c.json") == ("cat", "dog")
