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


def _rec(boxes, labels, ignore=None, image="a.jpg"):
    n = len(labels)
    return Record(image, 100, 100, np.asarray(boxes, dtype=float).reshape(-1, 4),
                  np.asarray(labels, dtype=np.int64),
                  np.zeros(n, bool) if ignore is None else np.asarray(ignore, bool), "g", "s")


@pytest.mark.parametrize("n", [0, 1, 3])
def test_independently_built_equal_records_compare_equal(n):
    boxes = [[float(i), float(i), float(i) + 5, float(i) + 5] for i in range(n)]
    a = _rec(boxes, list(range(n)))
    b = _rec([list(x) for x in boxes], list(range(n)))
    assert a is not b
    assert a == b


def test_record_differing_in_one_box_is_not_equal():
    a = _rec([[0., 0., 5., 5.], [1., 1., 6., 6.], [2., 2., 7., 7.]], [0, 1, 2])
    b = _rec([[0., 0., 5., 5.], [1., 1., 6., 6.], [2., 2., 7., 8.]], [0, 1, 2])
    assert a != b


def test_equal_records_hash_equal_and_dedupe_in_set():
    a = _rec([[0., 0., 5., 5.], [1., 1., 6., 6.]], [0, 1])
    b = _rec([[0., 0., 5., 5.], [1., 1., 6., 6.]], [0, 1])
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_eq_with_non_record_is_false():
    assert _rec([[0., 0., 5., 5.]], [0]) != "not a record"


@pytest.mark.parametrize("bad", [[np.nan, 0., 5., 5.], [0., 0., np.inf, 5.], [-np.inf, 0., 5., 5.]])
def test_non_finite_box_rejected(bad):
    with pytest.raises(ValueError):
        Record("a.jpg", 10, 10, np.array([bad]), np.array([0]), np.array([False]), "g", "s")
