import json

import pytest

from cua.gates import GateFailure, class_box_stats, cooccurrence_rate, reference_split
from cua.prepare import prepare_dataset
from cua.records import Record, read_classes, read_records


def test_cooccurrence_rate_counts_only_target_images(mk):
    recs = [mk(labels=[0, 1]), mk(labels=[0]), mk(labels=[])]
    assert cooccurrence_rate(recs) == 0.5


def test_class_box_stats(mk):
    import numpy as np

    big = Record("a.jpg", 100, 100, np.array([[0, 0, 80, 80.0], [0, 0, 10, 10.0]]),
                 np.array([0, 1]), np.zeros(2, bool), "g", "s")
    st = class_box_stats([big], 2)
    assert st[0] == {"count": 1, "large_fraction": 1.0}
    assert st[1] == {"count": 1, "large_fraction": 0.0}


def test_reference_split_is_group_disjoint_and_meets_quota(synthetic_train):
    train = synthetic_train(n_groups=400, per_group=3, n_classes=4, seed=1)
    rest, ref = reference_split(train, 4, n_target=100, min_per_class=20)
    assert {r.group for r in rest}.isdisjoint({r.group for r in ref})
    assert len(rest) + len(ref) == len(train) and len(ref) >= 100
    assert all(sum((r.labels == c).sum() for r in ref) >= 20 for c in range(4))


def test_reference_split_deterministic(synthetic_train):
    train = synthetic_train(400, 3, 4, seed=1)
    assert reference_split(train, 4, 100, 20, seed=0)[1] == reference_split(train, 4, 100, 20, seed=0)[1]


def test_gate_failure_below_threshold(mk, tmp_path):
    single = {"train": [mk(labels=[0])] * 50, "val": [mk(labels=[0])], "test": [mk(labels=[0])]}
    with pytest.raises(GateFailure):
        prepare_dataset("toy", single, ("a", "b"), tmp_path, tmp_path)


def test_prepare_dataset_writes_outputs(synthetic_train, mk, tmp_path):
    splits = {"train": synthetic_train(400, 3, 4, seed=1), "val": [mk(labels=[0, 1])],
              "test": [mk(labels=[1, 2])]}
    out = tmp_path / "out"
    gates = prepare_dataset("toy", splits, ("a", "b", "c", "d"), out, tmp_path)
    for f in ("train.jsonl", "reference.jsonl", "val.jsonl", "test.jsonl", "classes.json",
              "source.json", "gates.json"):
        assert (out / f).exists()
    assert read_classes(out / "classes.json") == ("a", "b", "c", "d")
    assert json.loads((out / "source.json").read_text())["image_root"]
    assert len(read_records(out / "train.jsonl")) + len(read_records(out / "reference.jsonl")) == 1200
    assert gates["cooccurrence"] == json.loads((out / "gates.json").read_text())["cooccurrence"]
