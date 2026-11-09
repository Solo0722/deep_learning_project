import numpy as np
import pytest

from cua.records import Record


def _mk(image="img/a.jpg", labels=(0,), group="g", source="s", width=100, height=100):
    labels = np.asarray(list(labels), dtype=np.int64)
    n = len(labels)
    boxes = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        boxes[i] = [10.0 + i, 10.0 + i, 40.0 + i, 40.0 + i]
    ignore = np.zeros(n, dtype=bool)
    return Record(image, width, height, boxes, labels, ignore, group, source)


@pytest.fixture
def mk():
    return _mk


def _synthetic_train(n_groups, per_group, n_classes, seed):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_groups):
        for k in range(per_group):
            n = int(rng.integers(1, 4))
            labels = rng.integers(0, n_classes, size=n).tolist()
            out.append(_mk(image=f"img/g{g}_{k}.jpg", labels=labels, group=f"g{g}"))
    return out


@pytest.fixture
def synthetic_train():
    return _synthetic_train
