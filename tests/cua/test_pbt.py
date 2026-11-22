"""Per-class PBT: table policy, selection rules, ids, and a tiny end-to-end run on CPU."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

import cua.pbt as PB
import cua.train as T
from cua.augment.menu import valid_menu
from cua.policy import TablePolicy
from cua.records import Record, write_classes, write_records

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
MENU = valid_menu("voc")


# ---- table policy ----

def test_table_policy_zero_logits_is_uniform_and_mixes_epsilon():
    pol = TablePolicy(("a", "b"), np.zeros((3, 2)), shared=False, epsilon=0.1)
    cands, p = pol.probs(frozenset({0}))
    assert cands == (None, "a", "b") and np.allclose(p, 1 / 3)
    pol = TablePolicy(("a", "b"), np.array([[0.0, 0.0], [50.0, -50.0], [0.0, 0.0]]), shared=False, epsilon=0.3)
    _, p0 = pol.probs(frozenset({0}))
    _, p01 = pol.probs(frozenset({0, 1}))  # class means cancel -> uniform
    assert p0[1] == pytest.approx(0.7 + 0.1) and np.allclose(p01, 1 / 3)


def test_table_policy_shared_ignores_class_set_and_validates_shape():
    pol = TablePolicy(("a",), np.array([[0.0], [1.0]]), shared=True, epsilon=0.0)
    assert np.allclose(pol.probs(frozenset({0}))[1], pol.probs(frozenset({5}))[1])
    with pytest.raises(ValueError):
        TablePolicy(("a",), np.zeros((3, 1)), shared=True)
    with pytest.raises(ValueError):
        TablePolicy(("a",), np.zeros((2, 2)), shared=True)


def test_table_fields_validate_and_stay_out_of_old_digests():
    base = dict(run_id="r", dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit=None)
    old = T.TrainConfig(**base, policy="uniform")
    import dataclasses, hashlib
    d = dataclasses.asdict(old)
    for f in dataclasses.fields(old):
        if f.name in T._PLAN2_FIELDS and d[f.name] == f.default:
            del d[f.name]
    assert "table_path" not in d and old.digest() == hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError):
        T._validate(T.TrainConfig(**base, policy="table"))
    with pytest.raises(ValueError):
        T._validate(T.TrainConfig(**base, policy="uniform", table_path="x.json"))


# ---- ids, windows, selection ----

def test_resolve_pbt_and_windows():
    p = PB.resolve_pbt("PBT-rdd-fasterrcnn-s0-pc")
    assert (p.dataset, p.detector, p.seed, p.shared, p.members, p.window, p.epochs) == ("rdd", "fasterrcnn", 0, False, 4, 2, 20)
    q = PB.resolve_pbt("PBT-dacl10k-retinanet-s2-shared-n6-w4")
    assert q.shared and q.members == 6 and q.window == 4 and q.epochs == 30 and q.n_windows == 8
    assert q.window_epochs(8) == (29, 30) and p.window_epochs(1) == (1, 2)
    for bad in ("PBT-rdd-yolo-s0-pc", "C-rdd-fasterrcnn-s0-cua", "PBT-rdd-fasterrcnn-s0-x"):
        with pytest.raises(KeyError):
            PB.resolve_pbt(bad)


def test_choose_per_class_and_shared():
    refs = [{"macro_ap50": 10.0, "per_class": {"a": 5.0, "b": 9.0, "c": None}},
            {"macro_ap50": 12.0, "per_class": {"a": 7.0, "b": 9.0, "c": None}},
            {"macro_ap50": 11.0, "per_class": {"a": 1.0, "b": 8.0, "c": None}}]
    p = PB.PBTConfig("x", "voc", "fasterrcnn", 0, members=3)
    assert PB._choose(p, refs, ("a", "b", "c")) == (1, [1, 0, None])  # tie on b -> lowest index
    ps = PB.PBTConfig("x", "voc", "fasterrcnn", 0, shared=True, members=3)
    assert PB._choose(ps, refs, ("a", "b", "c")) == (1, [1])


def test_explore_keeps_member0_and_is_deterministic():
    p = PB.PBTConfig("x", "voc", "fasterrcnn", 3, members=3, sigma=1.0)
    base = np.arange(6.0).reshape(3, 2)
    a, b = PB._explore(p, base, 2), PB._explore(p, base, 2)
    assert np.array_equal(a[0], base) and all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], base) and not np.array_equal(a[1], a[2])
    assert not np.array_equal(PB._explore(p, base, 3)[1], a[1])


# ---- end to end (CPU, tiny synthetic voc copy) ----

@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    (images / "img").mkdir(parents=True)
    recs = []
    for i in range(8):
        w, h = 96, 64
        rgb = rng.integers(20, 200, size=(h, w, 3), dtype=np.uint8)
        rgb[10:40, 10 + i:50 + i] = (230, 30, 30)
        Image.fromarray(rgb).save(images / "img" / f"{i}.png")
        boxes = [[10.0 + i, 10.0, 50.0 + i, 40.0], [55.0, 20.0, 90.0, 60.0]]
        recs.append(Record(f"img/{i}.png", w, h, np.array(boxes), np.array([0, 1]),
                           np.array([False, False]), f"g{i}", "s"))
    d = tmp_path / "data" / "voc"
    d.mkdir(parents=True)
    write_records(d / "train.jsonl", recs[:3])
    write_records(d / "val.jsonl", recs[3:5])
    write_records(d / "reference.jsonl", recs[5:8])
    write_classes(d / "classes.json", ("a", "b"))
    (d / "source.json").write_text(json.dumps({"image_root": "../../images"}), encoding="utf-8")
    return tmp_path / "data", tmp_path / "runs"


@pytest.fixture
def tiny(monkeypatch):
    """Small, fast member configs: no pretrained weights, batch 2, no workers, short warm-up."""
    orig = PB.member_config

    def small(p, runs_root, k, j, winner):
        import dataclasses
        return dataclasses.replace(orig(p, runs_root, k, j, winner), pretrained=False, batch_size=2, workers=0,
                                   lr_warmup_iters=2)

    monkeypatch.setattr(PB, "member_config", small)
    orig_ref = PB.reference_metrics
    monkeypatch.setattr(PB, "reference_metrics", lambda p, dr, rd, dev, workers=4: orig_ref(p, dr, rd, dev, 0))


@pytest.mark.slow
def test_pbt_runs_end_to_end_selects_and_is_idempotent(data, tiny):
    data_root, runs_root = data
    p = PB.PBTConfig("PBT-voc-fasterrcnn-s0-pc", "voc", "fasterrcnn", 0, members=2, window=1, epochs=2)
    assert PB.run_all(p, data_root, runs_root) == 0
    root = runs_root / p.pbt_id
    rows = [json.loads(x) for x in (root / "decisions.jsonl").read_text().splitlines()]
    assert [r["window"] for r in rows] == [1, 2] and "final" in rows[1]
    assert json.loads((root / "final.json").read_text())["member"] in (0, 1)
    # window 2 started from window 1's winner and member 0 used the selected table unchanged
    w = rows[0]["weights_from"]
    cfg2 = json.loads((root / "w2-m0" / "config.json").read_text())["config"]
    assert cfg2["init_run"] == f"{p.pbt_id}/w1-m{w}" and cfg2["first_epoch"] == 2
    t20 = json.loads((root / "tables" / "w2-m0.json").read_text())["logits"]
    assert np.allclose(t20, rows[0]["selected"])
    # checkpoints of window 1 are freed; the final window keeps its best.pt files
    assert not any((root / f"w1-m{j}" / "last.pt").exists() for j in range(2))
    assert all((root / f"w2-m{j}" / "best.pt").exists() for j in range(2))
    # re-running does nothing new
    before = (root / "w2-m1" / "history.jsonl").stat().st_mtime_ns
    assert PB.run_all(p, data_root, runs_root) == 0
    assert (root / "w2-m1" / "history.jsonl").stat().st_mtime_ns == before
    assert len((root / "decisions.jsonl").read_text().splitlines()) == 2
