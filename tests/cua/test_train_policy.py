"""Part C policies in the trainer (CPU, untrained Faster R-CNN, 640 canvas, tiny synthetic voc copy)."""
import json
import time

import numpy as np
import pytest
import torch
from PIL import Image

import cua.train as T
from cua.augment.menu import ArmTransform, valid_menu
from cua.data import DetectionDataset
from cua.policy import IdentityPolicy, UniformPolicy
from cua.policy_sampler import PolicyBatchSampler
from cua.records import Record, read_records, write_classes, write_records
from cua.train import TrainConfig, read_history, train
from helpers import toy_fetch, toy_sample

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
MENU = valid_menu("voc")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def cfg(**kw):
    base = dict(run_id="p", dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit=None,
                lr_warmup_iters=2, batch_size=2, amp=True, workers=0, pretrained=False, size=640,
                augment=True, policy="uniform")
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture
def data(tmp_path):
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
    (d / "test.jsonl").write_text("this is not json\n", encoding="utf-8")
    write_classes(d / "classes.json", ("a", "b"))
    (d / "source.json").write_text(json.dumps({"image_root": "../../images"}), encoding="utf-8")
    return tmp_path / "data", tmp_path / "runs"


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _spy_reads(monkeypatch):
    seen = []
    orig = T.read_records

    def spy(p):
        seen.append(p.name)
        return orig(p)

    monkeypatch.setattr(T, "read_records", spy)
    return seen


# ---- fast ----

def test_apply_edit_none_equals_base_and_validates():
    t = ArmTransform("voc", None, seed=0)
    s = toy_sample()
    base = t.base(s, 1, 0)
    out = t.apply(s, 1, 0, toy_fetch, 5, None)
    assert np.array_equal(out.rgb, base.rgb) and np.array_equal(out.boxes, base.boxes)
    assert not np.array_equal(t.apply(s, 1, 0, toy_fetch, 5, "blur").rgb, base.rgb)
    with pytest.raises(ValueError):
        t.apply(s, 1, 0, toy_fetch, 5, "vflip")


def test_dataset_accepts_index_edit_pairs(data):
    data_root, _ = data
    recs = read_records(data_root / "voc" / "train.jsonl")
    ds = DetectionDataset(recs, data_root.parent / "images", ArmTransform("voc", None, 0), 640)
    a, ta = ds[(1, None)]
    b, tb = ds[(1, "brightness")]
    assert a.shape == b.shape == (3, 640, 640) and ta["image_id"] == tb["image_id"] == 1
    assert torch.equal(ta["boxes"], tb["boxes"]) and not torch.equal(a, b)


def test_policy_sampler_pairs_and_counts(mk):
    recs = [mk(image=f"img/{i}.jpg", labels=[i % 2]) for i in range(5)]

    def make(policy_fn, bs, order):
        return PolicyBatchSampler(recs, policy_fn, batch_size=bs, order=order, seed=0, epoch=1,
                                  menu=MENU, classes=("a", "b"))

    s = make(lambda: UniformPolicy(MENU), 2, [4, 3, 2, 1, 0])
    batches = list(s)
    assert [len(b) for b in batches] == [2, 2, 1] and len(s) == 3
    assert [i for b in batches for i, _ in b] == [4, 3, 2, 1, 0]
    assert all(e is None or e in MENU for b in batches for _, e in b)
    s2 = make(lambda: UniformPolicy(MENU), 2, [4, 3, 2, 1, 0])
    assert list(s2) == batches
    stats = s2.stats()
    assert sum(stats["edit_counts"].values()) == 5
    assert sum(stats["edit_by_class"]["a"].values()) == 3
    ident = make(IdentityPolicy, 5, list(range(5)))
    assert [e for b in ident for _, e in b] == [None] * 5


def test_measured_requires_matching_table(data):
    data_root, runs_root = data
    with pytest.raises(FileNotFoundError):
        train(cfg(policy="measured"), data_root, runs_root)
    bad = {"edits": list(MENU), "classes": ["a", "zzz"], "mean": [[0.0, 0.0]] * len(MENU),
           "se": [[0.0, 0.0]] * len(MENU), "n_seeds": 3}
    (data_root / "voc" / "measured_fasterrcnn.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="classes"):
        train(cfg(policy="measured", run_id="p2"), data_root, runs_root)


def test_policy_requires_augment():
    with pytest.raises(ValueError):
        T._validate(cfg(policy="uniform", augment=False, size=320))


# ---- slow: real (tiny) training ----

@pytest.mark.slow
def test_uniform_policy_trains_logs_counts_and_skips_reference(data, monkeypatch):
    data_root, runs_root = data
    seen = _spy_reads(monkeypatch)
    assert train(cfg(), data_root, runs_root) == 0
    row = read_history(runs_root / "p")[0]
    assert sum(row["policy"]["edit_counts"].values()) == 3
    assert "reference.jsonl" not in seen and "test.jsonl" not in seen
    assert not (runs_root / "p" / "utility.jsonl").exists()


@pytest.mark.slow
def test_cua_writes_utility_rows_and_reads_reference(data, monkeypatch):
    data_root, runs_root = data
    seen = _spy_reads(monkeypatch)
    c = cfg(policy="cua", est_every=1, est_start_epoch=1, est_batch=2, ref_batch=2)
    assert train(c, data_root, runs_root) == 0
    rows = _rows(runs_root / "p" / "utility.jsonl")
    assert len(rows) == 2  # 3 images, batch 2 -> 2 steps, an estimate after each
    assert np.asarray(rows[0]["u"]).shape == (len(MENU), 2)
    assert "reference.jsonl" in seen and "test.jsonl" not in seen
    pol = read_history(runs_root / "p")[0]["policy"]
    assert pol["estimates"] == 2 and pol["estimator_seconds"] > 0


@pytest.mark.slow
def test_cua_resume_equals_uninterrupted(data):
    data_root, runs_root = data
    c = cfg(policy="cua", est_every=1, est_start_epoch=1, est_batch=2, ref_batch=2, epoch_lr=(1.0, 0.1))
    assert train(c, data_root, runs_root / "straight") == 0
    split = runs_root / "split"
    assert train(c, data_root, split, deadline=time.monotonic() - 1.0) == 3
    assert train(c, data_root, split, resume=True) == 0
    a, b = read_history(runs_root / "straight" / "p"), read_history(split / "p")
    assert [h["policy"]["edit_counts"] for h in a] == [h["policy"]["edit_counts"] for h in b]
    for ha, hb in zip(a, b):
        for k in ha["losses"]:
            assert ha["losses"][k] == pytest.approx(hb["losses"][k], abs=1e-6), k
    ua, ub = _rows(runs_root / "straight" / "p" / "utility.jsonl"), _rows(split / "p" / "utility.jsonl")
    assert len(ua) == len(ub) == 4
    for x, y in zip(ua, ub):
        assert np.allclose(x["mean"], y["mean"], atol=1e-9)
    ma = torch.load(runs_root / "straight" / "p" / "last.pt", weights_only=True)["model"]
    mb = torch.load(split / "p" / "last.pt", weights_only=True)["model"]
    assert all(torch.equal(ma[k], mb[k]) for k in ma)


@pytest.mark.slow
def test_fallbacks_are_logged_per_epoch(data, monkeypatch):
    """Review finding: the policy's fallback counter is cumulative; history must log each epoch's own count."""
    class CountingUniform(UniformPolicy):
        fallbacks = 0

        def probs(self, class_set):
            self.fallbacks += 1
            return super().probs(class_set)

    monkeypatch.setattr(T, "UniformPolicy", CountingUniform)
    data_root, runs_root = data
    assert train(cfg(epoch_lr=(1.0, 0.1)), data_root, runs_root) == 0
    assert [h["policy"]["fallbacks"] for h in read_history(runs_root / "p")] == [3, 3]


@pytest.mark.slow
def test_cua_v2_lookahead_with_probe_prior_trains(data):
    data_root, runs_root = data
    probe = {"edits": list(MENU), "classes": ["a", "b"], "mean": [[1.0, -1.0]] * len(MENU),
             "se": [[0.5, 0.5]] * len(MENU), "n_seeds": 3}
    c = cfg(policy="cua", estimator="lookahead", lookahead_steps=2, est_every=1, est_start_epoch=1,
            est_batch=2, ref_batch=2, min_t=2.0, prior="probe", beta=0.98)
    with pytest.raises(FileNotFoundError):
        train(c, data_root, runs_root)
    (data_root / "voc" / "probe_fasterrcnn.json").write_text(json.dumps(probe), encoding="utf-8")
    import shutil
    shutil.rmtree(runs_root / "p", ignore_errors=True)
    assert train(c, data_root, runs_root) == 0
    rows = _rows(runs_root / "p" / "utility.jsonl")
    assert len(rows) == 2 and np.asarray(rows[0]["u"]).shape == (len(MENU), 2)
    assert np.isfinite(np.asarray(rows[0]["u"])).all()
