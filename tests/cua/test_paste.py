"""Class-adaptive copy-paste: bank, class probabilities, pasting and the trainer wiring."""
import json
import time

import numpy as np
import pytest
import torch
from PIL import Image

import cua.train as T
from cua.augment.paste import CopyPaste, adaptive_probs, build_bank, content_of
from cua.experiments.specs import resolve
from cua.image import letterbox
from cua.records import Record, write_classes, write_records

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")


def _rec(i, boxes, labels, ignore=None, w=320, h=240):
    n = len(boxes)
    return Record(f"img/{i}.png", w, h, np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(labels),
                  np.zeros(n, bool) if ignore is None else np.array(ignore), f"g{i}", "s")


def _recs():
    return [_rec(0, [[10, 10, 60, 50], [100, 100, 102, 103]], [0, 1]),   # tiny class-1 box is not banked
            _rec(1, [[50, 60, 150, 140]], [1]),
            _rec(2, [[200, 20, 300, 120], [5, 5, 40, 40]], [0, 2], [False, True]),  # ignored box not banked
            _rec(3, [], [])]


def _sample(rec, rng):
    rgb = rng.integers(0, 255, size=(rec.height, rec.width, 3), dtype=np.uint8)
    return letterbox(rgb, rec, 640)[0]


def test_bank_skips_small_and_ignored_boxes():
    bank = build_bank(_recs(), 3)
    assert bank[0].tolist() == [[0, 0], [2, 0]] and bank[1].tolist() == [[1, 0]] and len(bank[2]) == 0


def test_adaptive_probs_favour_weak_classes_with_floor():
    p = adaptive_probs([80.0, 20.0, None], np.array([True, True, True]), 0.2)
    assert p[1] > p[2] > p[0] and np.isclose(p.sum(), 1) and p.min() >= 0.2 / 3 - 1e-12
    q = adaptive_probs([80.0, 20.0, 50.0], np.array([True, True, False]), 0.0)
    assert q[2] == 0 and q[1] > q[0]
    assert np.allclose(adaptive_probs([50.0, 50.0], np.array([True, True]), 0.2), 0.5)


def test_modes_initial_probs_and_set_probs_rules():
    recs = _recs()
    r = CopyPaste(recs, 3, "random", 0)
    b = CopyPaste(recs, 3, "balanced", 0)
    a = CopyPaste(recs, 3, "adaptive", 0)
    assert np.allclose(r.probs, [2 / 3, 1 / 3, 0]) and np.allclose(b.probs, [0.5, 0.5, 0])
    a.set_probs([0.2, 0.8, 0.0])
    with pytest.raises(ValueError):
        a.set_probs([0.2, 0.4, 0.4])  # class 2 has no objects
    with pytest.raises(ValueError):
        b.set_probs([0.5, 0.5, 0.0])
    with pytest.raises(ValueError):
        CopyPaste(recs, 3, "other", 0)


def test_plan_is_keyed_and_respects_probabilities():
    a = CopyPaste(_recs(), 3, "adaptive", 7, p=1.0, max_objects=3)
    assert a.plan(2, 3) == a.plan(2, 3)
    a.set_probs([0.0, 1.0, 0.0])
    plans = [a.plan(1, i % 4) for i in range(40)]
    assert all(c == 1 and (j, b) == (1, 0) for pl in plans for c, j, b in pl)
    assert all(j != i % 4 for i, pl in enumerate(plans) for _, j, _ in pl)  # never into itself
    none = CopyPaste(_recs(), 3, "random", 7, p=0.0)
    assert all(none.plan(1, i) == [] for i in range(10))


def test_paste_adds_labelled_boxes_inside_content_without_covering_objects():
    recs = _recs()
    rng = np.random.default_rng(0)
    samples = [_sample(r, rng) for r in recs]
    cp = CopyPaste(recs, 3, "balanced", 3, p=1.0, max_objects=3)
    for epoch in range(1, 6):
        for i in range(4):
            out, pasted = cp(samples[i], epoch, i, lambda j: samples[j])
            again, _ = cp(samples[i], epoch, i, lambda j: samples[j])
            assert np.array_equal(out.rgb, again.rgb)
            n0 = len(samples[i].boxes)
            assert len(out.boxes) == n0 + len(pasted) and out.labels[n0:].tolist() == pasted
            assert not out.ignore[n0:].any() and np.array_equal(out.boxes[:n0], samples[i].boxes)
            c = content_of(recs[i], 640)
            new = out.boxes[n0:]
            assert np.all(new[:, 0] >= c.left) and np.all(new[:, 2] <= c.left + c.width)
            assert np.all(new[:, 1] >= c.top) and np.all(new[:, 3] <= c.top + c.height)
            if not pasted:
                assert out is samples[i]
                continue
            mask = np.zeros((640, 640), bool)
            for x0, y0, x1, y1 in new.astype(int):
                mask[y0:y1, x0:x1] = True
            assert np.array_equal(out.rgb[~mask], samples[i].rgb[~mask])  # only pasted areas change
            for k, box in enumerate(out.boxes):
                for m, cand in enumerate(new):
                    if k == n0 + m:
                        continue
                    iw = max(0, min(box[2], cand[2]) - max(box[0], cand[0]))
                    ih = max(0, min(box[3], cand[3]) - max(box[1], cand[1]))
                    if k < n0 + m:  # earlier boxes are never covered by more than 30%
                        assert iw * ih / ((box[2] - box[0]) * (box[3] - box[1])) <= 0.3 + 1e-9


def test_paste_specs_and_validation():
    c = resolve("P-rdd-fasterrcnn-s2-cpada")
    assert (c.policy, c.cp_mode, c.seed) == ("uniform", "adaptive", 2)
    assert resolve("P-voc-retinanet-s0-cpbal").cp_mode == "balanced"
    with pytest.raises(KeyError):
        resolve("P-rdd-fasterrcnn-s0-cpfoo")
    base = dict(run_id="r", dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit=None)
    T._validate(T.TrainConfig(**base, policy="uniform", cp_mode="random"))
    for bad in (dict(policy="uniform", cp_mode="x"), dict(cp_mode="random", augment=False, size=320),
                dict(policy="uniform", cp_mode="random", cp_p=2.0), dict(policy="uniform", cp_mode="random", cp_max=0)):
        with pytest.raises(ValueError):
            T._validate(T.TrainConfig(**base, **bad))
    # the new fields stay out of existing digests (value recorded before copy-paste existed)
    assert resolve("C-rdd-fasterrcnn-s0-uniform").digest().startswith("46be8801b746")


# ---- trainer wiring (CPU, tiny synthetic data) ----

@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    (images / "img").mkdir(parents=True)
    recs = []
    for i in range(9):
        w, h = 96, 64
        rgb = rng.integers(20, 200, size=(h, w, 3), dtype=np.uint8)
        rgb[10:40, 10 + i:50 + i] = (230, 30, 30)
        Image.fromarray(rgb).save(images / "img" / f"{i}.png")
        recs.append(Record(f"img/{i}.png", w, h, np.array([[10.0 + i, 10.0, 50.0 + i, 40.0], [55.0, 20.0, 90.0, 60.0]]),
                           np.array([0, 1]), np.array([False, False]), f"g{i}", "s"))
    d = tmp_path / "data" / "voc"
    d.mkdir(parents=True)
    write_records(d / "train.jsonl", recs[:4])
    write_records(d / "val.jsonl", recs[4:6])
    write_records(d / "reference.jsonl", recs[6:9])
    write_classes(d / "classes.json", ("a", "b"))
    (d / "source.json").write_text(json.dumps({"image_root": "../../images"}), encoding="utf-8")
    return tmp_path / "data", tmp_path / "runs"


def _cfg(run_id):
    return T.TrainConfig(run_id=run_id, dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0, 1.0),
                         edit=None, policy="uniform", cp_mode="adaptive", cp_p=1.0, pretrained=False,
                         batch_size=2, workers=0, lr_warmup_iters=2, amp=False)


@pytest.mark.slow
def test_adaptive_paste_trains_logs_and_resumes_exactly(data):
    data_root, runs_root = data
    assert T.train(_cfg("full"), data_root, runs_root) == 0
    full = T.read_history(runs_root / "full")
    assert all("paste" in r and "next_probs" in r["paste"] and len(r["paste"]["ref_ap50"]) == 2 for r in full)
    assert full[0]["paste"]["probs"] == [0.5, 0.5]  # balanced before any AP is known
    assert full[1]["paste"]["probs"] == full[0]["paste"]["next_probs"]
    assert sum(full[0]["paste"]["planned"].values()) > 0
    # stop after epoch 1, then resume: same trajectory as the uninterrupted run
    assert T.train(_cfg("split"), data_root, runs_root, deadline=time.monotonic()) == T.STOPPED_AT_DEADLINE
    assert T.train(_cfg("split"), data_root, runs_root, resume=True) == 0
    split = T.read_history(runs_root / "split")
    assert [r["paste"] for r in split] == [r["paste"] for r in full]
    assert [r["metrics"]["macro_ap50"] for r in split] == [r["metrics"]["macro_ap50"] for r in full]
