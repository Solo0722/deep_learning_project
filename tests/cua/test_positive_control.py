"""Positive control: the planted edit (deletes one class's annotations), its specs and its read-out."""
import json

import numpy as np
import pytest

from cua.analysis.control import plant_rates, plant_utility
from cua.augment.menu import PLANT, ArmTransform, drop_class, valid_menu
from cua.experiments.specs import control_specs, part_c_specs, resolve
from cua.records import Record, write_classes, write_records
from cua.train import TrainConfig, _validate, resolve_plant

from helpers import toy_fetch, toy_sample


def _rec(i, labels, ignore=None):
    n = len(labels)
    return Record(f"img/{i}.png", 96, 64, np.tile([[10.0, 10.0, 50.0, 40.0]], (n, 1)), np.array(labels, dtype=np.int64),
                  np.zeros(n, bool) if ignore is None else np.array(ignore), f"g{i}", "s")


def test_drop_class_removes_only_target_boxes_and_keeps_pixels():
    s = toy_sample(boxes=[[10, 10, 50, 50], [60, 60, 90, 90], [100, 100, 150, 150]])
    s.labels[:] = [2, 0, 2]
    out = drop_class(s, 2)
    assert out.labels.tolist() == [0] and out.boxes.tolist() == [[60, 60, 90, 90]] and out.ignore.tolist() == [False]
    assert np.array_equal(out.rgb, s.rgb)
    same = drop_class(s, 1)
    assert np.array_equal(same.boxes, s.boxes) and np.array_equal(same.labels, s.labels)


def test_arm_transform_accepts_plant_only_when_planted():
    s = toy_sample()  # labels 0, 1
    with pytest.raises(ValueError):
        ArmTransform("dacl10k", None, seed=0).apply(s, 1, 0, toy_fetch, 10, PLANT)
    t = ArmTransform("dacl10k", None, seed=0, plant_class=1)
    assert t.menu == valid_menu("dacl10k") + (PLANT,)
    out = t.apply(s, 1, 0, toy_fetch, 10, PLANT)
    base = t.apply(s, 1, 0, toy_fetch, 10, None)
    assert out.labels.tolist() == [0] and np.array_equal(out.rgb, base.rgb)
    assert np.array_equal(out.boxes, base.boxes[base.labels == 0])


def test_resolve_plant():
    classes = ("a", "b", "c")
    recs = [_rec(0, [1, 1, 2]), _rec(1, [2, 0]), _rec(2, [1], ignore=[True])]
    assert resolve_plant("none", classes, recs) is None
    assert resolve_plant("frequent", classes, recs) == 1  # b: 2 counted (one ignored), c: 2 -> tie, lowest index
    assert resolve_plant("c", classes, recs) == 2
    with pytest.raises(ValueError):
        resolve_plant("zzz", classes, recs)


def test_control_specs_and_digests():
    ks = control_specs("dacl10k", "fasterrcnn", (0,))
    cs = part_c_specs("dacl10k", "fasterrcnn", (0,), ("uniform", "shared", "cua", "shared2", "cua2"))
    assert [k.run_id for k in ks] == [c.run_id.replace("C-", "K-", 1) for c in cs]
    for k, c in zip(ks, cs):
        assert k.plant == "frequent"
        assert {f for f in TrainConfig.__dataclass_fields__ if getattr(k, f) != getattr(c, f)} == {"run_id", "plant"}
    assert resolve("K-dacl10k-fasterrcnn-s2-cua2") == control_specs("dacl10k", "fasterrcnn", (2,), ("cua2",))[0]
    # existing runs keep their digests
    assert cs[0].digest()[:12] == "2ad52d017282"
    with pytest.raises(ValueError):
        control_specs("dacl10k", "fasterrcnn", (0,), ("measured",))


def test_plant_needs_an_edit_policy():
    cfg = TrainConfig(run_id="r", dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit=None,
                      policy="base", plant="frequent")
    with pytest.raises(ValueError):
        _validate(cfg)


def test_plant_rates():
    hist = [{"epoch": 4, "policy": {"edit_counts": {"plant": 100, "none": 100}, "edit_by_class": {"a": {"plant": 50}}}},
            {"epoch": 5, "policy": {"edit_counts": {"plant": 30, "none": 170, "blur": 100},
                                    "edit_by_class": {"a": {"plant": 2, "none": 98}}}}]
    r = plant_rates(hist, "a")
    assert r["n_with"] == 100 and r["n_without"] == 200
    assert r["rate_with"] == pytest.approx(0.02) and r["rate_without"] == pytest.approx(28 / 200)
    assert r["ratio"] == pytest.approx(0.02 / 0.14)


def test_plant_utility_finds_the_planted_cell():
    rng = np.random.default_rng(0)
    menu = ["blur", PLANT]
    rows = []
    for _ in range(40):
        u = rng.normal(0, 0.1, (2, 3))
        u[1, 2] -= 1.0
        rows.append({"u": u.tolist(), "valid": np.ones((2, 3), bool).tolist()})
    rows.append({"u": np.zeros((2, 3)).tolist(), "valid": np.ones((2, 3), bool).tolist(), "skipped": True})
    p = plant_utility(rows, menu, 2)
    assert p["n_estimates"] == 40 and p["target_is_min"]
    assert p["t_target"] < -10 and p["contrast"] < -0.8
    q = plant_utility(rows, menu, 0)
    assert not q["target_is_min"] and q["contrast"] > 0


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
@pytest.mark.parametrize("policy", ["uniform", "cua"])
def test_trainer_runs_with_planted_edit(tmp_path, monkeypatch, policy):
    import torch
    from PIL import Image
    import cua.train as T
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(0)
    (tmp_path / "images" / "img").mkdir(parents=True)
    recs = []
    for i in range(9):
        Image.fromarray(rng.integers(20, 200, size=(64, 96, 3), dtype=np.uint8)).save(tmp_path / "images" / "img" / f"{i}.png")
        recs.append(Record(f"img/{i}.png", 96, 64, np.array([[10.0, 10.0, 50.0, 40.0], [55.0, 20.0, 90.0, 60.0]]),
                           np.array([1, i % 2]), np.array([False, False]), f"g{i}", "s"))
    d = tmp_path / "data" / "voc"
    d.mkdir(parents=True)
    write_records(d / "train.jsonl", recs[:5])
    write_records(d / "val.jsonl", recs[5:7])
    write_records(d / "reference.jsonl", recs[7:9])
    write_classes(d / "classes.json", ("a", "b"))
    (d / "source.json").write_text(json.dumps({"image_root": "../../images"}), encoding="utf-8")
    cfg = T.TrainConfig(run_id="r", dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit=None,
                        policy=policy, plant="frequent", pretrained=False, batch_size=2, workers=0,
                        lr_warmup_iters=2, amp=False, est_every=1, est_batch=2, ref_batch=2, est_start_epoch=1)
    assert T.train(cfg, tmp_path / "data", tmp_path / "runs") == 0
    out = tmp_path / "runs" / "r"
    assert json.loads((out / "config.json").read_text())["plant_class"] == "b"
    hist = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines() if x.strip()]
    assert "edit_by_class" in hist[-1]["policy"]
    if policy == "cua":
        rows = [json.loads(x) for x in (out / "utility.jsonl").read_text().splitlines() if x.strip()]
        assert rows and len(rows[0]["u"]) == len(valid_menu("voc")) + 1


class _FixedPlantPolicy:
    """Plants with probability q on every image (class-blind), or never on images with class 0 (avoid=True)."""

    def __init__(self, q, avoid=False):
        self.q, self.avoid = q, avoid

    def probs(self, class_set):
        q = 0.0 if self.avoid and 0 in class_set else self.q
        return (None, PLANT), np.array([1.0 - q, q])


def _replayed_history(records, policy, menu, seed, epochs):
    from cua.policy_sampler import PolicyBatchSampler
    hist = []
    for ep in epochs:
        s = PolicyBatchSampler(records, lambda: policy, 32, list(range(len(records))), seed, ep, menu, ("a", "b"))
        for _ in s:
            pass
        hist.append({"epoch": ep, "policy": s.stats()})
    return hist


@pytest.mark.parametrize("avoid", [False, True])
def test_plant_rates_removes_target_free_images_exactly(avoid):
    from cua.analysis.control import target_free_draws
    menu = ("blur", PLANT)
    recs = [_rec(i, [0]) for i in range(600)] + [_rec(600 + i, [1]) for i in range(600)] + \
           [_rec(1200 + i, []) for i in range(600)]
    seed, epochs = 3, range(5, 11)
    hist = _replayed_history(recs, _FixedPlantPolicy(0.05, avoid), menu, seed, epochs)
    r = plant_rates(hist, "a", free_draws=lambda ep: target_free_draws(recs, menu, seed, ep))
    assert r["n_target_free"] == 600 * len(epochs) and r["n_without"] == 600 * len(epochs)
    # target-free images draw the plant at 0.5 / len(menu) = 0.25, which drags the uncorrected ratio far below 1
    assert r["ratio_incl_target_free"] < 0.4
    if avoid:
        assert r["rate_with"] == 0.0 and r["ratio"] == 0.0
    else:
        assert 0.75 < r["ratio"] < 1.33 and r["rate_without"] == pytest.approx(0.05, abs=0.015)


def test_plant_rates_rejects_inconsistent_replay():
    hist = [{"epoch": 5, "policy": {"edit_counts": {"plant": 1, "none": 9}, "edit_by_class": {"a": {"none": 5}}}}]
    with pytest.raises(ValueError, match="target-free"):
        plant_rates(hist, "a", free_draws=lambda ep: (6, 0))
