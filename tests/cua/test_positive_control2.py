"""Positive control 2: plant_mosaic (mosaic minus the anchor tile's c* boxes), its specs, jobs and read-out."""
import json
from pathlib import Path

import numpy as np
import pytest

from cua.augment.menu import PLANT, PLANT_MOSAIC, ArmTransform, planted_menu, valid_menu
from cua.augment.rng import keyed_rng
from cua.experiments.specs import CONTROL2_ARMS, control2_specs, part_c_specs, resolve
from cua.policy import UniformPolicy, choose
from cua.records import Record, write_classes, write_records
from cua.train import TrainConfig, _validate, resolve_plant

from helpers import toy_fetch, toy_sample

RUNS = Path(__file__).resolve().parents[2] / "results" / "cua_runs"


def _rec(i, labels):
    n = len(labels)
    return Record(f"img/{i}.png", 96, 64, np.tile([[10.0, 10.0, 50.0, 40.0]], (n, 1)), np.array(labels, dtype=np.int64),
                  np.zeros(n, bool), f"g{i}", "s")


def test_planted_menu_puts_plant_mosaic_in_mosaics_slot():
    m = planted_menu("voc", "mosaic")
    v = valid_menu("voc")
    assert len(m) == len(v) and "mosaic" not in m and m.index(PLANT_MOSAIC) == v.index("mosaic")
    assert planted_menu("voc", "drop") == v + (PLANT,)
    with pytest.raises(ValueError):
        planted_menu("voc", "zzz")


def test_uniform_draws_are_the_same_as_without_the_plant():
    m, v = planted_menu("voc", "mosaic"), valid_menu("voc")
    for i in range(300):
        a = choose(UniformPolicy(v), frozenset({1}), keyed_rng(0, 3, i, "policy"), v)
        b = choose(UniformPolicy(m), frozenset({1}), keyed_rng(0, 3, i, "policy"), m)
        assert b == (PLANT_MOSAIC if a == "mosaic" else a)


def test_plant_mosaic_is_mosaic_without_the_anchors_target_boxes():
    s = toy_sample(boxes=[[100, 100, 300, 300], [350, 350, 600, 600], [50, 400, 250, 620]])
    s.labels[:] = [2, 0, 2]
    t = ArmTransform("voc", None, seed=4, plant_class=2, plant_kind="mosaic")
    assert t.menu == planted_menu("voc", "mosaic")
    with pytest.raises(ValueError):  # clean mosaic is no longer in the menu
        t.apply(s, 1, 0, toy_fetch, 50, "mosaic")
    seen = 0
    for idx in range(6):
        clean = ArmTransform("voc", None, seed=4).apply(s, 1, idx, toy_fetch, 50, "mosaic")
        out = t.apply(s, 1, idx, toy_fetch, 50, PLANT_MOSAIC)
        assert np.array_equal(out.rgb, clean.rgb)  # same tiles, same centre, pixels unchanged
        # toy_fetch tiles carry label 0 only, so every label-2 box comes from the anchor and must be gone
        seen += int(2 in clean.labels.tolist())
        assert 2 not in out.labels.tolist()
        keep = clean.labels != 2
        assert np.array_equal(out.boxes, clean.boxes[keep]) and np.array_equal(out.labels, clean.labels[keep])
    assert seen >= 4  # the anchor's class-2 boxes were visible in the mosaic in most draws


def test_plant_mosaic_keeps_target_boxes_from_the_other_tiles():
    def fetch(j):  # other tiles: one box of class 2 each
        x = toy_fetch(j)
        x.labels[:] = 2
        return x
    s = toy_sample(boxes=[[100, 100, 300, 300]])
    s.labels[:] = [0]
    t = ArmTransform("voc", None, seed=1, plant_class=2, plant_kind="mosaic")
    clean = ArmTransform("voc", None, seed=1).apply(s, 2, 7, fetch, 50, "mosaic")
    out = t.apply(s, 2, 7, fetch, 50, PLANT_MOSAIC)
    # the anchor has no class 2: plant_mosaic is exactly mosaic
    assert np.array_equal(out.rgb, clean.rgb) and np.array_equal(out.boxes, clean.boxes)
    assert np.array_equal(out.labels, clean.labels) and (out.labels == 2).sum() >= 1


def test_resolve_plant_second():
    classes = ("a", "b", "c", "d")
    recs = [_rec(0, [1, 1, 1, 2]), _rec(1, [2, 0]), _rec(2, [3, 3])]
    assert resolve_plant("frequent", classes, recs) == 1
    assert resolve_plant("second", classes, recs) == 2  # c: 2 and d: 2 tie -> lowest index


def test_validate_plant_kind():
    base = dict(run_id="r", dataset="voc", detector="retinanet", seed=0, epoch_lr=(1.0,), edit=None, policy="uniform")
    with pytest.raises(ValueError):
        _validate(TrainConfig(**base, plant_kind="mosaic"))  # no planted class
    with pytest.raises(ValueError):
        _validate(TrainConfig(**base, plant="second", plant_kind="zzz"))
    _validate(TrainConfig(**base, plant="second", plant_kind="mosaic"))


def test_control2_specs_and_digests():
    ks = control2_specs()
    cs = part_c_specs("voc", "retinanet", (0, 1, 2), CONTROL2_ARMS)
    assert [k.run_id for k in ks] == [c.run_id.replace("C-", "K2-", 1) for c in cs]
    for k, c in zip(ks, cs):
        assert k.plant == "second" and k.plant_kind == "mosaic"
        assert {f for f in TrainConfig.__dataclass_fields__ if getattr(k, f) != getattr(c, f)} == \
            {"run_id", "plant", "plant_kind"}
        assert resolve(k.run_id) == k
        assert k.digest() != c.digest()
    with pytest.raises(ValueError):
        control2_specs(arms=("cua",))
    # existing C- runs keep the digests recorded when they were trained
    for c in cs:
        cfg = RUNS / c.run_id / "config.json"
        if cfg.exists():
            assert json.loads(cfg.read_text(encoding="utf-8"))["digest"] == c.digest(), c.run_id


def test_make_jobs_control2_and_confirm_seed_guard(tmp_path):
    from cua.ccast.make_jobs import main
    out = tmp_path / "jobs"
    assert main(["--family", "control2", "--dataset", "voc", "--detector", "retinanet", "--out", str(out)]) == 0
    sub = (out / "submit.sh").read_text()
    assert sub.count("qsub") == 9 and "qsub K2-voc-retinanet-s2-cua2.pbs" in sub
    with pytest.raises(SystemExit):
        main(["--family", "control2", "--dataset", "dacl10k", "--detector", "fasterrcnn", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        main(["--family", "confirm", "--dataset", "dacl10k", "--detector", "fasterrcnn", "--seeds", "2,3",
              "--out", str(tmp_path / "y")])


def test_test_eval_accepts_k2(tmp_path):
    from cua.test_eval import evaluate_run
    d = tmp_path / "K2-voc-retinanet-s0-cua2"
    d.mkdir()
    (d / "COMPLETE.json").write_text("{}")
    (d / "config.json").write_text(json.dumps({"digest": "wrong"}))
    with pytest.raises(RuntimeError, match="digest"):  # past the prefix check, stopped by the digest check
        evaluate_run(d, tmp_path)


# ---- read-out on a synthetic world ----

CLASSES = ("a", "b", "c")  # c* = "b"


def _world(tmp_path, cua_misses_target: bool):
    """Test set of 40 images, one box each (classes cycle a, b, c). Every run detects every box with a
    random score, except that K2-shared2 (and K2-cua2 when cua_misses_target) misses half the class-b boxes."""
    from cua.analysis.ap50 import _macro, image_matches
    data = tmp_path / "data" / "voc"
    data.mkdir(parents=True)
    write_classes(data / "classes.json", CLASSES)
    test = [Record(f"t/{i}.png", 100, 100, np.array([[10.0, 10.0, 60.0, 60.0]]), np.array([i % 3]),
                   np.array([False]), f"g{i}", "s") for i in range(40)]
    write_records(data / "test.jsonl", test)
    train = [_rec(i, [i % 3]) for i in range(30)] + [_rec(30 + i, []) for i in range(5)]
    write_records(data / "train.jsonl", train)
    runs = tmp_path / "runs"
    rng = np.random.default_rng(0)
    menu = planted_menu("voc", "mosaic")
    for kind in ("C", "K2"):
        for arm in CONTROL2_ARMS:
            for s in (0, 1, 2):
                d = runs / f"{kind}-voc-retinanet-s{s}-{arm}"
                d.mkdir(parents=True)
                miss = kind == "K2" and (arm == "shared2" or (arm == "cua2" and cua_misses_target))
                preds = []
                for i, r in enumerate(test):
                    hit = not (miss and r.labels[0] == 1 and i % 2 == 0)
                    preds.append({"image_index": i, "boxes": [r.boxes[0].tolist()] if hit else [],
                                  "scores": [float(rng.uniform(0.5, 1))] if hit else [],
                                  "labels": [int(r.labels[0])] if hit else []})
                    preds[-1]["boxes"].append([70.0, 70.0, 95.0, 95.0])  # a false positive per image
                    preds[-1]["scores"].append(float(rng.uniform(0, 0.6)))
                    preds[-1]["labels"].append(int(rng.integers(0, 3)))
                (d / "test_predictions.json").write_text(json.dumps(preds))
                m = image_matches(test, preds, 3)
                (d / "test_metrics.json").write_text(json.dumps({"macro_ap50": _macro(m, np.arange(40))}))
                (d / "config.json").write_text(json.dumps({"plant_class": "b", "max_train_images": None}))
                (d / "COMPLETE.json").write_text("{}")
                if kind == "K2":
                    by_c = {"b": {PLANT_MOSAIC: 0 if arm == "cua2" else 40, "none": 60},
                            "a": {PLANT_MOSAIC: 50, "none": 50}}
                    hist = {"epoch": 6, "policy": {"edit_counts": {PLANT_MOSAIC: (50 if arm == "cua2" else 90),
                                                                   "none": 110},
                                                   "edit_by_class": by_c}}
                    (d / "history.jsonl").write_text(json.dumps(hist) + "\n")
                    if arm != "uniform":
                        k = menu.index(PLANT_MOSAIC)
                        rows = []
                        for _ in range(30):
                            u = rng.normal(0, 0.01, (len(menu), 3))
                            u[k, 1] -= 0.2
                            rows.append({"u": u.tolist(), "valid": np.ones((len(menu), 3), bool).tolist()})
                        (d / "utility.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return runs, tmp_path / "data"


def test_control2_readout_passes_when_cua_avoids_the_harm(tmp_path, monkeypatch):
    import cua.analysis.control2 as C2
    monkeypatch.setattr(C2, "target_free_draws", lambda recs, menu, seed, epoch, edit: (0, 0))
    runs, data = _world(tmp_path, cua_misses_target=False)
    res = C2.summarize(runs, data, n_boot=200)
    assert res["complete"] and res["target"] == "b"
    assert res["pass"]["O3"] and res["pass"]["O0"] and res["pass"]["O1"] and res["pass"]["O2"]
    assert res["o4"]["t"]["mean"] > 0 and res["o3"]["t"]["mean"] > 10
    C2.write_report(res, tmp_path / "out")
    text = (tmp_path / "out" / "CONTROL2.md").read_text(encoding="utf-8")
    assert "O4 macro" in text and "plant_mosaic" in text


def test_control2_readout_fails_when_cua_suffers_the_harm_too(tmp_path, monkeypatch):
    import cua.analysis.control2 as C2
    monkeypatch.setattr(C2, "target_free_draws", lambda recs, menu, seed, epoch, edit: (0, 0))
    runs, data = _world(tmp_path, cua_misses_target=True)
    res = C2.summarize(runs, data, n_boot=200)
    assert not res["pass"]["O4"] and not res["pass"]["O3"]


def test_control2_readout_reports_missing_runs(tmp_path):
    import cua.analysis.control2 as C2
    data = tmp_path / "data" / "voc"
    data.mkdir(parents=True)
    write_classes(data / "classes.json", CLASSES)
    res = C2.summarize(tmp_path / "runs_empty", tmp_path / "data")
    assert not res["complete"] and len(res["missing_runs"]) == 9
    C2.write_report(res, tmp_path / "out")


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
@pytest.mark.parametrize("policy", ["uniform", "cua2"])
def test_trainer_runs_with_plant_mosaic(tmp_path, monkeypatch, policy):
    import torch
    from PIL import Image
    import cua.train as T
    from cua.experiments.specs import V2_SETTINGS
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
    menu = valid_menu("voc")
    extra = {}
    if policy == "cua2":
        # probe prior: mosaic strongly positive for every class, everything else 0
        mean = np.zeros((len(menu), 2))
        mean[menu.index("mosaic")] = 5.0
        (d / "probe_retinanet.json").write_text(json.dumps({"classes": ["a", "b"], "edits": list(menu),
                                                            "mean": mean.tolist(), "se": np.ones_like(mean).tolist(),
                                                            "n_seeds": 3}))
        extra = {**V2_SETTINGS, "est_every": 1, "est_batch": 4, "ref_batch": 2, "lookahead_steps": 2}
    cfg = T.TrainConfig(run_id="r", dataset="voc", detector="retinanet", seed=0, epoch_lr=(1.0,), edit=None,
                        policy="cua" if policy == "cua2" else "uniform", plant="second", plant_kind="mosaic",
                        pretrained=False, batch_size=2, workers=0, lr_warmup_iters=2, amp=False,
                        est_start_epoch=1, **extra)
    assert T.train(cfg, tmp_path / "data", tmp_path / "runs") == 0
    out = tmp_path / "runs" / "r"
    assert json.loads((out / "config.json").read_text())["plant_class"] == "a"  # b has 7 boxes, a has 3
    hist = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines() if x.strip()]
    counts = hist[-1]["policy"]["edit_counts"]
    assert "mosaic" not in counts
    if policy == "cua2":
        rows = [json.loads(x) for x in (out / "utility.jsonl").read_text().splitlines() if x.strip()]
        assert rows and len(rows[0]["u"]) == len(menu)
