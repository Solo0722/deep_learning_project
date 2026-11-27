"""Object-local pixel edits (reports/cua_local/PROTOCOL.md): composite, transform, specs, jobs and read-out."""
import json

import numpy as np
import pytest

from cua.analysis.local import interaction_size, selection_gain
from cua.augment.base import base_geometry, base_geometry_matrix
from cua.augment.boxes import hflip_matrix
from cua.augment.local import object_mask
from cua.augment.menu import ArmTransform
from cua.augment.rng import keyed_rng
from cua.experiments.specs import LOCAL_EDITS, local_specs, part_a_specs, resolve
from cua.records import Record, write_classes, write_records
from cua.train import TrainConfig, _validate

from helpers import toy_fetch, toy_sample


def _rec640():
    return Record("img/0.png", 640, 640, np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool), "g", "s")


def _rect(b):
    x0, y0, x1, y1 = b
    return [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)]


def test_object_mask_follows_letterbox_and_matrix():
    rec = _rec640()
    m = object_mask([_rect([100, 200, 300, 400])], rec, np.eye(3), 640)
    assert m[300, 200] == pytest.approx(1.0) and m[50, 50] == 0.0 and m[300, 500] == 0.0
    f = object_mask([_rect([100, 200, 300, 400])], rec, hflip_matrix(640), 640)
    assert f[300, 640 - 200] == pytest.approx(1.0) and f[300, 200] == 0.0
    # a 320x160 image is letterboxed by 2x with 160 px of padding above and below
    small = Record("img/1.png", 320, 160, np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool), "g", "s")
    s = object_mask([_rect([0, 0, 50, 40])], small, np.eye(3), 640)
    assert s[160 + 40, 50] == pytest.approx(1.0) and s[100, 50] == 0.0


def test_base_geometry_matrix_matches_base_geometry():
    s = toy_sample()
    a = base_geometry(s, keyed_rng(0, 3, 7, "base"))
    b, m = base_geometry_matrix(s, keyed_rng(0, 3, 7, "base"))
    assert np.array_equal(a.rgb, b.rgb) and np.allclose(a.boxes, b.boxes)
    corners = np.array([[s.boxes[0][0], s.boxes[0][1], 1.0], [s.boxes[0][2], s.boxes[0][3], 1.0]]) @ m.T
    xs, ys = sorted(corners[:, 0]), sorted(corners[:, 1])
    assert np.allclose([xs[0], ys[0], xs[1], ys[1]], b.boxes[0])


@pytest.mark.parametrize("edit", ["contrast", "noise", "blur"])
def test_object_scope_edits_only_object_pixels_with_the_same_strength(edit):
    s = toy_sample()
    polys = [[_rect(b) for b in s.boxes]]
    whole = ArmTransform("dacl10k", None, seed=0)
    local = ArmTransform("dacl10k", None, seed=0, scope="object", polygons=polys, records=[_rec640()])
    for epoch in range(4):
        base = whole.apply(s, epoch, 0, toy_fetch, 10, None)
        w = whole.apply(s, epoch, 0, toy_fetch, 10, edit)
        out = local.apply(s, epoch, 0, toy_fetch, 10, edit)
        assert np.array_equal(out.boxes, base.boxes) and np.array_equal(out.labels, base.labels)
        inside = np.zeros(s.rgb.shape[:2], bool)
        near = np.zeros(s.rgb.shape[:2], bool)
        for x0, y0, x1, y1 in base.boxes:
            inside[int(y0) + 4:int(y1) - 4, int(x0) + 4:int(x1) - 4] = True
            near[max(0, int(y0) - 4):int(y1) + 5, max(0, int(x0) - 4):int(x1) + 5] = True
        assert np.abs(out.rgb[inside].astype(int) - w.rgb[inside].astype(int)).max() <= 1
        assert np.array_equal(out.rgb[~near], base.rgb[~near])
        assert not np.array_equal(out.rgb, w.rgb)


def test_object_scope_with_no_objects_is_identity_and_coin_matches():
    s = toy_sample(boxes=np.zeros((0, 4)))
    t = ArmTransform("dacl10k", "contrast", seed=1, scope="object", polygons=[[]], records=[_rec640()])
    base = ArmTransform("dacl10k", None, seed=1)
    for epoch in range(6):
        assert np.array_equal(t(s, epoch, 0, toy_fetch, 10).rgb, base(s, epoch, 0, toy_fetch, 10).rgb)


def test_object_scope_validation():
    with pytest.raises(ValueError):
        ArmTransform("dacl10k", "contrast", seed=0, scope="object")
    with pytest.raises(ValueError):
        ArmTransform("dacl10k", "mosaic", seed=0, scope="object", polygons=[[]], records=[_rec640()])
    with pytest.raises(ValueError):
        ArmTransform("dacl10k", "contrast", seed=0, scope="box")
    common = dict(run_id="r", dataset="dacl10k", detector="fasterrcnn", seed=0, epoch_lr=(1.0,))
    with pytest.raises(ValueError):
        _validate(TrainConfig(**common, edit="mosaic", edit_scope="object"))
    with pytest.raises(ValueError):
        _validate(TrainConfig(**common, edit=None, policy="uniform", edit_scope="object"))
    _validate(TrainConfig(**common, edit="hue", edit_scope="object"))


def test_local_specs_and_digests():
    ls = local_specs()
    assert len(ls) == 24 and LOCAL_EDITS == ("brightness", "contrast", "saturation", "hue", "blur", "noise",
                                             "jpeg", "sharpen")
    twins = {a.run_id: a for a in part_a_specs("dacl10k", "fasterrcnn")}
    for s in ls:
        a = twins["A" + s.run_id[1:]]
        assert {f for f in TrainConfig.__dataclass_fields__ if getattr(s, f) != getattr(a, f)} == {"run_id",
                                                                                                    "edit_scope"}
        assert s.init_run == a.init_run and s.init_run.endswith("-warm")
    assert resolve("L-dacl10k-fasterrcnn-s1-jpeg") == local_specs(seeds=(1,), edits=("jpeg",))[0]
    # Part A runs keep their digests
    a0 = part_a_specs("dacl10k", "fasterrcnn", (0,))
    assert [s.digest()[:12] for s in a0[:3]] == ["572fa110d1c6", "ff84440f8e6b", "d32141ef901b"]
    with pytest.raises(ValueError):
        local_specs("rdd")
    with pytest.raises(ValueError):
        local_specs(edits=("mosaic",))


def test_make_jobs_local_family_submits_without_warm(tmp_path):
    from cua.ccast.make_jobs import make_jobs
    submit = make_jobs(local_specs(seeds=(0,), edits=("hue", "blur")), tmp_path, project="/p",
                       external_init=True).read_text()
    assert "qsub L-dacl10k-fasterrcnn-s0-hue.pbs" in submit and "depend" not in submit
    with pytest.raises(ValueError):
        make_jobs(local_specs(seeds=(0,), edits=("hue",)), tmp_path / "x", project="/p")


def _rows(fn, seeds=(0, 1, 2), classes=("a", "b", "c", "d"), edits=("e1", "e2", "e3")):
    rng = np.random.default_rng(0)
    return [{"dataset": "dacl10k", "detector": "fasterrcnn", "seed": k, "edit": e, "cls": c,
             "delta": fn(e, c) + rng.normal(0, 0.05)} for k in seeds for e in edits for c in classes]


def test_selection_gain_and_interaction_size():
    best = {"a": "e1", "b": "e2", "c": "e3", "d": "e1"}
    specific = _rows(lambda e, c: 1.0 if best[c] == e else -1.0)
    g = selection_gain(specific)
    assert g["all_positive"] and g["mean"] > 1.0 and len(g["folds"]) == 3
    assert interaction_size(specific) > 0.5
    shared = _rows(lambda e, c: {"e1": 0.5, "e2": 0.0, "e3": -0.5}[e])
    s = selection_gain(shared)
    assert abs(s["mean"]) < 0.1 and interaction_size(shared) < 0.1


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
def test_trainer_runs_object_local_edit(tmp_path, monkeypatch):
    import torch
    from PIL import Image
    import cua.train as T
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(0)
    (tmp_path / "images" / "img").mkdir(parents=True)
    recs = []
    for i in range(7):
        Image.fromarray(rng.integers(20, 200, size=(64, 96, 3), dtype=np.uint8)).save(tmp_path / "images" / "img" / f"{i}.png")
        recs.append(Record(f"img/{i}.png", 96, 64, np.array([[10.0, 10.0, 50.0, 40.0]]), np.array([i % 2]),
                           np.array([False]), f"g{i}", "s"))
    d = tmp_path / "data" / "voc"
    d.mkdir(parents=True)
    write_records(d / "train.jsonl", recs[:4])
    write_records(d / "val.jsonl", recs[4:7])
    write_classes(d / "classes.json", ("a", "b"))
    (d / "source.json").write_text(json.dumps({"image_root": "../../images"}), encoding="utf-8")
    tri = [[[10, 40], [50, 40], [30, 10]]]
    (d / "train_masks.json").write_text(json.dumps({"version": 1, "masks": {r.image: [tri] for r in recs[:4]}}))
    cfg = T.TrainConfig(run_id="r", dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit="contrast",
                        edit_p=1.0, edit_scope="object", pretrained=False, batch_size=2, workers=0,
                        lr_warmup_iters=2, amp=False)
    assert T.train(cfg, tmp_path / "data", tmp_path / "runs") == 0
    assert json.loads((tmp_path / "runs" / "r" / "config.json").read_text())["config"]["edit_scope"] == "object"
