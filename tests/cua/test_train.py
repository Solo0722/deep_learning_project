"""Trainer tests: CPU, pretrained=False, size 320, augment=False, 6 tiny synthetic images."""
import json
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from cua.records import Record, read_records, write_classes, write_records
from cua.train import TrainConfig, full_schedule, read_history, train

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")


def tiny_cfg(**kw):
    base = TrainConfig(run_id="tiny", dataset="tiny", detector="fasterrcnn", seed=0, epoch_lr=(1.0,),
                       edit=None, lr_warmup_iters=2, batch_size=3, amp=True, workers=0,
                       pretrained=False, size=320, augment=False)
    return replace(base, **kw)


def after_first_epoch():
    """A deadline that has already passed: the run's very first epoch is never deadline-checked,
    so exactly one epoch runs and the check before the second epoch stops the run."""
    return time.monotonic() - 1.0


@pytest.fixture
def tiny_data(tmp_path):
    """data_root/tiny/{train(3),val(2)}.jsonl over 6 images (one unused); garbage test/reference."""
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    (images / "img").mkdir(parents=True)
    recs = []
    for i in range(6):
        w, h = 96, 64
        rgb = rng.integers(20, 200, size=(h, w, 3), dtype=np.uint8)
        rgb[10:40, 10 + i:50 + i] = (230, 30, 30)
        Image.fromarray(rgb).save(images / "img" / f"{i}.png")
        boxes = [[10.0 + i, 10.0, 50.0 + i, 40.0], [55.0, 20.0, 90.0, 60.0]]
        recs.append(Record(f"img/{i}.png", w, h, np.array(boxes), np.array([0, 1]),
                           np.array([False, i == 2]), f"g{i}", "s"))
    data_root = tmp_path / "data"
    d = data_root / "tiny"
    d.mkdir(parents=True)
    write_records(d / "train.jsonl", recs[:3])
    write_records(d / "val.jsonl", recs[3:5])
    (d / "test.jsonl").write_text("this is not json\n", encoding="utf-8")
    (d / "reference.jsonl").write_text("{garbage\n", encoding="utf-8")
    write_classes(d / "classes.json", ("a", "b"))
    (d / "source.json").write_text(json.dumps({"image_root": "../../images"}), encoding="utf-8")
    runs_root = tmp_path / "runs"
    return data_root, runs_root


def _state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


# ---- brief tests ----

def test_full_schedule_20():
    assert full_schedule(20) == (1.0,) * 12 + (0.1,) * 4 + (0.01,) * 4


def test_train_writes_history_and_complete(tiny_data):
    assert train(tiny_cfg(epoch_lr=(1.0, 1.0)), *tiny_data) == 0
    assert [h["epoch"] for h in read_history(tiny_data[1] / "tiny")] == [1, 2]
    assert (tiny_data[1] / "tiny" / "COMPLETE.json").exists()
    # extra: artefacts and history content
    run = tiny_data[1] / "tiny"
    for name in ("config.json", "last.pt", "best.pt", "val_predictions_last.json",
                 "val_predictions_best.json", "history.jsonl"):
        assert (run / name).exists(), name
    hist = read_history(run)
    for h in hist:
        assert set(h) >= {"epoch", "lr_mult", "losses", "metrics", "seconds"}
        assert "macro_ap50" in h["metrics"] and "per_class" in h["metrics"]
        assert "total" in h["losses"] and all(np.isfinite(v) for v in h["losses"].values())
    preds = json.loads((run / "val_predictions_last.json").read_text())
    assert sorted(p["image_index"] for p in preds) == [0, 1]
    complete = json.loads((run / "COMPLETE.json").read_text())
    assert complete["final_epoch"] == 2
    cfg_json = json.loads((run / "config.json").read_text())
    assert cfg_json["digest"] == tiny_cfg(epoch_lr=(1.0, 1.0)).digest()
    ck = _state(run / "last.pt")
    assert ck["epoch"] == 2 and ck["cfg_digest"] == cfg_json["digest"] and ck["global_iter"] == 2
    assert set(ck) >= {"model", "optimizer", "scaler", "epoch", "global_iter", "cfg_digest", "best_macro"}


def test_refuses_overwrite_and_resume_of_complete(tiny_data):
    train(tiny_cfg(epoch_lr=(1.0,)), *tiny_data)
    with pytest.raises(RuntimeError):
        train(tiny_cfg(epoch_lr=(1.0,)), *tiny_data)
    with pytest.raises(RuntimeError):
        train(tiny_cfg(epoch_lr=(1.0,)), *tiny_data, resume=True)


def test_resume_rejects_changed_config(tiny_data):
    assert train(tiny_cfg(epoch_lr=(1.0, 1.0)), *tiny_data, deadline=after_first_epoch()) == 3
    with pytest.raises(RuntimeError):
        train(tiny_cfg(epoch_lr=(1.0, 0.1)), *tiny_data, resume=True)


def test_deadline_stop_then_resume_continues(tiny_data):
    assert train(tiny_cfg(epoch_lr=(1.0, 1.0, 1.0)), *tiny_data, deadline=after_first_epoch()) == 3
    assert train(tiny_cfg(epoch_lr=(1.0, 1.0, 1.0)), *tiny_data, resume=True) == 0
    assert [h["epoch"] for h in read_history(tiny_data[1] / "tiny")] == [1, 2, 3]


# ---- controller-requested tests ----

def test_resume_is_equivalent_to_uninterrupted_run(tiny_data):
    data_root, runs_root = tiny_data
    cfg = tiny_cfg(epoch_lr=(1.0, 1.0, 0.1))
    assert train(cfg, data_root, runs_root / "straight") == 0
    split = runs_root / "split"
    assert train(cfg, data_root, split, deadline=after_first_epoch()) == 3
    assert [h["epoch"] for h in read_history(split / "tiny")] == [1]
    assert train(cfg, data_root, split, resume=True) == 0
    a, b = read_history(runs_root / "straight" / "tiny"), read_history(split / "tiny")
    assert [h["epoch"] for h in a] == [h["epoch"] for h in b] == [1, 2, 3]
    for ha, hb in zip(a, b):
        assert ha["lr_mult"] == hb["lr_mult"]
        assert ha["losses"].keys() == hb["losses"].keys()
        for k in ha["losses"]:
            assert ha["losses"][k] == pytest.approx(hb["losses"][k], abs=1e-6), k
        assert json.dumps(ha["metrics"], sort_keys=True) == json.dumps(hb["metrics"], sort_keys=True)
    ma = _state(runs_root / "straight" / "tiny" / "last.pt")["model"]
    mb = _state(split / "tiny" / "last.pt")["model"]
    assert ma.keys() == mb.keys()
    for k in ma:
        assert torch.equal(ma[k], mb[k]), k


def test_resume_truncates_history_beyond_last_checkpoint(tiny_data):
    data_root, runs_root = tiny_data
    cfg = tiny_cfg(epoch_lr=(1.0, 1.0))
    assert train(cfg, data_root, runs_root, deadline=after_first_epoch()) == 3
    hist_path = runs_root / "tiny" / "history.jsonl"
    stale = dict(read_history(runs_root / "tiny")[0], epoch=2, stale=True)
    with hist_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stale) + "\n")
    # resumed first epoch is deadline-checked (duration from history); a far deadline lets it run
    assert train(cfg, data_root, runs_root, resume=True, deadline=time.monotonic() + 1e6) == 0
    hist = read_history(runs_root / "tiny")
    assert [h["epoch"] for h in hist] == [1, 2] and "stale" not in hist[1]


def test_resume_without_checkpoint_starts_fresh_or_raises(tiny_data):
    data_root, runs_root = tiny_data
    cfg = tiny_cfg(epoch_lr=(1.0,))
    (runs_root / "tiny").mkdir(parents=True)  # empty: killed before the first checkpoint
    assert train(cfg, data_root, runs_root, resume=True) == 0
    other = runs_root / "other"
    other.mkdir()
    (other / "unrelated.txt").write_text("x")
    with pytest.raises(RuntimeError):
        train(tiny_cfg(run_id="other"), data_root, runs_root, resume=True)


def _record_lrs(monkeypatch):
    lrs = []
    orig = torch.optim.SGD.step

    def spy(self, *a, **k):
        lrs.append(self.param_groups[0]["lr"])
        return orig(self, *a, **k)

    monkeypatch.setattr(torch.optim.SGD, "step", spy)
    return lrs


def test_warmup_starts_at_one_thousandth(tiny_data, monkeypatch):
    lrs = _record_lrs(monkeypatch)
    cfg = tiny_cfg(epoch_lr=(0.5,), batch_size=1, lr_warmup_iters=4, base_lr=0.01)
    assert train(cfg, *tiny_data, max_train_images=2) == 0
    assert len(lrs) == 2
    assert lrs[0] == pytest.approx(0.01 * 0.5 * 0.001)
    assert lrs[1] == pytest.approx(0.01 * 0.5 * (0.001 + 0.999 * 1 / 4))


def test_init_run_loads_warm_weights_and_skips_warmup(tiny_data, monkeypatch):
    data_root, runs_root = tiny_data
    assert train(tiny_cfg(run_id="warm", epoch_lr=(1.0,)), data_root, runs_root) == 0
    warm = _state(runs_root / "warm" / "last.pt")["model"]

    import cua.train as T
    seen = {}
    orig = T._train_one_epoch

    def spy(model, *a, **k):
        if "state" not in seen:
            seen["state"] = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        return orig(model, *a, **k)

    monkeypatch.setattr(T, "_train_one_epoch", spy)
    lrs = _record_lrs(monkeypatch)
    arm = tiny_cfg(run_id="arm", epoch_lr=(0.1,), init_run="warm", first_epoch=2, lr_warmup_iters=1000)
    assert train(arm, data_root, runs_root) == 0
    assert seen["state"].keys() == warm.keys()
    for k in warm:
        assert torch.equal(seen["state"][k], warm[k]), k
    assert lrs == [pytest.approx(arm.base_lr * 0.1)]  # no warm-up
    assert [h["epoch"] for h in read_history(runs_root / "arm")] == [2]
    assert _state(runs_root / "arm" / "last.pt")["global_iter"] == 1


def test_init_run_requires_complete_and_matching_config(tiny_data):
    data_root, runs_root = tiny_data
    (runs_root / "warm").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        train(tiny_cfg(run_id="arm", init_run="warm", first_epoch=2), data_root, runs_root)
    assert not (runs_root / "arm" / "last.pt").exists()


def test_init_run_rejects_mismatched_architecture(tiny_data):
    data_root, runs_root = tiny_data
    warm_cfg = tiny_cfg(run_id="warm")
    run = runs_root / "warm"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"config": {**json.loads(json.dumps(
        warm_cfg.__dict__)), "detector": "retinanet"}, "digest": "x"}))
    (run / "COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError):
        train(tiny_cfg(run_id="arm", init_run="warm", first_epoch=2), data_root, runs_root)


def test_edit_without_augment_raises(tiny_data):
    with pytest.raises(ValueError):
        train(tiny_cfg(edit="translate"), *tiny_data)


def test_augment_requires_size_640(tiny_data):
    with pytest.raises(ValueError):
        train(tiny_cfg(augment=True), *tiny_data)


def test_never_reads_test_or_reference_jsonl(tiny_data):
    data_root, runs_root = tiny_data
    with pytest.raises(Exception):
        read_records(data_root / "tiny" / "test.jsonl")
    with pytest.raises(Exception):
        read_records(data_root / "tiny" / "reference.jsonl")
    assert train(tiny_cfg(epoch_lr=(1.0,)), data_root, runs_root) == 0


def test_digest_is_stable_and_sensitive():
    a, b = tiny_cfg(), tiny_cfg()
    assert a.digest() == b.digest() and len(a.digest()) == 64
    assert tiny_cfg(seed=1).digest() != a.digest()
