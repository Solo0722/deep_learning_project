"""One-time test evaluation of Part C runs (cua.test_eval)."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

import cua.test_eval as TE
from cua.records import Record, write_classes, write_records
from cua.test_eval import evaluate_run, fp_images, main
from cua.train import TrainConfig, train

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
RID = "C-voc-fasterrcnn-s0-base"


def _rec(i, boxes, labels):
    return Record(f"img/{i}.png", 96, 64, np.array(boxes, dtype=np.float64).reshape(-1, 4),
                  np.array(labels, dtype=np.int64), np.zeros(len(labels), dtype=bool), f"g{i}", "s")


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    (images / "img").mkdir(parents=True)
    for i in range(9):
        rgb = rng.integers(20, 200, size=(64, 96, 3), dtype=np.uint8)
        rgb[10:40, 10 + i:50 + i] = (230, 30, 30)
        Image.fromarray(rgb).save(images / "img" / f"{i}.png")
    recs = [_rec(i, [[10.0 + i, 10.0, 50.0 + i, 40.0], [55.0, 20.0, 90.0, 60.0]], [0, 1]) for i in range(9)]
    d = tmp_path / "data" / "voc"
    d.mkdir(parents=True)
    write_records(d / "train.jsonl", recs[:3])
    write_records(d / "val.jsonl", recs[3:5])
    write_records(d / "reference.jsonl", recs[5:6])
    write_records(d / "test.jsonl", [recs[6], recs[7], _rec(8, [], [])])
    write_classes(d / "classes.json", ("a", "b"))
    (d / "source.json").write_text(json.dumps({"image_root": "../../images"}), encoding="utf-8")
    return tmp_path / "data", tmp_path / "runs"


def _cfg():
    return TrainConfig(run_id=RID, dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit=None,
                       lr_warmup_iters=2, batch_size=2, amp=True, workers=0, pretrained=False, size=640,
                       augment=True, policy="base")


def _stub_run(runs, rid, digest="0" * 64, complete=True, metrics=None):
    d = runs / rid
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"digest": digest}), encoding="utf-8")
    if complete:
        (d / "COMPLETE.json").write_text("{}", encoding="utf-8")
    torch.save({"model": {}, "epoch": 1, "best_macro": 0.0, "best_epoch": 1}, d / "best.pt")
    if metrics is not None:
        (d / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return d


@pytest.mark.slow
def test_evaluates_best_checkpoint_once(data, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data_root, runs = data
    assert train(_cfg(), data_root, runs) == 0
    monkeypatch.setattr(TE, "resolve", lambda rid: _cfg())
    m = evaluate_run(runs / RID, data_root, device="cpu", workers=0)
    assert (runs / RID / "test_predictions.json").exists() and (runs / RID / "test_metrics.json").exists()
    assert m["best_epoch"] == 1 and m["n_test_images"] == 3 and m["run_id"] == RID
    assert m["fp_images"]["n_target_free"] == 1 and "macro_ap50" in m


def test_skips_already_evaluated_run(tmp_path, capsys):
    d = _stub_run(tmp_path / "runs", RID, metrics={"sentinel": 1})
    before = (d / "test_metrics.json").read_bytes()
    assert main(["--runs-root", str(tmp_path / "runs"), "--data-root", str(tmp_path)]) == 0
    assert (d / "test_metrics.json").read_bytes() == before
    assert "already evaluated" in capsys.readouterr().out


def test_refuses_non_part_c_runs(tmp_path):
    d = _stub_run(tmp_path / "runs", "T-rdd-fasterrcnn-s0-default")
    with pytest.raises(ValueError, match="Part C"):
        evaluate_run(d, tmp_path, device="cpu")
    with pytest.raises(SystemExit) as e:
        main(["--runs-root", str(tmp_path / "runs"), "--data-root", str(tmp_path), "--prefix", "T-"])
    assert e.value.code == 2


def test_refuses_digest_mismatch(tmp_path, capsys):
    runs = tmp_path / "runs"
    d = _stub_run(runs, RID)
    _stub_run(runs, "C-voc-fasterrcnn-s1-base", metrics={"sentinel": 1})
    with pytest.raises(RuntimeError, match="digest"):
        evaluate_run(d, tmp_path, device="cpu")
    assert main(["--runs-root", str(runs), "--data-root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "already evaluated" in out and "refused" in out
    assert not (d / "test_metrics.json").exists()


def test_incomplete_run_is_reported_not_evaluated(tmp_path, capsys):
    d = _stub_run(tmp_path / "runs", RID, complete=False)
    assert main(["--runs-root", str(tmp_path / "runs"), "--data-root", str(tmp_path)]) == 0
    assert "incomplete" in capsys.readouterr().out
    assert not (d / "test_metrics.json").exists() and not (d / "test_predictions.json").exists()


def test_fp_images_none_without_target_free_images():
    recs = [_rec(0, [[1, 1, 5, 5]], [0])]
    preds = [{"image_index": 0, "boxes": [[1, 1, 5, 5]], "scores": [0.9], "labels": [0]}]
    assert fp_images(recs, preds) == {"count": 0, "n_target_free": 0, "fraction": None}


def test_fp_images_counts_confident_detections_only():
    recs = [_rec(0, [], []), _rec(1, [], []), _rec(2, [[1, 1, 5, 5]], [0])]
    preds = [{"image_index": 0, "boxes": [[1, 1, 5, 5]], "scores": [0.6], "labels": [0]},
             {"image_index": 1, "boxes": [[1, 1, 5, 5]], "scores": [0.4], "labels": [1]},
             {"image_index": 2, "boxes": [[1, 1, 5, 5]], "scores": [0.9], "labels": [0]}]
    assert fp_images(recs, preds) == {"count": 1, "n_target_free": 2, "fraction": 0.5}
