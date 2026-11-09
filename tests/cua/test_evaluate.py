import time

import numpy as np
import pytest
import torch
from PIL import Image

from cua.analysis.ap50 import ap50_from_matches, image_matches, paired_bootstrap
from cua.data import DetectionDataset
from cua.evaluate import coco_metrics, predict
from cua.records import Record
from helpers import problem_where_class_2_has_no_gt, random_detection_problem

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")


def test_fast_ap50_matches_pycocotools():
    recs, preds = random_detection_problem(n_images=60, n_classes=3, seed=4, with_ignored=True)
    coco = coco_metrics(recs, preds, ("a", "b", "c"))
    fast = ap50_from_matches(image_matches(recs, preds, 3), np.arange(60))
    for k, name in enumerate("abc"):
        assert abs(fast[k] - coco["per_class"][name]["ap50"]) < 1e-9


@pytest.mark.parametrize("seed,n_images,n_classes,with_ignored", [
    (7, 50, 3, False), (11, 80, 3, True), (23, 70, 4, True), (31, 40, 2, False),
])
def test_fast_ap50_matches_pycocotools_more_seeds(seed, n_images, n_classes, with_ignored):
    recs, preds = random_detection_problem(n_images, n_classes, seed, with_ignored=with_ignored)
    names = tuple("abcd"[:n_classes])
    coco = coco_metrics(recs, preds, names)
    fast = ap50_from_matches(image_matches(recs, preds, n_classes), np.arange(n_images))
    for k, name in enumerate(names):
        assert abs(fast[k] - coco["per_class"][name]["ap50"]) < 1e-9


def test_fast_ap50_with_exactly_tied_scores_across_images():
    recs, preds = random_detection_problem(60, 3, seed=5, with_ignored=True)
    for p in preds:
        p["scores"] = np.round(p["scores"] * 3) / 3  # only {0, 1/3, 2/3, 1}: many exact cross-image ties
    coco = coco_metrics(recs, preds, ("a", "b", "c"))
    fast = ap50_from_matches(image_matches(recs, preds, 3), np.arange(60))
    for k, name in enumerate("abc"):
        assert abs(fast[k] - coco["per_class"][name]["ap50"]) < 1e-9


def test_ignored_gt_is_scanned_last():
    # Crowd (ignored) GT listed before a regular GT of the same class; the detection overlaps both,
    # with crowd IoU 1.0 > regular IoU 0.9. COCO scans regular GTs first, so this is a true positive.
    rec = Record("img/x.jpg", 200, 200, np.array([[0., 0., 100., 100.], [0., 0., 50., 100.]]),
                 np.array([0, 0]), np.array([True, False]), "g", "s")
    pred = {"image_index": 0, "boxes": np.array([[0., 0., 50., 90.]]), "scores": np.array([0.9]),
            "labels": np.array([0])}
    coco = coco_metrics([rec], [pred], ("a",))
    assert coco["per_class"]["a"]["ap50"] == pytest.approx(100.0)
    m = image_matches([rec], [pred], 1)
    assert m.tp[0].tolist() == [True] and m.ignored[0].tolist() == [False]
    assert ap50_from_matches(m, np.arange(1))[0] == coco["per_class"]["a"]["ap50"]


def test_maxdet_truncation_is_exercised():
    recs, preds = random_detection_problem(10, 3, seed=4, with_ignored=True)
    assert (preds[0]["labels"] == 0).sum() > 100
    m = image_matches(recs, preds, 3)
    assert m.offsets[0][1] - m.offsets[0][0] == 100


def test_duplicate_resample_matches_pycocotools_on_rebuilt_set():
    recs, preds = random_detection_problem(40, 3, seed=8, with_ignored=True)
    idx = np.random.default_rng(0).integers(0, 40, 40)
    assert len(np.unique(idx)) < 40
    fast = ap50_from_matches(image_matches(recs, preds, 3), idx)
    r2 = [recs[i] for i in idx]
    p2 = [dict(preds[i], image_index=j) for j, i in enumerate(idx)]
    coco = coco_metrics(r2, p2, ("a", "b", "c"))
    for k, name in enumerate("abc"):
        ref = coco["per_class"][name]["ap50"]
        if ref is None:
            assert np.isnan(fast[k])
        else:
            assert abs(fast[k] - ref) < 1e-9


def test_class_without_gt_is_none_and_excluded():
    recs, preds = problem_where_class_2_has_no_gt()
    out = coco_metrics(recs, preds, ("a", "b", "c"))
    assert out["per_class"]["c"]["ap50"] is None
    assert out["per_class"]["c"]["n_gt"] == 0
    assert out["macro_ap50"] == pytest.approx(np.mean([out["per_class"][n]["ap50"] for n in "ab"]))
    fast = ap50_from_matches(image_matches(recs, preds, 3), np.arange(len(recs)))
    assert np.isnan(fast[2])


def test_perfect_predictions_score_100_and_keys():
    recs, _ = random_detection_problem(20, 3, seed=2, with_ignored=True)
    preds = [{"image_index": i, "boxes": r.boxes[~r.ignore], "scores": np.ones(int((~r.ignore).sum())),
              "labels": r.labels[~r.ignore]} for i, r in enumerate(recs)]
    out = coco_metrics(recs, preds, ("a", "b", "c"))
    assert set(out) >= {"macro_ap50", "ap50_95", "ap75", "ap_small", "ap_medium", "ap_large", "per_class"}
    assert out["macro_ap50"] == pytest.approx(100.0)
    assert out["ap50_95"] == pytest.approx(100.0)
    for name in "abc":
        pc = out["per_class"][name]
        assert pc["n_gt"] == sum(int(((r.labels == "abc".index(name)) & ~r.ignore).sum()) for r in recs)
        assert pc["ap50"] == pytest.approx(100.0) and pc["ap50_95"] == pytest.approx(100.0)


def test_zero_detections_give_zero_ap():
    recs, _ = random_detection_problem(10, 3, seed=6)
    empty = [{"image_index": i, "boxes": np.zeros((0, 4)), "scores": np.zeros(0),
              "labels": np.zeros(0, dtype=np.int64)} for i in range(10)]
    out = coco_metrics(recs, empty, ("a", "b", "c"))
    assert out["macro_ap50"] == 0.0 and out["ap50_95"] == 0.0
    for name in "abc":
        assert out["per_class"][name]["ap50"] in (0.0, None)
    fast = ap50_from_matches(image_matches(recs, empty, 3), np.arange(10))
    assert np.all((fast == 0) | np.isnan(fast))


def test_bootstrap_of_identical_models_is_zero():
    recs, preds = random_detection_problem(40, 2, seed=1)
    m = image_matches(recs, preds, 2)
    r = paired_bootstrap(m, m, n_boot=200)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def _degraded(preds, seed):
    rng = np.random.default_rng(seed)
    out = []
    for p in preds:
        keep = rng.random(len(p["scores"])) > 0.3
        out.append({"image_index": p["image_index"], "boxes": p["boxes"][keep],
                    "scores": p["scores"][keep] * rng.uniform(0.5, 1.0, keep.sum()), "labels": p["labels"][keep]})
    return out


def test_bootstrap_ci_contains_point_and_is_deterministic():
    recs, preds = random_detection_problem(80, 3, seed=12, with_ignored=True)
    ma = image_matches(recs, preds, 3)
    mb = image_matches(recs, _degraded(preds, 1), 3)
    r1 = paired_bootstrap(ma, mb, n_boot=300, seed=5)
    r2 = paired_bootstrap(ma, mb, n_boot=300, seed=5)
    r3 = paired_bootstrap(ma, mb, n_boot=300, seed=6)
    assert r1 == r2 and r1 != r3
    assert r1["diff"] != 0 and r1["lo"] <= r1["diff"] <= r1["hi"]
    full = np.arange(80)
    expected = np.nanmean(ap50_from_matches(mb, full)) - np.nanmean(ap50_from_matches(ma, full))
    assert r1["diff"] == pytest.approx(expected)


def test_bootstrap_rejects_different_ground_truth():
    recs, preds = random_detection_problem(20, 2, seed=1)
    ma = image_matches(recs, preds, 2)
    other, _ = random_detection_problem(20, 2, seed=2)
    mb = image_matches(other, [dict(p, labels=p["labels"] % 2) for p in preds], 2)
    assert not np.array_equal(ma.gt_count, mb.gt_count)
    with pytest.raises(ValueError, match="gt_count"):
        paired_bootstrap(ma, mb, n_boot=10)


def test_bootstrap_rejects_mismatched_image_counts():
    recs, preds = random_detection_problem(20, 2, seed=1)
    ma = image_matches(recs, preds, 2)
    mb = image_matches(recs[:10], preds[:10], 2)
    with pytest.raises(ValueError):
        paired_bootstrap(ma, mb, n_boot=10)


def _disk_records(tmp_path):
    rng = np.random.default_rng(0)
    recs = []
    for k, (w, h) in enumerate([(80, 60), (50, 90)]):
        name = f"im{k}.png"
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(tmp_path / name)
        recs.append(Record(name, w, h, np.array([[5., 5., 30., 40.]]), np.array([1]), np.array([False]), "g", "s"))
    return recs


class _StubDetector(torch.nn.Module):
    """Eval-mode detector returning fixed canvas-space outputs (1-based labels) for every image."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.seen_devices = []

    def forward(self, images):
        assert not self.training
        self.seen_devices += [im.device for im in images]
        return [self.outputs[i % len(self.outputs)] for i in range(len(images))]


def test_predict_restores_clips_and_relabels_exactly(tmp_path):
    # 80x60 image at size 640: scale 8, resized 640x480, pad_left 0, pad_top 80.
    recs = _disk_records(tmp_path)[:1]
    out_canvas = {
        "boxes": torch.tensor([[80., 160., 240., 400.],    # -> [10, 10, 30, 40]
                               [600., 40., 700., 200.],    # -> [75, -5, 87.5, 15] -> clipped [75, 0, 80, 15]
                               [0., 80., 640., 560.]]),    # label 0: dropped
        "scores": torch.tensor([0.9, 0.5, 0.99]),
        "labels": torch.tensor([2, 3, 0]),
    }
    model = _StubDetector([out_canvas])
    model.train()
    ds = DetectionDataset(recs, tmp_path, transform=None, size=640)
    g = ds[0][1]["geometry"]
    assert (g["resized_width"], g["resized_height"], g["pad_left"], g["pad_top"]) == (640, 480, 0, 80)
    out = predict(model, ds, "cpu")
    assert model.training  # mode restored
    assert len(out) == 1 and out[0]["image_index"] == 0
    np.testing.assert_allclose(out[0]["boxes"], [[10., 10., 30., 40.], [75., 0., 80., 15.]], rtol=0, atol=1e-9)
    assert out[0]["boxes"].dtype == np.float64
    np.testing.assert_array_equal(out[0]["labels"], [1, 2])
    assert out[0]["labels"].dtype == np.int64
    np.testing.assert_allclose(out[0]["scores"], [0.9, 0.5], rtol=1e-6)
    assert all(d.type == "cpu" for d in model.seen_devices)


def test_predict_handles_no_detections(tmp_path):
    recs = _disk_records(tmp_path)
    empty = {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}
    out = predict(_StubDetector([empty]), DetectionDataset(recs, tmp_path, transform=None, size=640), "cpu")
    assert [p["image_index"] for p in out] == [0, 1]
    for p in out:
        assert p["boxes"].shape == (0, 4) and p["scores"].shape == (0,) and p["labels"].shape == (0,)


def test_predict_real_detector_smoke(tmp_path):
    from cua.models import build_detector

    recs = _disk_records(tmp_path)
    model = build_detector("retinanet", 3, size=640, pretrained=False)
    out = predict(model, DetectionDataset(recs, tmp_path, transform=None, size=640), "cpu", batch_size=2)
    assert [p["image_index"] for p in out] == [0, 1]
    for p in out:
        n = len(p["scores"])
        assert p["boxes"].shape == (n, 4) and p["boxes"].dtype == np.float64
        assert p["labels"].shape == (n,) and p["labels"].dtype == np.int64


def test_predict_rejects_transform(tmp_path):
    from cua.augment.menu import ArmTransform

    recs = _disk_records(tmp_path)
    ds = DetectionDataset(recs, tmp_path, ArmTransform("rdd", None, seed=0, p=1.0), size=640)
    with pytest.raises(ValueError):
        predict(torch.nn.Identity(), ds, "cpu")


@pytest.mark.slow
def test_bootstrap_speed():
    recs, preds = random_detection_problem(2000, 4, seed=0, with_ignored=True)
    ma = image_matches(recs, preds, 4)
    mb = image_matches(recs, _degraded(preds, 2), 4)
    t0 = time.perf_counter()
    paired_bootstrap(ma, mb, n_boot=1000, seed=0)
    assert time.perf_counter() - t0 < 30.0
