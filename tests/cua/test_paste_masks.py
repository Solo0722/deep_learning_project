"""Mask-based copy-paste: polygon sidecar for dacl10k and pasting only object pixels."""
import json

import numpy as np
import pytest

from cua.augment.paste import CopyPaste, content_of
from cua.datasets.dacl10k import parse_fiftyone_sample, DACL_DAMAGE
from cua.datasets.dacl10k_masks import build_masks, load_masks, main as masks_main
from cua.experiments.specs import resolve
from cua.image import letterbox
from cua.records import Record, write_classes, write_records


def _rec(i, boxes, labels, w=320, h=240):
    return Record(f"img/{i}.png", w, h, np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(labels),
                  np.zeros(len(labels), bool), f"g{i}", "s")


def _triangle(x0, y0, x1, y1):
    return [np.array([[x0, y1], [x1, y1], [(x0 + x1) / 2, y0]], dtype=np.float64)]


def test_mask_paste_changes_only_polygon_pixels_and_boxes_the_mask():
    recs = [_rec(0, [[20, 20, 120, 120]], [0]), _rec(1, [[150, 100, 300, 220]], [1]), _rec(2, [], [])]
    masks = [[_triangle(20, 20, 120, 120)], [_triangle(150, 100, 300, 220)], []]
    rng = np.random.default_rng(0)
    samples = []
    for r in recs:
        rgb = np.full((r.height, r.width, 3), 40, dtype=np.uint8)
        for (x0, y0, x1, y1), tri in zip(r.boxes.astype(int), [m[0] for m in masks[recs.index(r)]]):
            rgb[y0:y1, x0:x1] = 250  # bright box; only the triangle inside it should be pasted
        samples.append(letterbox(rgb, r, 640)[0])
    cp = CopyPaste(recs, 2, "balanced", 1, p=1.0, max_objects=2, masks=masks)
    seen = 0
    for epoch in range(1, 10):
        out, pasted = cp(samples[2], epoch, 2, lambda j: samples[j])
        if not pasted:
            continue
        seen += len(pasted)
        changed = np.any(out.rgb != samples[2].rgb, axis=2)
        inside = np.zeros_like(changed)
        for x0, y0, x1, y1 in out.boxes.astype(int):
            inside[y0:y1, x0:x1] = True
            region = changed[y0:y1, x0:x1]
            # the box is tight around the pasted pixels, and a triangle fills about half its box
            assert region[0].any() or region[:, 0].any()
            assert 0.3 < region.mean() < 0.75
        assert not (changed & ~inside).any()
        c = content_of(recs[2], 640)
        assert np.all(out.boxes[:, 0] >= c.left) and np.all(out.boxes[:, 2] <= c.left + c.width)
    assert seen > 0


def test_bank_needs_a_polygon_and_masks_must_align():
    recs = [_rec(0, [[20, 20, 120, 120], [130, 20, 230, 120]], [0, 1]), _rec(1, [[10, 10, 90, 90]], [1])]
    cp = CopyPaste(recs, 2, "balanced", 0, masks=[[_triangle(20, 20, 120, 120), []], [_triangle(10, 10, 90, 90)]])
    assert cp.bank[0].tolist() == [[0, 0]] and cp.bank[1].tolist() == [[1, 0]]
    with pytest.raises(ValueError, match="one polygon list per box"):
        CopyPaste(recs, 2, "balanced", 0, masks=[[[]], [[]]])


def _sample(path, polys, w=200, h=100, tags=()):
    return {"filepath": path, "tags": list(tags), "metadata": {"width": w, "height": h},
            "semseg_ground_truth_filled": {"polylines": [
                {"label": lab, "points": [[[x / w, y / h] for x, y in ring] for ring in rings]} for lab, rings in polys]}}


def test_sidecar_matches_prepare_boxes_and_round_trips(tmp_path):
    samples = [
        _sample("data/a.jpg", [("Crack", [[(10, 10), (60, 10), (60, 40)]]),
                               ("Cavity", [[(0, 0), (5, 5), (10, 10)]]),           # collinear: dropped
                               ("Bearing", [[(1, 1), (9, 1), (9, 9)]]),            # not damage: dropped
                               ("ACrack", [[(20, 20), (40, 20), (40, 40)]]),       # excluded class: dropped
                               ("Rust", [[(150, 50), (250, 50), (250, 90)], [(160, 60), (170, 60), (170, 70)]])]),
        _sample("data/b.jpg", []),
    ]
    retained = tuple(c for c in DACL_DAMAGE if c != "ACrack")
    remap = {DACL_DAMAGE.index(c): i for i, c in enumerate(retained)}
    recs = []
    for s in samples:
        r = parse_fiftyone_sample(s, DACL_DAMAGE)
        keep = np.array([int(c) in remap for c in r.labels], dtype=bool)
        recs.append(Record(r.image, r.width, r.height, r.boxes[keep],
                           np.array([remap[int(c)] for c in r.labels[keep]], dtype=np.int64), r.ignore[keep], "g", "dacl10k"))
    masks = build_masks(samples, recs, retained)
    assert [len(masks[r.image]) for r in recs] == [2, 0]
    assert len(masks["data/a.jpg"][1]) == 2  # both rings of the Rust polygon kept
    d = tmp_path / "dacl10k"
    d.mkdir()
    write_records(d / "train.jsonl", recs)
    write_classes(d / "classes.json", retained)
    (tmp_path / "samples.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")
    assert masks_main(["--root", str(tmp_path), "--data-dir", str(d)]) == 0
    loaded = load_masks(d / "train_masks.json", recs)
    assert np.allclose(loaded[0][0][0], [[10, 10], [60, 10], [60, 40]])
    bad = [Record(recs[0].image, 200, 100, recs[0].boxes[:1], recs[0].labels[:1], recs[0].ignore[:1], "g", "dacl10k")]
    with pytest.raises(ValueError):
        build_masks(samples, bad, retained)
    with pytest.raises(ValueError):
        load_masks(d / "train_masks.json", bad)


def test_mask_arm_specs():
    c = resolve("P-dacl10k-fasterrcnn-s1-cpbalm")
    assert (c.cp_mode, c.cp_mask, c.cp_region, c.policy) == ("balanced", True, "full", "uniform")
    assert resolve("P-dacl10k-fasterrcnn-s1-cpbal").cp_mask is False
    with pytest.raises((KeyError, ValueError)):
        resolve("P-rdd-fasterrcnn-s0-cpbalm")
    # the rectangle arms keep their digests (values recorded before cp_mask existed)
    assert resolve("P-dacl10k-fasterrcnn-s0-cpbal").digest().startswith("4a46d3daa0b0")
    assert resolve("C-dacl10k-fasterrcnn-s0-uniform").digest().startswith("2ad52d017282")


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
def test_trainer_pastes_masks_from_sidecar(tmp_path, monkeypatch):
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
    cfg = T.TrainConfig(run_id="r", dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit=None,
                        policy="uniform", cp_mode="balanced", cp_p=1.0, cp_mask=True, pretrained=False,
                        batch_size=2, workers=0, lr_warmup_iters=2, amp=False)
    with pytest.raises((FileNotFoundError, OSError)):
        T.train(cfg, tmp_path / "data", tmp_path / "runs")
    tri = [[[10, 40], [50, 40], [30, 10]]]
    (d / "train_masks.json").write_text(json.dumps({"version": 1, "masks": {r.image: [tri] for r in recs[:4]}}))
    assert T.train(cfg, tmp_path / "data", tmp_path / "runs2") == 0
    hist = [json.loads(x) for x in (tmp_path / "runs2" / "r" / "history.jsonl").read_text().splitlines() if x.strip()]
    assert sum(hist[-1]["paste"]["planned"].values()) > 0
