"""Strength levels of pixel edits (reports/cua_strength/PROTOCOL.md): edits, transform, specs, jobs and read-out."""
import json

import numpy as np
import pytest

from cua.analysis.strength import (arm_name, load_arms, load_part_a_reference, run_name, shrunk_selection_gain,
                                   spread, summarize)
from cua.augment import pixel
from cua.augment.menu import ArmTransform
from cua.augment.rng import keyed_rng
from cua.experiments.specs import STRENGTH_EDITS, part_a_specs, resolve, strength_specs
from cua.records import Record, write_classes, write_records
from cua.train import TrainConfig, _validate

from helpers import toy_fetch, toy_sample


def _orig_contrast(sample, rng):  # the pre-strength implementation, verbatim
    from PIL import ImageEnhance
    factor = float(rng.uniform(0.7, 1.3))
    return pixel._pil(sample, lambda im: ImageEnhance.Contrast(im).enhance(factor))


def _orig_noise(sample, rng):
    noisy = sample.rgb.astype(np.float64) + rng.normal(0.0, pixel.NOISE_SIGMA, size=sample.rgb.shape)
    return pixel._with_rgb(sample, np.clip(np.rint(noisy), 0, 255).astype(np.uint8))


@pytest.mark.parametrize("edit", STRENGTH_EDITS)
def test_level_1_is_bit_identical_to_the_default(edit):
    s = toy_sample()
    for k in range(5):
        a = pixel.PIXEL_EDITS[edit](s, np.random.default_rng(k))
        b = pixel.PIXEL_EDITS[edit](s, np.random.default_rng(k), level=1)
        assert np.array_equal(a.rgb, b.rgb)


def test_level_1_matches_the_original_implementation():
    s = toy_sample()
    for k in range(5):
        assert np.array_equal(pixel.contrast(s, np.random.default_rng(k)).rgb,
                              _orig_contrast(s, np.random.default_rng(k)).rgb)
        assert np.array_equal(pixel.noise(s, np.random.default_rng(k)).rgb,
                              _orig_noise(s, np.random.default_rng(k)).rgb)


def test_ranges_and_paired_quantile():
    assert pixel._RANGES["contrast"] == {1: (0.7, 1.3), 2: (0.5, 1.5), 3: (0.3, 1.7)}
    assert pixel._NOISE_SCALE[1] == 1.0 and pixel.NOISE_SIGMA * pixel._NOISE_SCALE[3] == pytest.approx(0.10 * 255)
    assert pixel._RANGES["blur"][3] == (1.5, 4.5)
    for edit in ("brightness", "contrast", "saturation"):  # factors: same centre, strictly wider
        r = pixel._RANGES[edit]
        for lv in (2, 3):
            assert sum(r[lv]) / 2 == pytest.approx(sum(r[1]) / 2)
            assert r[lv][1] - r[lv][0] > r[lv - 1][1] - r[lv - 1][0]
    assert min(pixel._RANGES[e][3][0] for e in ("brightness", "contrast")) >= 0.3  # objects stay visible
    for lv in (2, 3):  # blur sigma (and noise sigma) scale multiplicatively
        assert pixel._RANGES["blur"][lv] == pytest.approx(tuple(lv * x for x in pixel._RANGES["blur"][1]))
    # one uniform draw per edit: the same rng state gives the same quantile at every level
    for k in range(20):
        u = [(np.random.default_rng(k).uniform(*pixel._RANGES["contrast"][lv]) - pixel._RANGES["contrast"][lv][0])
             / (pixel._RANGES["contrast"][lv][1] - pixel._RANGES["contrast"][lv][0]) for lv in (1, 2, 3)]
        assert np.allclose(u, u[0])


@pytest.mark.parametrize("edit", STRENGTH_EDITS)
def test_higher_levels_change_the_image_more(edit):
    s = toy_sample()
    dev = {lv: [] for lv in (1, 2, 3)}
    for k in range(12):
        for lv in (1, 2, 3):
            out = pixel.PIXEL_EDITS[edit](s, np.random.default_rng(k), level=lv)
            dev[lv].append(np.abs(out.rgb.astype(float) - s.rgb.astype(float)).mean())
            assert np.array_equal(out.boxes, s.boxes) and np.array_equal(out.labels, s.labels)
    assert np.mean(dev[1]) < np.mean(dev[2]) < np.mean(dev[3])


def test_saturation_level_3_is_clipped_to_greyscale():
    s = toy_sample()
    for k in range(40):
        out = pixel.saturation(s, np.random.default_rng(k), level=3)
        assert out.rgb.min() >= 0 and out.rgb.max() <= 255


def test_bad_level_rejected():
    with pytest.raises(ValueError):
        pixel.contrast(toy_sample(), np.random.default_rng(0), level=4)
    with pytest.raises(ValueError):
        pixel.noise(toy_sample(), np.random.default_rng(0), level=0)


def test_transform_strength_coin_and_pairing():
    s = toy_sample()
    x1 = ArmTransform("dacl10k", "contrast", seed=3)
    x3 = ArmTransform("dacl10k", "contrast", seed=3, strength=3)
    base = ArmTransform("dacl10k", None, seed=3)
    applied = 0
    for epoch in range(10):
        b = base(s, epoch, 0, toy_fetch, 10)
        a1, a3 = x1(s, epoch, 0, toy_fetch, 10), x3(s, epoch, 0, toy_fetch, 10)
        same1 = np.array_equal(a1.rgb, b.rgb)
        assert same1 == np.array_equal(a3.rgb, b.rgb)  # identical coin across levels
        if not same1:
            applied += 1
            # level 3 = the level-3 edit of the same base, with the same edit rng
            er = keyed_rng(3, epoch, 0, "edit")
            er.random()
            ref = pixel.contrast(base.apply(s, epoch, 0, toy_fetch, 10, None), er, level=3)
            assert np.array_equal(a3.rgb, ref.rgb)
    assert applied > 0


def test_transform_and_config_validation():
    with pytest.raises(ValueError):
        ArmTransform("dacl10k", "hue", seed=0, strength=2)
    with pytest.raises(ValueError):
        ArmTransform("dacl10k", None, seed=0, strength=2)
    with pytest.raises(ValueError):
        ArmTransform("dacl10k", "contrast", seed=0, strength=5)
    common = dict(run_id="r", dataset="dacl10k", detector="fasterrcnn", seed=0, epoch_lr=(1.0,))
    with pytest.raises(ValueError):
        _validate(TrainConfig(**common, edit="mosaic", edit_strength=2))
    with pytest.raises(ValueError):
        _validate(TrainConfig(**common, edit=None, policy="uniform", edit_strength=2))
    with pytest.raises(ValueError):
        _validate(TrainConfig(**common, edit="blur", edit_strength=4))
    _validate(TrainConfig(**common, edit="blur", edit_strength=3))


def test_strength_specs_and_digests():
    ss = strength_specs()
    assert len(ss) == 45 and STRENGTH_EDITS == ("brightness", "contrast", "saturation", "blur", "noise")
    twins = {a.run_id: a for a in part_a_specs("dacl10k", "fasterrcnn")}
    for s in ss:
        stem, lv = s.run_id.rsplit("-x", 1)
        a = twins["A" + stem[1:]]
        changed = {f for f in TrainConfig.__dataclass_fields__ if getattr(s, f) != getattr(a, f)}
        assert changed == ({"run_id", "edit_p"} if lv == "1" else {"run_id", "edit_p", "edit_strength"})
        assert s.edit_p == 1.0 and a.edit_p == 0.5
        assert s.edit_strength == int(lv) in (1, 2, 3) and s.init_run == a.init_run and s.digest() != a.digest()
    assert len({s.digest() for s in ss}) == 45
    assert resolve("S-dacl10k-fasterrcnn-s1-blur-x3") == strength_specs(seeds=(1,), edits=("blur",), levels=(3,))[0]
    a0 = part_a_specs("dacl10k", "fasterrcnn", (0,))
    assert [s.digest()[:12] for s in a0[:3]] == ["572fa110d1c6", "ff84440f8e6b", "d32141ef901b"]
    with pytest.raises(ValueError):
        strength_specs(edits=("hue",))
    with pytest.raises(ValueError):
        strength_specs(levels=(4,))


def test_make_jobs_strength_family(tmp_path):
    from cua.ccast.make_jobs import main
    assert main(["--family", "strength", "--dataset", "dacl10k", "--detector", "fasterrcnn", "--out",
                 str(tmp_path), "--project", "/p", "--edits", "contrast", "--seeds", "0"]) == 0
    submit = (tmp_path / "submit.sh").read_text()
    assert all(f"qsub S-dacl10k-fasterrcnn-s0-contrast-x{lv}.pbs" in submit for lv in (1, 2, 3))
    assert "depend" not in submit and submit.count("qsub") == 3


def _rows(fn, seeds=(0, 1, 2), classes=("a", "b", "c", "d"), noise=0.05):
    rng = np.random.default_rng(0)
    out = []
    for k in seeds:
        for c in classes:
            out.append({"dataset": "dacl10k", "detector": "fasterrcnn", "seed": k, "edit": "base", "cls": c, "ap50": 50.0})
            for e in ("contrast", "blur"):
                for lv in (1, 2, 3):
                    out.append({"dataset": "dacl10k", "detector": "fasterrcnn", "seed": k, "edit": arm_name(e, lv),
                                "cls": c, "ap50": 50.0 + fn(e, lv, c) + rng.normal(0, noise)})
    return out


def test_summary_detects_a_class_specific_strength_effect():
    best = {"a": ("contrast", 3), "b": ("blur", 2), "c": ("contrast", 1), "d": ("blur", 3)}
    rows = _rows(lambda e, lv, c: 2.0 if best[c] == (e, lv) else -1.0)
    s = summarize(rows, ("contrast", "blur"))
    assert s["S1_pass"] and s["S2_pass"] and s["go"] and s["S2"]["mean"] > 1.0
    assert s["S4"]["mean"] > 1.0 and s["S4"]["mean_lambda"] > 0.5


def test_summary_rejects_a_shared_effect_and_noise():
    shared = _rows(lambda e, lv, c: {"contrast": 0.5, "blur": -0.5}[e] * lv)
    s = summarize(shared, ("contrast", "blur"))
    assert not s["S2_pass"] and abs(s["S2"]["mean"]) < 0.2
    noisy = _rows(lambda e, lv, c: 0.0, noise=1.0)
    n = summarize(noisy, ("contrast", "blur"))
    assert not n["go"]
    assert len(n["S3"]) == 6 and np.mean([r["lambda"] for r in n["S3"]]) < 0.5
    assert spread([], ("contrast",)) == []


def test_load_arms_names_and_missing(tmp_path):
    assert run_name("contrast", 1, 0) == "S-dacl10k-fasterrcnn-s0-contrast-x1"
    assert run_name("contrast", 2, 1) == "S-dacl10k-fasterrcnn-s1-contrast-x2"
    hist = {"epoch": 6, "metrics": {"per_class": {"Rust": {"ap50": 30.0}, "Crack": {"ap50": 40.0}}}}
    for name in ("A-dacl10k-fasterrcnn-s0-base", "A-dacl10k-fasterrcnn-s0-contrast", "S-dacl10k-fasterrcnn-s0-contrast-x1",
                 "S-dacl10k-fasterrcnn-s0-contrast-x2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "history.jsonl").write_text(json.dumps(hist) + "\n")
        (tmp_path / name / "COMPLETE.json").write_text("{}")
    rows, missing = load_arms(tmp_path, ("contrast",), levels=(1, 2, 3), seeds=(0,))
    assert {r["edit"] for r in rows} == {"base", "contrast@x1", "contrast@x2"}
    assert missing == ["S-dacl10k-fasterrcnn-s0-contrast-x3"]
    ref = load_part_a_reference(tmp_path, ("contrast",), seeds=(0,))
    assert {r["edit"] for r in ref} == {"base", "contrast@A"}


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
def test_trainer_runs_strength_level(tmp_path, monkeypatch):
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
    cfg = T.TrainConfig(run_id="r", dataset="voc", detector="fasterrcnn", seed=0, epoch_lr=(1.0,), edit="contrast",
                        edit_p=1.0, edit_strength=3, pretrained=False, batch_size=2, workers=0,
                        lr_warmup_iters=2, amp=False)
    assert T.train(cfg, tmp_path / "data", tmp_path / "runs") == 0
    assert json.loads((tmp_path / "runs" / "r" / "config.json").read_text())["config"]["edit_strength"] == 3
