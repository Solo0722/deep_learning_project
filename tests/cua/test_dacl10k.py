import json
from pathlib import Path

import pytest
from PIL import Image

from cua.datasets.dacl10k import DACL_DAMAGE, load_dacl10k, parse_dacl_json
from cua.gates import retained_classes

FIX = Path(__file__).parent / "fixtures" / "dacl10k"


def test_polygon_to_box_and_component_labels_dropped():
    rec = parse_dacl_json(FIX / "img1.json", DACL_DAMAGE)
    assert rec.boxes.tolist() == [[10.0, 20.0, 110.0, 60.0]]
    assert [DACL_DAMAGE[i] for i in rec.labels] == ["Rust"]          # "Bearing" (component) dropped
    assert (rec.width, rec.height) == (200, 100) and not rec.ignore.any()


def test_degenerate_polygons_are_dropped_and_counted():
    counts = {}
    rec = parse_dacl_json(FIX / "img_degenerate.json", DACL_DAMAGE, counts=counts)
    # diagonal collinear, single point, repeated single point are dropped; the square stays
    assert rec.boxes.tolist() == [[10.0, 10.0, 40.0, 30.0]]
    assert counts["dropped_degenerate"] == 3


def test_points_outside_image_are_clipped_and_empty_boxes_dropped():
    counts = {}
    rec = parse_dacl_json(FIX / "img_clip.json", DACL_DAMAGE, counts=counts)
    assert rec.boxes.tolist() == [[0.0, 0.0, 200.0, 50.0]]
    assert counts == {"clipped_boxes": 1, "dropped_degenerate": 1}  # fully-outside box is empty


@pytest.mark.parametrize("text", [
    "{not json",
    '{"imageWidth": 10}',
    '{"imageWidth": 0, "imageHeight": 5, "imagePath": "a", "shapes": []}',
])
def test_bad_json_raises_value_error_naming_file(tmp_path, text):
    p = tmp_path / "broken_ann.json"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="broken_ann.json"):
        parse_dacl_json(p, DACL_DAMAGE)


def test_malformed_points_raise_value_error_naming_file(tmp_path):
    p = tmp_path / "bad_points.json"
    p.write_text(json.dumps({"imageWidth": 10, "imageHeight": 10, "imagePath": "a.png",
                             "shapes": [{"label": "Crack", "points": [[1, 2, 3], [4, 5, 6]]}]}),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="bad_points.json"):
        parse_dacl_json(p, DACL_DAMAGE)


def test_class_exclusion_rule():
    stats = [{"count": 299, "large_fraction": 0.0}, {"count": 5000, "large_fraction": 0.6}, {"count": 400, "large_fraction": 0.1}]
    assert retained_classes(("a", "b", "c"), stats) == ("c",)


def test_class_exclusion_boundaries_are_inclusive():
    stats = [{"count": 300, "large_fraction": 0.5}, {"count": 300, "large_fraction": 0.5001}]
    assert retained_classes(("a", "b"), stats) == ("a",)


def _write_root(root: Path, noise_image, brighten):
    """40 distinct images (35 with 30 Crack boxes + 1 Rust box, 5 Rust-only) plus a brightened duplicate."""
    (root / "annotations" / "train").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)

    def add(name, img, shapes):
        Image.fromarray(img).save(root / "images" / "train" / f"{name}.png")
        (root / "annotations" / "train" / f"{name}.json").write_text(json.dumps({
            "imageWidth": img.shape[1], "imageHeight": img.shape[0], "imagePath": f"{name}.png",
            "shapes": shapes}), encoding="utf-8")

    crack = [{"label": "Crack", "points": [[i, 2], [i + 5, 2], [i + 5, 9], [i, 9]]} for i in range(30)]
    rust = [{"label": "Rust", "points": [[1, 1], [20, 1], [20, 20]]}]
    for k in range(40):
        add(f"im{k:02d}", noise_image(seed=k), (crack if k >= 5 else []) + rust)
    add("im_dup", brighten(noise_image(seed=7), 1.05), crack)


def test_load_dacl10k_pools_groups_splits_and_excludes(tmp_path, noise_image, brighten):
    _write_root(tmp_path, noise_image, brighten)
    splits, retained = load_dacl10k(tmp_path, seed=0)
    assert retained == ("Crack",)                      # Rust has 40 boxes < 300
    everything = splits["train"] + splits["val"] + splits["test"]
    assert len(everything) == 41
    assert all((r.labels == 0).all() for r in everything)
    assert sum(len(r.boxes) == 0 for r in everything) == 5   # Rust-only images stay as zero-box records
    assert all(r.image.startswith("images/train/") for r in everything)
    by_name = {Path(r.image).stem: (r, s) for s, rs in splits.items() for r in rs}
    assert by_name["im07"][0].group == by_name["im_dup"][0].group
    assert by_name["im07"][1] == by_name["im_dup"][1]  # duplicates never straddle splits
    assert by_name["im07"][0].group.startswith("dup-")
    g = [{r.group for r in splits[s]} for s in ("train", "val", "test")]
    assert g[0].isdisjoint(g[1]) and g[0].isdisjoint(g[2]) and g[1].isdisjoint(g[2])


def test_load_dacl10k_is_deterministic_and_reports_stats(tmp_path, noise_image, brighten):
    _write_root(tmp_path, noise_image, brighten)
    stats = {}
    a, _ = load_dacl10k(tmp_path, seed=3, stats=stats)
    b, _ = load_dacl10k(tmp_path, seed=3)
    assert a == b
    assert list(stats["excluded"]) == [c for c in DACL_DAMAGE if c != "Crack"]
    assert stats["excluded"]["Rust"]["count"] < 300
    assert stats["class_stats"][0]["count"] >= 300


def test_dacl10k_cli_passes_exclusions_into_extra(tmp_path, monkeypatch, noise_image, brighten):
    import cua.prepare as prepare

    _write_root(tmp_path / "root", noise_image, brighten)
    captured = {}

    def fake_prepare(name, splits, classes, out_dir, image_root, *, extra=None, **kw):
        captured.update(name=name, classes=classes, extra=extra, image_root=image_root)
        return {"reference_shortfall": []}

    monkeypatch.setattr(prepare, "prepare_dataset", fake_prepare)
    prepare.main(["dacl10k", "--root", str(tmp_path / "root"), "--out", str(tmp_path / "o")])
    assert captured["name"] == "dacl10k" and captured["classes"] == ("Crack",)
    extra = captured["extra"]
    assert extra["dacl10k_retained_classes"] == ["Crack"]
    assert "Rust" in extra["dacl10k_excluded_classes"]
    assert set(extra["dacl10k_excluded_classes"]["Rust"]) == {"count", "large_fraction"}
