from pathlib import Path

import numpy as np

from cua.datasets.voc import VOC_CLASSES, load_voc, parse_voc_xml, split_trainval

FIX = Path(__file__).parent / "fixtures" / "voc"


def fake_ids(n):
    return [f"2007/{i:06d}" for i in range(n)]


def test_voc_classes_are_the_twenty_standard_names():
    assert VOC_CLASSES == (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor",
    )


def test_difficult_objects_are_ignored():
    recs = parse_voc_xml(FIX / "000005.xml", year="2007")
    assert recs.ignore.tolist() == [False, True]


def test_voc_boxes_are_zero_based_continuous():
    rec = parse_voc_xml(FIX / "000005.xml", year="2007")
    assert rec.boxes[0].tolist() == [262.0, 210.0, 324.0, 339.0]  # xmin-1, ymin-1, xmax, ymax


def test_parse_voc_xml_reads_size_labels_and_group():
    rec = parse_voc_xml(FIX / "000005.xml", year="2007")
    assert (rec.width, rec.height) == (500, 375)
    assert rec.labels.tolist() == [VOC_CLASSES.index("chair"), VOC_CLASSES.index("person")]
    assert rec.source == "VOC2007"
    assert rec.group == "2007/000005"


def test_val_split_is_deterministic_and_disjoint():
    a, b = split_trainval(fake_ids(1000), 0.10, 0), split_trainval(fake_ids(1000), 0.10, 0)
    assert a == b and set(a[0]).isdisjoint(a[1]) and 80 <= len(a[1]) <= 120


def _write_devkit(root: Path) -> Path:
    """Tiny fake VOCdevkit: 4 trainval ids per year, 2 VOC2007 test ids."""
    def ann(year, img_id, width=352, height=500):
        d = root / f"VOC{year}" / "Annotations"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{img_id}.xml").write_text(
            "<annotation><size>"
            f"<width>{width}</width><height>{height}</height><depth>3</depth>"
            "</size><object><name>car</name><difficult>0</difficult><bndbox>"
            "<xmin>10</xmin><ymin>20</ymin><xmax>60</xmax><ymax>90</ymax>"
            "</bndbox></object></annotation>",
            encoding="utf-8",
        )

    def ids_file(year, split, ids):
        d = root / f"VOC{year}" / "ImageSets" / "Main"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        (root / f"VOC{year}" / "JPEGImages").mkdir(parents=True, exist_ok=True)

    t07 = ["000001", "000002", "000003", "000004"]
    t12 = ["2012_000001", "2012_000002", "2012_000003", "2012_000004"]
    te07 = ["000005", "000006"]
    for i in t07:
        ann(2007, i)
    for i in t12:
        ann(2012, i)
    for i in te07:
        ann(2007, i)
    ids_file(2007, "trainval", t07)
    ids_file(2007, "test", te07)
    ids_file(2012, "trainval", t12)
    return root


def test_load_voc_splits_trainval_and_test(tmp_path):
    devkit = _write_devkit(tmp_path / "VOCdevkit")
    splits = load_voc(devkit, val_fraction=0.5, seed=0)

    assert set(splits) == {"train", "val", "test"}
    trainval = splits["train"] + splits["val"]
    groups = {r.group for r in trainval}
    assert groups == {f"2007/{i}" for i in ["000001", "000002", "000003", "000004"]} | {
        f"2012/{i}" for i in ["2012_000001", "2012_000002", "2012_000003", "2012_000004"]
    }
    assert len(trainval) == 8
    assert {r.group for r in splits["train"]}.isdisjoint({r.group for r in splits["val"]})

    assert sorted(r.group for r in splits["test"]) == ["2007/000005", "2007/000006"]
    assert all(r.source == "VOC2007" for r in splits["test"])
    assert all(r.image == f"VOC2007/JPEGImages/{r.group.split('/')[1]}.jpg" for r in splits["test"])
    assert all(r.image == f"VOC2012/JPEGImages/{r.group.split('/')[1]}.jpg"
               for r in trainval if r.source == "VOC2012")


def test_load_voc_is_deterministic(tmp_path):
    devkit = _write_devkit(tmp_path / "VOCdevkit")
    a = load_voc(devkit, val_fraction=0.5, seed=0)
    b = load_voc(devkit, val_fraction=0.5, seed=0)
    assert a == b
    assert [r.group for r in a["val"]] == [r.group for r in b["val"]]


def test_border_boxes_are_clamped_not_rejected():
    counts = {"clamped_boxes": 0, "dropped_boxes": 0}
    rec = parse_voc_xml(FIX / "000010.xml", year="2007", counts=counts)
    # xmin=0 -> -1 after conversion, clamped to 0; xmax=100 equals width, kept as is
    assert rec.boxes[0].tolist() == [0.0, 0.0, 100.0, 80.0]
    assert rec.boxes[1].tolist() == [0.0, 0.0, 50.0, 40.0]
    assert rec.labels.tolist() == [VOC_CLASSES.index("cat"), VOC_CLASSES.index("dog")]


def test_kept_boxes_overflowing_edges_are_clamped_and_counted():
    counts = {"clamped_boxes": 0, "dropped_boxes": 0}
    rec = parse_voc_xml(FIX / "000012.xml", year="2007", counts=counts)
    # cat: xmax=105 on width 100 -> kept, clamped to 100; xmin 50 -> 49
    assert rec.boxes[0].tolist() == [49.0, 9.0, 100.0, 20.0]
    # dog: ymax=95 on height 80 -> kept, clamped to 80
    assert rec.boxes[1].tolist() == [9.0, 59.0, 40.0, 80.0]
    assert counts == {"clamped_boxes": 2, "dropped_boxes": 0}


def test_degenerate_boxes_after_clamp_are_dropped_and_counted():
    counts = {"clamped_boxes": 0, "dropped_boxes": 0}
    rec = parse_voc_xml(FIX / "000010.xml", year="2007", counts=counts)
    # the third object lies fully outside the image (x1 = 100 = width) and is dropped
    assert len(rec.labels) == 2
    assert counts == {"clamped_boxes": 1, "dropped_boxes": 1}


def test_nested_parts_are_not_counted_as_objects():
    counts = {"clamped_boxes": 0, "dropped_boxes": 0}
    rec = parse_voc_xml(FIX / "000011.xml", year="2007", counts=counts)
    assert rec.labels.tolist() == [VOC_CLASSES.index("person")]
    assert rec.boxes.tolist() == [[19.0, 29.0, 120.0, 150.0]]


def test_missing_difficult_defaults_to_not_ignored():
    rec = parse_voc_xml(FIX / "000007.xml", year="2007")
    assert rec.ignore.tolist() == [False]


def test_truncated_and_occluded_do_not_change_boxes_or_ignore():
    rec = parse_voc_xml(FIX / "000007.xml", year="2007")
    assert rec.boxes[0].tolist() == [49.0, 59.0, 120.0, 180.0]
    assert rec.ignore.tolist() == [False]


def test_malformed_xml_error_names_the_file(tmp_path):
    import pytest

    bad = tmp_path / "000099.xml"
    bad.write_text("<annotation><object><name>cat</name></object></annotation>", encoding="utf-8")
    with pytest.raises(ValueError, match="000099.xml"):
        parse_voc_xml(bad, year="2007")


def test_load_voc_reports_box_fix_totals(tmp_path):
    devkit = _write_devkit(tmp_path / "VOCdevkit")
    # add one clamp+drop image to VOC2007 trainval
    (devkit / "VOC2007" / "Annotations" / "000010.xml").write_text(
        (FIX / "000010.xml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (devkit / "VOC2007" / "ImageSets" / "Main" / "trainval.txt").write_text(
        "000001\n000002\n000003\n000004\n000010\n", encoding="utf-8"
    )
    stats = {}
    splits = load_voc(devkit, val_fraction=0.5, seed=0, stats=stats)
    assert stats == {"clamped_boxes": 1, "dropped_boxes": 1}
    assert len(splits["train"]) + len(splits["val"]) == 9


def test_voc_cli_writes_box_fixes_into_gates(tmp_path, monkeypatch):
    import json

    import cua.prepare as prepare

    devkit = _write_devkit(tmp_path / "VOCdevkit")
    captured = {}

    def fake_prepare(name, splits, classes, out_dir, image_root, *, extra=None, **kw):
        captured.update(name=name, extra=extra, out_dir=out_dir, image_root=image_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gates.json").write_text(json.dumps({"voc_box_fixes": extra}), encoding="utf-8")
        return {"reference_shortfall": []}

    monkeypatch.setattr(prepare, "prepare_dataset", fake_prepare)
    monkeypatch.setattr(prepare, "_voc_duplicate_counts", lambda splits, devkit: {"stub": 1})
    prepare.main(["voc", "--devkit", str(devkit), "--out", str(tmp_path / "o")])
    assert captured["name"] == "voc"
    assert captured["image_root"] == devkit
    assert captured["extra"] == {
        "voc_box_fixes": {"clamped_boxes": 0, "dropped_boxes": 0},
        "voc_cross_split_duplicates": {"stub": 1},
    }


def test_prepare_dataset_writes_voc_box_fixes_into_real_gates_json(tmp_path, mk):
    import json

    from cua.prepare import prepare_dataset

    # every record has two classes, so the co-occurrence gate passes (rate 1.0)
    train = [mk(image=f"VOC2007/JPEGImages/t{i}.jpg", labels=[0, 1], group=f"2007/t{i}")
             for i in range(6)]
    fixes = {"clamped_boxes": 3, "dropped_boxes": 1}
    out = tmp_path / "out"
    gates = prepare_dataset(
        "voc",
        {"train": train, "val": [], "test": []},
        VOC_CLASSES,
        out,
        tmp_path,
        n_target=4,
        min_per_class=2,
        extra={"voc_box_fixes": fixes},
    )
    written = json.loads((out / "gates.json").read_text(encoding="utf-8"))
    assert written["voc_box_fixes"] == fixes
    assert written["dataset"] == "voc"
    assert written["cooccurrence"] == 1.0
    assert gates["voc_box_fixes"] == fixes


def test_voc_cli_subcommand_parses_devkit_and_out(tmp_path):
    from cua.prepare import build_parser

    args = build_parser().parse_args(["voc", "--devkit", str(tmp_path), "--out", str(tmp_path / "o")])
    assert args.dataset == "voc"
    assert args.devkit == str(tmp_path)
    assert args.out == str(tmp_path / "o")


def test_voc_cli_records_cross_split_duplicates(tmp_path, monkeypatch, noise_image, brighten):
    from PIL import Image

    import cua.prepare as prepare

    devkit = _write_devkit(tmp_path / "VOCdevkit")
    ids = {"2007": ["000001", "000002", "000003", "000004", "000005", "000006"],
           "2012": ["2012_000001", "2012_000002", "2012_000003", "2012_000004"]}
    def smooth(s):  # structured image: coarse noise upscaled, so the 9x8 hash is stable
        small = Image.fromarray(noise_image(seed=s, size=(12, 16)))
        return np.asarray(small.resize((352, 500), Image.Resampling.BICUBIC))

    seed = 0
    for year, names in ids.items():
        (devkit / f"VOC{year}" / "JPEGImages").mkdir(parents=True, exist_ok=True)
        for name in names:
            seed += 1
            img = smooth(seed)
            if name == "000005":  # a test image that duplicates trainval image 000001 (seed 1)
                img = brighten(smooth(1), 1.05)
            Image.fromarray(img).save(
                devkit / f"VOC{year}" / "JPEGImages" / f"{name}.jpg", quality=95
            )
    captured = {}

    def fake_prepare(name, splits, classes, out_dir, image_root, *, extra=None, **kw):
        captured["extra"] = extra
        return {"reference_shortfall": []}

    monkeypatch.setattr(prepare, "prepare_dataset", fake_prepare)
    prepare.main(["voc", "--devkit", str(devkit), "--out", str(tmp_path / "o")])
    dup = captured["extra"]["voc_cross_split_duplicates"]
    assert set(dup) == {"train_vs_test", "val_vs_test"}
    assert dup["train_vs_test"] + dup["val_vs_test"] == 1
    assert "voc_box_fixes" in captured["extra"]
