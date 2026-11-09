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


def test_voc_cli_subcommand_parses_devkit_and_out(tmp_path):
    from cua.prepare import build_parser

    args = build_parser().parse_args(["voc", "--devkit", str(tmp_path), "--out", str(tmp_path / "o")])
    assert args.dataset == "voc"
    assert args.devkit == str(tmp_path)
    assert args.out == str(tmp_path / "o")
