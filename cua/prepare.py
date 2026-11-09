"""Dataset preparation CLI: python -m cua.prepare <dataset> ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cua.gates import GateFailure, class_box_stats, cooccurrence_rate, reference_split
from cua.records import Record, write_classes, write_records

MIN_COOCCURRENCE = 0.20


def _class_totals(records: list[Record], n_classes: int) -> list[int]:
    return [s["count"] for s in class_box_stats(records, n_classes)]


def prepare_dataset(name: str, splits: dict, classes: tuple, out_dir: Path, image_root: Path) -> dict:
    """Run data gates, carve the reference split, and write the dataset directory."""
    n_classes = len(classes)
    train = splits["train"]
    co = cooccurrence_rate(train)
    if co < MIN_COOCCURRENCE:
        raise GateFailure(f"{name}: co-occurrence {co:.3f} < {MIN_COOCCURRENCE}")
    rest, ref = reference_split(train, n_classes)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records(out_dir / "train.jsonl", rest)
    write_records(out_dir / "reference.jsonl", ref)
    write_records(out_dir / "val.jsonl", splits["val"])
    write_records(out_dir / "test.jsonl", splits["test"])
    write_classes(out_dir / "classes.json", tuple(classes))
    (out_dir / "source.json").write_text(
        json.dumps({"image_root": str(Path(image_root))}), encoding="utf-8"
    )
    gates = {
        "dataset": name,
        "cooccurrence": co,
        "class_stats": class_box_stats(train, n_classes),
        "reference_class_counts": _class_totals(ref, n_classes),
        "split_sizes": {
            "train": len(rest),
            "reference": len(ref),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
    }
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    return gates


def _cmd_rdd(args: argparse.Namespace) -> None:
    from cua.datasets.rdd import RDD_CLASSES, load_rdd

    prepare_dataset("rdd", load_rdd(Path(args.splits)), RDD_CLASSES, Path(args.out), Path(args.images))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cua.prepare")
    sub = p.add_subparsers(dest="dataset", required=True)
    rdd = sub.add_parser("rdd", help="prepare the RDD dataset")
    rdd.add_argument("--splits", required=True)
    rdd.add_argument("--images", required=True)
    rdd.add_argument("--out", required=True)
    rdd.set_defaults(func=_cmd_rdd)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except GateFailure as e:
        print(f"GATE FAILURE: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
