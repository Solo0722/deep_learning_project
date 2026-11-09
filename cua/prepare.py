"""Dataset preparation CLI: python -m cua.prepare <dataset> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cua.gates import GateFailure, class_box_stats, cooccurrence_rate, reference_split
from cua.records import Record, write_classes, write_records

MIN_COOCCURRENCE = 0.20


def _class_totals(records: list[Record], n_classes: int) -> list[int]:
    return [s["count"] for s in class_box_stats(records, n_classes)]


def prepare_dataset(
    name: str,
    splits: dict,
    classes: tuple,
    out_dir: Path,
    image_root: Path,
    *,
    n_target: int = 1000,
    min_per_class: int = 100,
    seed: int = 0,
    extra: dict | None = None,
) -> dict:
    """Run data gates, carve the reference split, and write the dataset directory."""
    n_classes = len(classes)
    train = splits["train"]
    co = cooccurrence_rate(train)
    if co < MIN_COOCCURRENCE:
        raise GateFailure(f"{name}: co-occurrence {co:.3f} < {MIN_COOCCURRENCE}")
    rest, ref = reference_split(train, n_classes, n_target, min_per_class, seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records(out_dir / "train.jsonl", rest)
    write_records(out_dir / "reference.jsonl", ref)
    write_records(out_dir / "val.jsonl", splits["val"])
    write_records(out_dir / "test.jsonl", splits["test"])
    write_classes(out_dir / "classes.json", tuple(classes))
    (out_dir / "source.json").write_text(
        json.dumps({"image_root": str(Path(image_root).resolve())}), encoding="utf-8"
    )
    ref_counts = _class_totals(ref, n_classes)
    gates = {
        "dataset": name,
        "cooccurrence": co,
        "class_stats": class_box_stats(train, n_classes),
        "reference_class_counts": ref_counts,
        "reference_shortfall": [classes[c] for c in range(n_classes) if ref_counts[c] < min_per_class],
        "split_sizes": {
            "train": len(rest),
            "reference": len(ref),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
    }
    gates.update(extra or {})
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    return gates


def _warn_shortfall(gates: dict) -> None:
    if gates["reference_shortfall"]:
        print(
            f"WARNING: reference quota unmet for classes {gates['reference_shortfall']}",
            file=sys.stderr,
        )


def _cmd_rdd(args: argparse.Namespace) -> None:
    from cua.datasets.rdd import RDD_CLASSES, load_rdd

    gates = prepare_dataset(
        "rdd", load_rdd(Path(args.splits)), RDD_CLASSES, Path(args.out), Path(args.images)
    )
    _warn_shortfall(gates)


def _voc_duplicate_counts(splits: dict, devkit: Path) -> dict:
    """Near-duplicate hash pairs between official VOC splits (reported, never regrouped)."""
    from cua.dedup import cross_split_duplicates, hash_records

    h = {k: hash_records(devkit, splits[k]) for k in ("train", "val", "test")}
    return {
        "train_vs_test": cross_split_duplicates(h["train"], h["test"]),
        "val_vs_test": cross_split_duplicates(h["val"], h["test"]),
    }


def _cmd_voc(args: argparse.Namespace) -> None:
    from cua.datasets.voc import VOC_CLASSES, load_voc

    devkit = Path(args.devkit)
    stats: dict = {}
    splits = load_voc(devkit, stats=stats)
    gates = prepare_dataset(
        "voc", splits, VOC_CLASSES, Path(args.out), devkit,
        extra={
            "voc_box_fixes": stats,
            "voc_cross_split_duplicates": _voc_duplicate_counts(splits, devkit),
        },
    )
    _warn_shortfall(gates)


def _cmd_dacl10k(args: argparse.Namespace) -> None:
    from cua.datasets.dacl10k import load_dacl10k

    root = Path(args.root)
    stats: dict = {}
    splits, retained = load_dacl10k(root, seed=args.seed, stats=stats)
    gates = prepare_dataset(
        "dacl10k", splits, retained, Path(args.out), root, seed=args.seed,
        extra={
            "dacl10k_retained_classes": list(retained),
            "dacl10k_excluded_classes": stats["excluded"],
            "dacl10k_parse_counts": {
                k: stats[k] for k in ("clipped_boxes", "dropped_degenerate")
            },
        },
    )
    _warn_shortfall(gates)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cua.prepare")
    sub = p.add_subparsers(dest="dataset", required=True)
    rdd = sub.add_parser("rdd", help="prepare the RDD dataset")
    rdd.add_argument("--splits", required=True)
    rdd.add_argument("--images", required=True)
    rdd.add_argument("--out", required=True)
    rdd.set_defaults(func=_cmd_rdd)
    voc = sub.add_parser("voc", help="prepare the PASCAL VOC dataset")
    voc.add_argument("--devkit", required=True, help="VOCdevkit directory (image root)")
    voc.add_argument("--out", required=True)
    voc.set_defaults(func=_cmd_voc)
    dacl = sub.add_parser("dacl10k", help="prepare the dacl10k dataset")
    dacl.add_argument("--root", required=True, help="dacl10k root (annotations/ and images/)")
    dacl.add_argument("--out", required=True)
    dacl.add_argument("--seed", type=int, default=0)
    dacl.set_defaults(func=_cmd_dacl10k)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except GateFailure as e:
        print(f"GATE FAILURE: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
