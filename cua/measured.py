"""Build the measured-effect policy tables from the edit-wise study's deltas.csv.

For each (dataset, detector) in the CSV, writes <data-root>/<dataset>/measured_<detector>.json with
the mean per-class dAP50 over seeds and its standard error (s.d. / sqrt(n_seeds)), laid out as
[edit][class] in valid_menu(dataset) and classes.json order.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from cua.augment.menu import valid_menu
from cua.records import read_classes


def measured_table(rows: list[dict], dataset: str, classes: tuple[str, ...]) -> dict:
    menu = valid_menu(dataset)
    cells: dict[tuple[str, str], dict[int, float]] = {}
    for r in rows:
        cells.setdefault((r["edit"], r["cls"]), {})[int(r["seed"])] = float(r["delta"])
    seeds = sorted({s for v in cells.values() for s in v})
    if len(seeds) < 2:
        raise ValueError(f"{dataset}: need deltas from at least 2 seeds for a standard error, got {seeds}")
    mean, se = [], []
    for e in menu:
        mrow, srow = [], []
        for c in classes:
            v = cells.get((e, c), {})
            if sorted(v) != seeds:
                raise ValueError(f"{dataset}: cell ({e}, {c}) has seeds {sorted(v)}, expected {seeds}")
            vals = [v[s] for s in seeds]
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
            mrow.append(m)
            srow.append(sd / math.sqrt(len(vals)))
        mean.append(mrow)
        se.append(srow)
    return {"edits": list(menu), "classes": list(classes), "mean": mean, "se": se, "n_seeds": len(seeds)}


def epoch_deltas(runs_root: Path, epoch: int) -> list[dict]:
    """Part A deltas (arm - base, per class AP50) at one arm epoch instead of the last one. With the
    first arm epoch this is the 1-epoch probe used as the CUA v2 prior (fix 4): a third of the
    Part A arm budget. Only complete A- runs are read."""
    from cua.analysis.part_a import _A_RE, deltas
    from cua.train import read_history

    rows = []
    for d in sorted(Path(runs_root).iterdir()):
        m = _A_RE.match(d.name) if d.is_dir() else None
        if not m or m[4] == "warm" or not (d / "COMPLETE.json").exists():
            continue
        hist = [h for h in read_history(d) if int(h["epoch"]) == epoch]
        if not hist:
            raise ValueError(f"{d.name}: no history row for epoch {epoch}")
        per = (hist[0].get("metrics") or {}).get("per_class") or {}
        for cls, v in per.items():
            if v.get("ap50") is not None:
                rows.append({"dataset": m[1], "detector": m[2], "seed": int(m[3]), "edit": m[4],
                             "cls": cls, "ap50": float(v["ap50"])})
    return deltas(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cua.measured")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--deltas", type=Path, help="Part A deltas.csv (last arm epoch) -> measured_<detector>.json")
    src.add_argument("--runs-root", type=Path, help="read A- runs at --epoch -> <name>_<detector>.json")
    ap.add_argument("--epoch", type=int, default=4)
    ap.add_argument("--name", default=None, help="table name (default: measured, or probe with --runs-root)")
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.deltas is not None:
        with open(a.deltas, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        name = a.name or "measured"
    else:
        rows = epoch_deltas(a.runs_root, a.epoch)
        name = a.name or "probe"
    for ds, det in sorted({(r["dataset"], r["detector"]) for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds and r["detector"] == det]
        classes = read_classes(a.data_root / ds / "classes.json")
        out = a.data_root / ds / f"{name}_{det}.json"
        out.write_text(json.dumps(measured_table(sub, ds, classes), indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
