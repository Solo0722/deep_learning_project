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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cua.measured")
    ap.add_argument("--deltas", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args(argv)
    with open(a.deltas, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for ds, det in sorted({(r["dataset"], r["detector"]) for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds and r["detector"] == det]
        classes = read_classes(a.data_root / ds / "classes.json")
        out = a.data_root / ds / f"measured_{det}.json"
        out.write_text(json.dumps(measured_table(sub, ds, classes), indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
