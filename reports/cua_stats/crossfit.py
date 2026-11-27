"""Cross-fitted per-class selection gain (paper Table "Cross-Fitted Per-Class Selection Gain", Eq. crossfit).

python reports/cua_stats/crossfit.py reports/cua_part_a_report/deltas.csv

For each dataset-detector pair and held-out seed k, each class's edit (or none) and one global edit (or none)
are chosen from the other two seeds; the fold gain is the held-out macro delta of the per-class choice minus
that of the global choice (cua.analysis.local.selection_gain). "Pixel edits" restricts the menu to the
colour and image-quality edits.
"""
import csv
import sys
from collections import defaultdict

from cua.analysis.local import selection_gain

PIXEL = {"brightness", "contrast", "saturation", "hue", "blur", "noise", "jpeg", "sharpen"}

rows = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    rows[(r["dataset"], r["detector"])].append(
        {"seed": int(r["seed"]), "edit": r["edit"], "cls": r["cls"], "delta": float(r["delta"])})

print(f"{'dataset':8s} {'det':11s} {'pixel folds':26s} {'pixel':>6s} {'all':>6s}")
for (ds, det), d in sorted(rows.items(), key=lambda kv: (["rdd", "voc", "dacl10k"].index(kv[0][0]), kv[0][1])):
    px = selection_gain([r for r in d if r["edit"] in PIXEL])
    al = selection_gain(d)
    folds = ", ".join(f"{f['gain']:+.2f}" for f in px["folds"])
    print(f"{ds:8s} {det:11s} {folds:26s} {px['mean']:+6.2f} {al['mean']:+6.2f}")
