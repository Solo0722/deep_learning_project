"""Fig. 3: per-class effect of each edit (edit-wise measurement study), 2 detectors x 3 datasets.

Reads reports/cua_part_a_report/deltas.csv and tests.json; writes research_paper/figures/measurement_heatmaps.pdf.
Cell colour = mean over seeds of Delta AP50 (edit - base, validation); dot = same sign in all three seeds.
Run from the repository root:  python research_paper/figures/build_measurement_heatmaps.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cua_part_a_report"
OUT = Path(__file__).resolve().parent / "measurement_heatmaps.pdf"

EDITS = ["translate", "rotate", "vflip", "brightness", "contrast", "saturation", "hue",
         "blur", "noise", "jpeg", "sharpen", "erase", "mosaic"]
EDIT_LABEL = {"translate": "Translation", "rotate": "Rotation", "vflip": "Vertical flip",
              "brightness": "Brightness", "contrast": "Contrast", "saturation": "Saturation", "hue": "Hue",
              "blur": "Blur", "noise": "Noise", "jpeg": "JPEG", "sharpen": "Sharpening",
              "erase": "Random erasing", "mosaic": "Mosaic"}
DATASETS = [("rdd", "RDD2022"), ("voc", "PASCAL VOC"), ("dacl10k", "dacl10k")]
DETECTORS = [("fasterrcnn", "Faster R-CNN"), ("retinanet", "RetinaNet")]
VMAX = 3.0
CMAP = LinearSegmentedColormap.from_list("cua", ["#A8322A", "#E58A7F", "#F7F7F7", "#7FC29B", "#1F6E43"])


def load():
    cells = defaultdict(list)
    order = defaultdict(list)
    with open(REPORT / "deltas.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cells[(r["dataset"], r["detector"], r["edit"], r["cls"])].append(float(r["delta"]))
            if r["cls"] not in order[r["dataset"]]:
                order[r["dataset"]].append(r["cls"])
    tests = json.loads((REPORT / "tests.json").read_text(encoding="utf-8"))
    return cells, order, tests


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else f"p = {p:.2f}"


def main() -> None:
    cells, order, tests = load()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5})
    widths = [max(len(order[d]), 7) for d, _ in DATASETS]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.7), gridspec_kw={"width_ratios": widths, "wspace": 0.06, "hspace": 0.12})
    norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0.0, vmax=VMAX)
    for i, (det, det_name) in enumerate(DETECTORS):
        for j, (ds, ds_name) in enumerate(DATASETS):
            ax = axes[i, j]
            classes = order[ds]
            grid = np.full((len(EDITS), len(classes)), np.nan)
            for a, e in enumerate(EDITS):
                for b, c in enumerate(classes):
                    v = cells.get((ds, det, e, c))
                    if v:
                        grid[a, b] = st.mean(v)
                        if len(v) == 3 and (all(x > 0 for x in v) or all(x < 0 for x in v)):
                            ax.plot(b, a, marker="o", ms=1.6, color="black", mew=0)
            ax.imshow(np.where(np.isnan(grid), 0, grid), cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
            for a in range(len(EDITS)):
                if np.isnan(grid[a]).all():
                    ax.add_patch(plt.Rectangle((-0.5, a - 0.5), len(classes), 1, color="#DDE2E6", lw=0))
                    ax.text((len(classes) - 1) / 2, a, "not valid", ha="center", va="center", fontsize=5, color="#6B7075")
            ax.set_xticks(np.arange(len(classes)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(EDITS)) - 0.5, minor=True)
            ax.grid(which="minor", color="white", lw=0.6)
            ax.tick_params(which="both", length=0)
            ax.set_yticks(range(len(EDITS)))
            ax.set_yticklabels([EDIT_LABEL[e] for e in EDITS] if j == 0 else [])
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes if i == 1 else [], rotation=90, fontsize=5.5)
            t = tests[f"{ds}-{det}"]["interaction"]
            title = (f"{ds_name}\n" if i == 0 else "") + f"F = {t['F']:.2f}, {fmt_p(t['p'])}"
            ax.set_title(title, fontsize=6.5, fontweight="bold" if t["p"] < 0.05 else "normal",
                         color="#1B5E4B" if t["p"] < 0.05 else "#6B7075", pad=3)
            if j == 0:
                ax.set_ylabel(det_name, fontsize=7.5, fontweight="bold", labelpad=4)
            for s in ax.spines.values():
                s.set_visible(False)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, extend="both")
    cb.set_label(r"$\Delta$AP50 (edit $-$ base)", fontsize=6.5)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.135, right=0.9, top=0.9, bottom=0.17)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
