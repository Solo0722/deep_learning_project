"""Fig. 1 (teaser) for the paper.

Writes, into research_paper/figures/:
  teaser_effects.pdf   per-class change in validation AP50 (edit - base recipe) for each chosen edit,
                       mean over seeds, error bars = +/-1 s.d. over seeds (needs >= 2 seeds).
  teaser_<name>.png    (optional, with --image) the chosen image unedited and under each edit,
                       rendered with the exact training edits in cua.augment.pixel.

Input: deltas.csv from the edit-wise measurement study
  python -m cua.analysis.part_a --runs-root <runs> --out <dir>      # writes <dir>/deltas.csv

Example (run from the repository root):
  python research_paper/figures/build_teaser.py --deltas <dir>/deltas.csv \
      --dataset rdd --detector fasterrcnn --edits saturation,blur --classes D00,D40 \
      --image <img.jpg> --boxes <boxes.json>

boxes.json: [{"box": [x1, y1, x2, y2], "label": "D00"}, ...] in the image's pixel coordinates.
Existing outputs are not overwritten unless --force is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "ooo"]
GREYS = ["0.15", "0.55", "0.85", "0.35", "0.7", "1.0"]
BOX_COLOURS = [(230, 60, 40), (40, 120, 230), (250, 190, 30), (60, 180, 90)]


def per_class_effects(path: Path, dataset: str, detector: str, edits: list[str], classes: list[str]):
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["detector"] == detector and r["edit"] in edits and r["cls"] in classes:
                cells[(r["edit"], r["cls"])].append(float(r["delta"]))
    missing = [(e, c) for e in edits for c in classes if not cells.get((e, c))]
    if missing:
        raise SystemExit(f"no rows in {path} for (edit, class): {missing}")
    mean = np.array([[np.mean(cells[(e, c)]) for c in classes] for e in edits])
    sd = np.array([[np.std(cells[(e, c)], ddof=1) if len(cells[(e, c)]) > 1 else 0.0 for c in classes] for e in edits])
    n = min(len(v) for v in cells.values())
    return mean, sd, n


def plot_effects(mean, sd, edits, classes, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "font.family": "serif", "hatch.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.45, 1.45))
    width = 0.8 / len(classes)
    x = np.arange(len(edits))
    for j, cls in enumerate(classes):
        ax.bar(x + (j - (len(classes) - 1) / 2) * width, mean[:, j], width, yerr=sd[:, j],
               color=GREYS[j % len(GREYS)], hatch=HATCHES[j % len(HATCHES)], edgecolor="black",
               linewidth=0.6, error_kw={"elinewidth": 0.6, "capsize": 1.5}, label=cls)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x, [e.replace("_", " ") for e in edits])
    ax.set_ylabel(r"$\Delta$AP50 vs. base")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=len(classes), loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(pad=0.2)
    fig.savefig(out)
    plt.close(fig)


def render_panels(image: Path, boxes: Path, edits: list[str], classes: list[str], seed: int, force: bool) -> None:
    sys.path.insert(0, str(ROOT))
    from PIL import Image, ImageDraw
    from cua.augment.pixel import PIXEL_EDITS
    from cua.image import Sample

    bad = [e for e in edits if e not in PIXEL_EDITS]
    if bad:
        raise SystemExit(f"panels support pixel edits only ({sorted(PIXEL_EDITS)}); got {bad}")
    ann = json.loads(boxes.read_text(encoding="utf-8"))
    with Image.open(image) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    sample = Sample(rgb, np.array([a["box"] for a in ann], dtype=np.float32),
                    np.zeros(len(ann), dtype=np.int64), np.zeros(len(ann), dtype=bool))
    views = {"unedited": sample}
    for e in edits:  # one fixed strength draw per edit, from the training strength range
        views[e] = PIXEL_EDITS[e](sample, np.random.default_rng(seed))
    for name, s in views.items():
        out = OUT_DIR / f"teaser_{name}.png"
        if out.exists() and not force:
            raise SystemExit(f"{out} exists; use --force to overwrite")
        canvas = Image.fromarray(s.rgb)
        draw = ImageDraw.Draw(canvas)
        lw = max(2, round(min(canvas.size) / 150))
        for a in ann:
            colour = BOX_COLOURS[classes.index(a["label"]) % len(BOX_COLOURS)] if a["label"] in classes else (255, 255, 255)
            draw.rectangle(a["box"], outline=colour, width=lw)
        canvas.save(out)
        print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deltas", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--detector", default="fasterrcnn")
    ap.add_argument("--edits", required=True, help="comma-separated edit names as in deltas.csv")
    ap.add_argument("--classes", required=True, help="comma-separated class names as in deltas.csv")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--boxes", type=Path)
    ap.add_argument("--seed", type=int, default=0, help="strength draw for the image panels")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    edits, classes = args.edits.split(","), args.classes.split(",")

    mean, sd, n = per_class_effects(args.deltas, args.dataset, args.detector, edits, classes)
    if n < 2:
        print("warning: fewer than 2 seeds per cell; error bars are zero", file=sys.stderr)
    out = OUT_DIR / "teaser_effects.pdf"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; use --force to overwrite")
    plot_effects(mean, sd, edits, classes, out)
    print(f"wrote {out} (seeds per cell: {n})")
    for e, row in zip(edits, mean):
        print("  " + e + ": " + ", ".join(f"{c} {v:+.2f}" for c, v in zip(classes, row)))
    if args.image:
        if not args.boxes:
            raise SystemExit("--image needs --boxes")
        render_panels(args.image, args.boxes, edits, classes, args.seed, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
