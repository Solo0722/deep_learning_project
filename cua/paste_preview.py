"""Save a few copy-paste examples with boxes drawn, to check pastes look sensible.

python -m cua.paste_preview --data-root DIR --dataset rdd --out DIR [--n 8] [--mode balanced]
Pasted boxes are drawn in red, original boxes in green.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cua.augment.paste import CopyPaste
from cua.image import letterbox, load_rgb
from cua.records import read_classes, read_records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cua.paste_preview")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--region", default="full", choices=("full", "lower"))
    a = ap.parse_args(argv)
    ds = a.data_root / a.dataset
    classes = read_classes(ds / "classes.json")
    root = Path(json.loads((ds / "source.json").read_text(encoding="utf-8"))["image_root"])
    root = root if root.is_absolute() else ds / root
    recs = read_records(ds / "train.jsonl")
    cp = CopyPaste(recs, len(classes), a.mode, 0, p=1.0, region=a.region)
    load = lambda j: letterbox(load_rgb(root, recs[j]), recs[j], 640)[0]  # noqa: E731
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    done = 0
    for i in rng.permutation(len(recs)):
        out, pasted = cp(load(int(i)), 1, int(i), load)
        if not pasted:
            continue
        im = Image.fromarray(out.rgb)
        d = ImageDraw.Draw(im)
        n0 = len(out.boxes) - len(pasted)
        for k, (box, lab) in enumerate(zip(out.boxes, out.labels)):
            d.rectangle(box.tolist(), outline=(255, 0, 0) if k >= n0 else (0, 255, 0), width=2)
            d.text((box[0] + 2, box[1] + 2), classes[int(lab)], fill=(255, 255, 0))
        im.save(a.out / f"paste_{done}.jpg", quality=85)
        done += 1
        if done >= a.n:
            break
    print(f"bank sizes: {dict(zip(classes, [len(b) for b in cp.bank]))}; wrote {done} previews to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
