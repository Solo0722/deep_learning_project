"""One-time test-set evaluation of Part C main-comparison runs (spec §4, §10) and of the
class-adaptive copy-paste pilot runs.

CLI: python -m cua.test_eval --runs-root DIR --data-root DIR [--device cuda] [--prefix C-|P-]

For every complete `C-<dataset>-<detector>-s<seed>-<policy>` (or, with --prefix P-,
`P-<dataset>-<detector>-s<seed>-<arm>`) run, loads best.pt (best validation macro AP50), predicts on
the dataset's test.jsonl and writes test_predictions.json and test_metrics.json into the run
directory. A run that already has test_metrics.json is skipped, so the test set is scored once per run
and resubmitting the job continues where it stopped. Only Part C and copy-paste ids are accepted:
tuning, ablation, Part A, budget and proxy runs never touch test.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from cua.data import DetectionDataset
from cua.evaluate import coco_metrics, predict
from cua.experiments.specs import resolve
from cua.models import build_detector
from cua.records import read_classes, read_records
from cua.train import _predictions_json, _write_json

PART_C_RE = re.compile(r"^C-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
PASTE_RE = re.compile(r"^P-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
FP_SCORE = 0.5


def fp_images(records, predictions, threshold: float = FP_SCORE) -> dict:
    """Target-free images (no non-ignored GT) with at least one detection scoring >= threshold."""
    free = {i for i, r in enumerate(records) if not (~np.asarray(r.ignore, dtype=bool)).any()}
    hit = {int(p["image_index"]) for p in predictions
           if int(p["image_index"]) in free and np.any(np.asarray(p["scores"], dtype=float) >= threshold)}
    return {"count": len(hit), "n_target_free": len(free),
            "fraction": (len(hit) / len(free)) if free else None}


def evaluate_run(run_dir: Path, data_root: Path, device: str = "cuda", batch_size: int = 8,
                 workers: int = 4) -> dict:
    run_dir, data_root = Path(run_dir), Path(data_root)
    rid = run_dir.name
    if not (run_dir / "COMPLETE.json").exists():
        raise RuntimeError(f"{rid}: incomplete (no COMPLETE.json)")
    if not (PART_C_RE.match(rid) or PASTE_RE.match(rid)):
        raise ValueError(f"{rid}: only Part C (C-...) and copy-paste (P-...) runs are evaluated on test")
    cfg = resolve(rid)
    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("digest")
    if stored != cfg.digest():
        raise RuntimeError(f"{rid}: config digest {stored} differs from the current spec's {cfg.digest()}")
    ds_dir = data_root / cfg.dataset
    classes = read_classes(ds_dir / "classes.json")
    image_root = Path(json.loads((ds_dir / "source.json").read_text(encoding="utf-8"))["image_root"])
    if not image_root.is_absolute():
        image_root = ds_dir / image_root
    records = read_records(ds_dir / "test.jsonl")
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model = build_detector(cfg.detector, len(classes), cfg.size, pretrained=False)
    model.load_state_dict(best["model"])
    preds = predict(model, DetectionDataset(records, image_root, None, cfg.size), device,
                    batch_size=batch_size, num_workers=workers)
    metrics = coco_metrics(records, preds, classes)
    metrics.update({"fp_images": fp_images(records, preds), "best_epoch": best.get("best_epoch"),
                    "best_val_macro_ap50": best.get("best_macro"), "n_test_images": len(records),
                    "run_id": rid, "digest": stored})
    _write_json(run_dir / "test_predictions.json", _predictions_json(preds), compact=True)
    _write_json(run_dir / "test_metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cua.test_eval")
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prefix", default="C-", choices=("C-", "P-"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    refused = 0
    for d in sorted(p for p in a.runs_root.iterdir() if p.is_dir() and p.name.startswith(a.prefix)):
        if (d / "test_metrics.json").exists():
            print(f"{d.name}: already evaluated")
        elif not (d / "COMPLETE.json").exists():
            print(f"{d.name}: incomplete, not evaluated")
        else:
            try:
                m = evaluate_run(d, a.data_root, device=a.device, workers=a.workers)
                print(f"{d.name}: evaluated, test macro AP50 {m['macro_ap50']}")
            except (ValueError, RuntimeError, KeyError) as e:
                refused += 1
                print(f"{d.name}: refused: {e}")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
