"""Offline check of the online class-utility estimator (diagnosis after Part C).

Part C found the estimator's utilities ranked edits almost opposite to Part A's measured effects
(strong edits such as mosaic and blur scored highest). Hypothesis: the dot product
<g_o - g_0, h_c> mostly grows with |g_o|, i.e. with how much harder an edit makes the batch, not
with whether training on it helps. This script recomputes the estimator's raw gradients on saved
checkpoints and scores four functions of them:

  dot      lr-free original:   <g_o - g_0, h_c> / (L_c + delta)
  normed   g_o rescaled to |g_0| first: <g_o |g_0|/|g_o| - g_0, h_c> / (L_c + delta)
  cosdiff  cos(g_o, h_c) - cos(g_0, h_c)
  ratio    |g_o| / |g_0|  (per edit, the same for every class)

Collect on the cluster (GPU):
  python -m cua.analysis.utility_check collect --runs-root R --data-root D --dataset rdd \
      --detector fasterrcnn --run A-rdd-fasterrcnn-s0-warm --ckpt last.pt --rounds 20 --out x.json
Summarise (anywhere):
  python -m cua.analysis.utility_check summarize --deltas deltas.csv --out summary.json x.json ...
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

VARIANTS = ("dot", "normed", "cosdiff", "ratio")


def variants(g0: torch.Tensor, go: torch.Tensor, h: torch.Tensor, lc: torch.Tensor,
             delta: float = 0.01) -> dict[str, np.ndarray]:
    """(|O|, C) arrays for each variant from g0 (P,), go (|O|, P), h (C, P), lc (C,)."""
    g0, go, h, lc = g0.double(), go.double(), h.double(), lc.double()
    n0 = g0.norm().clamp(min=1e-30)
    no = go.norm(dim=1).clamp(min=1e-30)
    nh = h.norm(dim=1).clamp(min=1e-30)
    den = (lc + delta).unsqueeze(0)
    dot = ((go - g0.unsqueeze(0)) @ h.T) / den
    normed = ((go * (n0 / no).unsqueeze(1) - g0.unsqueeze(0)) @ h.T) / den
    cosdiff = (go @ h.T) / (no.unsqueeze(1) * nh.unsqueeze(0)) - ((h @ g0) / (n0 * nh)).unsqueeze(0)
    ratio = (no / n0).unsqueeze(1).expand(-1, h.shape[0])
    return {k: v.cpu().numpy() for k, v in
            {"dot": dot, "normed": normed, "cosdiff": cosdiff, "ratio": ratio}.items()}


# ---- collect (GPU) ----

def collect(runs_root: Path, data_root: Path, dataset: str, detector: str, run: str, ckpt: str,
            rounds: int, seed: int, theta: str, out: Path, workers: int) -> None:
    from cua.augment.menu import ArmTransform, valid_menu
    from cua.augment.rng import keyed_rng
    from cua.data import DetectionDataset
    from cua.estimator.utility import Estimator
    from cua.models import build_detector
    from cua.records import read_classes, read_records

    ds_dir = data_root / dataset
    classes = read_classes(ds_dir / "classes.json")
    image_root = Path(json.loads((ds_dir / "source.json").read_text(encoding="utf-8"))["image_root"])
    if not image_root.is_absolute():
        image_root = ds_dir / image_root
    train_recs = read_records(ds_dir / "train.jsonl")
    ref_recs = read_records(ds_dir / "reference.jsonl")
    train_ds = DetectionDataset(train_recs, image_root, ArmTransform(dataset, None, seed, 0.5), 640)
    train_ds.set_epoch(1)
    ref_ds = DetectionDataset(ref_recs, image_root, None, 640)
    menu = valid_menu(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector(detector, len(classes), 640, pretrained=False).to(device)
    state = torch.load(runs_root / run / ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.train()
    est = Estimator(model, detector, theta, menu, len(classes), device=device, workers=workers)

    rows = []
    for r in range(rounds):
        t0 = time.monotonic()
        g0, go, h, lc, valid_c = est.gradients(lambda i, e: train_ds[(i, e)], lambda i: ref_ds[i],
                                               len(train_ds), ref_recs, keyed_rng(seed, 0, r, "utility_check"))
        v = variants(g0, go, h, lc, est.delta)
        rows.append({"round": r, "valid": valid_c.tolist(), "Lc": lc.cpu().tolist(),
                     **{k: a.tolist() for k, a in v.items()}})
        print(f"{run}/{ckpt} round {r}: {time.monotonic() - t0:.1f} s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": dataset, "detector": detector, "run": run, "ckpt": ckpt,
                               "theta": theta, "menu": list(menu), "classes": list(classes),
                               "rows": rows}), encoding="utf-8")


# ---- summarise ----

def _rank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(a)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def part_a_means(deltas: Path, dataset: str, detector: str) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    with deltas.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["detector"] == detector and row["delta"] not in ("", "nan"):
                acc[(row["edit"], row["cls"])].append(float(row["delta"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def summarize_one(d: dict, part_a: dict[tuple[str, str], float]) -> dict:
    menu, classes = d["menu"], d["classes"]
    valid = np.array([r["valid"] for r in d["rows"]], bool)  # (R, C)
    means = {}
    for k in VARIANTS:
        a = np.array([r[k] for r in d["rows"]], float)  # (R, O, C)
        m = np.where(valid[:, None, :], a, np.nan)
        means[k] = np.nanmean(m, axis=0)  # (O, C)
    cells = [(i, j) for i, e in enumerate(menu) for j, c in enumerate(classes)
             if (e, c) in part_a and np.isfinite(means["dot"][i, j])]
    pa = np.array([part_a[(menu[i], classes[j])] for i, j in cells])
    out = {"run": d["run"], "ckpt": d["ckpt"], "n_cells": len(cells), "n_rounds": len(d["rows"]),
           "all_cells": {}, "within_edit": {}, "edit_level": {}, "positive_share": {}}
    for k in VARIANTS:
        u = np.array([means[k][i, j] for i, j in cells])
        out["all_cells"][k] = spearman(u, pa)
        # class signal only: remove each edit's mean from both sides
        ei = np.array([i for i, _ in cells])
        ud, pd = u.copy(), pa.copy()
        for i in set(ei.tolist()):
            ud[ei == i] -= ud[ei == i].mean()
            pd[ei == i] -= pd[ei == i].mean()
        out["within_edit"][k] = spearman(ud, pd) if k != "ratio" else float("nan")
        edits = sorted(set(ei.tolist()))
        out["edit_level"][k] = spearman([u[ei == i].mean() for i in edits], [pa[ei == i].mean() for i in edits])
        out["positive_share"][k] = float(np.mean(u > (1.0 if k == "ratio" else 0.0)))
    out["dot_vs_ratio_edit_level"] = spearman([means["dot"][i].mean() for i in range(len(menu))],
                                              [means["ratio"][i, 0] for i in range(len(menu))])
    out["per_edit"] = {menu[i]: {"dot": float(np.nanmean(means["dot"][i])),
                                 "ratio": float(means["ratio"][i, 0]),
                                 "cosdiff": float(np.nanmean(means["cosdiff"][i])),
                                 "part_a": float(np.mean([part_a[(menu[i], c)] for c in classes
                                                          if (menu[i], c) in part_a] or [np.nan]))}
                       for i in range(len(menu))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cua.analysis.utility_check")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs-root", type=Path, required=True)
    c.add_argument("--data-root", type=Path, required=True)
    c.add_argument("--dataset", required=True)
    c.add_argument("--detector", required=True)
    c.add_argument("--run", required=True)
    c.add_argument("--ckpt", default="best.pt")
    c.add_argument("--rounds", type=int, default=20)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--theta", default="head_fpn")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--deltas", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("inputs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.runs_root, a.data_root, a.dataset, a.detector, a.run, a.ckpt, a.rounds, a.seed, a.theta,
                a.out, a.workers)
        return 0
    res = []
    for p in a.inputs:
        d = json.loads(p.read_text(encoding="utf-8"))
        r = summarize_one(d, part_a_means(a.deltas, d["dataset"], d["detector"]))
        r.update(dataset=d["dataset"], detector=d["detector"])
        res.append(r)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
