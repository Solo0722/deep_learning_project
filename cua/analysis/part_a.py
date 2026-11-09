"""Part A analysis: per-class augmentation deltas, interaction F-tests (H1), budget plateau, proxy agreement."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

from cua.augment.menu import MENU_ORDER
from cua.train import read_history

_A_RE = re.compile(r"^A-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
_PROXY_RE = re.compile(r"^proxy-rdd-fasterrcnn-e(\d+)-([^-]+)$")


# ---- loading ----

def _final_rows(run_dir: Path) -> list[tuple[str, float]]:
    hist = read_history(run_dir)
    if not hist:
        return []
    per_class = (hist[-1].get("metrics") or {}).get("per_class") or {}
    return [(c, float(v["ap50"])) for c, v in per_class.items() if v.get("ap50") is not None]


def load_final(runs_root: Path, prefix: str) -> list[dict]:
    """Last-epoch per-class AP50 of every complete Part A arm (warm runs skipped; base arm has edit 'base')."""
    out: list[dict] = []
    incomplete = 0
    no_data = 0
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        m = _A_RE.match(d.name)
        if not m or m[4] == "warm":
            continue
        if not (d / "COMPLETE.json").exists():
            incomplete += 1
            continue
        rows = _final_rows(d)
        if not rows:
            no_data += 1
        for cls, ap in rows:
            out.append({"dataset": m[1], "detector": m[2], "seed": int(m[3]), "edit": m[4],
                        "cls": cls, "ap50": ap})
    if incomplete:
        print(f"warning: skipped {incomplete} incomplete run(s) under {runs_root}", file=sys.stderr)
    if no_data:
        print(f"warning: {no_data} complete run(s) under {runs_root} have no per-class AP50 data", file=sys.stderr)
    return out


def deltas(rows: list[dict]) -> list[dict]:
    """arm - base matched on (dataset, detector, seed, cls); cells without a base are skipped."""
    base = {(r["dataset"], r["detector"], r["seed"], r["cls"]): r["ap50"] for r in rows if r["edit"] == "base"}
    out = []
    missing = 0
    for r in rows:
        if r["edit"] == "base":
            continue
        b = base.get((r["dataset"], r["detector"], r["seed"], r["cls"]))
        if b is None:
            missing += 1
            continue
        out.append({"dataset": r["dataset"], "detector": r["detector"], "seed": r["seed"],
                    "edit": r["edit"], "cls": r["cls"], "delta": r["ap50"] - b})
    if missing:
        print(f"warning: skipped {missing} arm cell(s) with no matching base", file=sys.stderr)
    return out


def proxy_deltas(runs_root: Path) -> list[dict]:
    """Per (edit, cls) delta of the proxy runs (last epoch, arm - base), one row each."""
    runs: dict[tuple[int, str], dict[str, float]] = {}
    for d in sorted(Path(runs_root).iterdir()):
        m = _PROXY_RE.match(d.name) if d.is_dir() else None
        if not m or not (d / "COMPLETE.json").exists():
            continue
        runs[(int(m[1]), m[2])] = dict(_final_rows(d))
    out = []
    for (n, edit), vals in runs.items():
        if edit == "base" or (n, "base") not in runs:
            continue
        for cls, ap in vals.items():
            if cls in runs[(n, "base")]:
                out.append({"epochs": n, "edit": edit, "cls": cls, "delta": ap - runs[(n, "base")][cls]})
    return out


# ---- statistics ----

def holm(pvalues: list[float]) -> list[float]:
    """Step-down Holm adjustment, monotone, capped at 1, in the original order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj


def _dummies(labels: list, drop_first: bool = True) -> list[np.ndarray]:
    levels = sorted(set(labels), key=str)
    if drop_first:
        levels = levels[1:]
    return [np.array([1.0 if x == lv else 0.0 for x in labels]) for lv in levels]


def _fit(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), int(y.size - rank)


def _f_test(y: np.ndarray, reduced: list[np.ndarray], full: list[np.ndarray]) -> dict:
    rss_r, df_r = _fit(np.column_stack(reduced), y)
    rss_f, df_f = _fit(np.column_stack(full), y)
    df1 = df_r - df_f
    if df1 <= 0 or df_f <= 0:
        raise ValueError(f"F-test has no degrees of freedom (df1={df1}, df2={df_f}): need more seeds/cells")
    num = max(rss_r - rss_f, 0.0) / df1
    if rss_f <= 1e-12 * max(1.0, float(y @ y)):
        return {"F": float("inf") if num > 0 else 0.0, "df1": df1, "df2": df_f, "p": 0.0 if num > 0 else 1.0}
    F = num / (rss_f / df_f)
    return {"F": float(F), "df1": df1, "df2": df_f, "p": float(stats.f.sf(F, df1, df_f))}


def _seed_cls(d: list[dict]) -> list[np.ndarray]:
    """seed:cls cell indicators (collinear with the seed and cls dummies; rank handles it)."""
    return _dummies([str(r["seed"]) + "|" + r["cls"] for r in d], drop_first=False)


def _single_pair(d: list[dict]) -> None:
    pairs = {(r["dataset"], r["detector"]) for r in d}
    if len(pairs) > 1:
        raise ValueError(f"Deltas mix dataset/detector pairs {sorted(pairs)}; test one pair at a time")


def interaction_test(d: list[dict]) -> dict:
    """Nested OLS F-test for H1 (one dataset x detector pair).

    Reduced: delta ~ seed + edit + cls + seed:cls; full: reduced + edit:cls. Every delta in a (seed, cls) cell is
    arm - base and shares that cell's base-run noise; the seed:cls dummies absorb it exactly (ruling R15).
    """
    _single_pair(d)
    y = np.array([r["delta"] for r in d], dtype=np.float64)
    main = [np.ones(len(d))] + _dummies([r["seed"] for r in d]) + _dummies([r["edit"] for r in d]) \
        + _dummies([r["cls"] for r in d]) + _seed_cls(d)
    inter = _dummies([r["edit"] + "\0" + r["cls"] for r in d], drop_first=False)
    return _f_test(y, main, main + inter)


def edit_class_tests(d: list[dict]) -> list[dict]:
    """Per edit: F-test for cls in delta ~ seed + cls; Holm across edits.

    Results are returned in first-appearance order of the edits in the input (each has F, df1, df2, p, p_holm).
    """
    _single_pair(d)
    res = []
    for e in dict.fromkeys(r["edit"] for r in d):
        sub = [r for r in d if r["edit"] == e]
        y = np.array([r["delta"] for r in sub], dtype=np.float64)
        base = [np.ones(len(sub))] + _dummies([r["seed"] for r in sub])
        res.append({"edit": e, **_f_test(y, base, base + _dummies([r["cls"] for r in sub]))})
    for r, p in zip(res, holm([r["p"] for r in res])):
        r["p_holm"] = p
    return res


def budget_plateaued(curve: list[float], tol: float = 0.3, tail_frac: float = 0.2) -> bool:
    if len(curve) < 2:
        raise ValueError(f"Budget curve needs at least 2 points, got {len(curve)}")
    return curve[-1] - curve[round((1 - tail_frac) * len(curve)) - 1] < tol


def budget_summary(runs_root: Path) -> dict:
    """Per budget-* run: macro_ap50 curve over epochs (history order), plateau flag, and completeness.
    `plateaued` is None (with an `error`) when the curve is too short or has missing values."""
    out: dict = {}
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir() or not d.name.startswith("budget-"):
            continue
        curve = [(h.get("metrics") or {}).get("macro_ap50") for h in read_history(d)]
        entry: dict = {"curve": curve, "plateaued": None, "complete": (d / "COMPLETE.json").exists()}
        if any(v is None for v in curve):
            entry["error"] = "curve has epochs without macro_ap50"
        else:
            try:
                entry["plateaued"] = bool(budget_plateaued([float(v) for v in curve]))
            except ValueError as e:
                entry["error"] = str(e)
        out[d.name] = entry
    return out


def proxy_agreement(short: list[dict], full: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman rho between Part A and proxy deltas over (edit, cls) cells, with a bootstrap CI over cells.

    `short`: Part A delta rows (short schedule, several seeds), averaged over seeds per (edit, cls).
    `full`: proxy-run delta rows (full schedule, one seed), one per (edit, cls) and one budget (`epochs`).
    Rows carrying several distinct `epochs` values raise ValueError.
    """
    budgets = {r["epochs"] for r in (*short, *full) if "epochs" in r}
    if len(budgets) > 1:
        raise ValueError(f"proxy_agreement needs a single budget, got epochs {sorted(budgets)}")
    acc: dict[tuple, list[float]] = {}
    for r in short:
        acc.setdefault((r["edit"], r["cls"]), []).append(r["delta"])
    fl = {(r["edit"], r["cls"]): r["delta"] for r in full}
    keys = [k for k in acc if k in fl]
    if len(keys) < 3:
        raise ValueError(f"Need at least 3 joined (edit, cls) cells, got {len(keys)}")
    a = np.array([np.mean(acc[k]) for k in keys])
    b = np.array([fl[k] for k in keys])
    rho = float(stats.spearmanr(a, b)[0])
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant resamples give NaN, dropped below
            r = stats.spearmanr(a[idx], b[idx])[0]
        if not np.isnan(r):
            rhos.append(r)
    lo, hi = (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))) if rhos else (float("nan"),) * 2
    return {"rho": rho, "lo": lo, "hi": hi, "n_cells": len(keys), "n_boot_valid": len(rhos)}


# ---- heatmap ----

def _edit_order(edits) -> list[str]:
    edits = list(dict.fromkeys(edits))
    return [e for e in MENU_ORDER if e in edits] + [e for e in edits if e not in MENU_ORDER]


def heatmap(d: list[dict], out_png: Path) -> None:
    """Mean delta per edit (rows, menu order) x class; bold where the seed-mean 95% CI excludes 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edits = _edit_order(r["edit"] for r in d)
    classes = list(dict.fromkeys(r["cls"] for r in d))
    cell: dict[tuple, list[float]] = {}
    for r in d:
        cell.setdefault((r["edit"], r["cls"]), []).append(r["delta"])
    mean = np.full((len(edits), len(classes)), np.nan)
    sig = np.zeros(mean.shape, dtype=bool)
    for i, e in enumerate(edits):
        for j, c in enumerate(classes):
            v = cell.get((e, c))
            if not v:
                continue
            mean[i, j] = np.mean(v)
            if len(v) > 1:
                half = stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
                sig[i, j] = abs(mean[i, j]) > half
    lim = max(float(np.nanmax(np.abs(mean))) if np.isfinite(mean).any() else 1.0, 1e-9)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.9 * len(classes) + 2), max(3.0, 0.5 * len(edits) + 1.5)))
    im = ax.imshow(mean, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(edits)), edits)
    for i in range(len(edits)):
        for j in range(len(classes)):
            if np.isfinite(mean[i, j]):
                ax.text(j, i, f"{mean[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(mean[i, j]) > 0.6 * lim else "black",
                        fontweight="bold" if sig[i, j] else "normal")
    fig.colorbar(im, ax=ax, label="mean delta AP50 (pts)")
    ax.set_title("Per-class AP50 delta (bold: seed-mean CI excludes 0)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---- CLI ----

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Part A analysis")
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="A-")
    ap.add_argument("--budget", action="store_true",
                    help="also write budget.json: macro_ap50 curve and plateau flag of every budget-* run")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    d = deltas(load_final(args.runs_root, args.prefix))
    with open(args.out / "deltas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "detector", "seed", "edit", "cls", "delta"])
        w.writeheader()
        w.writerows(d)

    tests: dict = {}
    for ds, det in sorted({(r["dataset"], r["detector"]) for r in d}):
        sub = [r for r in d if r["dataset"] == ds and r["detector"] == det]
        entry: dict = {}
        for name, fn in (("interaction", interaction_test), ("edit_class", edit_class_tests)):
            try:
                entry[name] = fn(sub)
            except ValueError as e:
                entry[name] = {"error": str(e)}
        tests[f"{ds}-{det}"] = entry
        heatmap(sub, args.out / f"heatmap_{ds}_{det}.png")
    (args.out / "tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")

    proxy_rows = proxy_deltas(args.runs_root)
    if proxy_rows:
        short = [r for r in d if r["dataset"] == "rdd" and r["detector"] == "fasterrcnn"]
        proxy: dict = {}
        for n in sorted({r["epochs"] for r in proxy_rows}):
            full = [r for r in proxy_rows if r["epochs"] == n]
            try:
                res = proxy_agreement(short, full)
            except ValueError as e:
                res = {"error": str(e)}
            proxy[f"e{n}"] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in res.items()}
        (args.out / "proxy.json").write_text(json.dumps(proxy, indent=2), encoding="utf-8")

    if args.budget:
        (args.out / "budget.json").write_text(json.dumps(budget_summary(args.runs_root), indent=2),
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
