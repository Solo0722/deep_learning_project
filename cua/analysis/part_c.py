"""Part C statistics (spec §10-11): arm summaries, H2/H3 comparisons with a seed-level paired t interval
and a seed-pooled paired image bootstrap, H4 estimator agreement, validation-only ablations, estimator
overhead and secondary metrics.

Reads, per run directory under runs_root: history.jsonl, utility.jsonl (shared/cua), and for Part C
main-comparison runs test_metrics.json and test_predictions.json written by cua.test_eval. Ablation
(X-) runs are compared on best validation macro AP50 only; they never have test files.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from cua.analysis.ap50 import Matches, image_matches, paired_bootstrap_seeds
from cua.analysis.part_a import proxy_agreement, read_history
from cua.augment.menu import valid_menu
from cua.records import read_classes, read_records

ARMS = ("base", "uniform", "measured", "shared", "cua")
SEEDS = (0, 1, 2)
_C_RE = re.compile(r"^C-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
_X_RE = re.compile(r"^X-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_runs(runs_root: Path) -> list[dict]:
    """One row per complete C- or X- run."""
    out = []
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir() or not (d / "COMPLETE.json").exists():
            continue
        m = _C_RE.match(d.name) or _X_RE.match(d.name)
        if not m:
            continue
        hist = read_history(d)
        vals = [(h.get("metrics") or {}).get("macro_ap50") for h in hist]
        vals = [float(v) for v in vals if v is not None]
        est = [(h.get("policy") or {}).get("estimator_seconds") for h in hist]
        secs = sum(float(h.get("seconds") or 0.0) for h in hist)
        overhead = (sum(float(e) for e in est if e) / secs) if any(est) and secs > 0 else None
        tm = d / "test_metrics.json"
        out.append({"run_id": d.name, "dir": d, "kind": d.name[0], "dataset": m[1], "detector": m[2],
                    "seed": int(m[3]), "arm": m[4], "best_val": max(vals) if vals else None,
                    "overhead": overhead,
                    "test": json.loads(tm.read_text(encoding="utf-8")) if tm.exists() else None})
    return out


def _value(row: dict, metric: str):
    if metric == "val":
        return row["best_val"]
    return None if row["test"] is None else row["test"].get("macro_ap50")


def _by_seed(rows, dataset, detector, arm, metric) -> dict[int, float]:
    out = {}
    for r in rows:
        if (r["dataset"], r["detector"], r["arm"]) == (dataset, detector, arm):
            v = _value(r, metric)
            if v is not None:
                out[r["seed"]] = float(v)
    return out


def seed_t_interval(diffs) -> dict:
    d = np.asarray(list(diffs), dtype=np.float64)
    n = len(d)
    mean = float(d.mean()) if n else None
    if n < 2:
        return {"mean": mean, "lo": None, "hi": None, "n": n}
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n))
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": n}


def _excludes_zero(lo, hi) -> int:
    """+1 if the interval is above 0, -1 if below, 0 otherwise."""
    if lo is None or hi is None:
        return 0
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def build_matches(runs_root: Path, data_root: Path, rows) -> dict:
    """Matches per (dataset, detector, arm, seed) from test_predictions.json for evaluated C- runs."""
    cache: dict[str, tuple] = {}
    out = {}
    for r in rows:
        if r["kind"] != "C" or r["test"] is None:
            continue
        p = r["dir"] / "test_predictions.json"
        if not p.exists():
            continue
        ds = r["dataset"]
        if ds not in cache:
            dsd = Path(data_root) / ds
            cache[ds] = (read_records(dsd / "test.jsonl"), len(read_classes(dsd / "classes.json")))
        recs, nc = cache[ds]
        out[(ds, r["detector"], r["arm"], r["seed"])] = image_matches(
            recs, json.loads(p.read_text(encoding="utf-8")), nc)
    return out


def compare(rows, dataset: str, detector: str, arm_a: str, arm_b: str, metric: str = "test",
            matches: dict | None = None, n_boot: int = 1000) -> dict:
    """b - a over seeds present in both arms. claim: the t interval (and, for test, the bootstrap)
    excludes 0 on the same side."""
    a = _by_seed(rows, dataset, detector, arm_a, metric)
    b = _by_seed(rows, dataset, detector, arm_b, metric)
    seeds = sorted(set(a) & set(b))
    t = seed_t_interval([b[s] - a[s] for s in seeds])
    out = {"a": arm_a, "b": arm_b, "metric": metric, "n_seeds": len(seeds), "seeds": seeds, "t": t,
           "boot": None, "claim": False, "note": ""}
    if len(seeds) < 2:
        out["note"] = "fewer than 2 paired seeds"
        return out
    side = _excludes_zero(t["lo"], t["hi"])
    if metric == "test" and matches is not None:
        pairs = [(matches[(dataset, detector, arm_a, s)], matches[(dataset, detector, arm_b, s)])
                 for s in seeds if (dataset, detector, arm_a, s) in matches and (dataset, detector, arm_b, s) in matches]
        if len(pairs) == len(seeds):
            out["boot"] = paired_bootstrap_seeds(pairs, n_boot=n_boot)
            side = side if _excludes_zero(out["boot"]["lo"], out["boot"]["hi"]) == side else 0
        else:
            out["note"] = "test predictions missing for some seeds; bootstrap skipped"
            side = 0
    out["claim"] = side != 0
    return out


def h3_baseline(rows, dataset: str, detector: str) -> str:
    means = {}
    for arm in ("base", "uniform"):
        v = _by_seed(rows, dataset, detector, arm, "test")
        means[arm] = np.mean(list(v.values())) if v else -math.inf
    return "uniform" if means["uniform"] > means["base"] else "base"


def h4_agreement(runs_root: Path, deltas_rows, dataset: str, detector: str, classes, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """Spearman rho between time-averaged u(o,c) of the cua runs (valid estimates only, averaged over
    seeds) and Part A dAP50 (mean over seeds), with a bootstrap CI over (o,c) cells."""
    menu = valid_menu(dataset)
    per_run = []
    for d in sorted(Path(runs_root).glob(f"C-{dataset}-{detector}-s*-cua")):
        if not (d / "COMPLETE.json").exists():
            continue
        util = _read_jsonl(d / "utility.jsonl")
        if not util:
            continue
        u = np.array([r["u"] for r in util], dtype=np.float64)
        v = np.array([r["valid"] for r in util], dtype=bool)
        with np.errstate(invalid="ignore"):
            per_run.append(np.where(v, u, 0.0).sum(0) / v.sum(0))
    if not per_run:
        raise ValueError(f"no cua utility logs for {dataset}-{detector}")
    ubar = np.nanmean(np.stack(per_run), axis=0)
    short = [{"edit": r["edit"], "cls": r["cls"], "delta": float(r["delta"])} for r in deltas_rows
             if r["dataset"] == dataset and r["detector"] == detector]
    if ubar.shape != (len(menu), len(classes)):
        raise ValueError(f"utility shape {ubar.shape} does not match menu x classes ({len(menu)}, {len(classes)})")
    full = [{"edit": e, "cls": c, "delta": float(ubar[k, j])} for k, e in enumerate(menu)
            for j, c in enumerate(classes) if np.isfinite(ubar[k, j])]
    res = proxy_agreement(short, full, n_boot=n_boot, seed=seed)
    res["n_runs"] = len(per_run)
    res["cells"] = [{"edit": f["edit"], "cls": f["cls"], "u": f["delta"]} for f in full]
    return res


def _mean_sd(values) -> dict:
    v = [float(x) for x in values]
    return {"mean": float(np.mean(v)) if v else None, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "n": len(v)}


def summarize(runs_root: Path, data_root: Path, deltas_csv: Path, n_boot: int = 1000) -> dict:
    rows = load_runs(runs_root)
    with open(deltas_csv, newline="", encoding="utf-8") as f:
        deltas_rows = list(csv.DictReader(f))
    matches = build_matches(runs_root, data_root, rows)
    combos = sorted({(r["dataset"], r["detector"]) for r in rows if r["kind"] == "C"})
    res: dict = {"arms": {}, "h2": {}, "h3": {}, "vs_measured": {}, "h4": {}, "ablations": {}, "overhead": {},
                 "secondary": {}, "missing": []}
    for ds, det in combos:
        key = f"{ds}-{det}"
        res["arms"][key] = {}
        res["overhead"][key] = {}
        res["secondary"][key] = {}
        for arm in ARMS:
            v = _by_seed(rows, ds, det, arm, "test")
            if v:
                res["arms"][key][arm] = {**_mean_sd(v.values()), "per_seed": {str(s): v[s] for s in sorted(v)}}
            arm_rows = [r for r in rows if r["kind"] == "C" and (r["dataset"], r["detector"], r["arm"]) == (ds, det, arm)]
            ov = [r["overhead"] for r in arm_rows if r["overhead"] is not None]
            if arm in ("shared", "cua") and ov:
                res["overhead"][key][arm] = float(np.mean(ov))
            tests = [r["test"] for r in arm_rows if r["test"] is not None]
            if tests:
                sec = {}
                for k, path in (("ap50_95", ("ap50_95",)), ("ap_small", ("ap_small",)), ("ap_medium", ("ap_medium",)),
                                ("ap_large", ("ap_large",)), ("fp_fraction", ("fp_images", "fraction"))):
                    vals = []
                    for t in tests:
                        x = t
                        for p in path:
                            x = (x or {}).get(p) if isinstance(x, dict) else None
                        if x is not None:
                            vals.append(float(x))
                    sec[k] = float(np.mean(vals)) if vals else None
                res["secondary"][key][arm] = sec
        h2 = compare(rows, ds, det, "shared", "cua", matches=matches, n_boot=n_boot)
        t = h2["t"]
        h2["escalate_to_5_seeds"] = bool(t["lo"] is not None and t["lo"] <= 0 <= t["hi"] and t["hi"] - t["lo"] > 1.0)
        res["h2"][key] = h2
        base = h3_baseline(rows, ds, det)
        res["h3"][key] = {**compare(rows, ds, det, base, "cua", matches=matches, n_boot=n_boot), "baseline": base}
        res["vs_measured"][key] = compare(rows, ds, det, "measured", "cua", matches=matches, n_boot=n_boot)
        try:
            res["h4"][key] = h4_agreement(runs_root, deltas_rows, ds, det,
                                          read_classes(Path(data_root) / ds / "classes.json"))
        except ValueError as e:
            res["h4"][key] = {"error": str(e)}
    for ds in sorted({r["dataset"] for r in rows if r["kind"] == "X"}):
        res["ablations"][ds] = {}
        for name in sorted({r["arm"] for r in rows if r["kind"] == "X" and r["dataset"] == ds}):
            res["ablations"][ds][name] = compare(rows, ds, "fasterrcnn", "cua", name, metric="val")
    present = {r["run_id"] for r in rows}
    for ds, det in combos:
        for s in SEEDS:
            for arm in ARMS:
                rid = f"C-{ds}-{det}-s{s}-{arm}"
                if rid not in present:
                    res["missing"].append(rid)
    res["missing"] += [r["run_id"] for r in rows if r["kind"] == "C" and r["test"] is None]
    res["missing"] = sorted(set(res["missing"]))
    return res
