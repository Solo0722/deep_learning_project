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
V2_ARMS = ("shared2", "cua2")  # CUA v2 (lookahead estimator, gate, probe prior); see cua.experiments.specs
V2_KINDS = ("v2_h2", "v2_h3", "v2_vs_measured", "v2_vs_v1")
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
        est_s = sum(float(e) for e in est if e)
        overhead = (est_s / secs) if est_s > 0 and secs > 0 else None
        # share: fraction of epoch wall time (training + estimator + validation) spent estimating;
        # extra: estimator time relative to the rest of the epoch
        overhead_extra = (est_s / (secs - est_s)) if overhead is not None and secs > est_s else None
        tm = d / "test_metrics.json"
        out.append({"run_id": d.name, "dir": d, "kind": d.name[0], "dataset": m[1], "detector": m[2],
                    "seed": int(m[3]), "arm": m[4], "best_val": max(vals) if vals else None,
                    "overhead": overhead, "overhead_extra": overhead_extra,
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
           "boot": None, "direction": 0, "claim": False, "note": ""}
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
    # direction: +1 b reliably better, -1 reliably worse, 0 not shown; claim means "b beats a"
    out["direction"] = side
    out["claim"] = side == 1
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
    pa: dict[tuple, list[float]] = {}
    for r in short:
        pa.setdefault((r["edit"], r["cls"]), []).append(r["delta"])
    joined = [f for f in full if (f["edit"], f["cls"]) in pa]
    res["x"] = [float(np.mean(pa[(f["edit"], f["cls"])])) for f in joined]
    res["y"] = [f["delta"] for f in joined]
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
        for arm in ARMS + V2_ARMS:
            v = _by_seed(rows, ds, det, arm, "test")
            if v:
                res["arms"][key][arm] = {**_mean_sd(v.values()), "per_seed": {str(s): v[s] for s in sorted(v)}}
            arm_rows = [r for r in rows if r["kind"] == "C" and (r["dataset"], r["detector"], r["arm"]) == (ds, det, arm)]
            ov = [r for r in arm_rows if r["overhead"] is not None]
            if arm in ("shared", "cua", *V2_ARMS) and ov:
                res["overhead"][key][arm] = {"share": float(np.mean([r["overhead"] for r in ov])),
                                             "extra": float(np.mean([r["overhead_extra"] for r in ov]))}
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
        res.setdefault("measured_vs_baseline", {})[key] = {
            **compare(rows, ds, det, base, "measured", matches=matches, n_boot=n_boot), "baseline": base}
        if _by_seed(rows, ds, det, "cua2", "test"):
            for kind, a in (("v2_h2", "shared2"), ("v2_h3", base), ("v2_vs_measured", "measured"), ("v2_vs_v1", "cua")):
                res.setdefault(kind, {})[key] = compare(rows, ds, det, a, "cua2", matches=matches, n_boot=n_boot)
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
        seen = {r["seed"] for r in rows if r["kind"] == "C" and (r["dataset"], r["detector"]) == (ds, det)}
        for s in sorted(set(SEEDS) | seen):
            for arm in ARMS:
                rid = f"C-{ds}-{det}-s{s}-{arm}"
                if rid not in present:
                    res["missing"].append(rid)
    res["missing"] += [r["run_id"] for r in rows if r["kind"] == "C" and r["test"] is None]
    res["missing"] = sorted(set(res["missing"]))
    return res


# ---- tables, figures, CLI ----

DS_NAME = {"rdd": "RDD2022", "voc": "VOC", "dacl10k": "dacl10k"}
DET_NAME = {"fasterrcnn": "Faster R-CNN", "retinanet": "RetinaNet"}
ARM_NAME = {"base": "Base", "uniform": "Uniform", "measured": "Measured", "shared": "Shared", "cua": "Class-utility"}


def _split_key(key: str) -> tuple[str, str]:
    ds, det = key.rsplit("-", 1)
    return ds, det


def write_tables(results: dict, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "main_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "detector"] + [f"{a}_{s}" for a in ARMS for s in ("mean", "sd")])
        for key, arms in results["arms"].items():
            ds, det = _split_key(key)
            w.writerow([ds, det] + [arms.get(a, {}).get(s) for a in ARMS for s in ("mean", "sd")])
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Test macro AP50 (\%), mean $\pm$ s.d. over seeds. Best mean per row in bold. "
             r"A superscript gives an arm's seed count where it is below the row's largest.}",
             r"\label{tab:main}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{ll" + "c" * len(ARMS) + "}",
             r"\toprule", "Dataset & Detector & " + " & ".join(ARM_NAME[a] for a in ARMS) + r" \\", r"\midrule"]
    for key, arms in results["arms"].items():
        ds, det = _split_key(key)
        means = {a: arms[a]["mean"] for a in ARMS if a in arms and arms[a]["mean"] is not None}
        best = max(means, key=means.get) if means else None
        n_max = max((arms[a]["n"] for a in means), default=0)
        cells = []
        for a in ARMS:
            if a not in means:
                cells.append("--")
                continue
            sd = arms[a]["sd"]
            txt = f"{means[a]:.1f}" + (rf" $\pm$ {sd:.1f}" if sd is not None else "")
            if arms[a]["n"] < n_max:
                txt += f"$^{{{arms[a]['n']}}}$"
            cells.append(rf"\textbf{{{txt}}}" if a == best else txt)
        lines.append(f"{DS_NAME.get(ds, ds)} & {DET_NAME.get(det, det)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out / "main_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "comparisons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "dataset", "detector", "a", "b", "metric", "n_seeds", "t_mean", "t_lo", "t_hi",
                    "boot_lo", "boot_hi", "claim", "note"])

        def row(kind, ds, det, c):
            b = c.get("boot") or {}
            w.writerow([kind, ds, det, c["a"], c["b"], c["metric"], c["n_seeds"], c["t"]["mean"], c["t"]["lo"],
                        c["t"]["hi"], b.get("lo"), b.get("hi"), c["claim"], c.get("note", "")])

        for kind in ("h2", "h3", "vs_measured", "measured_vs_baseline", *V2_KINDS):
            for key, c in results.get(kind, {}).items():
                row(kind, *_split_key(key), c)
        for ds, abl in results["ablations"].items():
            for c in abl.values():
                row("ablation", ds, "fasterrcnn", c)


def write_figures(results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for kind, label in (("h2", "CUA - shared"), ("h3", "CUA - best simple")):
        for key, c in results[kind].items():
            if c["t"]["mean"] is None:
                continue
            ds, det = _split_key(key)
            boot = (c["boot"]["lo"], c["boot"]["hi"]) if c.get("boot") else None
            name = f"{DS_NAME.get(ds, ds)} / {DET_NAME.get(det, det)}: {label}"
            if kind == "h3":
                name += f" ({c['baseline']})"
            items.append((name, c["t"]["mean"], (c["t"]["lo"], c["t"]["hi"]), boot, c["claim"]))
    fig, ax = plt.subplots(figsize=(6, 0.4 * max(len(items), 2) + 1))
    # thin line: seed-level t interval; thick line: image bootstrap; filled marker: claim holds
    for i, (name, m, t_iv, boot, claim) in enumerate(items):
        if t_iv[0] is not None:
            ax.plot(list(t_iv), [i, i], color="black", linewidth=1.0)
        if boot is not None:
            ax.plot(list(boot), [i, i], color="black", linewidth=3.0)
        ax.plot([m], [i], marker="o", color="black", markersize=5,
                markerfacecolor="black" if claim else "white")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Difference in test macro AP50 (points)")
    fig.tight_layout()
    fig.savefig(out / "fig_differences.png", dpi=200)
    plt.close(fig)
    for key, h in results["h4"].items():
        if "cells" not in h:
            continue
        ds, det = _split_key(key)
        # x: Part A delta (seed mean), y: time-averaged u
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.scatter(h["x"], h["y"], s=10, facecolors="none", edgecolors="black")
        ax.set_xlabel(r"Part A $\Delta$AP50 (points)")
        ax.set_ylabel("Mean estimated utility")
        ax.set_title(f"{DS_NAME.get(ds, ds)} / {DET_NAME.get(det, det)}: "
                     rf"$\rho$ = {h['rho']:.2f} [{h['lo']:.2f}, {h['hi']:.2f}]", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"fig_h4_{ds}_{det}.png", dpi=200)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="cua.analysis.part_c")
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--deltas", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    res = summarize(a.runs_root, a.data_root, a.deltas, n_boot=a.n_boot)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_tables(res, a.out)
    write_figures(res, a.out)
    print(f"wrote {a.out}; missing or unevaluated runs: {len(res['missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
