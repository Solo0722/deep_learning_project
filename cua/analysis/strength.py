"""Read-out of the strength pilot (reports/cua_strength/PROTOCOL.md).

Arms: the Part A base arm A-dacl10k-fasterrcnn-s<k>-base and the strength arms S-...-s<k>-<edit>-x<level>
(edit applied to every image, levels 1-3). Each (edit, level) is one arm "<edit>@x<level>"; deltas are arm - base.
The Part A arms A-...-s<k>-<edit> (level 1, p = 0.5) are read separately, as a reference only.

  S1: (edit, level) x class interaction F-test (Part A model)                go if p < 0.05
  S2: cross-fitted per-class selection gain over none + the 15 arms           go if mean >= 0.50 and every fold > 0
  S3: per edit and level: SD of the seed-mean delta across classes, mean seed SD, signal share lambda (descriptive)
  S4: S2 with empirical-Bayes shrinkage of each class toward the pooled arm effect (descriptive)

The thresholds are constants here and are not changed after the runs.

    python -m cua.analysis.strength --runs-root <cua_runs> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cua.analysis.local import selection_gain, table
from cua.analysis.part_a import _final_rows, deltas, interaction_test

S1_P = 0.05
S2_MEAN = 0.50
PAIR = ("dacl10k", "fasterrcnn")
LEVELS = (1, 2, 3)


def arm_name(edit: str, level: int) -> str:
    return f"{edit}@x{level}"


def run_name(edit: str, level: int, seed: int) -> str:
    return f"S-dacl10k-fasterrcnn-s{seed}-{edit}-x{level}"


def load_arms(runs_root: Path, edits: tuple[str, ...], levels: tuple[int, ...] = LEVELS,
              seeds: tuple[int, ...] = (0, 1, 2)) -> tuple[list[dict], list[str]]:
    """(final-epoch per-class AP50 rows for base + every (edit, level) arm, missing run ids)."""
    rows, missing = [], []
    for seed in seeds:
        runs = [(f"A-dacl10k-fasterrcnn-s{seed}-base", "base")]
        runs += [(run_name(e, lv, seed), arm_name(e, lv)) for e in edits for lv in levels]
        for name, arm in runs:
            run = Path(runs_root) / name
            if not (run / "COMPLETE.json").exists():
                missing.append(name)
                continue
            for cls, ap in _final_rows(run):
                rows.append({"dataset": PAIR[0], "detector": PAIR[1], "seed": seed, "edit": arm, "cls": cls,
                             "ap50": ap})
    return rows, missing


def load_part_a_reference(runs_root: Path, edits: tuple[str, ...], seeds: tuple[int, ...] = (0, 1, 2)) -> list[dict]:
    """Final-epoch per-class AP50 of the A- base and level-1 arms (p = 0.5), arms named '<edit>@A'."""
    rows = []
    for seed in seeds:
        for name, arm in [(f"A-dacl10k-fasterrcnn-s{seed}-base", "base")] +                 [(f"A-dacl10k-fasterrcnn-s{seed}-{e}", f"{e}@A") for e in edits]:
            run = Path(runs_root) / name
            if (run / "COMPLETE.json").exists():
                rows += [{"dataset": PAIR[0], "detector": PAIR[1], "seed": seed, "edit": arm, "cls": c, "ap50": ap}
                         for c, ap in _final_rows(run)]
    return rows


def shrunk_selection_gain(d: list[dict]) -> dict:
    """selection_gain with each class's arm effect shrunk toward the pooled arm effect.

    Per held-out seed k, from the other seeds: pooled(a) = class mean of the seed-mean deltas; noise(a) = class mean
    of the variance of a seed mean; tau2(a) = max(0, var over classes of the class deviations - noise(a));
    lambda(a) = tau2 / (tau2 + noise); u(a, c) = pooled(a) + lambda(a) (dbar(a, c) - pooled(a)). 'none' is 0.
    """
    t = table(d)
    seeds = sorted(t)
    folds, lams = [], []
    for k in seeds:
        train = [s for s in seeds if s != k]
        classes = sorted(set.intersection(*(set(t[s]) for s in seeds)))
        arms = sorted(set.intersection(*(set(t[s][c]) for s in seeds for c in classes)))
        u, pooled_of = {}, {}
        for a in arms:
            vals = np.array([[t[s][c][a] for c in classes] for s in train])
            dbar = vals.mean(axis=0)
            pooled = float(dbar.mean())
            pooled_of[a] = pooled
            lam = 1.0
            if a != "none" and len(train) > 1:
                noise = float(np.mean(vals.var(axis=0, ddof=1) / len(train)))
                tau2 = max(0.0, float(np.var(dbar - pooled, ddof=1)) - noise)
                lam = tau2 / (tau2 + noise) if tau2 + noise > 0 else 0.0
                lams.append(lam)
            for i, c in enumerate(classes):
                u[(a, c)] = pooled + lam * (dbar[i] - pooled)
        glob = max(arms, key=lambda a: (pooled_of[a], a == "none"))
        per = {c: max(arms, key=lambda a, c=c: (u[(a, c)], a == "none")) for c in classes}
        pc = float(np.mean([t[k][c][per[c]] for c in classes]))
        gl = float(np.mean([t[k][c][glob] for c in classes]))
        folds.append({"held_out": k, "gain": pc - gl, "global_edit": glob, "choice": per})
    gains = [f["gain"] for f in folds]
    return {"folds": folds, "mean": float(np.mean(gains)) if gains else float("nan"),
            "all_positive": bool(gains) and all(g > 0 for g in gains),
            "mean_lambda": float(np.mean(lams)) if lams else None}


def spread(d: list[dict], edits: tuple[str, ...], levels: tuple[int, ...] = LEVELS) -> list[dict]:
    """S3: per (edit, level): mean delta, SD across classes of the seed-mean delta, mean seed SD, lambda."""
    out = []
    for e in edits:
        for lv in levels:
            a = arm_name(e, lv)
            cells: dict[str, list[float]] = {}
            for r in d:
                if r["edit"] == a:
                    cells.setdefault(r["cls"], []).append(r["delta"])
            if not cells or min(len(v) for v in cells.values()) < 2:
                continue
            means = np.array([np.mean(v) for v in cells.values()])
            n = min(len(v) for v in cells.values())
            seed_sd = float(np.mean([np.std(v, ddof=1) for v in cells.values()]))
            noise = float(np.mean([np.var(v, ddof=1) / len(v) for v in cells.values()]))
            between = float(np.var(means, ddof=1))
            tau2 = max(0.0, between - noise)
            out.append({"edit": e, "level": lv, "mean_delta": float(means.mean()),
                        "class_sd": float(np.sqrt(between)), "seed_sd": seed_sd, "n_seeds": n,
                        "lambda": tau2 / (tau2 + noise) if tau2 + noise > 0 else 0.0})
    return out


def summarize(rows: list[dict], edits: tuple[str, ...], reference_rows: list[dict] | None = None) -> dict:
    d = deltas(rows)
    d_ref = deltas(reference_rows) if reference_rows else []
    s1 = interaction_test(d) if d else None
    s2 = selection_gain(d) if d else None
    lvl1 = [r for r in d if r["edit"].endswith("@x1")]
    ref = selection_gain(lvl1) if lvl1 else None
    s4 = shrunk_selection_gain(d) if d else None
    pass1 = bool(s1 is not None and s1["p"] < S1_P)
    pass2 = bool(s2 is not None and s2["mean"] >= S2_MEAN and s2["all_positive"])
    return {"S1": s1, "S1_pass": pass1, "S2": s2, "S2_pass": pass2, "S2_level1_reference": ref,
            "S2_part_a_reference": selection_gain(d_ref) if d_ref else None,
            "S3": spread(d, edits), "S4": s4, "go": pass1 and pass2,
            "thresholds": {"S1_p": S1_P, "S2_mean": S2_MEAN}}


def _folds(g: dict) -> str:
    return ", ".join(f"{f['gain']:+.2f}" for f in g["folds"]) + f"; mean {g['mean']:+.2f}"


def write_report(s: dict, missing: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "strength.json").write_text(json.dumps({**s, "missing_runs": missing}, indent=2, default=str),
                                       encoding="utf-8")
    lines = ["# Strength pilot read-out", "",
             "Protocol: `reports/cua_strength/PROTOCOL.md`. Generated by `cua/analysis/strength.py`.", ""]
    if missing:
        lines += [f"**Incomplete: {len(missing)} run(s) missing:** " + ", ".join(missing), ""]
    s1 = s["S1"]
    lines += ["| Check | Result | Pass |", "|---|---|---|",
              "| S1 interaction | " + (f"F={s1['F']:.2f}, p={s1['p']:.3g}" if s1 else "n/a")
              + f" | {'yes' if s['S1_pass'] else 'no'} |"]
    if s["S2"]:
        lines.append(f"| S2 selection gain, edit + strength | folds {_folds(s['S2'])} | "
                     f"{'yes' if s['S2_pass'] else 'no'} |")
    if s["S2_level1_reference"]:
        lines.append(f"| reference: same gain, level 1 only (p = 1) | folds {_folds(s['S2_level1_reference'])} | — |")
    if s.get("S2_part_a_reference"):
        lines.append(f"| reference: Part A level 1 (p = 0.5) | folds {_folds(s['S2_part_a_reference'])} | — |")
    if s["S4"]:
        lam = s["S4"]["mean_lambda"]
        lines.append(f"| S4 shrunk selection gain | folds {_folds(s['S4'])}"
                     + (f"; mean λ {lam:.2f}" if lam is not None else "") + " | — |")
    lines += ["", f"**Verdict: {'GO to step 2' if s['go'] else 'NO-GO'}.**", ""]
    if s["S3"]:
        lines += ["## S3: class spread by strength", "",
                  "| Edit | Level | Mean Δ | Class SD | Seed SD | λ |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['edit']} | x{r['level']} | {r['mean_delta']:+.2f} | {r['class_sd']:.2f} | "
                  f"{r['seed_sd']:.2f} | {r['lambda']:.2f} |" for r in s["S3"]]
    if s["S2"]:
        lines += ["", "## S2 choices per fold", ""]
        for f in s["S2"]["folds"]:
            ch = ", ".join(f"{c}: {a}" for c, a in sorted(f["choice"].items()))
            lines.append(f"- held-out seed {f['held_out']} (global: {f['global_edit']}): {ch}")
    (out / "STRENGTH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from cua.experiments.specs import STRENGTH_EDITS
    ap = argparse.ArgumentParser(prog="cua.analysis.strength")
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rows, missing = load_arms(a.runs_root, STRENGTH_EDITS)
    if missing:
        print(f"warning: {len(missing)} run(s) incomplete or missing", file=sys.stderr)
    s = summarize(rows, STRENGTH_EDITS, load_part_a_reference(a.runs_root, STRENGTH_EDITS))
    write_report(s, missing, a.out)
    print(f"wrote {a.out / 'STRENGTH.md'}; go={s['go']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
