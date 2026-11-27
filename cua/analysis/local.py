"""Read-out of the object-local pilot (reports/cua_local/PROTOCOL.md).

Object-local arms L-dacl10k-fasterrcnn-s<k>-<edit> are compared with the Part A base arm A-...-s<k>-base
(delta, the same short-run protocol) and with their whole-image twins A-...-s<k>-<edit>.

  L1: edit x class interaction F-test on the object-local deltas (Part A model)      go if p < 0.05
  L2: cross-fitted per-class selection gain (pixel edits + none)                      go if mean >= 0.50 and
                                                                                      every fold > 0
  L3: object-local minus whole-image deltas, and the size of the interaction part   descriptive

The thresholds are constants here and are not changed after the runs.

    python -m cua.analysis.local --runs-root <cua_runs> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cua.analysis.part_a import _final_rows, deltas, interaction_test

L1_P = 0.05
L2_MEAN = 0.50
PAIR = ("dacl10k", "fasterrcnn")


def load_arms(runs_root: Path, edits: tuple[str, ...]) -> tuple[list[dict], list[dict], list[str]]:
    """(rows for A- base + whole-image arms, rows for L- arms, missing run ids). Rows: final-epoch per-class AP50."""
    whole, local, missing = [], [], []
    for scope, prefix in (("A", whole), ("L", local)):
        names = (["base"] if scope == "A" else []) + list(edits)
        for seed in (0, 1, 2):
            for e in names:
                run = Path(runs_root) / f"{scope}-dacl10k-fasterrcnn-s{seed}-{e}"
                if not (run / "COMPLETE.json").exists():
                    missing.append(run.name)
                    continue
                for cls, ap in _final_rows(run):
                    prefix.append({"dataset": PAIR[0], "detector": PAIR[1], "seed": seed, "edit": e,
                                   "cls": cls, "ap50": ap})
    return whole, local, missing


def table(d: list[dict]) -> dict[int, dict[str, dict[str, float]]]:
    """seed -> cls -> edit -> delta, with 'none' = 0 added for every (seed, cls)."""
    out: dict[int, dict[str, dict[str, float]]] = {}
    for r in d:
        out.setdefault(r["seed"], {}).setdefault(r["cls"], {"none": 0.0})[r["edit"]] = r["delta"]
    return out


def selection_gain(d: list[dict]) -> dict:
    """Cross-fitted gain of choosing each class's edit over choosing one edit for all classes.

    For each held-out seed k: from the other seeds, pick per class the edit (or none) with the highest mean delta,
    and one edit (or none) with the highest macro mean delta. The fold gain is the held-out macro delta of the
    per-class choice minus that of the global choice.
    """
    t = table(d)
    seeds = sorted(t)
    folds = []
    for k in seeds:
        train = [s for s in seeds if s != k]
        classes = sorted(set.intersection(*(set(t[s]) for s in seeds)))
        edits = sorted(set.intersection(*(set(t[s][c]) for s in seeds for c in classes)))
        mean = {c: {e: float(np.mean([t[s][c][e] for s in train])) for e in edits} for c in classes}
        per = {c: max(edits, key=lambda e, c=c: (mean[c][e], e == "none")) for c in classes}
        glob = max(edits, key=lambda e: (float(np.mean([mean[c][e] for c in classes])), e == "none"))
        pc = float(np.mean([t[k][c][per[c]] for c in classes]))
        gl = float(np.mean([t[k][c][glob] for c in classes]))
        folds.append({"held_out": k, "gain": pc - gl, "per_class": pc, "global": gl, "global_edit": glob,
                      "choice": per})
    gains = [f["gain"] for f in folds]
    return {"folds": folds, "mean": float(np.mean(gains)) if gains else float("nan"),
            "all_positive": bool(gains) and all(g > 0 for g in gains)}


def interaction_size(d: list[dict]) -> float:
    """RMS of the edit x class part of the seed-averaged delta table (cell - edit mean - class mean + grand)."""
    t = table(d)
    classes = sorted(set.intersection(*(set(v) for v in t.values())))
    edits = sorted({r["edit"] for r in d})
    m = np.array([[np.mean([t[s][c][e] for s in t]) for c in classes] for e in edits])
    resid = m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + m.mean()
    return float(np.sqrt(np.mean(resid ** 2)))


def summarize(whole_rows: list[dict], local_rows: list[dict], edits: tuple[str, ...]) -> dict:
    base = [r for r in whole_rows if r["edit"] == "base"]
    d_whole = [r for r in deltas(whole_rows) if r["edit"] in edits]
    d_local = deltas(base + local_rows)
    l1 = interaction_test(d_local) if d_local else None
    l2 = selection_gain(d_local) if d_local else None
    ref = selection_gain(d_whole) if d_whole else None
    w = {(r["seed"], r["edit"], r["cls"]): r["delta"] for r in d_whole}
    diff = {}
    for r in d_local:
        key = (r["seed"], r["edit"], r["cls"])
        if key in w:
            diff.setdefault((r["edit"], r["cls"]), []).append(r["delta"] - w[key])
    l3 = [{"edit": e, "cls": c, "local_minus_whole": float(np.mean(v)), "n_seeds": len(v)}
          for (e, c), v in sorted(diff.items())]
    go_l1 = bool(l1 is not None and l1["p"] < L1_P)
    go_l2 = bool(l2 is not None and l2["mean"] >= L2_MEAN and l2["all_positive"])
    return {
        "L1": l1, "L1_pass": go_l1,
        "L2": l2, "L2_pass": go_l2, "L2_whole_image_reference": ref,
        "L3": l3,
        "interaction_rms": {"local": interaction_size(d_local) if d_local else None,
                            "whole": interaction_size(d_whole) if d_whole else None},
        "go": go_l1 and go_l2,
        "thresholds": {"L1_p": L1_P, "L2_mean": L2_MEAN},
    }


def write_report(s: dict, missing: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "local.json").write_text(json.dumps({**s, "missing_runs": missing}, indent=2, default=str),
                                    encoding="utf-8")
    lines = ["# Object-local pilot read-out", "",
             "Protocol: `reports/cua_local/PROTOCOL.md`. Generated by `cua/analysis/local.py`.", ""]
    if missing:
        lines += [f"**Incomplete: {len(missing)} run(s) missing:** " + ", ".join(missing), ""]
    l1 = s["L1"]
    lines += ["| Check | Result | Pass |", "|---|---|---|",
              f"| L1 interaction | " + (f"F={l1['F']:.2f}, p={l1['p']:.3g}" if l1 else "n/a")
              + f" | {'yes' if s['L1_pass'] else 'no'} |"]
    l2 = s["L2"]
    if l2:
        folds = ", ".join(f"{f['gain']:+.2f}" for f in l2["folds"])
        lines.append(f"| L2 selection gain (local) | folds {folds}; mean {l2['mean']:+.2f} | "
                     f"{'yes' if s['L2_pass'] else 'no'} |")
    ref = s["L2_whole_image_reference"]
    if ref:
        folds = ", ".join(f"{f['gain']:+.2f}" for f in ref["folds"])
        lines.append(f"| reference: same gain, whole-image | folds {folds}; mean {ref['mean']:+.2f} | — |")
    rms = s["interaction_rms"]
    if rms["local"] is not None and rms["whole"] is not None:
        lines.append(f"| L3 interaction RMS | local {rms['local']:.2f} vs whole-image {rms['whole']:.2f} AP50 | — |")
    lines += ["", f"**Verdict: {'GO to step 2' if s['go'] else 'NO-GO'}.**", ""]
    if s["L3"]:
        lines += ["## L3: object-local minus whole-image delta (mean over seeds)", "", "| Edit | Class | Δ |",
                  "|---|---|---|"]
        lines += [f"| {r['edit']} | {r['cls']} | {r['local_minus_whole']:+.2f} |" for r in s["L3"]]
    (out / "LOCAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from cua.experiments.specs import LOCAL_EDITS
    ap = argparse.ArgumentParser(prog="cua.analysis.local")
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    whole, local, missing = load_arms(a.runs_root, LOCAL_EDITS)
    if missing:
        print(f"warning: {len(missing)} run(s) incomplete or missing", file=sys.stderr)
    s = summarize(whole, local, LOCAL_EDITS)
    write_report(s, missing, a.out)
    print(f"wrote {a.out / 'LOCAL.md'}; go={s['go']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
