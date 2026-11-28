"""Positive control 2 read-out (K2- runs; protocol in reports/cua_control2/PROTOCOL.md).

The K2- arms are the VOC RetinaNet C- arms with mosaic replaced by plant_mosaic (cua.augment.menu.PLANT_MOSAIC):
mosaic with the anchor tile's boxes of c* deleted. It helps the other classes (it is mosaic on images
without c*) and hurts c*, so a class-blind policy must accept the harm or lose the help.

  O4 per-class advantage (primary): macro test AP50, K2-cua2 - K2-shared2, both intervals above 0.
  O3 mechanism:     test AP50 of c*, K2-cua2 - K2-shared2, both intervals above 0.
  O0 manipulation:  test AP50 of c*, K2-shared2 - C-shared2, both intervals below 0.
  O1 detection:     cua2's utility of (plant_mosaic, c*): t <= -2 and the row minimum in every seed;
                    the seed interval of its contrast with the other classes below 0.
  O2 action:        cua2's plant_mosaic rate on images with c* / on images with other classes only <= 1/3
                    in every seed (target-free images removed by exact replay), epochs >= 5.
  O5 vs default:    macro test AP50, K2-cua2 - K2-uniform, both intervals above 0.

python -m cua.analysis.control2 --runs-root $P/cua_runs --data-root $P/cua_data --out reports/cua_control2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cua.analysis.ap50 import ap50_from_matches, image_matches
from cua.analysis.control import (O1_T, O2_RATIO, POLICY_FROM_EPOCH, _cfg, _jsonl, class_bootstrap, plant_rates,
                                  plant_utility, t_interval, target_free_draws)
from cua.augment.menu import PLANT_MOSAIC, planted_menu
from cua.experiments.specs import CONTROL2_ARMS, CONTROL2_PAIR
from cua.records import read_classes, read_records

SEEDS = (0, 1, 2)
DATASET, DETECTOR = CONTROL2_PAIR
EDIT = PLANT_MOSAIC
ESTIMATING = ("shared2", "cua2")


def _run(runs_root: Path, kind: str, seed: int, arm: str) -> Path:
    return Path(runs_root) / f"{kind}-{DATASET}-{DETECTOR}-s{seed}-{arm}"


def _macro_test(run_dir: Path) -> float | None:
    p = run_dir / "test_metrics.json"
    return float(json.loads(p.read_text(encoding="utf-8"))["macro_ap50"]) if p.exists() else None


def compare(matches: dict, macros: dict, a: tuple, b: tuple, target: int | None, n_boot: int) -> dict | None:
    """b - a over the seeds both have: seed t interval and seed-pooled paired image bootstrap.
    target None = macro AP50 (seed values from test_metrics.json), else AP50 of that class."""
    seeds = [s for s in SEEDS if a + (s,) in matches and b + (s,) in matches]
    if len(seeds) < 2:
        return None
    if target is None:
        va, vb = [macros[a + (s,)] for s in seeds], [macros[b + (s,)] for s in seeds]
    else:
        va = [float(ap50_from_matches(matches[a + (s,)], np.arange(matches[a + (s,)].n_images))[target]) for s in seeds]
        vb = [float(ap50_from_matches(matches[b + (s,)], np.arange(matches[b + (s,)].n_images))[target]) for s in seeds]
    boot = class_bootstrap([(matches[a + (s,)], matches[b + (s,)]) for s in seeds], target, n_boot)
    t = t_interval(np.subtract(vb, va))
    return {"a": "-".join(a), "b": "-".join(b), "seeds": seeds, "a_values": va, "b_values": vb, "t": t,
            "boot": boot, "above": (t["lo"] is not None and t["lo"] > 0 and boot["lo"] > 0),
            "below": (t["hi"] is not None and t["hi"] < 0 and boot["hi"] < 0)}


def summarize(runs_root: Path, data_root: Path, n_boot: int = 1000) -> dict:
    runs_root, data_root = Path(runs_root), Path(data_root)
    classes = list(read_classes(data_root / DATASET / "classes.json"))
    menu = list(planted_menu(DATASET, "mosaic"))
    done = {(s, arm): _run(runs_root, "K2", s, arm) for s in SEEDS for arm in CONTROL2_ARMS
            if (_run(runs_root, "K2", s, arm) / "COMPLETE.json").exists()}
    missing = [f"K2-{DATASET}-{DETECTOR}-s{s}-{arm}" for s in SEEDS for arm in CONTROL2_ARMS if (s, arm) not in done]
    if not done:
        return {"missing_runs": missing, "complete": False}
    names = {_cfg(d)["plant_class"] for d in done.values()}
    if len(names) != 1:
        raise ValueError(f"K2- runs disagree on the planted class: {sorted(names)}")
    target_name = names.pop()
    target = classes.index(target_name)
    out: dict = {"dataset": DATASET, "detector": DETECTOR, "target": target_name, "edit": EDIT,
                 "missing_runs": missing, "complete": not missing, "o1": {}, "o2": {}}

    for arm in ESTIMATING:
        per = [{"seed": s, **plant_utility(_jsonl(done[(s, arm)] / "utility.jsonl"), menu, target, EDIT)}
               for s in SEEDS if (s, arm) in done]
        if per:
            ci = t_interval([p["contrast"] for p in per])
            out["o1"][arm] = {"seeds": per, "contrast_t_interval": ci,
                              "pass": len(per) == len(SEEDS)
                              and all((p["t_target"] or 0) <= O1_T and p["target_is_min"] for p in per)
                              and ci["hi"] is not None and ci["hi"] < 0}

    train = read_records(data_root / DATASET / "train.jsonl")
    cache: dict = {}

    def free_draws_for(seed):
        def f(epoch):
            if (seed, epoch) not in cache:
                cache[(seed, epoch)] = target_free_draws(train, menu, seed, epoch, EDIT)
            return cache[(seed, epoch)]
        return f

    for arm in CONTROL2_ARMS:
        per = []
        for s in SEEDS:
            if (s, arm) not in done:
                continue
            if _cfg(done[(s, arm)]).get("max_train_images"):
                raise ValueError(f"{done[(s, arm)].name}: trained on a subset; target-free draws cannot be replayed")
            per.append({"seed": s, **plant_rates(_jsonl(done[(s, arm)] / "history.jsonl"), target_name,
                                                 free_draws=free_draws_for(s), edit=EDIT)})
        if per:
            ratios = [p["ratio"] for p in per if p["ratio"] is not None]
            out["o2"][arm] = {"seeds": per, "ratio_mean": float(np.mean(ratios)) if ratios else None,
                              "pass": len(ratios) == len(SEEDS) and all(r <= O2_RATIO for r in ratios)}

    recs = read_records(data_root / DATASET / "test.jsonl")
    matches, macros = {}, {}
    for kind in ("C", "K2"):
        for arm in CONTROL2_ARMS:
            for s in SEEDS:
                d = _run(runs_root, kind, s, arm)
                p = d / "test_predictions.json"
                if p.exists() and _macro_test(d) is not None:
                    matches[(kind, arm, s)] = image_matches(recs, json.loads(p.read_text(encoding="utf-8")),
                                                            len(classes))
                    macros[(kind, arm, s)] = _macro_test(d)
    c = lambda a, b, t: compare(matches, macros, a, b, t, n_boot)  # noqa: E731
    out["o4"] = c(("K2", "shared2"), ("K2", "cua2"), None)
    out["o3"] = c(("K2", "shared2"), ("K2", "cua2"), target)
    out["o0"] = c(("C", "shared2"), ("K2", "shared2"), target)
    out["o5"] = c(("K2", "uniform"), ("K2", "cua2"), None)
    out["reported"] = {
        **{f"c_star_{arm}_K2_minus_C": c(("C", arm), ("K2", arm), target) for arm in ("uniform", "cua2")},
        **{f"macro_{arm}_K2_minus_C": c(("C", arm), ("K2", arm), None) for arm in CONTROL2_ARMS},
        "macro_C_cua2_minus_C_shared2": c(("C", "shared2"), ("C", "cua2"), None),
    }
    out["pass"] = {"O4": bool(out["o4"] and out["o4"]["above"]), "O3": bool(out["o3"] and out["o3"]["above"]),
                   "O0": bool(out["o0"] and out["o0"]["below"]),
                   "O1": bool(out["o1"].get("cua2", {}).get("pass")), "O2": bool(out["o2"].get("cua2", {}).get("pass")),
                   "O5": bool(out["o5"] and out["o5"]["above"])}
    return out


def write_report(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "control2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    f = lambda x, d=2: "n/a" if x is None else f"{x:+.{d}f}"  # noqa: E731
    g = lambda x, d: "n/a" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = ["# Positive control 2: VOC RetinaNet, mosaic that deletes the anchor's "
             f"{res.get('target', '?')} boxes", "",
             "Protocol: `reports/cua_control2/PROTOCOL.md`. Generated by `cua/analysis/control2.py`.", ""]
    if res["missing_runs"]:
        lines += [f"**Incomplete: {len(res['missing_runs'])} run(s) missing:** " + ", ".join(res["missing_runs"]), ""]
    if "target" not in res:
        (out / "CONTROL2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    def cmp_line(label, cmp, rule):
        if not cmp:
            return f"- {label}: n/a (fewer than 2 seeds with test predictions)"
        t, b = cmp["t"], cmp["boot"]
        per = ", ".join(f(y - x) for x, y in zip(cmp["a_values"], cmp["b_values"]))
        return (f"- {label}: {f(t['mean'])} (per seed {per}), seed t [{f(t['lo'])}, {f(t['hi'])}], "
                f"bootstrap [{f(b['lo'])}, {f(b['hi'])}]" + (f". Pass: {cmp[rule]}" if rule else ""))

    lines += ["## Pre-registered checks", "",
              "| Check | Pass |", "|---|---|"] + [f"| {k} | {v} |" for k, v in res["pass"].items()] + [""]
    lines += ["## Accuracy (test AP50)", "",
              cmp_line("O4 macro, K2-cua2 − K2-shared2", res["o4"], "above"),
              cmp_line(f"O3 {res['target']}, K2-cua2 − K2-shared2", res["o3"], "above"),
              cmp_line(f"O0 {res['target']}, K2-shared2 − C-shared2", res["o0"], "below"),
              cmp_line("O5 macro, K2-cua2 − K2-uniform", res["o5"], "above"), "", "Reported without a pass rule:", ""]
    lines += [cmp_line(k, v, None) for k, v in res["reported"].items()]
    lines += ["", "## O1 detection (raw utility of plant_mosaic)", "",
              "| Arm | Seed | u(plant_mosaic, c*) | t | mean other classes | c* most negative |", "|---|---|---|---|---|---|"]
    for arm, v in res["o1"].items():
        for p in v["seeds"]:
            lines.append(f"| {arm} | {p['seed']} | {p['mean_target']:.3g} | {f(p['t_target'])} | "
                         f"{p['mean_others']:.3g} | {p['target_is_min']} |")
    for arm, v in res["o1"].items():
        ci = v["contrast_t_interval"]
        iv = "n/a" if ci.get("lo") is None else f"[{ci['lo']:.3g}, {ci['hi']:.3g}]"
        lines.append(f"\n{arm}: contrast seed t {iv}. Pass: {v['pass']}")
    lines += ["", f"## O2 action (plant_mosaic rate from epoch {POLICY_FROM_EPOCH}; images with c* / images with "
              "other classes only)", "",
              "| Arm | Seed | rate with c* | rate without | ratio | ratio incl. target-free |", "|---|---|---|---|---|---|"]
    for arm, v in res["o2"].items():
        for p in v["seeds"]:
            lines.append(f"| {arm} | {p['seed']} | {g(p['rate_with'], 4)} | {g(p['rate_without'], 4)} | "
                         f"{g(p['ratio'], 3)} | {g(p.get('ratio_incl_target_free'), 3)} |")
    lines += [""] + [f"{arm}: mean ratio {g(v['ratio_mean'], 3)}. Pass (all ≤ {O2_RATIO:.2f}): {v['pass']}"
                     for arm, v in res["o2"].items()]
    (out / "CONTROL2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cua.analysis.control2")
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    write_report(summarize(a.runs_root, a.data_root, a.n_boot), a.out)
    print((a.out / "CONTROL2.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
