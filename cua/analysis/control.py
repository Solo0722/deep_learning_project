"""Positive-control read-out (K- runs; protocol in reports/cua_control/PROTOCOL.md).

The K- arms train with one planted edit added to the menu (cua.augment.menu.PLANT), which deletes the
annotations of a target class c*. Its effect is known by construction: it harms c* and nothing else.
Four pre-registered checks:

  O0 manipulation: the planted edit hurts c* under uniform sampling (K-uniform vs C-uniform, test AP50 of c*).
  O1 detection:    the estimator's utility of (plant, c*) is negative and below the other classes' (plant, c).
  O2 action:       CUA applies the planted edit far less often to images that contain c* than to images
                   that do not (rate ratio), unlike uniform and shared. Images without any target object
                   bypass the policy (cua.policy.choose samples them uniformly), so they are removed from
                   the "without c*" group; their draws are recomputed exactly from the keyed RNG.
  O3 benefit:      CUA beats uniform on test AP50 of c* when the planted edit is in the menu.

python -m cua.analysis.control --runs-root $P/cua_runs --data-root $P/cua_data --dataset dacl10k \
    --detector fasterrcnn --out reports/cua_control
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from cua.analysis.ap50 import _macro, ap50_from_matches, image_matches
from cua.augment.menu import PLANT, valid_menu
from cua.augment.rng import keyed_rng
from cua.policy import choose
from cua.records import read_classes, read_records

SEEDS = (0, 1, 2)
ESTIMATING = ("shared", "cua", "shared2", "cua2")
POLICY_FROM_EPOCH = 5  # O2 counts edits from this epoch on (estimates start at epoch 2)
O1_T = -2.0
O2_RATIO = 1.0 / 3.0


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _cfg(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def t_interval(d) -> dict:
    d = np.asarray(list(d), dtype=np.float64)
    n = len(d)
    if n < 2:
        return {"mean": float(d.mean()) if n else None, "lo": None, "hi": None, "n": n}
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n))
    return {"mean": float(d.mean()), "lo": float(d.mean()) - half, "hi": float(d.mean()) + half, "n": n}


# ---- O1: estimator detection ----

def plant_utility(utility_rows: list[dict], menu: list[str], target: int, edit: str = PLANT) -> dict:
    """Per-run summary of the raw per-estimate utilities u[plant, c] (valid cells only): mean, standard
    error over estimates, t, the contrast u[plant, c*] - mean over other classes, and whether c* is the
    most negative class of the plant row."""
    k = menu.index(edit)
    rows = [r for r in utility_rows if not r.get("skipped")]
    if not rows:
        raise ValueError("no estimates")
    u = np.array([np.asarray(r["u"], dtype=np.float64)[k] for r in rows])
    v = np.array([np.asarray(r["valid"], dtype=bool)[k] for r in rows])
    u = np.where(v, u, np.nan)
    n = np.sum(v, axis=0)
    mean = np.nanmean(u, axis=0)
    se = np.nanstd(u, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    others = [c for c in range(u.shape[1]) if c != target and n[c] > 0]
    contrast = u[:, target] - np.nanmean(u[:, others], axis=1)
    ok = np.isfinite(contrast)
    return {"n_estimates": int(n[target]), "mean_target": float(mean[target]), "se_target": float(se[target]),
            "t_target": float(mean[target] / se[target]) if se[target] > 0 else None,
            "mean_others": float(np.nanmean(mean[others])),
            "contrast": float(contrast[ok].mean()),
            "contrast_t": float(contrast[ok].mean() / (contrast[ok].std(ddof=1) / math.sqrt(ok.sum())))
            if ok.sum() > 1 else None,
            "target_is_min": bool(np.nanargmin(np.where(n > 0, mean, np.nan)) == target),
            "row_mean": mean.tolist()}


# ---- O2: policy action ----

def target_free_draws(train_records, menu, seed: int, epoch: int, edit: str = PLANT) -> tuple[int, int]:
    """(number of target-free training images, how many of them drew the planted edit) in one epoch.

    Images without a target object never reach the policy: cua.policy.choose samples them from a fixed
    rule with keyed_rng(seed, epoch, index, "policy"), which is replayed here exactly."""
    free = [i for i, r in enumerate(train_records) if not r.class_set()]
    hits = sum(choose(None, frozenset(), keyed_rng(seed, epoch, i, "policy"), tuple(menu)) == edit for i in free)
    return len(free), int(hits)


def plant_rates(history: list[dict], target_name: str, from_epoch: int = POLICY_FROM_EPOCH,
                free_draws=None, edit: str = PLANT) -> dict:
    """Rate of the planted edit on images with c* and on images without it, summed over epochs >= from_epoch,
    and their ratio. With free_draws (epoch -> (n, planted) from target_free_draws), the "without" group is
    restricted to images that contain other target classes, i.e. to images the policy actually decides; the
    rate that includes target-free images is kept as ratio_incl_target_free."""
    with_p = with_n = all_p = all_n = free_p = free_n = 0
    for h in history:
        if h["epoch"] < from_epoch:
            continue
        pol = h.get("policy") or {}
        counts, by_c = pol.get("edit_counts") or {}, (pol.get("edit_by_class") or {}).get(target_name) or {}
        with_p += by_c.get(edit, 0)
        with_n += sum(by_c.values())
        all_p += counts.get(edit, 0)
        all_n += sum(counts.values())
        if free_draws is not None:
            n, k = free_draws(h["epoch"])
            free_n, free_p = free_n + n, free_p + k
    wo_p, wo_n = all_p - with_p, all_n - with_n
    r_with = with_p / with_n if with_n else None
    r_incl = wo_p / wo_n if wo_n else None
    if free_draws is not None:
        if free_n > wo_n or free_p > wo_p:
            raise ValueError(f"target-free draws ({free_p}/{free_n}) exceed the logged non-c* draws ({wo_p}/{wo_n})")
        wo_p, wo_n = wo_p - free_p, wo_n - free_n
    r_wo = wo_p / wo_n if wo_n else None
    return {"rate_with": r_with, "rate_without": r_wo,
            "ratio": (r_with / r_wo) if r_with is not None and r_wo else None,
            "rate_without_incl_target_free": r_incl,
            "ratio_incl_target_free": (r_with / r_incl) if r_with is not None and r_incl else None,
            "n_with": with_n, "n_without": wo_n, "n_target_free": free_n}


# ---- O0 / O3: test AP50 of c* ----

def class_bootstrap(pairs, target: int | None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Seed-pooled paired image bootstrap of the AP50 difference (b - a) of class `target`, or of the macro
    AP50 (mean over classes with ground truth in the resample) when target is None."""
    n = pairs[0][0].n_images
    stat = _macro if target is None else (lambda m, idx: float(ap50_from_matches(m, idx)[target]))

    def diff(idx):
        return float(np.mean([stat(mb, idx) - stat(ma, idx) for ma, mb in pairs]))

    d = diff(np.arange(n))
    rng = np.random.default_rng(seed)
    ds = np.array([diff(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return {"diff": d, "lo": float(lo), "hi": float(hi)}


def summarize(runs_root: Path, data_root: Path, dataset: str, detector: str, n_boot: int = 1000) -> dict:
    runs_root, data_root = Path(runs_root), Path(data_root)
    classes = read_classes(data_root / dataset / "classes.json")
    rid = lambda kind, s, arm: runs_root / f"{kind}-{dataset}-{detector}-s{s}-{arm}"  # noqa: E731
    first = _cfg(rid("K", SEEDS[0], "uniform"))
    target_name = first["plant_class"]
    target = list(classes).index(target_name)
    out: dict = {"dataset": dataset, "detector": detector, "target": target_name, "o1": {}, "o2": {}}
    menu = [*valid_menu(dataset), PLANT]

    for arm in ESTIMATING:
        per = []
        for s in SEEDS:
            d = rid("K", s, arm)
            if not (d / "COMPLETE.json").exists():
                continue
            if _cfg(d)["plant_class"] != target_name:
                raise ValueError(f"{d.name}: planted class differs from {target_name}")
            # shared runs log the per-class utilities too (they pool them only for the policy)
            rows = _jsonl(d / "utility.jsonl")
            per.append({"seed": s, **plant_utility(rows, menu, target)})
        if per:
            out["o1"][arm] = {"seeds": per,
                              "contrast_t_interval": t_interval([p["contrast"] for p in per]),
                              "pass": all((p["t_target"] or 0) <= O1_T and p["target_is_min"] for p in per)
                              and (t_interval([p["contrast"] for p in per])["hi"] or 0) < 0}

    train = read_records(data_root / dataset / "train.jsonl")
    cache: dict = {}

    def free_draws_for(seed):
        def f(epoch):
            if (seed, epoch) not in cache:
                cache[(seed, epoch)] = target_free_draws(train, menu, seed, epoch)
            return cache[(seed, epoch)]
        return f

    for arm in ("uniform", *ESTIMATING):
        per = []
        for s in SEEDS:
            d = rid("K", s, arm)
            if (d / "COMPLETE.json").exists():
                if _cfg(d).get("max_train_images"):
                    raise ValueError(f"{d.name}: trained on a subset; target-free draws cannot be replayed")
                per.append({"seed": s, **plant_rates(_jsonl(d / "history.jsonl"), target_name,
                                                     free_draws=free_draws_for(s))})
        if per:
            ratios = [p["ratio"] for p in per if p["ratio"] is not None]
            out["o2"][arm] = {"seeds": per, "ratio_mean": float(np.mean(ratios)) if ratios else None,
                              "pass": len(ratios) == len(SEEDS) and all(r <= O2_RATIO for r in ratios)}

    recs = read_records(data_root / dataset / "test.jsonl")
    matches = {}
    for kind, arm in [("C", "uniform"), ("K", "uniform"), *[("K", a) for a in ESTIMATING]]:
        for s in SEEDS:
            p = rid(kind, s, arm) / "test_predictions.json"
            if p.exists():
                matches[(kind, arm, s)] = image_matches(recs, json.loads(p.read_text(encoding="utf-8")), len(classes))
    allidx = np.arange(len(recs))

    def compare(a, b):
        seeds = [s for s in SEEDS if a + (s,) in matches and b + (s,) in matches]
        if len(seeds) < 2:
            return None
        ta = [float(ap50_from_matches(matches[a + (s,)], allidx)[target]) for s in seeds]
        tb = [float(ap50_from_matches(matches[b + (s,)], allidx)[target]) for s in seeds]
        boot = class_bootstrap([(matches[a + (s,)], matches[b + (s,)]) for s in seeds], target, n_boot)
        t = t_interval(np.subtract(tb, ta))
        return {"a": "-".join(a), "b": "-".join(b), "seeds": seeds, "ap_a": ta, "ap_b": tb, "t": t, "boot": boot,
                "excludes_zero_below": (t["hi"] or 0) < 0 and boot["hi"] < 0,
                "excludes_zero_above": (t["lo"] or 0) > 0 and boot["lo"] > 0}

    out["o0"] = compare(("C", "uniform"), ("K", "uniform"))
    out["o3"] = {arm: compare(("K", "uniform"), ("K", arm)) for arm in ESTIMATING}
    return out


def write_report(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "control.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    f = lambda x, d=2: "n/a" if x is None else f"{x:+.{d}f}"  # noqa: E731
    lines = [f"# Positive control: {res['dataset']} {res['detector']}, planted class {res['target']}", ""]
    o0 = res["o0"]
    lines += ["## O0 manipulation check (K-uniform − C-uniform, test AP50 of the planted class)", ""]
    if o0:
        lines.append(f"{f(o0['t']['mean'])}, seed t [{f(o0['t']['lo'])}, {f(o0['t']['hi'])}], bootstrap "
                     f"[{f(o0['boot']['lo'])}, {f(o0['boot']['hi'])}]. Pass: {o0['excludes_zero_below']}")
    lines += ["", "## O1 detection (raw utility of the planted edit)", "",
              "| Arm | Seed | u(plant, c*) | t | mean other classes | c* most negative |", "|---|---|---|---|---|---|"]
    for arm, v in res["o1"].items():
        for p in v["seeds"]:
            lines.append(f"| {arm} | {p['seed']} | {p['mean_target']:.3g} | {f(p['t_target'])} | "
                         f"{p['mean_others']:.3g} | {p['target_is_min']} |")
    for arm, v in res["o1"].items():
        ci = v["contrast_t_interval"]
        iv = "n/a (needs 2+ seeds)" if ci is None or ci.get("lo") is None else f"[{ci['lo']:.3g}, {ci['hi']:.3g}]"
        lines.append(f"\n{arm}: contrast seed t {iv}. Pass: {v['pass']}")
    lines += ["", "## O2 action (planted-edit rate, images with c* / images with other classes only)", "",
              "Target-free images bypass the policy and are excluded from the denominator group; the last column "
              "keeps them in, as in the first read-out.", "",
              "| Arm | Seed | rate with c* | rate without | ratio | ratio incl. target-free |", "|---|---|---|---|---|---|"]
    for arm, v in res["o2"].items():
        for p in v["seeds"]:
            g = lambda x, d: "n/a" if x is None else f"{x:.{d}f}"  # noqa: E731
            lines.append(f"| {arm} | {p['seed']} | {g(p['rate_with'], 4)} | {g(p['rate_without'], 4)} | "
                         f"{g(p['ratio'], 3)} | {g(p.get('ratio_incl_target_free'), 3)} |")
    lines += [""] + [f"{arm}: mean ratio "
                     + ("n/a (no planted edit on any image the policy decides)" if v["ratio_mean"] is None
                        else f"{v['ratio_mean']:.3f}")
                     + f". Pass (all ≤ {O2_RATIO:.2f}): {v['pass']}" for arm, v in res["o2"].items()]
    lines += ["", "## O3 benefit (arm − K-uniform, test AP50 of c*)", ""]
    for arm, c in res["o3"].items():
        if c:
            lines.append(f"- {arm}: {f(c['t']['mean'])}, seed t [{f(c['t']['lo'])}, {f(c['t']['hi'])}], bootstrap "
                         f"[{f(c['boot']['lo'])}, {f(c['boot']['hi'])}]. Pass: {c['excludes_zero_above']}")
    (out / "CONTROL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cua.analysis.control")
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset", default="dacl10k")
    ap.add_argument("--detector", default="fasterrcnn")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    write_report(summarize(a.runs_root, a.data_root, a.dataset, a.detector, a.n_boot), a.out)
    print((a.out / "CONTROL.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
