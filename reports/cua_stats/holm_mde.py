"""Holm-adjusted p-values per hypothesis family and minimum detectable effects (MDE) per pair.

Seed-level test: paired t on 3 per-seed differences (df = 2), two-sided p = 1 - |t| / sqrt(t^2 + 2) (exact for df 2).
Image bootstrap: p approximated from the stored 95% percentile interval assuming normality,
se = (hi - lo) / 3.92, p = 2 (1 - Phi(|diff| / se)).
H4: Spearman rho, p from z = rho * sqrt(n - 1) (normal approximation; cells are not independent).
MDE (80% power, two-sided alpha 0.05): seed t -> noncentral-t delta (simulated) * sd_diff / sqrt(3);
bootstrap -> 2.80 / 1.96 * half-width of the 95% interval.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
R = json.loads((ROOT / "reports/cua_part_c_v2_report/results.json").read_text())
ARMS = R["arms"]
PAIRS = list(ARMS)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def t_p(d):
    d = np.asarray(d, float)
    m, s = d.mean(), d.std(ddof=1)
    if s == 0:
        return m, s, 0.0
    t = m / (s / math.sqrt(len(d)))
    assert len(d) == 3
    return m, s, 1 - abs(t) / math.sqrt(t * t + 2)


def boot_p(b):
    if not b:
        return None
    se = (b["hi"] - b["lo"]) / 3.92
    return 2 * (1 - phi(abs(b["diff"]) / se)) if se > 0 else 0.0


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]
    order = sorted(idx, key=lambda i: ps[i])
    out, run = [None] * len(ps), 0.0
    m = len(idx)
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


def per_seed(pair, arm):
    return [ARMS[pair][arm]["per_seed"][str(s)] for s in (0, 1, 2)]


def best_baseline(pair):
    return max(("base", "uniform"), key=lambda a: ARMS[pair][a]["mean"])


def comp_boot(key, pair):
    v = R.get(key, {}).get(pair)
    return v.get("boot") if v else None


# simulated noncentral-t delta for 80% power, df 2, two-sided 0.05
rng = np.random.default_rng(0)
Z, C = rng.standard_normal(400_000), np.sqrt(rng.chisquare(2, 400_000) / 2)
TCRIT = 4.302652729911275
lo, hi = 0.0, 20.0
for _ in range(60):
    mid = (lo + hi) / 2
    power = np.mean(np.abs((Z + mid) / C) > TCRIT)
    lo, hi = (mid, hi) if power < 0.8 else (lo, mid)
DELTA = (lo + hi) / 2

families = {
    "H2 v1: cua - shared": [(p, "shared", "cua", "h2") for p in PAIRS],
    "H3 v1: cua - best baseline": [(p, best_baseline(p), "cua", "h3") for p in PAIRS],
    "H2 v2: cua2 - shared2": [(p, "shared2", "cua2", "v2_h2") for p in PAIRS],
    "H3 v2: cua2 - best baseline": [(p, best_baseline(p), "cua2", "v2_h3") for p in PAIRS],
    "Measured - best baseline": [(p, best_baseline(p), "measured", "measured_vs_baseline") for p in PAIRS],
}
PASTE = {  # test macro AP50 per seed, from cua.test_eval on CCAST (job 817465)
    ("dacl10k", "uniform"): [22.321, 22.334, 22.468], ("dacl10k", "cpbal"): [22.943, 22.800, 22.851],
    ("dacl10k", "cpada"): [22.999, 22.766, 22.824], ("dacl10k", "cprand"): [22.314, 22.502, 22.413],
    ("rdd", "uniform"): [50.574, 50.525, 50.537], ("rdd", "cpbal"): [50.219, 49.978, 49.776],
    ("rdd", "cpada"): [50.009, 49.693, 49.594], ("rdd", "cprand"): [49.965, 49.838, 50.268],
}
PASTE_BOOT = {("dacl10k", "cpbal"): {"diff": 0.4778, "lo": -0.0623, "hi": 0.9391},
              ("dacl10k", "cpada"): {"diff": 0.4899, "lo": -0.0511, "hi": 0.9871}}

report = {"delta_80": DELTA, "families": {}}
for name, rows in families.items():
    recs = []
    for pair, a, b, key in rows:
        d = np.array(per_seed(pair, b)) - np.array(per_seed(pair, a))
        m, s, pt = t_p(d)
        bt = comp_boot(key, pair)
        recs.append({"pair": pair, "a": a, "b": b, "diff": m, "sd": s, "p_t": pt, "boot": bt, "p_b": boot_p(bt)})
    report["families"][name] = recs
recs = []
for ds in ("dacl10k", "rdd"):
    for arm in ("cpbal", "cpada", "cprand"):
        d = np.array(PASTE[(ds, arm)]) - np.array(PASTE[(ds, "uniform")])
        m, s, pt = t_p(d)
        bt = PASTE_BOOT.get((ds, arm))
        recs.append({"pair": f"{ds}-fasterrcnn", "a": "uniform", "b": arm, "diff": m, "sd": s, "p_t": pt,
                     "boot": bt, "p_b": boot_p(bt)})
report["families"]["Copy-paste - uniform"] = recs

for name, recs in report["families"].items():
    ht, hb = holm([r["p_t"] for r in recs]), holm([r["p_b"] for r in recs])
    for r, a, b in zip(recs, ht, hb):
        r["p_t_holm"], r["p_b_holm"] = a, b
        r["claim_holm"] = (a is not None and a < 0.05 and b is not None and b < 0.05
                           and (r["boot"]["lo"] > 0) == (r["diff"] > 0))

h4 = []
for pair in PAIRS:
    v = R["h4"][pair]
    z = v["rho"] * math.sqrt(v["n_cells"] - 1)
    h4.append({"pair": pair, "rho": v["rho"], "lo": v["lo"], "hi": v["hi"], "n": v["n_cells"],
               "p": 2 * (1 - phi(abs(z)))})
for r, a in zip(h4, holm([r["p"] for r in h4])):
    r["p_holm"] = a
report["H4"] = h4

# MDE per pair: median sd of per-seed differences across the pair's comparisons, and median bootstrap half-width
mde = []
for pair in PAIRS:
    sds, hws = [], []
    for recs in report["families"].values():
        for r in recs:
            if r["pair"] == pair:
                sds.append(r["sd"])
                if r["boot"]:
                    hws.append((r["boot"]["hi"] - r["boot"]["lo"]) / 2)
    sd, hw = float(np.median(sds)), float(np.median(hws)) if hws else None
    mde.append({"pair": pair, "sd_diff_median": sd, "mde_seed_t": DELTA * sd / math.sqrt(3),
                "boot_halfwidth_median": hw, "mde_boot": (2.8 / 1.96 * hw) if hw else None,
                "n_comparisons": len(sds)})
report["MDE"] = mde
print(json.dumps(report, indent=1, default=float))
