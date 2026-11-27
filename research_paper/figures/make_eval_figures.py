"""Forest plot of paired test differences and estimator-agreement scatter for main_evaluation_paper.tex.

Inputs: reports/cua_stats/holm_mde.json, reports/cua_part_c_v2_report/results.json, reports/cua_part_a_report/deltas.csv.
Run from the repository root: python research_paper/figures/make_eval_figures.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BLUE, ORANGE, PURPLE, GREY, TEXT = "#2F6DAA", "#E0832B", "#8068A8", "#B8BEC3", "#6B7075"
NEG = "#C4473D"
POS = "#2E8B57"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

PAIRS = ["rdd-fasterrcnn", "rdd-retinanet", "dacl10k-fasterrcnn", "dacl10k-retinanet", "voc-fasterrcnn", "voc-retinanet"]
LABEL = {"rdd": "RDD2022", "dacl10k": "dacl10k", "voc": "VOC", "fasterrcnn": "F", "retinanet": "R"}
T975 = stats.t.ppf(0.975, 2)


def pair_label(p):
    d, m = p.split("-")
    return f"{LABEL[d]} {LABEL[m]}"


def forest():
    fam = json.loads((ROOT / "reports/cua_stats/holm_mde.json").read_text())["families"]
    panels = [("vs. class-blind control",
               [("H2 v1: cua - shared", "CUA", BLUE), ("H2 v2: cua2 - shared2", "CUA-v2", ORANGE)]),
              ("vs. best simple baseline",
               [("H3 v1: cua - best baseline", "CUA", BLUE), ("H3 v2: cua2 - best baseline", "CUA-v2", ORANGE),
                ("Measured - best baseline", "Measured-effect", PURPLE)])]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.75), sharey=True)
    for ax, (title, rows) in zip(axes, panels):
        n = len(rows)
        offs = np.linspace(0.27, -0.27, n) if n > 1 else [0.0]
        for (key, name, col), off in zip(rows, offs):
            by = {e["pair"]: e for e in fam[key]}
            for i, p in enumerate(PAIRS):
                e = by[p]
                y = len(PAIRS) - 1 - i + off
                hw = T975 * e["sd"] / np.sqrt(3)
                ax.plot([e["diff"] - hw, e["diff"] + hw], [y, y], color=col, lw=0.6, alpha=0.75, solid_capstyle="butt")
                b = e["boot"]
                ax.plot([b["lo"], b["hi"]], [y, y], color=col, lw=2.0, solid_capstyle="butt")
                claim = bool(e.get("claim_holm"))
                ax.plot(e["diff"], y, "o", ms=3.0, color=col if claim else "white", mec=col, mew=0.8, zorder=3,
                        label=None)
        ax.axvline(0, color=TEXT, lw=0.6, ls="--", zorder=1)
        ax.set_title(title, pad=3)
        ax.set_xlim(-3.0, 2.5)
        ax.set_xlabel("Test macro AP50 difference")
        ax.grid(axis="x", color="#E3E7EA", lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(PAIRS)))
    axes[0].set_yticklabels([pair_label(p) for p in reversed(PAIRS)])
    axes[0].set_ylim(-0.6, len(PAIRS) - 0.4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=2.0, marker="o", ms=3.0, mfc=c, mec=c) for c in (BLUE, ORANGE, PURPLE)]
    fig.legend(h, ["CUA", "CUA-v2", "Measured-effect"], loc="lower center", ncol=3, frameon=False,
               handlelength=1.4, columnspacing=1.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "forest_differences.pdf")
    plt.close(fig)
    clipped = [(k, e["pair"]) for _, rows in panels for k, _, _ in rows for e in fam[k]
               if e["diff"] - T975 * e["sd"] / np.sqrt(3) < -3.0 or e["diff"] + T975 * e["sd"] / np.sqrt(3) > 2.5]
    print("forest: seed intervals clipped at +-3 for", clipped)


def agreement():
    res = json.loads((ROOT / "reports/cua_part_c_v2_report/results.json").read_text())["h4"]
    acc = defaultdict(list)
    with open(ROOT / "reports/cua_part_a_report/deltas.csv") as f:
        for r in csv.DictReader(f):
            acc[(r["dataset"], r["detector"], r["edit"], r["cls"])].append(float(r["delta"]))
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for ax, pair in zip(axes, ["rdd-fasterrcnn", "voc-fasterrcnn"]):
        d, m = pair.split("-")
        h = res[pair]
        xs, ys, mos = [], [], []
        for c in h["cells"]:
            v = acc.get((d, m, c["edit"], c["cls"]))
            if not v:
                continue
            xs.append(c["u"] * 1e3)
            ys.append(float(np.mean(v)))
            mos.append(c["edit"] == "mosaic")
        xs, ys, mos = map(np.asarray, (xs, ys, mos))
        rho = stats.spearmanr(xs, ys).statistic
        ax.scatter(xs[~mos], ys[~mos], s=6, color=BLUE, alpha=0.55, lw=0, label="other edits")
        ax.scatter(xs[mos], ys[mos], s=9, color=NEG, lw=0, label="mosaic")
        ax.axhline(0, color=TEXT, lw=0.5, ls="--")
        ax.axvline(0, color=TEXT, lw=0.5, ls="--")
        ax.set_title(f"{pair_label(pair)}: $\\rho={h['rho']:.2f}$ [{h['lo']:.2f}, {h['hi']:.2f}]", pad=3)
        ax.set_xlabel(r"Mean estimated utility ($\times10^{-3}$)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        print(f"agreement {pair}: n={len(xs)} (stored {h['n_cells']}), recomputed rho={rho:.3f}, stored {h['rho']:.3f}")
    axes[0].set_ylabel(r"Measured $\bar\Delta(o,c)$ (AP50)")
    axes[0].legend(loc="lower left", frameon=False, handletextpad=0.1, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "estimator_agreement.pdf")
    plt.close(fig)


def teaser(name="img3"):
    """Fig. 1: one dacl10k training image, unedited and under contrast, with the per-class effect of contrast."""
    import sys
    sys.path.insert(0, str(OUT))
    import build_real_figures as brf
    from matplotlib.patches import Rectangle
    classes = brf.META["dacl10k"]["classes"]
    acc = defaultdict(list)
    with open(ROOT / "reports/cua_part_a_report/deltas.csv") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "dacl10k" and r["detector"] == "fasterrcnn" and r["edit"] == "contrast":
                acc[r["cls"]].append(float(r["delta"]))
    o, mo = brf.img("dacl10k", name, "orig")
    key = brf.pick_draw("dacl10k", name, "contrast")
    e, me = brf.img("dacl10k", name, key)
    present = sorted({classes[l] for l in mo["labels"]})
    # crop the letterbox padding (value 114) so the photo fills the panel
    rows = np.where(np.any(np.abs(o.astype(int) - 114).sum(axis=2) > 6, axis=1))[0]
    cols = np.where(np.any(np.abs(o.astype(int) - 114).sum(axis=2) > 6, axis=0))[0]
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    fig = plt.figure(figsize=(3.5, 1.95))
    gs = fig.add_gridspec(1, 2, wspace=0.05, left=0.01, right=0.56, top=0.9, bottom=0.17)
    pal = {c: col for c, col in zip(present, ["#ffd400", "#00e5ff", "#ff4fd8"])}
    for k, (rgb, m, ttl) in enumerate(((o, mo, "Unedited"), (e, me, "Contrast"))):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(rgb[y0:y1, x0:x1])
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4), sp.set_color(TEXT)
        for b, lab in zip(m["boxes"], m["labels"]):
            c = pal[classes[lab]]
            ax.add_patch(Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=0.8))
        ax.set_title(ttl, pad=2)
    handles = [Rectangle((0, 0), 1, 1, fc="none", ec=pal[c], lw=1.0) for c in present]
    fig.legend(handles, present, loc="lower left", ncol=len(present), frameon=False, bbox_to_anchor=(0.0, 0.0),
               handlelength=1.0, columnspacing=0.8, fontsize=6)
    ax = fig.add_axes([0.765, 0.17, 0.225, 0.73])
    order = sorted(acc, key=lambda c: np.mean(acc[c]))
    ys = np.arange(len(order))
    means = [np.mean(acc[c]) for c in order]
    sames = [all(v > 0 for v in acc[c]) or all(v < 0 for v in acc[c]) for c in order]
    ax.barh(ys, means, color=[POS if m > 0 else NEG for m in means], alpha=0.85, height=0.7)
    for y, c, m, sm in zip(ys, order, means, sames):
        for v in acc[c]:
            ax.plot(v, y, "|", color="black", ms=3, mew=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=5.6)
    for t, c in zip(ax.get_yticklabels(), order):
        if c in present:
            t.set_fontweight("bold")
    ax.axvline(0, color=TEXT, lw=0.5)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_title(r"Contrast: $\Delta$AP50", pad=2)
    ax.tick_params(axis="x", labelsize=5.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(OUT / "teaser_contrast.pdf", dpi=300)
    plt.close(fig)
    print("teaser:", name, key, "present", present, {c: round(float(np.mean(acc[c])), 2) for c in present})


if __name__ == "__main__":
    forest()
    agreement()
    teaser()
