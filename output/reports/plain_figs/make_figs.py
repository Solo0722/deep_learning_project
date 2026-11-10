"""Black-and-white figures for the Nov 2 and Nov 9 progress reports."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "..", "..", "..", "reports")
plt.rcParams.update({"font.family": "serif", "font.size": 10})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, name), dpi=220)
    plt.close()


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


# Training curves, five 20-epoch runs
rows = list(csv.DictReader(open(os.path.join(REP, "fasterrcnn_class_adaptive_v1", "comparison_curves.csv"))))
ep = [int(r["epoch"]) for r in rows]
runs = [("identity", "No augmentation", "-", "o"), ("image_level", "Fixed image-level", "--", "s"),
        ("local", "Fixed local", ":", "^"), ("online_shared", "Online shared", "-", "D"),
        ("class_adaptive", "Online class-adaptive", "--", "x")]
greys = ["0.0", "0.35", "0.5", "0.2", "0.0"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for (k, lab, ls, mk), g in zip(runs, greys):
    ax.plot(ep, [float(r[f"{k}_AP50_95"]) for r in rows], ls=ls, marker=mk, ms=3.5, color=g, lw=1.1, label=lab)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AP50:95 (%)")
ax.set_xticks(range(2, 21, 2)); ax.legend(fontsize=8, frameon=False, ncol=2); clean(ax)
save("curves_20epoch.png")

# Paired epoch-8 run
rows = list(csv.DictReader(open(os.path.join(REP, "rdd_v2_pair8_v1", "epoch_comparison.csv"))))
ep = [int(r["epoch"]) for r in rows]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9))
axs[0].plot(ep, [float(r["geometry_AP50_95"]) for r in rows], "k-o", ms=4, label="Geometry only")
axs[0].plot(ep, [float(r["class_AP50_95"]) for r in rows], "k--s", ms=4, mfc="white", label="Class-adaptive")
axs[0].set_ylabel("Validation AP50:95 (%)"); axs[0].legend(fontsize=8, frameon=False)
axs[1].plot(ep, [float(r["class_minus_geometry_AP50_95_points"]) for r in rows], "k-o", ms=4)
axs[1].axhline(0, color="k", lw=0.7)
axs[1].set_ylabel("Difference (AP points)")
for a in axs:
    a.set_xlabel("Epoch"); clean(a)
save("curves_epoch8.png")

# Per-class AP
cls = ["D00", "D10", "D20", "D40"]
s1 = [("No augmentation", [24.744, 20.206, 29.975, 16.367], "white", ""),
      ("Online shared", [24.905, 20.050, 30.162, 17.120], "0.6", ""),
      ("Class-adaptive", [24.235, 19.446, 30.069, 16.502], "0.15", "")]
s2 = [("Geometry only", [22.3915, 16.5069, 24.7822, 15.0576], "white", ""),
      ("Class-adaptive", [21.5166, 15.7426, 22.7198, 13.7686], "0.15", "")]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9))
for ax, data, title in [(axs[0], s1, "(a) 20-epoch runs"), (axs[1], s2, "(b) Epoch-8 paired run")]:
    w = 0.8 / len(data)
    for i, (lab, v, fc, h) in enumerate(data):
        ax.bar([j + (i - (len(data) - 1) / 2) * w for j in range(4)], v, w, color=fc, edgecolor="k",
               lw=0.7, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels(cls); ax.set_ylim(10, 32)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7, frameon=False); clean(ax)
axs[0].set_ylabel("Validation AP50:95 (%)")
save("per_class.png")

# Ablation differences
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9))
for ax, labs, vals, title in [
        (axs[0], ["No aug.", "Fixed\nimage", "Fixed\nlocal", "Online\nshared", "Class-\nadaptive"],
         [22.823, 22.810, 22.555, 23.059, 22.563], "(a) 20 epochs, vs no augmentation"),
        (axs[1], ["Geometry\nonly", "+ fixed\nlocal", "+ online\nshared", "+ class-\nadaptive"],
         [19.808, 18.861, 19.366, 19.608], "(b) Epoch 4, vs geometry only")]:
    d = [v - vals[0] for v in vals]
    ax.bar(range(len(d)), d, color="0.55", edgecolor="k", lw=0.7)
    for i, x in enumerate(d):
        ax.text(i, x + (0.03 if x >= 0 else -0.1), f"{x:+.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.7); ax.set_ylim(-1.15, 0.45)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_title(title, fontsize=10); clean(ax)
axs[0].set_ylabel("AP50:95 difference (points)")
save("ablation.png")

# Method diagram
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 50); ax.axis("off")


def box(x, y, w, h, t, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc="white", ec="k", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9,
            fontweight="bold" if bold else "normal")


def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1, color="k"))


ax.text(1, 47.5, "Training loop", fontsize=10, fontweight="bold")
box(1, 34, 17, 10, "Training image\ncontaining\nclasses S")
box(24, 34, 20, 10, "Choose edit o for S\n(no edit, or one of\nthe menu edits)", bold=True)
box(50, 34, 17, 10, "Flip and scale\n+ edit o")
box(73, 34, 17, 10, "Detector\n(Faster R-CNN or\nRetinaNet)")
box(95, 34, 13, 10, "SGD\nupdate")
for a, b in [(18.3, 23.7), (44.3, 49.7), (67.3, 72.7), (90.3, 94.7)]:
    arr(a, 39, b, 39)
ax.text(1, 20.5, "Online benefit estimate (every 200 steps)", fontsize=10, fontweight="bold")
box(1, 4, 21, 13, "Small training batch,\nunedited and with\neach edit")
box(28, 4, 22, 13, "Head gradients:\ng0 (unedited)\ngo (edited)")
box(56, 4, 22, 13, "Held-out reference\nimages of class c:\ngradient hc")
box(84, 4, 24, 13, "Benefit of o for c:\nη<go - g0, hc> / (Lc + δ)\n(smoothed, with SE)", bold=True)
for a, b in [(22.3, 27.7), (50.3, 55.7), (78.3, 83.7)]:
    arr(a, 10.5, b, 10.5)
box(84, 23, 24, 6.5, "Edit x class benefit table")
arr(96, 17.3, 96, 22.7)
arr(83.7, 26.2, 34, 33.7)
ax.text(46, 25.0, "average benefit over classes in S,\nblock edits that hurt any class in S,\n"
        "sample (softmax + 10% exploration)", ha="center", fontsize=8, style="italic")
save("method.png")
print("figures written to", HERE)
