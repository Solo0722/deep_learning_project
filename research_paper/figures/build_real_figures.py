"""Figures built from real dataset images rendered by the training code.

Inputs: research_paper/figures/assets/ (from make_assets.py on CCAST) and the estimated class utilities of
the CUA runs in reports/cua_part_c_v2_report/results.json (h4 cells, dacl10k Faster R-CNN, mean of 3 seeds).
Outputs (research_paper/figures/):
  cua_framework_real.pdf   Fig. 2, the CUA framework with real images and the real utility table
  augmentation_gallery.pdf one real image per dataset under the edits of the menu
  copy_paste_examples.pdf  box-rectangle vs polygon-mask copy-paste, same paste plan
Run from the repository root:  python research_paper/figures/build_real_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A = HERE / "assets"
META = json.loads((A / "meta.json").read_text(encoding="utf-8"))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
INK, GREY, GREEN, ORANGE, RED = "#263b4a", "#6b7075", "#1b5e4b", "#e0832b", "#c4473d"
CMAP = LinearSegmentedColormap.from_list("rg", ["#c4473d", "#f4f4f4", "#2e8b57"])
BOX_COLORS = ["#ffd400", "#00e5ff", "#ff4fd8", "#7CFC00"]
EDIT_LABEL = {"orig": "Original", "translate": "Translation", "rotate": "Rotation", "vflip": "Vertical flip",
              "brightness": "Brightness", "contrast": "Contrast", "saturation": "Saturation", "hue": "Hue",
              "blur": "Blur", "noise": "Noise", "jpeg": "JPEG", "sharpen": "Sharpening",
              "erase": "Random erasing", "mosaic": "Mosaic"}
MENU = ["translate", "rotate", "vflip", "brightness", "contrast", "saturation", "hue",
        "blur", "noise", "jpeg", "sharpen", "erase", "mosaic"]


def img(ds: str, name: str, key: str) -> tuple[np.ndarray, dict]:
    m = META[ds]["images"][name][key]
    return np.asarray(Image.open(A / ds / name / m["file"]).convert("RGB")), m


def pick_draw(ds: str, name: str, edit: str) -> str:
    """Of the two rendered draws, the one that changes the image more (pixel edits are moderate)."""
    if edit in ("mosaic", "translate", "rotate", "vflip", "erase"):
        return f"{edit}_1"
    o = img(ds, name, "orig")[0].astype(float)
    d = [np.abs(img(ds, name, f"{edit}_{k}")[0].astype(float) - o).mean() for k in (0, 1)]
    return f"{edit}_{int(np.argmax(d))}"


def show(ax, rgb, meta, classes=None, color_of=None, labels=True, lw=0.8, highlight_from=None):
    ax.imshow(rgb)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4), s.set_color(GREY)
    for k, (b, lab) in enumerate(zip(meta["boxes"], meta["labels"])):
        pasted = highlight_from is not None and k >= highlight_from
        c = RED if pasted else (color_of(lab) if color_of else BOX_COLORS[0])
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=lw,
                               ls="--" if pasted else "-"))
        if labels and classes is not None:
            ax.text(b[0] + 3, b[1] + 3, classes[lab], color="black", fontsize=4.5, va="top", ha="left",
                    bbox=dict(boxstyle="square,pad=0.1", fc=c, ec="none", alpha=0.85))
    ax.set_xlim(0, rgb.shape[1]), ax.set_ylim(rgb.shape[0], 0)


def colors_for(classes, labels):
    uniq = sorted(set(labels))
    return lambda lab: BOX_COLORS[uniq.index(lab) % len(BOX_COLORS)] if lab in uniq else "#ffffff"


# --------------------------------------------------------------------------- gallery
def gallery():
    rows = [("dacl10k", "img1", "dacl10k"), ("rdd", "img3", "RDD2022"), ("voc", "img3", "PASCAL VOC")]
    cols = ["orig", "translate", "rotate", "contrast", "saturation", "blur", "noise", "jpeg", "erase", "mosaic"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(7.16, 2.55),
                             gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for r, (ds, name, title) in enumerate(rows):
        classes = META[ds]["classes"]
        _, m0 = img(ds, name, "orig")
        col = colors_for(classes, m0["labels"])
        for c, edit in enumerate(cols):
            key = "orig" if edit == "orig" else pick_draw(ds, name, edit)
            rgb, m = img(ds, name, key)
            if edit == "mosaic":  # 640x640 canvas: crop the central 4:3 band instead of squashing it
                h0 = (rgb.shape[0] - rgb.shape[1] * 3 // 4) // 2
                rgb = rgb[h0:rgb.shape[0] - h0]
                m = dict(m, boxes=[[b[0], b[1] - h0, b[2], b[3] - h0] for b in m["boxes"]])
            ax = axes[r, c]
            show(ax, rgb, m, classes, col, labels=(edit == "orig"), lw=0.5 if edit == "mosaic" else 0.6)
            ax.set_aspect("auto")
            if r == 0:
                ax.set_title(EDIT_LABEL[edit], fontsize=6.5, pad=2)
            if c == 0:
                ax.set_ylabel(title, fontsize=6.5)
    fig.savefig(HERE / "augmentation_gallery.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- copy-paste
def copy_paste():
    exs = ["ex2", "ex4", "ex1"]
    meta = META["paste_dacl10k"]
    classes = meta["classes"]
    fig, axes = plt.subplots(len(exs), 3, figsize=(3.45, 2.75), gridspec_kw=dict(wspace=0.04, hspace=0.05))
    for r, ex in enumerate(exs):
        e = meta["examples"][ex]
        n0 = len(e["orig"]["boxes"])
        for c, key in enumerate(("orig", "rect", "mask")):
            m = e[key]
            rgb = np.asarray(Image.open(A / "paste_dacl10k" / m["file"]).convert("RGB"))
            show(axes[r, c], rgb, m, classes, lambda lab: "#7CFC00", labels=False, lw=0.6, highlight_from=n0)
            axes[r, c].set_aspect("auto")
            if r == 0:
                axes[r, c].set_title(["Original", "Box-rectangle paste", "Polygon-mask paste"][c], fontsize=6.5, pad=2)
    fig.savefig(HERE / "copy_paste_examples.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- framework
def utility_table():
    r = json.loads((ROOT / "reports/cua_part_c_v2_report/results.json").read_text(encoding="utf-8"))
    cells = r["h4"]["dacl10k-fasterrcnn"]["cells"]
    classes = META["dacl10k"]["classes"]
    U = np.full((len(MENU), len(classes)), np.nan)
    for c in cells:
        U[MENU.index(c["edit"]), classes.index(c["cls"])] = c["u"]
    return U, classes


def framework():
    W, H = 7.16, 3.75
    fig = plt.figure(figsize=(W, H))

    def ax_at(x, y, w, h):  # inches from bottom-left
        return fig.add_axes([x / W, y / H, w / W, h / H])

    over = fig.add_axes([0, 0, 1, 1])
    over.set_xlim(0, W), over.set_ylim(0, H), over.axis("off")

    def arrow(x0, y0, x1, y1, color=GREY, lw=0.9, ls="-", rad=0.0, z=5):
        over.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                       ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))

    def label(x, y, text, sub=None):
        over.text(x, y, text, ha="center", va="bottom", fontsize=7, fontweight="bold")
        if sub:
            over.text(x, y - 0.03, sub, ha="center", va="top", fontsize=5.8, color=GREY)

    ds, name = "dacl10k", "img1"
    classes = META[ds]["classes"]
    rgb, m = img(ds, name, "orig")
    col = colors_for(classes, m["labels"])
    present = sorted(set(m["labels"]))
    U, ucls = utility_table()

    # ---------------- top row: one training step
    ty, th, tw = 2.55, 0.78, 1.04
    lab_y = ty + th + 0.20
    a = ax_at(0.08, ty, tw, th); show(a, rgb, m, classes, col, lw=0.7); a.set_aspect("auto")
    label(0.08 + tw / 2, lab_y, "Training image $x$", "$S(x)$ = {" + ", ".join(classes[c] for c in present) + "}")

    flip = rgb[:, ::-1]
    mf = dict(m)
    mf["boxes"] = [[rgb.shape[1] - b[2], b[1], rgb.shape[1] - b[0], b[3]] for b in m["boxes"]]
    a = ax_at(1.42, ty, tw, th); show(a, flip, mf, classes, col, labels=False, lw=0.7); a.set_aspect("auto")
    label(1.42 + tw / 2, lab_y, "Base recipe", "flip, scale jitter (every policy)")

    # policy scores for this image, from the real utility table
    s_all = np.concatenate([[0.0], np.nanmean(U[:, present], axis=1) * 1e4])
    names = ["No edit"] + [EDIT_LABEL[e] for e in MENU]
    srt = [int(i) for i in np.argsort(-s_all)]
    order = list(dict.fromkeys(srt[:5] + [0] + srt[-2:]))
    a = ax_at(3.42, ty + 0.08, 0.78, th - 0.04)
    vals = s_all[order]
    a.barh(range(len(order))[::-1], vals, height=0.68,
           color=[GREY if i == 0 else (GREEN if v > 0 else RED) for i, v in zip(order, vals)])
    a.set_yticks(range(len(order))[::-1]); a.set_yticklabels([names[i] for i in order], fontsize=5.2)
    a.axvline(0, color=INK, lw=0.4)
    a.tick_params(axis="x", labelsize=4.6, length=1.5, pad=1); a.tick_params(axis="y", length=0, pad=1)
    a.set_xlabel(r"$s(o\,|\,S)\times10^{4}$", fontsize=5.0, labelpad=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    label(3.55, lab_y, "Image policy", "average over $S(x)$, harm block, sample")

    erg, em = img(ds, name, pick_draw(ds, name, "mosaic"))
    a = ax_at(4.55, ty - 0.03, 0.84, 0.84); show(a, erg, em, classes, col, labels=False, lw=0.5)
    label(4.97, lab_y, "Apply one edit $o$", "top-scored here: mosaic")

    dx, dy = 5.72, ty + 0.10
    for t, x0, w, fc in (("ResNet-50", 0.0, 0.48, "#dfe7ef"), ("FPN", 0.54, 0.34, "#dfe7ef"),
                         ("Head", 0.94, 0.42, "#cfe6dc")):
        over.add_patch(FancyBboxPatch((dx + x0, dy + 0.08), w, 0.48, boxstyle="round,pad=0.01,rounding_size=0.04",
                                      fc=fc, ec=INK, lw=0.6))
        over.text(dx + x0 + w / 2, dy + 0.32, t, ha="center", va="center", fontsize=5.8)
    arrow(dx + 0.48, dy + 0.32, dx + 0.54, dy + 0.32, lw=0.6)
    arrow(dx + 0.88, dy + 0.32, dx + 0.94, dy + 0.32, lw=0.6)
    over.add_patch(Rectangle((dx + 0.74, dy + 0.02), 0.66, 0.60, fill=False, ec=GREEN, lw=0.9, ls="--"))
    over.text(dx + 1.07, dy - 0.02, r"$\theta_s$: top FPN level + head", ha="center", va="top", fontsize=5.4,
              color=GREEN)
    label(dx + 0.70, lab_y, "Detector update", "Faster R-CNN or RetinaNet")
    for x0, x1 in ((1.15, 1.39), (2.49, 2.88), (4.23, 4.52), (5.42, 5.69)):
        arrow(x0, ty + th / 2, x1, ty + th / 2)

    # ---------------- bottom: estimator
    over.add_patch(FancyBboxPatch((0.03, 0.04), 7.10, 2.12, boxstyle="round,pad=0.0,rounding_size=0.06",
                                  fill=False, ec=GREY, lw=0.6, ls=(0, (3, 2))))
    over.text(0.14, 2.16, "Class-utility estimator", fontsize=7.2, fontweight="bold", color=GREEN, va="center",
              bbox=dict(fc="white", ec="none", pad=1.5))
    over.text(1.50, 2.16, r"every $K$ optimizer steps; gradients on $\theta_s$ only; the model is not changed",
              fontsize=5.8, color=GREY, va="center", ha="left", bbox=dict(fc="white", ec="none", pad=1.5))
    arrow(0.60, ty - 0.03, 0.60, 2.05, ls="--")
    over.text(0.66, 2.36, "sampled", fontsize=5.2, color=GREY, ha="left")

    label(0.85, 1.86, "Estimation batch")
    for k, (nm, ed) in enumerate((("img0", "contrast"), ("img2", "blur"), ("img3", "hue"))):
        for j, key in enumerate(("orig", pick_draw(ds, nm, ed))):
            rr, mm = img(ds, nm, key)
            a = ax_at(0.14 + 0.50 * k, 1.42 - 0.40 * j, 0.47, 0.36)
            show(a, rr, mm, labels=False, lw=0.3, color_of=lambda _: BOX_COLORS[0]); a.set_aspect("auto")
            if j == 1:
                for sp in a.spines.values():
                    sp.set_color(ORANGE), sp.set_linewidth(1.0)
    over.text(1.66, 1.60, "unedited $B$", fontsize=5.4, color=GREY, ha="left", va="center")
    over.text(1.66, 1.20, "edited $T_o(B)$", fontsize=5.4, color=ORANGE, ha="left", va="center")

    label(0.85, 0.86, "Reference batch $R$")
    for k, cname in enumerate(sorted(META["ref_dacl10k"])[:12]):
        r_, c_ = divmod(k, 6)
        im = np.asarray(Image.open(A / "ref_dacl10k" / f"{cname}.jpg").convert("RGB"))
        a = ax_at(0.14 + 0.25 * c_, 0.44 - 0.33 * r_, 0.24, 0.31)
        a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_aspect("auto")
        for sp in a.spines.values():
            sp.set_linewidth(0.3)
        short = {"Efflorescence": "Efflor.", "ExposedRebars": "Exp. rebars", "Restformwork": "Restform."}
        a.text(0.5, 0.02, short.get(cname, cname), transform=a.transAxes, ha="center", va="bottom", fontsize=3.9,
               color="white",
               bbox=dict(fc="black", ec="none", alpha=0.6, pad=0.4))
    over.text(1.66, 0.45, "held out;\nevery class\npresent", fontsize=5.2, color=GREY, ha="left", va="center")

    cx, cy = 2.48, 0.98
    arrow(2.18, 1.40, cx - 0.17, cy + 0.17)
    over.text(2.12, 1.44, r"$g_0,\ g_o$", fontsize=6.2, ha="center", va="bottom")
    arrow(2.10, 0.45, cx - 0.17, cy - 0.17)
    over.text(2.40, 0.50, "$h_c$ per class", fontsize=6.2, ha="left", va="top")
    over.add_patch(plt.Circle((cx, cy), 0.22, fc="white", ec=GREEN, lw=1.2))
    over.text(cx, cy, r"$\langle\cdot,\cdot\rangle$", ha="center", va="center", fontsize=9, color=GREEN)
    over.text(cx, cy - 0.28, "$u(o,c)$, Eq. (1)", ha="center", va="top", fontsize=5.8)
    arrow(cx + 0.23, cy, 3.30, cy)

    v = float(np.nanpercentile(np.abs(U), 95))
    norm = TwoSlopeNorm(0, -v, v)
    a = ax_at(3.95, 0.42, 2.30, 1.24)
    a.imshow(np.clip(U, -v, v), cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
    a.set_yticks(range(len(MENU))); a.set_yticklabels([EDIT_LABEL[e] for e in MENU], fontsize=4.8)
    a.set_xticks(range(len(ucls)))
    a.set_xticklabels(ucls, fontsize=4.5, rotation=40, ha="right", rotation_mode="anchor")
    a.tick_params(length=0, pad=1)
    for c in present:
        a.add_patch(Rectangle((c - 0.5, -0.5), 1, len(MENU), fill=False, ec=INK, lw=0.9))
    over.text(5.10, 1.94, r"Utility table $\bar u(o,c)$", ha="center", fontsize=7, fontweight="bold")
    over.text(5.10, 1.91, "dacl10k, Faster R-CNN: run mean of the estimates over 3 seeds;\n"
              "outlined columns are the classes of the image above", ha="center", va="top", fontsize=5.2, color=GREY)

    a2 = ax_at(6.62, 0.42, 0.17, 1.24)
    a2.imshow(np.clip(np.nanmean(U, axis=1, keepdims=True), -v, v), cmap=CMAP, norm=norm, aspect="auto",
              interpolation="nearest")
    a2.set_xticks([]); a2.set_yticks([])
    over.text(6.70, 0.36, "Shared-\nutility", ha="center", va="top", fontsize=5.6)
    arrow(6.28, 1.11, 6.58, 1.11, ls="--")
    over.text(6.43, 1.15, "pool", ha="center", fontsize=5.2, color=GREY)

    arrow(5.10, 2.12, 3.85, ty + 0.02, color=GREEN, lw=1.3, rad=0.2, z=6)
    over.text(5.25, 2.30, "utilities", fontsize=6.2, color=GREEN, fontweight="bold", ha="left",
              bbox=dict(fc="white", ec="none", pad=0.5))
    fig.savefig(HERE / "cua_framework_real.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    gallery()
    copy_paste()
    framework()
    print("wrote augmentation_gallery.pdf, copy_paste_examples.pdf, cua_framework_real.pdf")
