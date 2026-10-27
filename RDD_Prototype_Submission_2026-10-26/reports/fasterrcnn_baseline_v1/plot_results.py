"""Plot only the verified baseline records; no model inference or test data."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
report = json.loads((OUT / "ccast_verification.json").read_text())
assert report["status"] == "verified_with_selected_and_final_metrics_recomputed"
with (OUT / "epoch_metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
epochs = [int(r["epoch"]) for r in rows]
assert epochs == list(range(1, 21))
paths = [OUT / f"learning_curves.{suffix}" for suffix in ("png", "pdf")]
if any(p.exists() for p in paths):
    raise SystemExit("Plot already exists; preserve the earlier artifact.")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
teal, orange = "#176B86", "#B85D14"
axes[0].plot(epochs, [float(r["train_total_loss"]) for r in rows], color=teal, marker="o", markersize=3)
axes[0].set(title="Mean training loss", xlabel="Completed epoch", ylabel="Total detection loss", ylim=(0, .3))
for epoch in (13, 17):
    axes[0].axvline(epoch, color="#777777", linestyle=":", linewidth=1)
    axes[0].text(epoch+.25, .285, "LR / 10", color="#555555", fontsize=8, rotation=90, va="top")
axes[1].plot(epochs, [100*float(r["val_AP50"]) for r in rows], color=teal, marker="o", markersize=3, label="AP50")
axes[1].plot(epochs, [100*float(r["val_AP50_95"]) for r in rows], color=orange, marker="o", markersize=3, label="AP50:95")
axes[1].axvline(report["best_epoch"], color="#555555", linestyle="--", linewidth=1, label="Selected epoch 14")
axes[1].set(title="Validation: 5,760 images", xlabel="Completed epoch", ylabel="AP (%)", ylim=(0, 65))
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
names = list(report["best_metrics"]["per_class"])
values = [100*report["best_metrics"]["per_class"][n]["AP50_95"] for n in names]
bars = axes[2].bar(names, values, color=[teal]*3+[orange], width=.65)
axes[2].bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=9)
axes[2].set(title="Per-class validation at epoch 14", xlabel="Damage class", ylabel="AP50:95 (%)", ylim=(0, 38))
for axis in axes:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DDDDDD", linewidth=.6)
for axis in axes[:2]:
    axis.set_xticks([1, 5, 10, 15, 20])
    axis.set_xlim(.5, 20.5)
fig.suptitle("Faster R-CNN baseline | Identity augmentation | Seed 42", fontsize=14, y=.98)
fig.text(.5, .035, "One development run. Epoch selected by validation AP50:95. Test data remain unopened.",
         ha="center", fontsize=9, color="#444444")
fig.tight_layout(rect=(0, .075, 1, .93))
fig.savefig(paths[0], dpi=180, facecolor="white")
fig.savefig(paths[1], facecolor="white")
plt.close(fig)
print("Saved:", *[str(p) for p in paths], sep="\n")
