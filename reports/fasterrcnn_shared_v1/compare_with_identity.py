"""Compare verified development runs without new inference or test-set access."""
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
IDENTITY = ROOT / "reports/fasterrcnn_baseline_v1"
METRICS = ("AP50_95", "AP50", "AP75", "AR100")
CLASSES = {"D00": "Longitudinal cracks", "D10": "Transverse cracks",
           "D20": "Alligator cracks", "D40": "Potholes"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name, content):
    path = OUT / name
    if path.exists():
        assert path.read_bytes() == content, f"Preserve existing {path}"
    else:
        with path.open("xb") as handle:
            handle.write(content)


def save_csv(name, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    save(name, stream.getvalue().encode())


def main():
    first, second = [read(p / "ccast_verification.json") for p in (IDENTITY, OUT)]
    assert first["status"] == second["status"] == "verified_with_selected_and_final_metrics_recomputed"
    histories = [read(OUT/n) for n in ("identity_history.json", "verified_history.json")]
    configs = [read(p / "verified_context.json") for p in (IDENTITY, OUT)]
    allowed = {"augmentation", "augmentation_policy", "version", "purpose", "recipe_status", "comparator", "sources"}
    assert {k: v for k, v in configs[0]["configuration"].items() if k not in allowed} == {
        k: v for k, v in configs[1]["configuration"].items() if k not in allowed}
    assert configs[0]["versions"] == configs[1]["versions"]
    assert configs[0]["raw_root"] == configs[1]["raw_root"] and configs[0]["split_root"] == configs[1]["split_root"]
    assert len(histories[0]) == len(histories[1]) == 20
    for left, right in zip(*histories):
        assert all(left[k] == right[k] for k in ("epoch", "step", "images_seen", "train_images", "val_images"))
    assert first["best_epoch"] == second["best_epoch"] == 14
    rows, classes, curves = [], [], []
    for selection in ("best", "final"):
        for metric in METRICS:
            a, b = [r[f"{selection}_metrics"][metric] for r in (first, second)]
            rows.append({"selection": selection, "metric": metric,
                         "identity_points": 100*a, "shared_points": 100*b, "change_pp": 100*(b-a)})
    for name, label in CLASSES.items():
        a, b = [r["best_metrics"]["per_class"][name] for r in (first, second)]
        assert a["target_count"] == b["target_count"]
        classes.append({"class": name, "description": label, "validation_targets": a["target_count"],
                        "identity_AP50_95": 100*a["AP50_95"], "shared_AP50_95": 100*b["AP50_95"],
                        "AP50_95_change_pp": 100*(b["AP50_95"]-a["AP50_95"]),
                        "identity_AP50": 100*a["AP50"], "shared_AP50": 100*b["AP50"],
                        "AP50_change_pp": 100*(b["AP50"]-a["AP50"])})
    for a, b in zip(*histories):
        curves.append({"epoch": a["epoch"], "identity_AP50_95": 100*a["validation"]["AP50_95"],
                       "shared_AP50_95": 100*b["validation"]["AP50_95"],
                       "change_pp": 100*(b["validation"]["AP50_95"]-a["validation"]["AP50_95"]),
                       "identity_AP50": 100*a["validation"]["AP50"], "shared_AP50": 100*b["validation"]["AP50"],
                       "applied_images": b["augmentation"]["counts"].get("applied_images", 0)})
    negatives = [h[13]["validation"]["no_target_predicted_boxes"] for h in histories]
    assert negatives[0].keys() == negatives[1].keys() and len(negatives[0]) == 2193
    ng = {"denominator": 2193, "confidence_threshold": 0.25,
          "identity_images_with_predictions": sum(v > 0 for v in negatives[0].values()),
          "shared_images_with_predictions": sum(v > 0 for v in negatives[1].values()),
          "identity_prediction_boxes": sum(negatives[0].values()), "shared_prediction_boxes": sum(negatives[1].values()),
          "newly_positive_images": sum(negatives[0][i] == 0 and negatives[1][i] > 0 for i in negatives[0]),
          "newly_clear_images": sum(negatives[0][i] > 0 and negatives[1][i] == 0 for i in negatives[0]),
          "scope": "Relative to retained annotations; no visual annotation completeness review"}
    ng["identity_fraction"] = ng["identity_images_with_predictions"] / ng["denominator"]
    ng["shared_fraction"] = ng["shared_images_with_predictions"] / ng["denominator"]
    ng["change_pp"] = 100*(ng["shared_fraction"]-ng["identity_fraction"])
    t1, t2 = first["run"]["elapsed_seconds"], second["run"]["elapsed_seconds"]
    report = {
        "status": "verified_descriptive_comparison_one_run_per_condition",
        "primary_comparison": "Validation AP50:95 at each run's declared selected checkpoint",
        "checkpoint_selection": {"identity_epoch": first["best_epoch"], "shared_epoch": second["best_epoch"],
                                 "rule": "Maximum full-validation AP50:95; earliest epoch wins ties"},
        "matched_configuration": True, "seed": 42, "train_images": 26863, "validation_images": 5760,
        "validation_targets": 8225, "epochs_each": 20, "updates_each": 134320,
        "aggregate_comparison": rows, "per_class_selected_comparison": classes,
        "negative_images_at_selected_epoch": ng,
        "runtime": {"identity_hours": t1/3600, "shared_hours": t2/3600, "difference_minutes": (t2-t1)/60,
                    "relative_difference_percent": 100*(t2/t1-1),
                    "identity_host": first["run"]["host"], "shared_host": second["run"]["host"],
                    "scope": "Whole invocation elapsed time on different A10 nodes; not isolated augmentation overhead"},
        "augmentation": second["augmentation_verification"],
        "unaugmented_epoch1": {**curves[0], "observed_trajectories_identical": False,
                               "interpretation": "Runs differ before augmentation starts. Cause is not established; matching seed/configuration did not produce an identical realized trajectory."},
        "verdict": "No observed gain in the primary selected-checkpoint metric for this fixed shared recipe; class-level changes are small and mixed.",
        "limits": ["One run per condition: training variability and statistical equivalence are not estimated.",
                   "Same validation set selected checkpoints and supplied comparison metrics; no untouched test-set estimate.",
                   "Warmup divergence prevents attributing tiny later differences exclusively to augmentation.",
                   "No adaptive or local augmentation controller was tested; no per-operation causal effect is identified.",
                   "Fixed recipe was an initial choice, not validation-tuned; this result does not rule out useful augmentation recipes.",
                   "Full checkpoint binaries and raw images were not independently reopened locally."],
        "next_action": "Preserve both controls. The next planned ablation is fixed local augmentation with the same bank, strengths, eligibility and probability; repeat matched runs before claiming a reliable small benefit.",
        "analysis_source_sha256": digest(Path(__file__)),
        "evidence": {"identity_archive_sha256": first["archive_sha256"], "shared_archive_sha256": second["archive_sha256"],
                     "identity_verification_sha256": digest(IDENTITY/"ccast_verification.json"),
                     "shared_verification_sha256": digest(OUT/"ccast_verification.json"),
                     "shared_recomputed_epochs": second["recomputed_epochs"],
                     "shared_metric_maximum_error": second["maximum_recomputed_metric_error"],
                     "identity_verification": "Original archive, prior audit source and report values rechecked; prior direct COCO recomputation retained"},
        "test_data_read": False, "new_training_or_inference_executed": False}
    save("comparison.json", (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())
    save_csv("aggregate_comparison.csv", rows)
    save_csv("per_class_comparison.csv", classes)
    save_csv("comparison_curves.csv", curves)

    plot_paths = [OUT/f"comparison.{extension}" for extension in ("png", "pdf")]
    assert not any(p.exists() for p in plot_paths), "Preserve existing plots"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for key, label, color in (("identity_AP50_95", "Identity", "#176B86"),
                              ("shared_AP50_95", "Fixed shared", "#BA5B24")):
        axes[0].plot([r["epoch"] for r in curves], [r[key] for r in curves], marker="o",
                     markersize=3, linewidth=1.7, color=color, label=label)
    axes[0].axvline(14, color="#888888", linestyle="--", linewidth=.9, label="Selected epoch (both)")
    axes[0].set(title="Validation learning curves", xlabel="Completed epoch", ylabel="AP50:95 (0–100)",
                ylim=(0, 27), xlim=(.5, 20.5), xticks=[1, 5, 10, 14, 20])
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    deltas = [r["AP50_95_change_pp"] for r in classes]
    bars = axes[1].bar(list(CLASSES), deltas, width=.6, color=["#377C72" if x >= 0 else "#BA5B24" for x in deltas])
    axes[1].bar_label(bars, labels=[f"{x:+.3f}" for x in deltas], padding=4, fontsize=9)
    axes[1].axhline(0, color="#444444", linewidth=.8)
    axes[1].set(title="Selected epoch: shared minus identity", xlabel="Damage class",
                ylabel="AP50:95 change (percentage points)", ylim=(-.45, .45))
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    fig.suptitle("Fixed shared augmentation | No overall AP gain in this seed", fontsize=14, y=.98)
    fig.text(.5, .028, "One run per condition. Epoch 1 unaugmented in both. Validation-selected results; test data unopened.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .065, 1, .93))
    fig.savefig(plot_paths[0], dpi=180, facecolor="white")
    fig.savefig(plot_paths[1], facecolor="white")
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("verdict", "aggregate_comparison", "per_class_selected_comparison",
                                          "negative_images_at_selected_epoch", "runtime", "unaugmented_epoch1")}, indent=2))


if __name__ == "__main__":
    main()
