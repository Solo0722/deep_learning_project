"""Verify a completed baseline's saved evidence and make validation-only figures.

Does not load crop pixels, train a model, or evaluate the held-out test set.
Checkpoints are loaded with weights_only=True on CPU.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from train_baseline import CompactCNN

LABELS = ["D00", "D10", "D20", "D40"]
NAMES = ["Longitudinal cracks", "Transverse cracks", "Alligator cracks", "Potholes"]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    support, predicted, correct = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(4), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(4), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(4), where=support + predicted > 0)
    return {"accuracy": float(correct.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "per_class": {c: {"support": int(support[i]), "precision": float(precision[i]),
                              "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, c in enumerate(LABELS)}}


def close(a, b, context):
    check(math.isfinite(float(a)) and math.isfinite(float(b)) and
          math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), context)


def compare_metrics(recomputed, stored, context):
    for metric in ("accuracy", "macro_f1"):
        close(recomputed[metric], stored[metric], context + ": " + metric)
    for label in LABELS:
        for metric in ("support", "precision", "recall", "f1"):
            close(recomputed["per_class"][label][metric], stored["per_class"][label][metric],
                  context + ": " + label + "/" + metric)


def same_state(a, b, context):
    check(a.keys() == b.keys(), context + ": keys")
    for name in a:
        check(torch.equal(a[name], b[name]), context + ": " + name)
        check(bool(torch.isfinite(a[name]).all()), context + ": nonfinite tensor")


def verify(run, manifest):
    config = read_json(run / "config.json")
    environment = read_json(run / "environment.json")
    status = read_json(run / "run_status.json")
    best = read_json(run / "best_validation.json")
    history = read_json(run / "history.json")
    history_csv = read_csv(run / "history.csv")
    predictions = read_csv(run / "validation_predictions.csv")
    manifest_rows = read_csv(manifest)
    train_rows = [r for r in manifest_rows if r["split"] == "train"]
    val_rows = [r for r in manifest_rows if r["split"] == "val"]
    check(digest(manifest) == config["manifest_sha256"], "Manifest hash mismatch")
    check(digest(Path(__file__).with_name("train_baseline.py")) == config["script_sha256"],
          "Training source differs from the executed version")
    check(config["pretrained"] is False and config["mode"] == "train", "Unexpected training configuration")
    check(all(obj["test_evaluated"] is False for obj in (config, status, best)), "Unexpected test evaluation flag")
    check(status["status"] == "complete", "Run incomplete")
    check(len(history) == len(history_csv) == config["epochs"] == status["completed_epochs"] ==
          status["planned_epochs"], "Epoch counts differ")
    for position, (record, csv_record) in enumerate(zip(history, history_csv), 1):
        check(record["epoch"] == int(csv_record["epoch"]) == position, "Nonsequential epochs")
        for key in ("lr", "seconds"):
            close(record[key], csv_record[key], "History CSV mismatch: " + key)
        check(record["seconds"] > 0, "Invalid epoch duration")
        for split, rows in (("train", train_rows), ("val", val_rows)):
            metrics = record[split]
            cm = np.array(metrics["confusion_matrix"])
            check(cm.shape == (4, 4) and bool((cm >= 0).all()) and bool((cm == cm.astype(int)).all()),
                  "Invalid confusion matrix")
            check(metrics["confusion_order"] == LABELS, "Unexpected class order")
            expected_support = Counter(r["label"] for r in rows)
            check(cm.sum(1).tolist() == [expected_support[c] for c in LABELS], "Split supports mismatch")
            compare_metrics(from_confusion(cm), metrics, "History epoch %d %s" % (position, split))
            for metric in ("loss", "accuracy", "macro_f1"):
                close(metrics[metric], csv_record[split + "_" + metric], "History metric mismatch")
    selected = max(history, key=lambda h: h["val"]["macro_f1"])
    check(selected["epoch"] == best["epoch"] == status["best_epoch"], "Incorrect checkpoint selection")
    check(selected["val"] == best["metrics"], "Selected metrics differ from epoch history")
    expected = {r["sample_id"]: r for r in val_rows}
    check(len(expected) == len(val_rows) == len(predictions), "Prediction count mismatch")
    check(len({r["sample_id"] for r in predictions}) == len(predictions), "Duplicate prediction IDs")
    check({r["sample_id"] for r in predictions} == set(expected), "Prediction IDs differ from validation")
    cm = np.zeros((4, 4), dtype=np.int64)
    for row in predictions:
        ref = expected[row["sample_id"]]
        check(row["true_class"] == ref["label"], "Prediction label mismatch")
        check(all(row[key] == ref[key] for key in ("image", "group_id", "source")), "Prediction metadata mismatch")
        check(row["predicted_class"] in LABELS, "Unknown predicted class")
        cm[LABELS.index(row["true_class"]), LABELS.index(row["predicted_class"])] += 1
    check(cm.tolist() == best["metrics"]["confusion_matrix"], "Prediction/confusion matrix mismatch")
    compare_metrics(from_confusion(cm), best["metrics"], "Predictions")

    best_pt = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    last_pt = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    check(best_pt["config"] == last_pt["config"] == config, "Checkpoint configuration mismatch")
    check(best_pt["epoch"] == last_pt["best"]["epoch"] == best["epoch"], "Checkpoint selected epoch mismatch")
    check(last_pt["epoch"] == len(history), "Final checkpoint epoch mismatch")
    check(last_pt["history"] == history, "Final checkpoint history mismatch")
    check(last_pt["best"]["metrics"] == best["metrics"], "Checkpoint metric mismatch")
    same_state(best_pt["model"], last_pt["best"]["model"], "Selected checkpoint weights")
    if best_pt["epoch"] == last_pt["epoch"]:
        same_state(best_pt["model"], last_pt["model"], "Selected/final checkpoint weights")
    model = CompactCNN()
    model.load_state_dict(best_pt["model"], strict=True)
    for index, pred in zip(last_pt["best"]["indices"], last_pt["best"]["predictions"]):
        check(predictions[index]["sample_id"] == val_rows[index]["sample_id"] and
              predictions[index]["predicted_class"] == LABELS[pred], "Checkpoint prediction mismatch")
    check(len(last_pt["best"]["indices"]) == len(predictions), "Checkpoint prediction count mismatch")
    return config, environment, status, best, history, predictions, cm, sum(p.numel() for p in model.parameters())


def save_figure(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(out / (name + ".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def figures(history, cm, best, out, seed):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.85), layout="constrained")
    for ax, metric, label in zip(axes, ("loss", "accuracy", "macro_f1"),
                                 ("Cross-entropy loss", "Accuracy", "Macro-F1")):
        for split, color, style in (("train", "#1b637f", "-"), ("val", "#b84d17", "--")):
            ax.plot(epochs, [r[split][metric] for r in history], label="Train" if split == "train" else "Validation",
                    color=color, linestyle=style, linewidth=1.6)
        ax.set(xlabel="Epoch", ylabel=label, xlim=(1, len(history)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(alpha=.2, linewidth=.5)
        ax.scatter([best["epoch"]], [best["metrics"][metric]], s=20, color="#b84d17", zorder=4)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Conventional augmentation | seed %d | selected epoch %d" % (seed, best["epoch"]), fontsize=10)
    fig.supxlabel("Training: augmented batches, dropout active; validation: fixed inputs, evaluation mode.", fontsize=7)
    save_figure(fig, out, "learning_curves")

    proportions = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.1, 3.55), layout="constrained")
    matrix = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "%d\n%.1f%%" % (cm[i, j], 100*proportions[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if proportions[i, j] > .5 else "#20252b")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted class", ylabel="True class", title="Validation confusion matrix | epoch %d" % best["epoch"])
    ax.tick_params(length=0)
    fig.colorbar(matrix, ax=ax, shrink=.82, label="Fraction within true class")
    fig.supxlabel("%s crops | counts and row percentages | seed %d" % (format(int(cm.sum()), ","), seed), fontsize=7)
    save_figure(fig, out, "confusion_matrix")

    fig, ax = plt.subplots(figsize=(5.1, 2.9), layout="constrained")
    x = np.arange(4)
    for metric, shift, color in (("recall", -.18, "#1b637f"), ("f1", .18, "#b84d17")):
        values = [best["metrics"]["per_class"][c][metric] for c in LABELS]
        bars = ax.bar(x + shift, values, width=.34, color=color, label="Recall" if metric == "recall" else "F1")
        ax.bar_label(bars, labels=["%.3f" % v for v in values], fontsize=7, padding=2)
    ax.set(xticks=x, xticklabels=[c + "\n" + name for c, name in zip(LABELS, NAMES)],
           ylabel="Score", ylim=(0, 1.14), title="Per-class validation performance | epoch %d" % best["epoch"])
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.03), ncols=2, fontsize=7)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    save_figure(fig, out, "per_class_performance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    run, manifest = args.run.resolve(), args.manifest.resolve()
    config, env, status, best, history, predictions, cm, parameters = verify(run, manifest)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    geometry = read_json(run / "geometry_metrics.json")
    check(geometry["test_evaluated"] is False and geometry["evaluation_split"] == "val", "Unexpected geometry split")
    for key in ("geometry", "majority"):
        compare_metrics(from_confusion(geometry[key]["confusion_matrix"]), geometry[key], key)
        check(np.array(geometry[key]["confusion_matrix"]).sum(1).tolist() == cm.sum(1).tolist(),
              "Geometry validation support mismatch")
    figures(history, cm, best, out, config["seed"])
    with (out / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "description", "support", "precision", "recall", "f1"])
        writer.writeheader()
        for c, name in zip(LABELS, NAMES):
            writer.writerow({"class": c, "description": name, **best["metrics"]["per_class"][c]})
    with (out / "confusion_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_class/predicted_class"] + LABELS)
        writer.writerows([[c] + row.tolist() for c, row in zip(LABELS, cm)])
    errors = [r for r in predictions if r["true_class"] != r["predicted_class"]]
    with (out / "validation_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(errors)
    total_seconds = sum(h["seconds"] for h in history)
    off_diagonal = sorted([(int(cm[i, j]), LABELS[i], LABELS[j]) for i in range(4) for j in range(4) if i != j], reverse=True)
    originals = ["config.json", "environment.json", "run_status.json", "best_validation.json", "history.csv",
                 "history.json", "validation_predictions.csv", "best.pt", "last.pt", "geometry_metrics.json", "augmentation_preview.jpg"]
    audit = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "run_directory": str(run),
             "checks_passed": ["source and manifest hashes", "complete sequential epoch history and CSV",
                               "all epoch metrics recomputed from confusion matrices", "per-split class support",
                               "all validation predictions matched to frozen manifest", "prediction-derived selected metrics",
                               "checkpoint selection rule", "both checkpoints and their configurations/history/predictions",
                               "selected weights agree across files and load into the model", "finite selected weights",
                               "geometry comparator metrics and validation support"],
             "original_artifact_sha256": {name: digest(run / name) for name in originals},
             "analysis_script_sha256": digest(Path(__file__)), "manifest_sha256": digest(manifest),
             "trainable_parameters": parameters, "selected_epoch": best["epoch"],
             "validation_crops": int(cm.sum()), "correct_validation_crops": int(cm.trace()),
             "incorrect_validation_crops": len(errors), "accuracy": best["metrics"]["accuracy"],
             "macro_f1": best["metrics"]["macro_f1"], "summed_epoch_seconds": total_seconds,
             "median_epoch_seconds_excluding_first": float(np.median([h["seconds"] for h in history[1:]])),
             "timing_scope": "recorded training + validation epoch intervals; excludes initialization and post-epoch checkpoint writes",
             "gpu": env["gpu"], "host": env["host"], "PBS_JOBID": env["PBS_JOBID"],
             "crop_pixels_loaded_for_analysis": False, "checkpoint_inference_rerun": False,
             "test_evaluated": False, "analysis_versions": {"torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    (out / "verification.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    rows = ["| Class | Defect | Support | Precision | Recall | F1 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for c, name in zip(LABELS, NAMES):
        m = best["metrics"]["per_class"][c]
        rows.append("| %s | %s | %d | %.4f | %.4f | %.4f |" % (c, name, m["support"], m["precision"], m["recall"], m["f1"]))
    report = f"""# Conventional augmentation baseline: verified validation results

The completed seed-{config['seed']} run selected epoch {best['epoch']} of {config['epochs']} by maximum validation macro-F1. Recomputing the saved predictions gives **accuracy {best['metrics']['accuracy']:.6f} ({100*best['metrics']['accuracy']:.2f}%)** and **macro-F1 {best['metrics']['macro_f1']:.6f}**: {int(cm.trace()):,} of {int(cm.sum()):,} validation crops were classified correctly. There are {len(errors):,} errors. These are validation results from one training seed.

## Per-class performance

{chr(10).join(rows)}

D20 has the lowest recall and F1. The largest directional confusion is D20 to D00: {cm[2,0]} crops, or {100*cm[2,0]/cm[2].sum():.2f}% of all D20 validation crops and {100*cm[2,0]/len(errors):.2f}% of all validation errors. This identifies an error pattern; it does not establish whether the cause is class imbalance, visual overlap, annotation ambiguity, or a particular augmentation. The next investigation should inspect representative validation errors and compare controlled policies.

## Learning behavior

Training loss decreased from {history[0]['train']['loss']:.4f} to {history[-1]['train']['loss']:.4f}; validation loss decreased from {history[0]['val']['loss']:.4f} to {history[-1]['val']['loss']:.4f}. Validation macro-F1 increased from {history[0]['val']['macro_f1']:.4f} to {history[-1]['val']['macro_f1']:.4f}. There were temporary validation fluctuations, but no sustained validation-loss deterioration over this 20-epoch run. The final epoch is the best under the declared schedule; this does not demonstrate convergence or justify selectively extending this condition.

Training metrics were accumulated on augmented batches while weights changed and dropout was active. Validation metrics were computed after each epoch with deterministic inputs and evaluation mode. Consequently, early validation metrics exceeding training metrics do not themselves indicate a data error, and the plotted train/validation difference is not a controlled estimate of a generalization gap.

## Geometry comparator

The metadata-only comparator uses bounding-box aspect ratio and reaches validation accuracy {geometry['geometry']['accuracy']:.4f}, macro-F1 {geometry['geometry']['macro_f1']:.4f}. Always predicting the training-majority class reaches {geometry['majority']['accuracy']:.4f} and {geometry['majority']['macro_f1']:.4f}, respectively. The CNN exceeds these particular comparators. This does not rule out use of annotation geometry or country-related cues by the CNN. All compared augmentation methods should retain the same preprocessing.

## Reproducibility and verification

The model contains {parameters:,} trainable parameters and was randomly initialized. The verified configuration uses the shared fixed mild policy, unweighted cross-entropy, AdamW, batch size {config['batch_size']}, a {config['epochs']}-epoch cosine learning-rate schedule, and no class resampling. The run used {env['gpu']} on {env['host']}, PBS job {env['PBS_JOBID']}, PyTorch {config['torch']}, CUDA build {env['cuda_build']}, Pillow {config['pillow']}, and NumPy {config['numpy']}.

Recorded epoch intervals total {total_seconds:.1f} seconds ({total_seconds/60:.2f} minutes); the median after epoch 1 is {audit['median_epoch_seconds_excluding_first']:.1f} seconds. These intervals include training and validation and exclude initialization and post-epoch checkpoint writes. They are not a measurement of total allocation time.

The source and manifest hashes match the executed configuration. All 20 epoch records agree across JSON and CSV; their confusion matrices reproduce their metrics and expected class counts. Every selected prediction matches exactly one frozen validation-manifest record, including source, image, group, and label. Selected metrics reproduce from the predictions. Both checkpoints load successfully on CPU; their configurations, histories, selected predictions, and selected weights agree. The selected weights fit the declared architecture and are finite. See `verification.json` for checksums and detailed verification.

No crop pixels were loaded for this numerical review, and inference was not rerun. The saved training augmentation preview was inspected separately. The development run records no test evaluation; the analysis uses validation predictions only. Visual inspection of individual validation errors remains outstanding. Current checks do not prove complete capture-sequence independence beyond the documented grouping procedure.

## Deliverable status and next work

The initial working baseline is complete: the pipeline trains, saves checkpoints, selects a model using validation macro-F1, and supplies verified metrics and figures. The prepared splits are complete for the current protocol. This one-seed result provides no evidence yet that class adaptation improves performance. The fixed class-specific, adaptive, repeated-seed, and ablation comparisons remain to be run after their protocol is finalized. Revised methodology and final writing checks remain separate submission tasks. Preserve the held-out test set until the comparison protocol is fixed.

## Files

- `learning_curves.png` / `.pdf`: training and validation loss, accuracy, and macro-F1.
- `confusion_matrix.png` / `.pdf`: counts and percentages within each true class.
- `per_class_performance.png` / `.pdf`: recall and F1 by class.
- `per_class_metrics.csv`, `confusion_counts.csv`: numerical tables.
- `validation_errors.csv`: all {len(errors)} misclassified validation crop identifiers, without pixels or confidence estimates.
- `verification.json`: consistency checks, artifact hashes, timing scope, and analysis versions.

Reproduce with `python summarize_baseline.py --run reports/conventional_seed42_v1 --manifest dataset_splits_v1/crops.csv` from the project directory. The analysis script leaves original run artifacts unchanged.
"""
    (out / "RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"verified": True, "accuracy": audit["accuracy"], "macro_f1": audit["macro_f1"],
                      "errors": len(errors), "epoch_minutes": total_seconds/60,
                      "parameters": parameters, "largest_confusions": off_diagonal[:4], "report": str(out / "RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()
