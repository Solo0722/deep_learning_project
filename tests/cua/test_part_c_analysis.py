"""Part C statistics, tables and figures on synthetic run directories (no training)."""
import csv
import json
import math

import numpy as np
import pytest

from cua.analysis.part_c import compare, h3_baseline, h4_agreement, load_runs, seed_t_interval, summarize
from cua.augment.menu import valid_menu
from cua.records import Record, write_classes, write_records

MENU = valid_menu("voc")
GT = [[10.0, 10.0, 40.0, 40.0], [50.0, 20.0, 80.0, 50.0]]


@pytest.fixture
def data_root(tmp_path):
    d = tmp_path / "data" / "voc"
    d.mkdir(parents=True)
    recs = [Record(f"img/{i}.png", 96, 64, np.array(GT), np.array([0, 1]), np.array([False, False]), f"g{i}", "s")
            for i in range(3)]
    write_records(d / "test.jsonl", recs)
    write_classes(d / "classes.json", ("a", "b"))
    return tmp_path / "data"


def _preds(quality: str):
    """'good': both GT boxes found on every image; 'poor': only class a found."""
    out = []
    for i in range(3):
        boxes, labels = (GT, [0, 1]) if quality == "good" else (GT[:1], [0])
        out.append({"image_index": i, "boxes": boxes, "scores": [0.9] * len(labels), "labels": labels})
    return out


def _run(runs, rid, test_ap=None, val=50.0, preds="good", utility=None, est_seconds=0.0, complete=True):
    d = runs / rid
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"digest": "x"}), encoding="utf-8")
    if complete:
        (d / "COMPLETE.json").write_text("{}", encoding="utf-8")
    hist = [{"epoch": e, "seconds": 100.0, "metrics": {"macro_ap50": val - 2 + e},
             "policy": {"estimator_seconds": est_seconds}} for e in (1, 2)]
    (d / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hist), encoding="utf-8")
    if test_ap is not None:
        (d / "test_metrics.json").write_text(json.dumps({
            "macro_ap50": test_ap, "ap50_95": test_ap / 2, "ap_small": None, "ap_medium": 1.0, "ap_large": 2.0,
            "fp_images": {"count": 0, "n_target_free": 0, "fraction": None}}), encoding="utf-8")
        (d / "test_predictions.json").write_text(json.dumps(_preds(preds)), encoding="utf-8")
    if utility is not None:
        (d / "utility.jsonl").write_text("".join(json.dumps(r) + "\n" for r in utility), encoding="utf-8")
    return d


def _deltas(path, value):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "detector", "seed", "edit", "cls", "delta"])
        w.writeheader()
        for s in (0, 1, 2):
            for k, e in enumerate(MENU):
                for j, c in enumerate(("a", "b")):
                    w.writerow({"dataset": "voc", "detector": "fasterrcnn", "seed": s, "edit": e, "cls": c,
                                "delta": value(k, j)})
    return path


def _util_rows():
    u = [[float(k + j) for j in range(2)] for k in range(len(MENU))]
    return [{"u": u, "valid": [[True, True]] * len(MENU)}]


@pytest.fixture
def runs(tmp_path):
    r = tmp_path / "runs"
    for s, (base, uni, meas, sh, cua) in enumerate([(40.0, 41.0, 41.5, 42.0, 43.0), (40.5, 41.2, 41.0, 42.1, 43.0),
                                                     (40.2, 41.4, 41.8, 41.9, 43.0)]):
        _run(r, f"C-voc-fasterrcnn-s{s}-base", base, preds="poor")
        _run(r, f"C-voc-fasterrcnn-s{s}-uniform", uni, preds="poor")
        _run(r, f"C-voc-fasterrcnn-s{s}-measured", meas, preds="poor")
        _run(r, f"C-voc-fasterrcnn-s{s}-shared", sh, preds="poor", est_seconds=20.0)
        _run(r, f"C-voc-fasterrcnn-s{s}-cua", cua + (s - 1) * 0.1, val=50.0, preds="good",
             utility=_util_rows(), est_seconds=30.0)
    return r


def test_seed_t_interval_known_values():
    t = seed_t_interval([1.0, 2.0, 3.0])
    half = 4.302653 * 1.0 / math.sqrt(3)
    assert t["mean"] == pytest.approx(2.0) and t["n"] == 3
    assert t["lo"] == pytest.approx(2.0 - half, abs=1e-5) and t["hi"] == pytest.approx(2.0 + half, abs=1e-5)
    one = seed_t_interval([1.0])
    assert one["lo"] is None and one["hi"] is None


def test_compare_pairs_seeds_and_claims(runs, data_root):
    from cua.analysis.part_c import build_matches
    rows = load_runs(runs)
    m = build_matches(runs, data_root, rows)
    c = compare(rows, "voc", "fasterrcnn", "shared", "cua", matches=m)
    assert c["n_seeds"] == 3 and c["seeds"] == [0, 1, 2]
    assert c["t"]["lo"] > 0 and c["boot"]["lo"] > 0 and c["claim"] is True


def test_unpaired_seed_is_dropped(tmp_path):
    r = tmp_path / "runs"
    for s in (0, 1, 2):
        _run(r, f"C-voc-fasterrcnn-s{s}-shared", 42.0 + s)
    for s in (0, 1):
        _run(r, f"C-voc-fasterrcnn-s{s}-cua", 43.0 + s * 1.1)
    c = compare(load_runs(r), "voc", "fasterrcnn", "shared", "cua")
    assert c["n_seeds"] == 2 and c["seeds"] == [0, 1]
    r2 = tmp_path / "runs2"
    _run(r2, "C-voc-fasterrcnn-s0-shared", 42.0)
    _run(r2, "C-voc-fasterrcnn-s0-cua", 43.0)
    c2 = compare(load_runs(r2), "voc", "fasterrcnn", "shared", "cua")
    assert c2["claim"] is False and c2["note"] == "fewer than 2 paired seeds"


def test_h3_baseline_is_better_mean(runs):
    assert h3_baseline(load_runs(runs), "voc", "fasterrcnn") == "uniform"


def test_escalation_flag(tmp_path, data_root):
    for name, diffs, expect in (("wide", [-0.6, 0.1, 0.9], True), ("tight", [0.9, 1.0, 1.1], False)):
        r = tmp_path / name
        for s, dlt in enumerate(diffs):
            _run(r, f"C-voc-fasterrcnn-s{s}-shared", 42.0)
            _run(r, f"C-voc-fasterrcnn-s{s}-cua", 42.0 + dlt)
        res = summarize(r, data_root, _deltas(tmp_path / f"{name}.csv", lambda k, j: k + j), n_boot=20)
        assert res["h2"]["voc-fasterrcnn"]["escalate_to_5_seeds"] is expect


def test_h4_uses_valid_cells_only(tmp_path, data_root):
    r = tmp_path / "runs"
    u1 = [[float(k + j) for j in range(2)] for k in range(len(MENU))]
    u2 = [row[:] for row in u1]
    u2[0][0] = 1e9
    valid2 = [[True, True] for _ in MENU]
    valid2[0][0] = False
    _run(r, "C-voc-fasterrcnn-s0-cua", 43.0, utility=[{"u": u1, "valid": [[True, True]] * len(MENU)},
                                                         {"u": u2, "valid": valid2}])
    with open(_deltas(tmp_path / "d.csv", lambda k, j: k + j), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    h = h4_agreement(r, rows, "voc", "fasterrcnn", ("a", "b"), n_boot=50)
    assert h["n_cells"] == 2 * len(MENU) and h["n_runs"] == 1
    assert h["rho"] == pytest.approx(1.0) and math.isfinite(h["lo"])


def test_ablation_compares_validation_only(runs, data_root, tmp_path):
    for s in (0, 1, 2):
        _run(runs, f"X-voc-fasterrcnn-s{s}-noharm", val=49.0 + 0.1 * s)
    res = summarize(runs, data_root, _deltas(tmp_path / "d.csv", lambda k, j: k + j), n_boot=20)
    ab = res["ablations"]["voc"]["noharm"]
    assert ab["boot"] is None and ab["n_seeds"] == 3 and ab["metric"] == "val"
    assert ab["t"]["mean"] == pytest.approx(np.mean([49.0 + 0.1 * s - 50.0 for s in (0, 1, 2)]))


def test_missing_lists_unevaluated_runs(runs, data_root, tmp_path):
    _run(runs, "C-voc-fasterrcnn-s3-cua")  # complete, not evaluated on test
    res = summarize(runs, data_root, _deltas(tmp_path / "d.csv", lambda k, j: k + j), n_boot=20)
    assert "C-voc-fasterrcnn-s3-cua" in res["missing"]


def test_summary_arms_overhead_and_secondary(runs, data_root, tmp_path):
    res = summarize(runs, data_root, _deltas(tmp_path / "d.csv", lambda k, j: k + j), n_boot=20)
    arms = res["arms"]["voc-fasterrcnn"]
    assert arms["cua"]["mean"] == pytest.approx(43.0) and arms["base"]["n"] == 3
    assert res["h3"]["voc-fasterrcnn"]["baseline"] == "uniform"
    assert res["overhead"]["voc-fasterrcnn"]["cua"]["share"] == pytest.approx(30.0 / 100.0)
    assert res["overhead"]["voc-fasterrcnn"]["shared"]["share"] == pytest.approx(0.2)
    sec = res["secondary"]["voc-fasterrcnn"]["cua"]
    assert sec["ap_small"] is None and sec["fp_fraction"] is None and sec["ap_large"] == pytest.approx(2.0)
    json.dumps(res)  # results.json must serialise


# ---- tables, figures, CLI (Task 4) ----

def test_cli_writes_tables_and_figures(runs, data_root, tmp_path):
    from cua.analysis.part_c import main
    out = tmp_path / "out"
    d = _deltas(tmp_path / "d.csv", lambda k, j: k + j)
    assert main(["--runs-root", str(runs), "--data-root", str(data_root), "--deltas", str(d), "--out", str(out),
                 "--n-boot", "20"]) == 0
    for name in ("results.json", "main_table.csv", "main_table.tex", "comparisons.csv", "fig_differences.png"):
        assert (out / name).exists(), name
    assert list(out.glob("fig_h4_*.png"))
    json.loads((out / "results.json").read_text(encoding="utf-8"))


def test_latex_bolds_row_best(runs, data_root, tmp_path):
    from cua.analysis.part_c import write_tables
    res = summarize(runs, data_root, _deltas(tmp_path / "d.csv", lambda k, j: k + j), n_boot=20)
    write_tables(res, tmp_path)
    rows = [x for x in (tmp_path / "main_table.tex").read_text(encoding="utf-8").splitlines() if x.startswith("VOC")]
    assert len(rows) == 1 and rows[0].count(r"\textbf{") == 1 and r"\textbf{43.0" in rows[0]
    with open(tmp_path / "main_table.csv", newline="", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    assert r[0]["dataset"] == "voc" and float(r[0]["cua_mean"]) == pytest.approx(43.0)


def test_comparisons_csv_has_h2_h3_rows(runs, data_root, tmp_path):
    from cua.analysis.part_c import write_tables
    write_tables(summarize(runs, data_root, _deltas(tmp_path / "d.csv", lambda k, j: k + j), n_boot=20), tmp_path)
    with open(tmp_path / "comparisons.csv", newline="", encoding="utf-8") as f:
        kinds = {(r["kind"], r["a"], r["b"]) for r in csv.DictReader(f)}
    assert ("h2", "shared", "cua") in kinds and ("h3", "uniform", "cua") in kinds


# ---- final-review fixes ----

def test_claim_is_directional(tmp_path):
    r = tmp_path / "runs"
    for s, d in enumerate([-1.0, -1.1, -0.9]):
        _run(r, f"C-voc-fasterrcnn-s{s}-shared", 42.0)
        _run(r, f"C-voc-fasterrcnn-s{s}-cua", 42.0 + d)
    c = compare(load_runs(r), "voc", "fasterrcnn", "shared", "cua")
    assert c["direction"] == -1 and c["claim"] is False


def test_forest_plot_draws_both_intervals(runs, data_root, tmp_path, monkeypatch):
    import matplotlib.axes
    from cua.analysis.part_c import write_figures
    res = summarize(runs, data_root, _deltas(tmp_path / "d.csv", lambda k, j: k + j), n_boot=20)
    widths = []
    orig = matplotlib.axes.Axes.plot

    def spy(self, *a, **k):
        widths.append(k.get("linewidth"))
        return orig(self, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy)
    write_figures(res, tmp_path / "figs")
    # per row: a t-interval line and a bootstrap line of different widths
    assert len({w for w in widths if w is not None and w > 0.9}) >= 2


def test_seed_count_follows_the_runs(tmp_path, data_root):
    from cua.analysis.part_c import write_tables
    r = tmp_path / "runs"
    for s in range(5):
        _run(r, f"C-voc-fasterrcnn-s{s}-shared", 42.0 + 0.1 * s)
        _run(r, f"C-voc-fasterrcnn-s{s}-cua", 43.0 + 0.1 * s)
    for s in range(3):
        _run(r, f"C-voc-fasterrcnn-s{s}-base", 40.0)
    res = summarize(r, data_root, _deltas(tmp_path / "d.csv", lambda k, j: k + j), n_boot=10)
    assert "C-voc-fasterrcnn-s4-base" in res["missing"]
    write_tables(res, tmp_path / "t")
    tex = (tmp_path / "t" / "main_table.tex").read_text(encoding="utf-8")
    assert "3 seeds" not in tex and "$^{3}$" in tex


def test_overhead_reports_share_and_extra(runs, data_root, tmp_path):
    res = summarize(runs, data_root, _deltas(tmp_path / "d.csv", lambda k, j: k + j), n_boot=20)
    o = res["overhead"]["voc-fasterrcnn"]["cua"]
    assert o["share"] == pytest.approx(0.30) and o["extra"] == pytest.approx(0.30 / 0.70)
