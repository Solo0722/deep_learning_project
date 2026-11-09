import json

import numpy as np
import pytest

from cua.analysis.part_a import (budget_plateaued, deltas, edit_class_tests, heatmap, holm, interaction_test,
                                 load_final, main, proxy_agreement, proxy_deltas)
from helpers import synthetic_deltas


def test_interaction_detected_when_classes_disagree():
    d = synthetic_deltas(edits=4, classes=4, seeds=3, conflict_edit=0, effect=3.0, noise=0.3, seed=0)
    assert interaction_test(d)["p"] < 0.01
    assert edit_class_tests(d)[0]["p_holm"] < 0.05


def test_no_interaction_under_null():
    d = synthetic_deltas(edits=4, classes=4, seeds=3, conflict_edit=None, effect=0.0, noise=0.3, seed=0)
    assert interaction_test(d)["p"] > 0.05


def test_budget_plateau_rule():
    assert budget_plateaued([10, 20, 30, 40, 41.0, 41.1, 41.2, 41.25, 41.3, 41.3])
    assert not budget_plateaued([10, 20, 30, 35, 38, 40, 42, 44, 46, 48.0])
    with pytest.raises(ValueError):
        budget_plateaued([1.0])


def test_holm_is_monotone():
    p = [0.01, 0.04, 0.03]
    adj = holm(p)
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def _row(edit, seed, cls, delta):
    return {"dataset": "d", "detector": "m", "seed": seed, "edit": edit, "cls": cls, "delta": delta}


def test_interaction_f_value_hand_computed():
    # Residuals (+0.5, -0.5) alternate sign across cells so the seed term cannot absorb them.
    cell = {("a", "x"): 1.0, ("a", "y"): 0.0, ("b", "x"): 0.0, ("b", "y"): 3.0}
    signs = [(0.5, -0.5), (-0.5, 0.5), (-0.5, 0.5), (0.5, -0.5)]
    rows = []
    for ((e, c), m), (r0, r1) in zip(cell.items(), signs):
        rows += [_row(e, 0, c, m + r0), _row(e, 1, c, m + r1)]
    out = interaction_test(rows)
    # Seed effect is exactly zero by symmetry, so the full-model residual is the within-cell residual:
    # 8 * 0.25 = 2 on df2 = 8 - 5 = 3. Interaction contrast 1-0-0+3 = 4 -> SS = 2 * 4^2 / 4 = 8 on df1 = 1.
    assert (out["df1"], out["df2"]) == (1, 3)
    assert out["F"] == pytest.approx(8.0 / (2.0 / 3.0))


def test_interaction_matches_statsmodels_if_available():
    sm = pytest.importorskip("statsmodels.formula.api")
    anova = pytest.importorskip("statsmodels.stats.anova")
    pd = pytest.importorskip("pandas")
    d = synthetic_deltas(edits=3, classes=3, seeds=3, conflict_edit=1, effect=1.0, noise=0.5, seed=2)
    df = pd.DataFrame(d)
    r = sm.ols("delta ~ C(seed) + C(edit) + C(cls)", df).fit()
    f = sm.ols("delta ~ C(seed) + C(edit) + C(cls) + C(edit):C(cls)", df).fit()
    tab = anova.anova_lm(r, f)
    out = interaction_test(d)
    assert out["F"] == pytest.approx(tab["F"].iloc[1])
    assert out["p"] == pytest.approx(tab["Pr(>F)"].iloc[1])


def test_mixed_pairs_raise():
    d = synthetic_deltas(3, 3, 3, None, 0.0, 0.3, 0)
    d += synthetic_deltas(3, 3, 3, None, 0.0, 0.3, 1, dataset="voc")
    with pytest.raises(ValueError):
        interaction_test(d)
    with pytest.raises(ValueError):
        edit_class_tests(d)


def test_degenerate_zero_residual():
    rows = [_row("a", s, c, v) for s in range(3) for c, v in (("x", 1.0), ("y", -1.0), ("z", 0.0))]
    out = edit_class_tests(rows)
    assert out[0]["p"] == 0.0 and out[0]["p_holm"] == 0.0
    rows = [_row("a", s, c, 2.0) for s in range(3) for c in "xyz"]
    assert edit_class_tests(rows)[0]["p"] == 1.0


def test_edit_class_tests_keep_input_order():
    d = synthetic_deltas(4, 4, 3, 2, 3.0, 0.3, 0)
    assert [r["edit"] for r in edit_class_tests(d)] == ["translate", "rotate", "vflip", "brightness"]


def test_proxy_agreement_identical_inputs():
    rng = np.random.default_rng(0)
    full = [{"edit": e, "cls": c, "delta": float(rng.normal())} for e in "abcd" for c in "wxyz"]
    short = [{**r, "seed": s} for r in full for s in range(3)]
    out = proxy_agreement(short, full, n_boot=300, seed=1)
    assert out["rho"] == pytest.approx(1.0)
    assert out["lo"] <= out["rho"] <= out["hi"]


def test_heatmap_writes_png(tmp_path):
    d = synthetic_deltas(4, 4, 3, 0, 3.0, 0.3, 0)
    p = tmp_path / "h.png"
    heatmap(d, p)
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _write_run(root, name, per_class, complete=True, early=None):
    d = root / name
    d.mkdir()
    rows = []
    if early is not None:
        rows.append({"epoch": 4, "metrics": {"per_class": {c: {"ap50": v} for c, v in early.items()}}})
    rows.append({"epoch": 6, "metrics": {"per_class": {c: {"ap50": v, "ap50_95": 1.0, "n_gt": 5}
                                                      for c, v in per_class.items()}}})
    (d / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if complete:
        (d / "COMPLETE.json").write_text("{}", encoding="utf-8")


def _make_root(root):
    rng = np.random.default_rng(0)
    for s in range(3):
        _write_run(root, f"A-rdd-fasterrcnn-s{s}-warm", {"x": 1.0, "y": 1.0})
        _write_run(root, f"A-rdd-fasterrcnn-s{s}-base", {"x": 50.0 + s, "y": 40.0, "z": None}, early={"x": 0.0})
        for e, shift in (("translate", 2.0), ("rotate", -1.0), ("vflip", 0.5)):
            _write_run(root, f"A-rdd-fasterrcnn-s{s}-{e}",
                       {"x": 50.0 + s + shift + rng.normal(0, .2), "y": 40.0 - shift + rng.normal(0, .2), "z": None})
    _write_run(root, "A-rdd-fasterrcnn-s0-hue", {"x": 99.0, "y": 99.0}, complete=False)


def test_load_final_and_deltas(tmp_path, capsys):
    _make_root(tmp_path)
    rows = load_final(tmp_path, "A-rdd-fasterrcnn")
    assert "1 incomplete" in capsys.readouterr().err
    assert {r["edit"] for r in rows} == {"base", "translate", "rotate", "vflip"}
    assert {r["cls"] for r in rows} == {"x", "y"}  # None ap50 dropped
    base = [r for r in rows if r["edit"] == "base" and r["seed"] == 1 and r["cls"] == "x"]
    assert base[0]["ap50"] == 51.0  # last history row, not the early one
    d = deltas(rows)
    assert len(d) == 3 * 3 * 2
    assert all(abs(r["delta"]) < 3 for r in d)
    no_base0 = deltas([r for r in rows if not (r["edit"] == "base" and r["seed"] == 0)])
    assert len(no_base0) == 2 * 3 * 2 and not any(r["seed"] == 0 for r in no_base0)


def test_cli_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_root(runs)
    for e, shift in (("base", 0.0), ("translate", 2.0), ("rotate", -1.0)):
        _write_run(runs, f"proxy-rdd-fasterrcnn-e20-{e}", {"x": 50.0 + shift, "y": 40.0 - shift})
    out = tmp_path / "out"
    assert main(["--runs-root", str(runs), "--out", str(out)]) == 0
    assert (out / "deltas.csv").exists()
    tests = json.loads((out / "tests.json").read_text())
    assert set(tests["rdd-fasterrcnn"]) == {"interaction", "edit_class"}
    assert (out / "heatmap_rdd_fasterrcnn.png").exists()
    assert (out / "proxy.json").exists()
    assert len(proxy_deltas(runs)) == 4
