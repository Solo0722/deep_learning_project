import json

import pytest

from cua.analysis.confirm import holm, one_sided, summarize, write_report
from cua.experiments.specs import CONFIRM_PICKS, CONFIRM_SEEDS, confirm_specs, part_a_specs, resolve

CLASSES = ("Cavity", "Hollowareas", "Rockpocket", "Crack", "Rust")


def _run(root, name, per_class):
    d = root / name
    d.mkdir()
    hist = {"epoch": 6, "metrics": {"per_class": {c: {"ap50": v} for c, v in per_class.items()}}}
    (d / "history.jsonl").write_text(json.dumps(hist) + "\n")
    (d / "COMPLETE.json").write_text("{}")


def _world(root, effect, noise_seed=0):
    """base AP 30 for every class; each pick arm adds `effect(k, pick_class, class)` per seed k."""
    import numpy as np
    rng = np.random.default_rng(noise_seed)
    for k in CONFIRM_SEEDS:
        _run(root, f"A-dacl10k-fasterrcnn-s{k}-base", {c: 30.0 for c in CLASSES})
        for e, lv, target in CONFIRM_PICKS:
            _run(root, f"S-dacl10k-fasterrcnn-s{k}-{e}-x{lv}",
                 {c: 30.0 + effect(k, target, c) + float(rng.normal(0, 0.05)) for c in CLASSES})


def test_confirm_specs_are_fresh_seeds_and_resolve():
    s = confirm_specs()
    assert len(s) == 15 and {x.seed for x in s} == {3, 4, 5}
    assert all(resolve(x.run_id).digest() == x.digest() for x in s)
    arms = [x for x in s if x.run_id.startswith("S-")]
    assert {(x.edit, x.edit_strength) for x in arms} == {(e, lv) for e, lv, _ in CONFIRM_PICKS}
    assert all(x.edit_p == 1.0 and x.init_run == f"A-dacl10k-fasterrcnn-s{x.seed}-warm" for x in arms)
    # Part A seeds 0-2 are untouched
    assert [x.digest()[:12] for x in part_a_specs("dacl10k", "fasterrcnn", (0,))[:3]] == \
        ["572fa110d1c6", "ff84440f8e6b", "d32141ef901b"]


def test_make_jobs_confirm_family_chains_on_warm(tmp_path):
    from cua.ccast.make_jobs import main
    out = tmp_path / "jobs"
    assert main(["--family", "confirm", "--dataset", "dacl10k", "--detector", "fasterrcnn", "--out", str(out)]) == 0
    sub = (out / "submit.sh").read_text()
    assert sub.count("qsub") == 15
    assert "qsub -W depend=afterok:$WARM_s4 S-dacl10k-fasterrcnn-s4-noise-x2.pbs" in sub


def test_holm_and_one_sided():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    r = one_sided([1.0, 1.2, 0.9])
    assert r["p"] < 0.01 and r["lo"] > 0


def test_class_specific_pick_is_confirmed(tmp_path):
    _world(tmp_path, lambda k, t, c: 1.5 if c == t else -0.2)
    res = summarize(tmp_path)
    assert all(r["confirmed"] for r in res["picks"]) and not res["missing_runs"]
    write_report(res, tmp_path / "out")
    assert "yes" in (tmp_path / "out" / "CONFIRM.md").read_text(encoding="utf-8")


def test_shared_gain_is_not_class_specific(tmp_path):
    _world(tmp_path, lambda k, t, c: 1.5)  # helps every class equally
    res = summarize(tmp_path)
    assert all(r["target"]["mean"] > 1 for r in res["picks"])
    assert not any(r["confirmed"] for r in res["picks"])


def test_harmful_or_noisy_pick_is_not_confirmed(tmp_path):
    _world(tmp_path, lambda k, t, c: (-1.0 if k == 3 else 1.0) if c == t else 0.0)
    assert not any(r["confirmed"] for r in summarize(tmp_path)["picks"])


def test_missing_runs_block_confirmation(tmp_path):
    _world(tmp_path, lambda k, t, c: 1.5 if c == t else -0.2)
    (tmp_path / "S-dacl10k-fasterrcnn-s5-noise-x2" / "COMPLETE.json").unlink()
    res = summarize(tmp_path)
    assert res["missing_runs"] == ["S-dacl10k-fasterrcnn-s5-noise-x2"]
    assert not next(r for r in res["picks"] if r["edit"] == "noise")["confirmed"]
