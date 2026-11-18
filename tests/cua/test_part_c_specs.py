"""Measured-effect table, Part C / tuning / ablation specs and their job scripts."""
import csv
import json
import math
from pathlib import Path

import pytest

from cua.augment.menu import valid_menu
from cua.ccast.make_jobs import main as jobs_main
from cua.experiments.specs import (BUDGET_EPOCHS, CUA_SETTINGS, ablation_specs, part_c_specs, resolve,
                                   tune_specs)
from cua.measured import main as measured_main
from cua.records import write_classes
from cua.train import full_schedule


def _deltas(path: Path, seeds, value):
    menu = valid_menu("voc")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "detector", "seed", "edit", "cls", "delta"])
        w.writeheader()
        for s in seeds:
            for e in menu:
                for c in ("a", "b"):
                    w.writerow({"dataset": "voc", "detector": "fasterrcnn", "seed": s, "edit": e, "cls": c,
                                "delta": value(s, e, c)})


def test_measured_table_mean_and_se(tmp_path):
    write_classes((tmp_path / "data" / "voc").mkdir(parents=True) or tmp_path / "data" / "voc" / "classes.json",
                  ("a", "b"))
    _deltas(tmp_path / "d.csv", [0, 1, 2], lambda s, e, c: float(s) + (1.0 if c == "b" else 0.0))
    assert measured_main(["--deltas", str(tmp_path / "d.csv"), "--data-root", str(tmp_path / "data")]) == 0
    t = json.loads((tmp_path / "data" / "voc" / "measured_fasterrcnn.json").read_text(encoding="utf-8"))
    assert t["edits"] == list(valid_menu("voc")) and t["classes"] == ["a", "b"] and t["n_seeds"] == 3
    assert t["mean"][0] == pytest.approx([1.0, 2.0])
    assert t["se"][0] == pytest.approx([1.0 / math.sqrt(3), 1.0 / math.sqrt(3)])


def test_measured_refuses_one_seed(tmp_path):
    write_classes((tmp_path / "data" / "voc").mkdir(parents=True) or tmp_path / "data" / "voc" / "classes.json",
                  ("a", "b"))
    _deltas(tmp_path / "d.csv", [0], lambda s, e, c: 0.5)
    with pytest.raises(ValueError, match="seeds"):
        measured_main(["--deltas", str(tmp_path / "d.csv"), "--data-root", str(tmp_path / "data")])


def test_part_c_ids_and_schedule():
    specs = part_c_specs("voc", "retinanet", seeds=(1,))
    assert [s.run_id for s in specs] == [f"C-voc-retinanet-s1-{p}" for p in
                                         ("base", "uniform", "measured", "shared", "cua")]
    s = resolve("C-voc-retinanet-s1-cua")
    assert s.policy == "cua" and s.epoch_lr == full_schedule(BUDGET_EPOCHS["voc"]) and s.init_run is None
    assert (s.est_every, s.theta, s.epsilon) == (CUA_SETTINGS["est_every"], CUA_SETTINGS["theta"],
                                                 CUA_SETTINGS["epsilon"])
    assert s.edit is None and s.augment


def test_base_policy_spec():
    s = resolve("C-rdd-fasterrcnn-s0-base")
    assert s.policy == "base" and s.edit is None


def test_tune_has_six_unique_runs():
    specs = tune_specs()
    assert len({s.run_id for s in specs}) == len({s.digest() for s in specs}) == 6
    assert all(s.run_id.startswith("T-rdd-fasterrcnn-s0-") and s.policy == "cua" for s in specs)
    by = {s.run_id.split("-")[-1]: s for s in specs}
    assert (by["k100"].est_every, by["k400"].est_every) == (100, 400)
    assert by["head"].theta == "head" and by["eps0.05"].epsilon == 0.05 and by["eps0.2"].epsilon == 0.2
    for s in specs:
        assert resolve(s.run_id) == s


def test_ablation_flags():
    by = {s.run_id: s for s in ablation_specs("dacl10k", seeds=(0,))}
    assert by["X-dacl10k-fasterrcnn-s0-noharm"].harm_block is False
    assert by["X-dacl10k-fasterrcnn-s0-noid"].allow_identity is False
    assert by["X-dacl10k-fasterrcnn-s0-invap"].weights == "inv_ap"
    for s in by.values():
        assert s.policy == "cua" and resolve(s.run_id) == s
    with pytest.raises(ValueError):
        ablation_specs("voc")


def test_make_jobs_part_c_writes_15_pbs_and_no_dependencies(tmp_path):
    out = tmp_path / "jobs"
    assert jobs_main(["--family", "part_c", "--dataset", "rdd", "--detector", "fasterrcnn", "--out", str(out)]) == 0
    assert len(list(out.glob("*.pbs"))) == 15
    submit = (out / "submit.sh").read_text(encoding="utf-8")
    assert "depend" not in submit and submit.count("qsub ") == 15


def test_make_jobs_policies_filter(tmp_path):
    out = tmp_path / "jobs"
    jobs_main(["--family", "part_c", "--dataset", "voc", "--detector", "retinanet", "--seeds", "0",
               "--policies", "cua,shared", "--out", str(out)])
    assert sorted(p.stem for p in out.glob("*.pbs")) == ["C-voc-retinanet-s0-cua", "C-voc-retinanet-s0-shared"]
