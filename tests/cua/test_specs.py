import re
import pytest

from cua import run as run_mod
from cua.augment.menu import MENU_ORDER, valid_menu
from cua.experiments.specs import BUDGET_EPOCHS, budget_specs, part_a_specs, proxy_specs, resolve
from cua.train import full_schedule

DATASETS = ("rdd", "voc", "dacl10k")
DETECTORS = ("fasterrcnn", "retinanet")


def test_part_a_rdd_counts_and_links():
    specs = part_a_specs("rdd", "fasterrcnn")
    assert len(specs) == 3 * (1 + 1 + 12)
    arm = next(s for s in specs if s.run_id == "A-rdd-fasterrcnn-s1-blur")
    assert arm.init_run == "A-rdd-fasterrcnn-s1-warm" and arm.first_epoch == 4 and arm.epoch_lr == (1.0, 1.0, 0.1)
    assert arm.edit == "blur" and arm.edit_p == 0.5 and arm.seed == 1


def test_run_ids_unique_across_families():
    ids = [s.run_id for d in DATASETS for s in part_a_specs(d, "fasterrcnn") + budget_specs(d, "fasterrcnn")]
    assert len(ids) == len(set(ids))


def test_warm_and_base_arm():
    specs = {s.run_id: s for s in part_a_specs("voc", "retinanet")}
    warm = specs["A-voc-retinanet-s0-warm"]
    assert warm.epoch_lr == (1.0, 1.0, 1.0) and warm.first_epoch == 1 and warm.edit is None and warm.init_run is None
    base = specs["A-voc-retinanet-s0-base"]
    assert base.edit is None and base.init_run == "A-voc-retinanet-s0-warm" and base.first_epoch == 4


def test_part_a_total_count():
    per_det = sum(len(part_a_specs(d, "fasterrcnn")) for d in DATASETS)
    assert per_det == 42 + 42 + 45 == 129
    assert per_det == sum(3 * (2 + len(valid_menu(d))) for d in DATASETS)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("detector", DETECTORS)
def test_arms_link_within_family_and_share_seeds(dataset, detector):
    specs = part_a_specs(dataset, detector)
    ids = {s.run_id for s in specs}
    arms = [s for s in specs if s.init_run is not None]
    assert all(a.init_run in ids for a in arms)
    by_seed = {}
    for s in specs:
        by_seed.setdefault(s.seed, set()).add(s.run_id)
    assert sorted(by_seed) == [0, 1, 2]
    assert len({len(v) for v in by_seed.values()}) == 1
    # the same edit set at every seed
    edits = {sd: {s.edit for s in specs if s.seed == sd} for sd in by_seed}
    assert edits[0] == edits[1] == edits[2] == {None, *valid_menu(dataset)}


def test_budget_schedules():
    assert BUDGET_EPOCHS == {"rdd": 20, "voc": 20, "dacl10k": 30}
    (r,) = budget_specs("rdd", "fasterrcnn")
    assert r.epoch_lr == full_schedule(20) and r.run_id == "budget-rdd-fasterrcnn-e20" and r.seed == 0
    assert r.edit is None and r.init_run is None and r.first_epoch == 1
    (d,) = budget_specs("dacl10k", "retinanet")
    assert d.epoch_lr == full_schedule(30) and d.run_id == "budget-dacl10k-retinanet-e30"
    (o,) = budget_specs("rdd", "fasterrcnn", epochs=30)
    assert o.epoch_lr == full_schedule(30) and o.run_id == "budget-rdd-fasterrcnn-e30"


def test_proxy_specs():
    specs = proxy_specs(["blur", "jpeg"], 20)
    assert specs[0].run_id == "proxy-rdd-fasterrcnn-e20-base" and specs[0].edit is None
    assert [s.run_id for s in specs[1:]] == ["proxy-rdd-fasterrcnn-e20-blur", "proxy-rdd-fasterrcnn-e20-jpeg"]
    assert all(s.dataset == "rdd" and s.detector == "fasterrcnn" and s.seed == 0 for s in specs)
    assert all(s.epoch_lr == full_schedule(20) for s in specs)
    with pytest.raises(ValueError):
        proxy_specs(["not_an_edit"], 20)


def test_unknown_names_rejected():
    with pytest.raises(ValueError):
        budget_specs("nope", "fasterrcnn")
    with pytest.raises(ValueError):
        budget_specs("rdd", "yolo")
    with pytest.raises(ValueError):
        part_a_specs("rdd", "yolo")
    with pytest.raises(ValueError):
        part_a_specs("nope", "fasterrcnn")


def test_names_contain_no_hyphens():
    for name in (*DATASETS, *DETECTORS, *MENU_ORDER):
        assert "-" not in name, name


def test_resolve_round_trips_every_family():
    for d in DATASETS:
        for det in DETECTORS:
            allspecs = list(part_a_specs(d, det)) + budget_specs(d, det) + budget_specs(d, det, 20) \
                + budget_specs(d, det, 30)
            for s in allspecs:
                assert resolve(s.run_id) == s
    menu = valid_menu("rdd")
    for n in (20, 30):
        for s in proxy_specs(list(menu), n):
            assert resolve(s.run_id) == s


def test_resolve_unknown():
    for bad in ("", "garbage", "A-rdd-fasterrcnn-s0-notanedit", "budget-rdd-yolo-e20", "proxy-rdd-fasterrcnn-e20-nope",
                "A-nope-fasterrcnn-s0-warm"):
        with pytest.raises(KeyError):
            resolve(bad)


def test_cli_passes_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run_mod.time, "monotonic", lambda: 1000.0)

    def fake_train(cfg, data_root, runs_root, resume=False, deadline=None, max_train_images=None):
        calls.append((cfg, data_root, runs_root, resume, deadline, max_train_images))
        return 3

    monkeypatch.setattr(run_mod, "train", fake_train)
    with pytest.raises(SystemExit) as e:
        run_mod.main(["--spec", "A-rdd-fasterrcnn-s1-blur", "--data-root", str(tmp_path / "d"),
                      "--runs-root", str(tmp_path / "r"), "--resume", "--deadline-seconds", "50",
                      "--max-train-images", "16"])
    assert e.value.code == 3
    cfg, dr, rr, resume, deadline, mti = calls[0]
    assert cfg.run_id == "A-rdd-fasterrcnn-s1-blur"
    assert str(dr) == str(tmp_path / "d") and str(rr) == str(tmp_path / "r")
    assert resume is True and deadline == 1050.0 and mti == 16


def test_cli_defaults_and_zero_exit(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run_mod, "train", lambda *a, **k: calls.append((a, k)) or 0)
    with pytest.raises(SystemExit) as e:
        run_mod.main(["--spec", "budget-rdd-fasterrcnn-e20", "--data-root", "d", "--runs-root", "r"])
    assert e.value.code == 0
    a, k = calls[0]
    assert k["resume"] is False and k["deadline"] is None and k["max_train_images"] is None


def test_cli_unknown_spec(monkeypatch, capsys):
    monkeypatch.setattr(run_mod, "train", lambda *a, **k: pytest.fail("train must not run"))
    with pytest.raises(SystemExit) as e:
        run_mod.main(["--spec", "bogus", "--data-root", "d", "--runs-root", "r"])
    assert e.value.code == 2
    assert "bogus" in capsys.readouterr().err


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("detector", DETECTORS)
def test_every_arm_continues_its_warm_run(dataset, detector):
    specs = part_a_specs(dataset, detector)
    by_id = {s.run_id: s for s in specs}
    arms = [s for s in specs if s.init_run is not None]
    assert len(arms) == 3 * (1 + len(valid_menu(dataset)))
    for arm in arms:
        warm = by_id[arm.init_run]
        assert (warm.dataset, warm.detector, warm.size, warm.pretrained, warm.seed) ==                (arm.dataset, arm.detector, arm.size, arm.pretrained, arm.seed)
        assert arm.first_epoch == warm.first_epoch + len(warm.epoch_lr)


def test_menu_has_no_reserved_arm_names():
    assert not {"warm", "base"} & set(MENU_ORDER)


def test_epochs_validation_and_duplicate_edits():
    for bad in (0, -3):
        with pytest.raises(ValueError):
            budget_specs("rdd", "fasterrcnn", epochs=bad)
        with pytest.raises(ValueError):
            proxy_specs(["blur"], bad)
    with pytest.raises(ValueError):
        proxy_specs(["blur", "blur"], 20)
