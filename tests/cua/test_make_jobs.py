import re
import shutil
import subprocess

import pytest

from cua.ccast.make_jobs import main as cli_main, make_jobs
from cua.experiments.specs import budget_specs, part_a_specs, proxy_specs

ROOT = "/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research"


def test_arm_jobs_depend_on_warm(tmp_path):
    submit = make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)), tmp_path).read_text()
    warm_line = submit.index("A-rdd-fasterrcnn-s0-warm.pbs"); arm_line = submit.index("A-rdd-fasterrcnn-s0-blur.pbs")
    assert warm_line < arm_line and "depend=afterok:$WARM_s0" in submit


def test_walltime_and_deadline(tmp_path):
    make_jobs(budget_specs("rdd", "fasterrcnn"), tmp_path)
    job = (tmp_path / "budget-rdd-fasterrcnn-e20.pbs").read_text()
    assert "walltime=24:00:00" in job and "--deadline-seconds 85200" in job


def test_a_family_walltime_and_resources(tmp_path):
    make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)), tmp_path)
    job = (tmp_path / "A-rdd-fasterrcnn-s0-blur.pbs").read_text()
    assert "walltime=06:00:00" in job and "--deadline-seconds 20400" in job
    assert "#PBS -q gpus" in job
    assert "#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:glist=a10" in job
    assert "#PBS -W group_list=x-ccast-prj-aaboah" in job and "#PBS -j oe" in job
    assert "conda activate pytorch" in job and ".venvs/rdd_detection_v1/bin/activate" in job
    assert f"cd {ROOT}/cua_code" in job and f"--data-root {ROOT}/cua_data" in job
    assert f"--runs-root {ROOT}/cua_runs " in job
    assert f"#PBS -o {ROOT}/cua_runs/_logs/A-rdd-fasterrcnn-s0-blur.log" in job
    assert "--spec A-rdd-fasterrcnn-s0-blur" in job


def test_proxy_walltime(tmp_path):
    make_jobs(proxy_specs(["blur"], 20), tmp_path)
    job = (tmp_path / "proxy-rdd-fasterrcnn-e20-blur.pbs").read_text()
    assert "walltime=24:00:00" in job and "--deadline-seconds 85200" in job


def test_smoke_applies_to_warm_and_arms(tmp_path):
    make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)), tmp_path, smoke=True)
    for rid in ("A-rdd-fasterrcnn-s0-warm", "A-rdd-fasterrcnn-s0-blur", "A-rdd-fasterrcnn-s0-base"):
        job = (tmp_path / f"{rid}.pbs").read_text()
        assert "--max-train-images 64" in job and "walltime=01:00:00" in job
        assert "--deadline-seconds 2400" in job
        assert f"--runs-root {ROOT}/cua_runs_smoke " in job and f"{ROOT}/cua_runs/" not in job
    normal = tmp_path / "n"
    make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)), normal)
    assert "--max-train-images" not in (normal / "A-rdd-fasterrcnn-s0-warm.pbs").read_text()
    assert f"{ROOT}/cua_runs_smoke/_logs" in (tmp_path / "submit.sh").read_text()


def test_pytest_only_in_non_arm_jobs(tmp_path):
    make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)), tmp_path)
    make_jobs(budget_specs("rdd", "fasterrcnn"), tmp_path)
    make_jobs(proxy_specs(["blur"], 20), tmp_path)
    cmd = 'pytest -q -x -m "not slow" tests/cua'
    assert cmd in (tmp_path / "A-rdd-fasterrcnn-s0-warm.pbs").read_text()
    assert cmd in (tmp_path / "budget-rdd-fasterrcnn-e20.pbs").read_text()
    assert cmd in (tmp_path / "proxy-rdd-fasterrcnn-e20-base.pbs").read_text()
    arms = [p for p in tmp_path.glob("A-*.pbs") if not p.name.endswith("-warm.pbs")]
    assert len(arms) == 13
    for p in arms:
        assert "pytest -q" not in p.read_text(), p.name


def test_pytest_failure_precedes_training(tmp_path):
    make_jobs(budget_specs("rdd", "fasterrcnn"), tmp_path)
    job = (tmp_path / "budget-rdd-fasterrcnn-e20.pbs").read_text()
    assert job.index("pytest") < job.index("python -m cua.run")
    assert "|| exit $?" in job


def test_resume_and_exit3_handling(tmp_path):
    make_jobs(budget_specs("rdd", "fasterrcnn"), tmp_path)
    job = (tmp_path / "budget-rdd-fasterrcnn-e20.pbs").read_text()
    assert "CUA_RESUME" in job and "--resume" in job
    assert "qsub -v CUA_RESUME=1" in job and "budget-rdd-fasterrcnn-e20.pbs" in job
    assert "exit 3" in job


def test_job_names_short_and_unique(tmp_path):
    specs = part_a_specs("voc", "retinanet")
    make_jobs(specs, tmp_path)
    names = []
    for s in specs:
        job = (tmp_path / f"{s.run_id}.pbs").read_text()
        name = re.search(r"^#PBS -N (\S+)$", job, re.M)[1]
        assert len(name) <= 15
        assert f"# run_id: {s.run_id}" in job
        names.append(name)
    assert len(set(names)) == len(names)


def test_one_pbs_per_spec_and_submitted_once(tmp_path):
    specs = part_a_specs("dacl10k", "fasterrcnn")
    submit_path = make_jobs(specs, tmp_path)
    assert submit_path.name == "submit.sh"
    assert sorted(p.name for p in tmp_path.glob("*.pbs")) == sorted(f"{s.run_id}.pbs" for s in specs)
    submit = submit_path.read_text()
    for s in specs:
        assert submit.count(f"{s.run_id}.pbs") == 1, s.run_id
    assert "set -euo pipefail" in submit and "mkdir -p" in submit and "_logs" in submit
    for seed in (0, 1, 2):
        assert submit.count(f"WARM_s{seed}=$(qsub ") == 1
    assert submit.count("depend=afterok:") == len(specs) - 3
    last_warm = max(submit.index(f"{s.run_id}.pbs") for s in specs if s.init_run is None)
    first_arm = min(submit.index(f"{s.run_id}.pbs") for s in specs if s.init_run is not None)
    assert last_warm < first_arm


def test_budget_family_submit_has_no_dependencies(tmp_path):
    submit = make_jobs(budget_specs("rdd", "fasterrcnn"), tmp_path).read_text()
    assert "depend" not in submit and "qsub " in submit


def test_mixed_pairs_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)) + part_a_specs("voc", "fasterrcnn", seeds=(0,)),
                  tmp_path)
    with pytest.raises(ValueError):
        make_jobs([], tmp_path)


def test_lf_only_and_project_override(tmp_path):
    make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)), tmp_path, project="/x/proj")
    for p in tmp_path.iterdir():
        assert b"\r" not in p.read_bytes(), p.name
    assert "/x/proj/cua_code" in (tmp_path / "A-rdd-fasterrcnn-s0-warm.pbs").read_text()
    assert "/x/proj/cua_runs/_logs" in (tmp_path / "submit.sh").read_text()


def _working_bash():
    exe = shutil.which("bash")
    if exe is None:
        return None
    try:
        r = subprocess.run([exe, "-c", "echo ok"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if r.returncode == 0 and r.stdout.strip() == "ok" else None


@pytest.mark.skipif(_working_bash() is None, reason="a working bash is not available")
def test_generated_scripts_pass_bash_syntax_check(tmp_path):
    make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)), tmp_path)
    make_jobs(budget_specs("rdd", "fasterrcnn"), tmp_path)
    for p in sorted(tmp_path.iterdir()):
        r = subprocess.run([_working_bash(), "-n", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, (p.name, r.stderr)


def test_cli_families(tmp_path):
    out = tmp_path / "a"
    assert cli_main(["--family", "part_a", "--dataset", "rdd", "--detector", "fasterrcnn", "--out", str(out),
                     "--seeds", "0", "--smoke"]) == 0
    assert (out / "submit.sh").exists() and len(list(out.glob("*.pbs"))) == 14
    out = tmp_path / "b"
    cli_main(["--family", "budget", "--dataset", "dacl10k", "--detector", "retinanet", "--out", str(out)])
    assert (out / "budget-dacl10k-retinanet-e30.pbs").exists()
    out = tmp_path / "b2"
    cli_main(["--family", "budget", "--dataset", "rdd", "--detector", "fasterrcnn", "--out", str(out),
              "--epochs", "5"])
    assert (out / "budget-rdd-fasterrcnn-e5.pbs").exists()
    out = tmp_path / "p"
    cli_main(["--family", "proxy", "--dataset", "rdd", "--detector", "fasterrcnn", "--out", str(out),
              "--epochs", "10", "--edits", "blur,jpeg", "--project", "/x/p"])
    assert sorted(p.name for p in out.glob("*.pbs")) == [
        "proxy-rdd-fasterrcnn-e10-base.pbs", "proxy-rdd-fasterrcnn-e10-blur.pbs",
        "proxy-rdd-fasterrcnn-e10-jpeg.pbs"]
    with pytest.raises(SystemExit):
        cli_main(["--family", "proxy", "--dataset", "voc", "--detector", "fasterrcnn", "--out", str(out),
                  "--epochs", "10", "--edits", "blur"])
