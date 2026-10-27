"""Freeze the tested shared-augmentation package without replacing a release."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fasterrcnn_shared_v1"
OUT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_new(path, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with path.open("xb") as handle:
        handle.write(data)


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    archive_path = ROOT / "fasterrcnn_shared_v1.tar.gz"
    for path in (archive_path, PACKAGE/"LOCAL_VALIDATION.json", PACKAGE/"SHA256SUMS", OUT/"release.json"):
        assert not path.exists(), f"Existing release artifact: {path}"
    paths = sorted(p for p in PACKAGE.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert not any(p.is_symlink() for p in paths)
    for path in paths:
        data = path.read_bytes()
        assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf"), path
        if path.suffix == ".py":
            ast.parse(data.decode(), feature_version=(3, 12))
    tests, actual, parity, exposure, shell = [read(n) for n in (
        "unit_tests.json", "local_model_check.json", "comparator_checks.json", "expected_exposure.json", "shell_checks.json")]
    assert tests["tests"] == 25 and not any(tests[k] for k in ("failures", "errors", "skipped"))
    assert actual["status"] == parity["status"] == shell["status"] == "passed"
    assert actual["initialization_matches_identity_model_exactly"]
    assert actual["fresh_model_checkpoint_prediction_replay"] == "exact"
    assert len(actual["operations_exercised"]) == 8
    assert exposure["total"]["applied_images"] == 158151 and exposure["total"]["images"] == 537260
    assert exposure["epochs"][0]["counts"].get("applied_images", 0) == 0
    for name, expected in shell["source_sha256"].items():
        assert digest((PACKAGE/name).read_bytes()) == expected
    for name in parity["unchanged_dependency_files"]:
        assert (PACKAGE/name).read_bytes() == (ROOT/"fasterrcnn_baseline_v1"/name).read_bytes()
    cfg = json.loads((PACKAGE/"protocol.json").read_text())
    assert digest((PACKAGE/"identity_protocol.json").read_bytes()) == cfg["comparator"]["identity_protocol_sha256"]
    assert digest(json.dumps(cfg["augmentation_policy"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) == exposure["policy_sha256"]
    validation = {
        "status": "passed_locally_ccast_execution_pending", "python": platform.python_version(),
        "focused_tests": {**tests, "command": "python -m unittest discover -s fasterrcnn_shared_v1/tests -p 'test_*.py' -v"},
        "actual_pretrained_model_check": actual, "comparator_checks": parity,
        "shell_syntax": shell, "python_3_12_grammar": "passed",
        "predicted_exposure": {"scope": exposure["scope"], "policy_sha256": exposure["policy_sha256"],
                               "total": exposure["total"], "smoke_assignments": exposure["smoke_assignments"]},
        "limits": ["Local library versions differ from CCAST.",
                   "No RDD pixels, CUDA execution, full training or augmentation efficacy measured locally.",
                   "CCAST job repeats focused tests and a 640-pixel real-data GPU smoke before full training.",
                   "The first fixed recipe has not been selected through validation tuning."],
        "source_sha256": {p.relative_to(PACKAGE).as_posix(): digest(p.read_bytes()) for p in paths},
        "local_evidence_sha256": {n: digest((OUT/n).read_bytes()) for n in (
            "unit_tests.json", "unit_tests.txt", "local_model_check.json", "comparator_checks.json", "expected_exposure.json", "shell_checks.json")}}
    save_new(PACKAGE/"LOCAL_VALIDATION.json", validation)
    paths = sorted([*paths, PACKAGE/"LOCAL_VALIDATION.json"])
    sums = {p.relative_to(PACKAGE).as_posix(): digest(p.read_bytes()) for p in paths}
    save_new(PACKAGE/"SHA256SUMS", "".join(f"{h}  {n}\n" for n, h in sums.items()).encode())
    paths = sorted([*paths, PACKAGE/"SHA256SUMS"])
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in paths:
            archive.add(path, arcname=f"{PACKAGE.name}/{path.relative_to(PACKAGE).as_posix()}", recursive=False)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        assert len(members) == len(paths)
        assert len({m.name for m in members}) == len(members)
        for member in members:
            name = PurePosixPath(member.name)
            assert member.isfile() and name.parts[0] == PACKAGE.name and ".." not in name.parts
            assert archive.extractfile(member).read() == ROOT.joinpath(*name.parts).read_bytes()
    record = {
        "status": "prepared_locally_ccast_submission_pending", "archive": archive_path.name,
        "sha256": digest(archive_path.read_bytes()), "bytes": archive_path.stat().st_size,
        "archive_files": len(paths), "checksum_entries": len(sums), "unit_tests_passed": tests["tests"],
        "actual_model_synthetic_cpu_check": "passed", "comparator_checks": "passed",
        "training_condition": "fixed_shared_image_level_augmentation", "recipe_status": cfg["recipe_status"],
        "policy_sha256": exposure["policy_sha256"], "epochs": cfg["epochs"], "seed": cfg["seed"],
        "training_images": cfg["expected_rows"]["train"], "validation_images": cfg["expected_rows"]["val"],
        "test_data_read": False, "total_updates": 134320, "predicted_applied_image_presentations": 158151,
        "pbs": {"queue": "gpus", "cpus": 4, "memory_gb": 32, "gpus": 1, "gpu_type": "a10", "walltime_hours": 24},
        "ccast_run": "pending", "ccast_tests": "pending", "ccast_smoke": "pending"}
    save_new(OUT/"release.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
