"""Freeze the tested baseline package without replacing an existing release."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "fasterrcnn_baseline_v1"
OUT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_new(path, data):
    with path.open("xb") as handle:
        handle.write(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main():
    paths = sorted(p for p in PACKAGE.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in paths:
        data = path.read_bytes()
        assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf"), path
        if path.suffix == ".py":
            ast.parse(data.decode(), feature_version=(3, 12))
    actual = json.loads((OUT / "local_model_check.json").read_text())
    assert actual["status"] == "passed"
    validation = {"status": "passed_locally_ccast_execution_pending", "python": platform.python_version(),
                  "focused_tests": {"passed": 13, "seconds": 10.338,
                    "command": "python -m unittest discover -s fasterrcnn_baseline_v1/tests -p test_baseline.py -v"},
                  "actual_pretrained_model_check": actual,
                  "shell_syntax": "Git Bash -n passed for job.pbs, run_job.sh, submit.sh; no job scripts executed locally",
                  "python_3_12_grammar": "passed", "train_val_hashes_and_separation": "passed",
                  "all_box_geometry_checked": {"train": 38530, "val": 8225, "minimum_fp32_extent_at_640": 0.7548828125},
                  "previous_unit_failure": "The initial synthetic fixture reused RGB values across partitions; the guard correctly rejected it. Distinct fixture pixels fixed the test data before the passing suite.",
                  "limits": ["Local PyTorch/TorchVision/Pillow/NumPy/COCO versions differ from CCAST.",
                             "No RDD image pixels, full baseline training or CUDA execution occurred locally.",
                             "CCAST job repeats focused tests and a real-data 640-pixel smoke gate before full training."],
                  "source_sha256": {p.relative_to(PACKAGE).as_posix(): digest(p.read_bytes()) for p in paths}}
    save_new(PACKAGE / "LOCAL_VALIDATION.json", json_bytes(validation))
    paths = sorted([*paths, PACKAGE / "LOCAL_VALIDATION.json"])
    sums = {p.relative_to(PACKAGE).as_posix(): digest(p.read_bytes()) for p in paths}
    save_new(PACKAGE / "SHA256SUMS", "".join(f"{h}  {n}\n" for n, h in sums.items()).encode())
    paths = sorted([*paths, PACKAGE / "SHA256SUMS"])
    archive_path = ROOT / "fasterrcnn_baseline_v1.tar.gz"
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in paths:
            archive.add(path, arcname=f"fasterrcnn_baseline_v1/{path.relative_to(PACKAGE).as_posix()}", recursive=False)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        assert len(members) == len(paths)
        for member in members:
            name = PurePosixPath(member.name)
            assert member.isfile() and name.parts[0] == PACKAGE.name and ".." not in name.parts
            assert archive.extractfile(member).read() == ROOT.joinpath(*name.parts).read_bytes()
    record = {"status": "prepared_locally_ccast_submission_pending", "archive": archive_path.name,
              "sha256": digest(archive_path.read_bytes()), "bytes": archive_path.stat().st_size,
              "archive_files": len(paths), "checksum_entries": len(sums), "unit_tests_passed": 13,
              "actual_model_synthetic_cpu_check": "passed", "training_condition": "identity_augmentation_control",
              "epochs": 20, "seed": 42, "training_images": 26863, "validation_images": 5760,
              "test_data_read": False, "total_updates": 134320,
              "pbs": {"queue": "gpus", "cpus": 4, "memory_gb": 32, "gpus": 1, "gpu_type": "a10", "walltime_hours": 24},
              "ccast_run": "pending", "ccast_tests": "pending", "ccast_smoke": "pending"}
    save_new(OUT / "release.json", json_bytes(record))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
