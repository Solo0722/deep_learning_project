"""Freeze the checked integration package without replacing earlier releases."""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import tarfile


root = Path(__file__).resolve().parents[2]
package = root / "detection_integration_v1"
archive = root / "detection_integration_v1.tar.gz"
record_path = Path(__file__).with_name("release.json")
for path in (archive, record_path, package / "LOCAL_VALIDATION.json", package / "SHA256SUMS"):
    if path.exists():
        raise FileExistsError(f"Existing release is preserved: {path}")
names = ["README.md", "integration.py", "protocol.json", "rdd_data.py", "reference.json",
         "selection.json", "select_training_cases.py", "tests/test_integration.py"]
for name in names:
    path = package / name
    content = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in content or content.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"Unexpected line endings/BOM: {name}")
    if path.suffix == ".py":
        ast.parse(content.decode("utf-8"), feature_version=(3, 12))
    if path.read_bytes() != content:
        path.write_bytes(content)
protocol = json.loads((package / "protocol.json").read_text(encoding="utf-8"))
for name, digest in protocol["loader_source_sha256"].items():
    assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
    assert (package / name).read_bytes() == (root / "detection_loader_v1" / name).read_bytes()
model_check_path = Path(__file__).with_name("local_model_check.json")
model_check = json.loads(model_check_path.read_text(encoding="utf-8"))
assert model_check["status"] == "passed"
validation = {
    "status": "prepared_locally_ccast_execution_pending",
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow", "ultralytics")},
    "unit_tests": {"command": "python -m unittest discover -s detection_integration_v1/tests -p test_integration.py -v",
                   "passed": 14, "skipped": 0, "reported_seconds": 12.464,
                   "evidence": "Completed local unittest output: Ran 14 tests, OK, process exit 0"},
    "unit_test_scope": "Synthetic adapter/gradient/output tests and mocked CCAST/CUDA CLI completion/failure paths",
    "actual_fasterrcnn_cpu_check": model_check,
    "local_yolo_execution": "Not run: installed Ultralytics 8.4.47 differs from pinned CCAST 8.4.182",
    "python_3_12_grammar": "passed",
    "unchanged_loader_source_files": 4,
    "metadata_scope": {"images": 15, "targets": 16, "batches_per_model": 8,
                       "batch_target_counts": [[2, 0], [2, 0], [0, 1], [1, 0], [0, 4], [0, 0], [0, 5], [1]]},
    "raw_rdd_images_read_locally": 0,
    "pending": ["CCAST transfer and source checks", "Focused tests in the pinned environment",
                "Both detector paths on the 15 real training images", "Returned run-record verification",
                "Bounded detector feasibility study and final architecture choice"]}
(package / "LOCAL_VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8", newline="\n")
names.append("LOCAL_VALIDATION.json")
checksums = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sorted(names)}
(package / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
                                       encoding="utf-8", newline="\n")
names.append("SHA256SUMS")
with archive.open("xb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for name in sorted(names):
                data = (package / name).read_bytes()
                info = tarfile.TarInfo(f"{package.name}/{name}")
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                bundle.addfile(info, io.BytesIO(data))
with tarfile.open(archive, "r:gz") as bundle:
    assert sorted(bundle.getnames()) == sorted(f"{package.name}/{name}" for name in names)
    for name in names:
        data = bundle.extractfile(f"{package.name}/{name}").read()
        assert data == (package / name).read_bytes() and b"\r" not in data
        if name in checksums:
            assert hashlib.sha256(data).hexdigest() == checksums[name]
report = {"status": "prepared_locally_not_yet_transferred", "archive": archive.name,
          "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
          "archive_files": len(names), "checksum_entries": len(checksums),
          "archive_member_bytes": "verified against released sources", "line_endings": "LF, no BOM",
          "local_unit_tests_passed": 14, "local_actual_model_check": "Faster R-CNN on CPU with synthetic inputs passed",
          "ccast_real_image_integration": "pending"}
record_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(report, indent=2))
