"""Freeze the locally checked v1 transfer package, refusing existing releases."""
import ast
from collections import Counter
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
package = root / "detection_loader_v1"
archive = root / "detection_loader_v1.tar.gz"
report_path = Path(__file__).with_name("release.json")
if any(path.exists() for path in (archive, package / "SHA256SUMS", package / "LOCAL_VALIDATION.json", report_path)):
    raise SystemExit("Refusing to replace an existing release or its verification records")

names = ["README.md", "rdd_data.py", "reference.json", "selection.json",
         "select_training_cases.py", "verify_training_loader.py", "tests/test_rdd_data.py"]
for name in names:
    path = package / name
    content = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in content or content.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"Unexpected line ending or BOM: {name}")
    if path.suffix == ".py":
        ast.parse(content.decode("utf-8"), feature_version=(3, 12))
    if content != path.read_bytes():
        path.write_bytes(content)

selection = json.loads((package / "selection.json").read_text(encoding="utf-8"))
selected = {entry["image"] for entry in selection["selected"]}
rows = [json.loads(line) for line in (root / "detection_splits_v1/train.jsonl").read_text(encoding="utf-8").splitlines()]
cases = [row for row in rows if row["image"] in selected]
validation = {
    "status": "local_software_checks_passed_ccast_pending",
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "test_evidence": "Completed local unittest output: 22 tests ran, OK, exit code 0; no skipped tests",
    "test_command": "python -m unittest discover -s detection_loader_v1/tests -p test_rdd_data.py -v",
    "tests_run": 22, "tests_passed": 22, "tests_skipped": 0,
    "python": platform.python_version(),
    "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow")},
    "python_3_12_grammar_parse": "passed",
    "synthetic_cli_tests": "Complete/archive/no-overwrite and failed-image/no-success-archive cases passed; PBS and environment references were synthetic fixtures, not real CCAST executions",
    "raw_rdd_images_read_locally": 0,
    "metadata_selection": {
        "manifest_sha256": selection["manifest_sha256"],
        "images": len(cases), "targets": sum(len(row["targets"]) for row in cases),
        "features": len(selection["required_features"]),
        "sources": dict(Counter(row["source"] for row in cases)),
        "categories": dict(Counter(row["annotation_category"] for row in cases)),
        "classes": dict(Counter(target["label"] for row in cases for target in row["targets"])),
    },
    "source_sha256": {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names},
    "pending": ["CCAST package checksums", "Unit tests in the pinned CCAST environment",
                "CPU verification of the 15 selected real training images", "Detector feasibility experiment and model choice"],
}
(package / "LOCAL_VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
names.append("LOCAL_VALIDATION.json")
checksums = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sorted(names)}
(package / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="utf-8", newline="\n")
names.append("SHA256SUMS")
with archive.open("xb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for name in sorted(names):
                content = (package / name).read_bytes()
                member = tarfile.TarInfo(f"{package.name}/{name}")
                member.size, member.mode, member.mtime = len(content), 0o644, 0
                bundle.addfile(member, io.BytesIO(content))

with tarfile.open(archive, "r:gz") as bundle:
    if sorted(bundle.getnames()) != sorted(f"{package.name}/{name}" for name in names):
        raise ValueError("Archive member set differs from release")
    for member in bundle.getmembers():
        content = bundle.extractfile(member).read()
        relative = member.name.removeprefix(package.name + "/")
        if content != (package / relative).read_bytes() or b"\r" in content:
            raise ValueError(f"Archive bytes or line endings differ: {relative}")
    for name, digest in checksums.items():
        if hashlib.sha256(bundle.extractfile(f"{package.name}/{name}").read()).hexdigest() != digest:
            raise ValueError(f"Archive checksum mismatch: {name}")

report = {
    "status": "prepared_locally_not_yet_deployed",
    "archive": str(archive), "bytes": archive.stat().st_size,
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "archive_files": len(names), "checksum_entries": len(checksums),
    "archive_member_bytes": "verified against local sources",
    "line_endings": "LF only, no UTF-8 BOM",
    "unit_tests": {"passed": 22, "skipped": 0},
    "ccast_execution": "pending",
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(report, indent=2))
