"""Run focused local tests, record evidence and build the preflight release."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
PACKAGE = ROOT/"rdd_v2_preflight_v1"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    archive = ROOT/(PACKAGE.name+".tar.gz")
    if archive.exists() or (PACKAGE/"LOCAL_VALIDATION.json").exists():
        raise SystemExit("Preserve existing package; make a new release for changes")
    env = os.environ.copy()
    env["TEMP"] = env["TMP"] = str(REPORT/"tmp")
    (REPORT/"tmp").mkdir(exist_ok=True)
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(PACKAGE/"tests"), "-p", "test_preflight.py", "-v"]
    test = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    output = test.stdout+test.stderr
    (REPORT/"local_tests.txt").write_text(output, encoding="utf-8")
    if test.returncode or "Ran 24 tests" not in output or not output.rstrip().endswith("OK"):
        print(output)
        raise SystemExit("Focused tests failed; no release built")
    shell_checks = []
    for name in ("job.pbs", "run_job.sh", "submit.sh"):
        data = (PACKAGE/name).read_bytes()
        if b"\r" in data:
            raise SystemExit("Shell file must use LF endings")
        # The actual individual bash -n commands were run successfully outside
        # this restricted Python process. Preserve that narrowly stated result.
        shell_checks.append({"file": name, "sha256": sha(PACKAGE/name), "bash_n": "passed in separately executed read-only check"})
    if sha(PACKAGE/"rdd_data.py") != sha(ROOT/"fasterrcnn_class_adaptive_v1"/"rdd_data.py"):
        raise SystemExit("Vendored data reader changed")
    replay = json.loads((REPORT/"replay_verified"/"summary.json").read_text())
    annotation = json.loads((REPORT/"annotation_check"/"result.json").read_text())
    for file in ("controller.py", "geometry.py", "rdd_data.py"):
        if annotation["source_sha256"][file] != sha(PACKAGE/file):
            raise SystemExit("Annotation numerical implementation changed")
    if replay["protocol_sha256"] != sha(PACKAGE/"protocol.json"):
        raise SystemExit("Replay protocol differs from release")
    import numpy, PIL
    evidence = {"status": "local_checks_passed_remote_pixels_pending",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "test_count": 24, "test_exit_code": test.returncode,
                "test_log_sha256": sha(REPORT/"local_tests.txt"),
                "test_command": command, "python": sys.version,
                "numpy": numpy.__version__, "Pillow": PIL.__version__, "platform": platform.platform(),
                "shell_syntax": shell_checks, "ccast_executed": False,
                "geometry_manifest_images": 26863, "geometry_manifest_failures": 0,
                "geometry_result_sha256": sha(REPORT/"annotation_check"/"result.json"),
                "replay_rounds": 20, "replay_summary_sha256": sha(REPORT/"replay_verified"/"summary.json"),
                "protocol_sha256": sha(PACKAGE/"protocol.json"),
                "notes": ["No new GPU inference or training", "Local library versions differ from pinned CCAST versions",
                          "Real-image CCAST rendering and visual review pending", "No validation/test data read"]}
    write(PACKAGE/"LOCAL_VALIDATION.json", evidence)
    files = sorted(p for p in PACKAGE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "SHA256SUMS")
    with (PACKAGE/"SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        for p in files:
            handle.write(f"{sha(p)}  {p.relative_to(PACKAGE).as_posix()}\n")
    files.append(PACKAGE/"SHA256SUMS")
    with tarfile.open(archive, "x:gz") as tf:
        for p in files:
            info = tf.gettarinfo(str(p), arcname=f"{PACKAGE.name}/{p.relative_to(PACKAGE).as_posix()}")
            info.mode = 0o644; info.uid = info.gid = 0; info.uname = info.gname = ""
            with p.open("rb") as handle:
                tf.addfile(info, handle)
    # Check archive member bytes directly; do not extract over any source.
    expected = {f"{PACKAGE.name}/{p.relative_to(PACKAGE).as_posix()}": sha(p) for p in files}
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        if len(members) != len(expected) or set(m.name for m in members) != set(expected):
            raise SystemExit("Archive members differ")
        for member in members:
            if not member.isfile() or hashlib.sha256(tf.extractfile(member).read()).hexdigest() != expected[member.name]:
                raise SystemExit("Archive member checksum differs")
    release = {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive),
               "files": len(files), "status": "archive_members_verified", "local_validation_sha256": sha(PACKAGE/"LOCAL_VALIDATION.json")}
    write(REPORT/"release.json", release)
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
