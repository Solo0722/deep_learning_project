"""Seal the reviewed source and generate copy/paste commands for this release."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT/'rdd_v2_pilot_v1'
REPORT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def sources():
    return sorted(p for p in PKG.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS')


def seal():
    (PKG/'SHA256SUMS').write_text(''.join(digest(p)+'  '+p.relative_to(PKG).as_posix()+'\n' for p in sources()), encoding='utf-8', newline='\n')


def main():
    log = (REPORT/'local_tests.txt').read_text(encoding='utf-16' if (REPORT/'local_tests.txt').read_bytes().startswith(b'\xff\xfe') else 'utf-8')
    match = re.search(r'Ran (\d+) tests in ([0-9.]+)s', log)
    assert match and re.search(r'\nOK\s*$', log), 'Local tests did not pass'
    checked = json.loads((REPORT/'tested_sources.json').read_text())
    for name, expected in checked.items(): assert digest(PKG/name) == expected, 'Code changed after final tests: '+name
    record = {'status': 'pass', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'unit_tests': int(match[1]), 'unit_test_seconds': float(match[2]), 'skips': 0,
              'versions': {k: version(k) for k in ('torch', 'torchvision', 'numpy', 'Pillow', 'pycocotools')},
              'python_scope': 'Windows Python 3.14 CPU; differs from pinned CCAST Python 3.12 / Torch 2.3 / TorchVision 0.18',
              'tests_log_sha256': digest(REPORT/'local_tests.txt'),
              'actual_cpu_model': json.loads((REPORT/'actual_model_check.json').read_text()),
              'metadata_preflight': json.loads((REPORT/'metadata_preflight.json').read_text()),
              'shell_syntax': {'job.pbs': 'bash -n PASS', 'run_job.sh': 'bash -n PASS', 'submit.sh': 'bash -n PASS', 'status.sh': 'bash -n PASS'},
              'not_yet_executed': ['New CCAST GPU smoke gate', 'Four-arm GPU training pilot'],
              'scientific_scope': 'Implementation and annotation safety checks only. No new accuracy result or superiority claim.'}
    save(PKG/'LOCAL_VALIDATION.json', record)
    seal()
    archive = ROOT/'rdd_v2_pilot_v1.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for p in sorted(sources()+[PKG/'SHA256SUMS']):
            tar.add(p, arcname='rdd_v2_pilot_v1/'+p.relative_to(PKG).as_posix(), recursive=False)
    expected = {p.relative_to(PKG).as_posix(): digest(p) for p in sources()+[PKG/'SHA256SUMS']}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(expected)
        for member in members:
            assert member.isfile() and not member.issym() and '..' not in Path(member.name).parts
            name = member.name.removeprefix('rdd_v2_pilot_v1/')
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[name]
    sha = digest(archive)
    save(REPORT/'release.json', {'archive': str(archive), 'bytes': archive.stat().st_size,
        'sha256': sha, 'files': len(expected), 'archive_payloads_verified': True,
        'manifest_sha256': digest(PKG/'SHA256SUMS'), 'unit_tests': int(match[1])})
    commands = f'''# CCAST four-arm pilot commands

Run block 1 in Windows PowerShell. Run block 2 in your existing CCAST login-node SSH session. No interactive GPU request or manual environment activation is needed. The PBS job runs the tests, GPU gate and bounded pilot sequentially.

## 1. Upload from Windows PowerShell

```powershell
& {{
    $ErrorActionPreference = "Stop"
    $bundle = "C:\\Users\\SOLOMON\\OneDrive - North Dakota University System\\deep_learning_project\\rdd_v2_pilot_v1.tar.gz"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash -ne "{sha}") {{
        throw "Package checksum mismatch."
    }}
    scp $bundle solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
    if ($LASTEXITCODE -ne 0) {{ throw "Upload failed." }}
}}
```

## 2. Verify, extract and submit on the CCAST login node

```bash
(
    set -euo pipefail
    cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    printf '%s  %s\\n' '{sha}' 'rdd_v2_pilot_v1.tar.gz' | sha256sum -c -
    if [ -e rdd_v2_pilot_v1 ]; then
        printf 'Package directory already exists; stop and inspect it.\\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf rdd_v2_pilot_v1.tar.gz
    cd rdd_v2_pilot_v1
    bash submit.sh start
)
```

## 3. Status and recent log

```bash
bash /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_pilot_v1/status.sh
```

The single batch allocation requests one A10, four CPUs and 32 GB RAM, with a 12-hour PBS ceiling and an 11-hour administrative cap. Each arm runs epochs 2–4 from the same epoch-1 state. The controller updates before epochs 2 and 4. A disconnect does not cancel the batch job. Do not submit a duplicate; reconnect and use block 3.

If the run stops at its time cap, inspect the log and status before explicitly using `bash submit.sh resume` from the package directory. This resumes the same bounded pilot, not a longer experiment. Failed tests or the smoke gate must be inspected, not bypassed.

Download the exact archive printed by the completed job and verify its printed SHA-256. The source archive and result archive have different filenames. Full checkpoint tensors remain on CCAST. No new pilot has been run locally or remotely by preparing this package.
'''
    (REPORT/'CCAST_COMMANDS.md').write_text(commands, encoding='utf-8', newline='\n')
    print(json.dumps({'sha256': sha, 'bytes': archive.stat().st_size, 'files': len(expected), 'tests': int(match[1])}))


if __name__ == '__main__': main()
