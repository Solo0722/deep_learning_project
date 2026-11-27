"""Seal tested diagnostic sources and generate the Windows/CCAST handoff."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
PACKAGE = ROOT/'rdd_v2_repro_v2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def files():
    return sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS')


def main():
    tests=(REPORT/'local_tests.txt').read_text(encoding='utf-8')
    match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',tests)
    assert match and int(match[1])==23 and re.search(r'\nOK\s*$',tests)
    tested=read(REPORT/'tested_sources.json')
    for name,digest in tested.items():
        assert sha(PACKAGE/name)==digest,'Source changed after validation: '+name
    actual=read(REPORT/'actual_model_check.json')
    assert actual['status']=='pass' and actual['model_optimizer_rng_gradients_exact']
    preflight=read(REPORT/'metadata_preflight.json');assert preflight['status']=='pass'
    shell=read(REPORT/'shell_validation.json')
    for name,record in shell['scripts'].items():
        assert record['status']=='pass' and record['lf_only'] and sha(PACKAGE/name)==record['sha256']
    validation={'status':'pass','created_utc':datetime.now(timezone.utc).isoformat(),
        'unit_tests':23,'unit_test_seconds':float(match[2]),'skips':0,
        'local_versions':{k:version(k) for k in ('torch','torchvision','numpy','Pillow','pycocotools')},
        'environment_boundary':'Windows CPU validation differs from pinned CCAST Python 3.12 / torch 2.3.0 / torchvision 0.18.0. No local CUDA result.',
        'actual_cpu_model':actual,'metadata_preflight':preflight,'shell_syntax':shell,
        'test_log_sha256':sha(REPORT/'local_tests.txt'),
        'failure_provenance':read(REPORT/'failure_provenance.json'),
        'old_hash_reproduction':read(REPORT/'hash_reproduction.json'),
        'repair_scope':'Diagnostic hash-layout compatibility; production collation regression coverage. Experimental data, model, optimizer and comparison conditions unchanged.',
        'pending':['Pinned CCAST runtime and raw-pixel replay','Actual seven-update four-condition CUDA diagnostic'],
        'scope':'Implementation validation only; no accuracy, causal-root-identification or superiority claim.'}
    save(PACKAGE/'LOCAL_VALIDATION.json',validation)
    # Include the original nested manifest as a payload, not merely its members.
    payload=files()+[PACKAGE/'source_pilot/SHA256SUMS']
    (PACKAGE/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(PACKAGE).as_posix()+'\n' for p in sorted(payload)),encoding='utf-8',newline='\n')
    payload=sorted(payload+[PACKAGE/'SHA256SUMS'])
    expected={p.relative_to(PACKAGE).as_posix():sha(p) for p in payload}
    archive=ROOT/'rdd_v2_repro_v2.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in payload:
            tar.add(p,arcname=PACKAGE.name+'/'+p.relative_to(PACKAGE).as_posix(),recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        found={}
        for member in tar:
            assert member.isfile() and not member.issym() and '..' not in Path(member.name).parts
            name=member.name.removeprefix(PACKAGE.name+'/')
            assert name not in found
            found[name]=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
        assert found==expected
    digest=sha(archive)
    save(REPORT/'release.json',{'archive':str(archive),'sha256':digest,'bytes':archive.stat().st_size,
        'files':len(expected),'archive_payloads_verified':True,'manifest_sha256':sha(PACKAGE/'SHA256SUMS'),
        'unit_tests':23,'actual_cpu_fasterrcnn_update_equivalence':'pass','ccast_job_submitted':False})
    commands=f'''# CCAST reproducibility diagnostic commands

Run block 1 in Windows PowerShell. Run block 2 in your CCAST login-node SSH session. The batch job requests one A10 and activates both environments automatically. Do not request an interactive GPU or activate an environment manually for submission.

## 1. Upload from Windows PowerShell

```powershell
& {{
    $ErrorActionPreference = "Stop"
    $bundle = "C:\\Users\\SOLOMON\\OneDrive - North Dakota University System\\deep_learning_project\\rdd_v2_repro_v2.tar.gz"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash -ne "{digest}") {{
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
    printf '%s  %s\\n' '{digest}' 'rdd_v2_repro_v2.tar.gz' | sha256sum -c -
    if [ -e rdd_v2_repro_v2 ]; then
        printf 'Package directory already exists; inspect it before continuing.\\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf rdd_v2_repro_v2.tar.gz
    cd rdd_v2_repro_v2
    bash submit.sh
)
```

## 3. Status and recent log on CCAST

```bash
bash /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_repro_v2/status.sh
```

The job has a one-hour PBS ceiling and a 40-minute diagnostic budget, checked between updates and probe forwards. It runs 28 optimizer updates in total. It neither evaluates accuracy nor extends the previous pilot. Metadata and bounded first-difference tensor evidence are archived for download; larger reference tensors remain on CCAST. A disconnect does not terminate the batch job. Reconnect and inspect status rather than submitting again.

## 4. Download after the job reports its records archive

Run this in Windows PowerShell only after the job finishes. A failed diagnostic may also produce an archive; its status remains a failure and must be inspected.

```powershell
& {{
    $ErrorActionPreference = "Stop"
    $folder = "C:\\Users\\SOLOMON\\OneDrive - North Dakota University System\\deep_learning_project"
    $archive = "rdd_v2_repro_check_v2_records.tar.gz"
    $destination = Join-Path $folder $archive
    $checksum = Join-Path $folder "$archive.sha256"
    if (-not (Test-Path -LiteralPath $checksum)) {{
        scp "solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/runs/$archive.sha256" $checksum
        if ($LASTEXITCODE -ne 0) {{ throw "Checksum download failed." }}
    }}
    $line = (Get-Content -LiteralPath $checksum -Raw).Trim()
    if ($line -notmatch '^([a-fA-F0-9]{{64}})  rdd_v2_repro_check_v2_records\\.tar\\.gz$') {{
        throw "Unexpected checksum file."
    }}
    $expected = $Matches[1]
    if (-not (Test-Path -LiteralPath $destination)) {{
        scp "solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/runs/$archive" $destination
        if ($LASTEXITCODE -ne 0) {{ throw "Records download failed." }}
    }}
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash
    if ($actual -ne $expected) {{ throw "Records checksum mismatch." }}
    Write-Output "Download verified: $actual"
}}
```

Source archive SHA-256: `{digest}`. This is different from the result archive's SHA-256, which is generated by the CCAST run. No job was submitted while preparing this local package.
'''
    (REPORT/'CCAST_COMMANDS.md').write_text(commands,encoding='utf-8',newline='\n')
    print(json.dumps({'sha256':digest,'bytes':archive.stat().st_size,'files':len(expected),'unit_tests':23},indent=2))


if __name__=='__main__':
    main()
