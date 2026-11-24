"""Seal the locally verified strict diagnostic and its CCAST handoff."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACKAGE = ROOT/'rdd_v2_strict_v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    tests = {}
    for file, expected in [('source_tests.txt',23),('local_tests.txt',19)]:
        content = (HERE/file).read_text(encoding='utf-8')
        match = re.search(r'Ran (\d+) tests in ([0-9.]+)s',content)
        assert match and int(match[1])==expected and re.search(r'\nOK\s*$',content)
        tests[file] = {'tests':expected,'seconds':float(match[2]),'sha256':sha(HERE/file),'skips':0}
    source = read(HERE/'source_provenance.json')
    for name, expected in source['unchanged_sources'].items():
        assert sha(PACKAGE/'source_repro'/name)==expected==sha(ROOT/'rdd_v2_repro_v2'/name)
    syntax = read(HERE/'shell_validation.json')
    for name, check in syntax['scripts'].items():
        assert check['status']=='pass' and check['lf_only'] and sha(PACKAGE/name)==check['sha256']
    actual = read(HERE/'actual_model_check.json')
    assert actual['status']=='pass' and actual['model_optimizer_rng_gradients_exact']
    assert actual['losses_exact'] and actual['gradient_norm_exact'] and actual['target_counts']==[4,0,0,0]
    protocol = read(PACKAGE/'protocol.json')
    base = read(PACKAGE/'source_repro/protocol.json')
    frozen = ('source_checkpoint','source_manifest_sha256','expected_pairing_sha256','expected_decision_sha256',
              'train_images','batch_size','epoch_index','initial_smoke_step','resume_after_update','conditions','phase_order')
    assert all(protocol[k]==base[k] for k in frozen)
    assert protocol['strict_settings']==actual['strict_flags']
    assert protocol['maximum_optimizer_updates']==sum(len(s['conditions'])*s['updates'] for s in protocol['stages'])==30
    old_inputs = read(ROOT/'reports/rdd_v2_repro_v2/returned_records/rdd_v2_repro_check_v2/materialized_inputs.json')
    assert old_inputs['augmentation']['pairing_sha256']==protocol['expected_pairing_sha256']
    assert old_inputs['augmentation']['decision_sha256']==protocol['expected_decision_sha256']
    preflight = {'status':'pass','same_frozen_input_fields':list(frozen),'unchanged_source_files':len(source['unchanged_sources']),
        'same_archived_input_and_decision_hashes':True,'maximum_updates':30,'gpu_result_pending':True}
    save(HERE/'metadata_preflight.json',preflight)
    validation = {'status':'pass','created_utc':datetime.now(timezone.utc).isoformat(),
        'unit_tests':42,'tests':tests,'actual_cpu_model':actual,'metadata_preflight':preflight,'shell_syntax':syntax,
        'local_versions':{k:version(k) for k in ('torch','torchvision','numpy','Pillow','pycocotools')},
        'environment_boundary':'Local Python 3.14 / torch 2.10 CPU differs from pinned CCAST Python 3.12 / torch 2.3 / torchvision 0.18 CUDA. No local CUDA available.',
        'pending':['Pinned CCAST unit tests','Strict CUDA RoIAlign execution, exactness, runtime and peak memory','Real training-image materialization under strict settings'],
        'scope':'Implementation validation only; no accuracy or full-experiment approval.'}
    save(PACKAGE/'LOCAL_VALIDATION.json',validation)
    payload = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=PACKAGE/'SHA256SUMS')
    (PACKAGE/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(PACKAGE).as_posix()+'\n' for p in payload),encoding='utf-8',newline='\n')
    payload = sorted(payload+[PACKAGE/'SHA256SUMS'])
    expected = {p.relative_to(PACKAGE).as_posix():sha(p) for p in payload}
    archive = ROOT/'rdd_v2_strict_v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in payload:
            tar.add(p,arcname=PACKAGE.name+'/'+p.relative_to(PACKAGE).as_posix(),recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        found = {}
        for member in tar:
            assert member.isfile() and member.name.startswith(PACKAGE.name+'/') and '..' not in Path(member.name).parts
            name = member.name.removeprefix(PACKAGE.name+'/')
            assert name not in found
            found[name] = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
        assert found==expected
    digest = sha(archive)
    save(HERE/'tested_sources.json',expected)
    save(HERE/'release.json',{'archive':str(archive),'sha256':digest,'bytes':archive.stat().st_size,
        'files':len(expected),'archive_payloads_verified':True,'unit_tests':42,
        'actual_cpu_fasterrcnn_gate':'pass','ccast_gpu_tested':False,'ccast_job_submitted':False})
    commands = f'''# Strict-determinism diagnostic: CCAST commands

Run block 1 in Windows PowerShell. Run blocks 2 and 3 in your CCAST login-node SSH session. The job requests one A10 and activates both environments. No interactive GPU request or manual environment activation is needed.

## 1. Upload from Windows PowerShell

```powershell
& {{
    $ErrorActionPreference = "Stop"
    $bundle = "C:\\Users\\SOLOMON\\OneDrive - North Dakota University System\\deep_learning_project\\rdd_v2_strict_v1.tar.gz"
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
    printf '%s  %s\\n' '{digest}' 'rdd_v2_strict_v1.tar.gz' | sha256sum -c -
    if [ -e rdd_v2_strict_v1 ]; then
        printf 'Package directory already exists; inspect it before continuing.\\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf rdd_v2_strict_v1.tar.gz
    cd rdd_v2_strict_v1
    bash submit.sh
)
```

## 3. Check status and recent log on CCAST

```bash
bash /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_strict_v1/status.sh
```

A disconnect does not cancel this batch job. Reconnect and inspect status instead of resubmitting. The package refuses duplicate submission and existing output. The one-hour allocation contains a 40-minute worker cap. There are at most 30 optimizer updates: two single-update gate runs, then the four seven-update conditions only if the gate passes. Strict mode has no automatic fallback. Any failure/timeout may still produce an archive; inspect its status rather than interpreting archive creation as success.

## 4. Download after the final archive line appears

Run this in Windows PowerShell after the job ends. Compare its reported hash to the CCAST archive hash too.

```powershell
& {{
    $ErrorActionPreference = "Stop"
    $folder = "C:\\Users\\SOLOMON\\OneDrive - North Dakota University System\\deep_learning_project"
    $archive = "rdd_v2_strict_check_v1_records.tar.gz"
    $destination = Join-Path $folder $archive
    $checksum = Join-Path $folder "$archive.sha256"
    if (-not (Test-Path -LiteralPath $checksum)) {{
        scp "solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/runs/$archive.sha256" $checksum
        if ($LASTEXITCODE -ne 0) {{ throw "Checksum download failed." }}
    }}
    $line = (Get-Content -LiteralPath $checksum -Raw).Trim()
    if ($line -notmatch '^([a-fA-F0-9]{{64}})  rdd_v2_strict_check_v1_records\\.tar\\.gz$') {{
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

Source package SHA-256: `{digest}`. This differs from the result archive hash generated by the job.

Local validation: 42 unit tests, an actual Faster R-CNN CPU gate and exact comparison with the unchanged baseline update, and Bash syntax checks. Strict CUDA execution remains pending on CCAST. Nothing was submitted by preparation of this local package.
'''
    (HERE/'CCAST_COMMANDS.md').write_text(commands,encoding='utf-8',newline='\n')
    print(json.dumps(read(HERE/'release.json'),indent=2))


if __name__ == '__main__':
    main()
