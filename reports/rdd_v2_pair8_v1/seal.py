"""Seal a reviewed package and produce platform-separated submission blocks."""
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
NAME = 'rdd_v2_pair8_v1'
PACKAGE = ROOT/NAME


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')


def main():
    archive = ROOT/(NAME+'.tar.gz')
    if archive.exists() or (PACKAGE/'SHA256SUMS').exists():
        raise SystemExit('Already sealed; inspect existing release rather than overwriting it.')
    log = (REPORT/'local_tests.txt').read_text(encoding='utf-8')
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$', log)
    assert match and int(match[1]) == 40
    actual = json.loads((REPORT/'actual_model_check.json').read_text())
    static = json.loads((REPORT/'static_checks.json').read_text())
    assert actual['status'] == static['status'] == 'pass'
    assert all(actual[k] for k in ('production_loop_exact','resume_exact','validation_predictions_exact'))
    cfg = json.loads((PACKAGE/'protocol.json').read_text())
    provenance = json.loads((REPORT/'source_provenance.json').read_text())
    changes = {}
    for name, item in provenance.items():
        source = ROOT/item['source']
        assert sha(source) == item['original_sha256'], name
        if (PACKAGE/name).is_file():
            changes[name] = {**item, 'current_sha256':sha(PACKAGE/name),
                'changed':sha(PACKAGE/name) != item['original_sha256']}
        else:
            changes[name] = {**item, 'removed_from_new_package':True}
    # Do not include validation metadata's changing digest in its own provenance.
    changes.pop('LOCAL_VALIDATION.json',None)
    write(PACKAGE/'SOURCE_PROVENANCE.json',{
        'source_package':'rdd_v2_pilot_v1','files':changes,
        'trace_state_source':'rdd_v2_repro_v2/trace_state.py',
        'trace_state_sha256':sha(PACKAGE/'trace_state.py'),
        'method_changes':'None: geometry, appearance bank, controller thresholds/routing/sampling and original recipe unchanged.',
        'execution_changes':['Two arms, fixed epoch 8, four frozen update panels.',
            'Strict backend settings, installed RoIAlign hash, production-loop equality gate.',
            'Defensive copy of optimizer state on warmup fork; immutable payload checked by gate.',
            'Class-adaptive runs first to expose full probe cost before training the control arm.',
            'Resource records, probe limit, raw gate evidence in main archives.'],
    })
    assert (PACKAGE/'trace_state.py').read_bytes() == (ROOT/'rdd_v2_repro_v2/trace_state.py').read_bytes()
    validation = {
        'created_utc':datetime.now(timezone.utc).isoformat(), 'status':'pass',
        'python':sys.version, 'device':'cpu',
        'versions':{k:version(k) for k in ['torch','torchvision','numpy','Pillow','pycocotools']},
        'unit_tests':int(match[1]), 'unit_test_seconds':float(match[2]),
        'failures':0,'errors':0,'skips':0,'tests_log_sha256':sha(REPORT/'local_tests.txt'),
        'initial_failure_log_sha256':sha(REPORT/'local_tests_initial.txt'),
        'resolved_local_findings':'Initial tests caught aliased CPU optimizer momentum on fork, fixed by deep copying; two new summary fixtures also had an incorrect class-constant reference, fixed before the successful rerun.',
        'actual_cpu_model':actual,
        'actual_cpu_model_log_sha256':sha(REPORT/'actual_model_check.txt'),
        'metadata_preflight':json.loads((REPORT/'metadata_preflight.json').read_text()),
        'static_checks':static,
        'shell_syntax':{n:'Git Bash -n PASS' for n in ['run_job.sh','job.pbs','submit.sh','status.sh']},
        'pending':['Pinned CCAST GPU production-loop gate','Two-arm epoch-8 accuracy comparison'],
        'evidence_boundary':'Local Python/torch/torchvision differ from pinned CCAST versions. Synthetic CPU checks establish implementation behavior only. No new CCAST job has been submitted by this preparation.',
        'post_test_change':'Serial full-run order was set to class-adaptive first; local gate and tested fit/update paths remain unchanged. Python 3.12 syntax and invariant checks passed after that change.'
    }
    write(PACKAGE/'LOCAL_VALIDATION.json',validation)
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert all(p.suffix not in ('.pt','.pyc') for p in files)
    sums = ''.join(f'{sha(p)}  {p.relative_to(PACKAGE).as_posix()}\n' for p in files)
    (PACKAGE/'SHA256SUMS').write_text(sums,encoding='ascii',newline='\n')
    files.append(PACKAGE/'SHA256SUMS')
    with archive.open('xb') as target:
        with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT) as tf:
                for p in sorted(files):
                    raw=p.read_bytes();info=tarfile.TarInfo(f'{NAME}/{p.relative_to(PACKAGE).as_posix()}')
                    info.size=len(raw);info.mode=0o755 if p.suffix in ('.sh','.pbs') else 0o644
                    info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
                    tf.addfile(info,io.BytesIO(raw))
    with tarfile.open(archive,'r:gz') as tf:
        assert len(tf.getmembers())==len(files)
        for member in tf.getmembers():
            assert member.isfile() and not member.issym() and not member.islnk()
            assert tf.extractfile(member).read()==(ROOT/member.name).read_bytes()
    digest=sha(archive)
    result={'package':str(archive),'sha256':digest,'bytes':archive.stat().st_size,
        'payload_files':len(files)-1,'archive_files':len(files),'members_verified':True,
        'endpoint':8,'arms':cfg['arms'],'additional_updates_each':47012,'local_validation':'pass',
        'remote_submission':'not performed'}
    write(REPORT/'release.json',result)
    (ROOT/(NAME+'.tar.gz.sha256')).write_text(f'{digest}  {NAME}.tar.gz\n',encoding='ascii')
    upload=f'''& {{
    $ErrorActionPreference = "Stop"
    $pair8Bundle = "{archive}"
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $pair8Bundle).Hash -ne "{digest}") {{
        throw "Package checksum mismatch."
    }}
    scp $pair8Bundle solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
    if ($LASTEXITCODE -ne 0) {{ throw "Upload failed." }}
}}
'''
    submit=f'''(
    set -euo pipefail
    cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
    printf '%s  %s\\n' '{digest}' '{NAME}.tar.gz' | sha256sum -c -
    if [ -e {NAME} ]; then
        printf 'Package directory already exists; inspect it before continuing.\\n' >&2
        exit 1
    fi
    tar --keep-old-files -xzf {NAME}.tar.gz
    cd {NAME}
    bash submit.sh start
)
'''
    status=f'bash /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/{NAME}/status.sh\n'
    (REPORT/'upload.ps1').write_text(upload,encoding='utf-8',newline='\n')
    (REPORT/'submit_block.sh').write_text(submit,encoding='ascii',newline='\n')
    (REPORT/'CCAST_COMMANDS.md').write_text(
        '# Epoch-8 paired experiment\n\nLocal validation passed; the CCAST GPU gate and accuracy run are pending. No remote submission has been performed.\n\n'
        '1. Run in Windows PowerShell:\n\n```powershell\n'+upload+'```\n\n'
        '2. Run on a CCAST login node after the upload completes. The batch job activates its environment and GPU allocation.\n\n```bash\n'+submit+'```\n\n'
        '3. Check the job and recent log:\n\n```bash\n'+status+'```\n\n'
        'The job runs its exactness gate, then class-adaptive and geometry-only training through epoch 8. An 11-hour deadline checkpoints and stops; inspect that output before explicit resume. No automatic resubmission. SSH may disconnect without cancelling the batch job.\n',
        encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
