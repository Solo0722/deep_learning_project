"""Freeze and verify the new evaluation-only diagnostic package; preserve releases."""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[2]
package=root/'detection_diagnostics_v1'
archive=root/'detection_diagnostics_v1.tar.gz'
report_path=Path(__file__).with_name('release.json')
for path in (archive,report_path,package/'LOCAL_VALIDATION.json',package/'SHA256SUMS'):
    if path.exists():
        raise FileExistsError(f'Preserve existing release: {path}')
names=['diagnose.py','protocol.json','README.md','tests/test_diagnose.py']
for name in names:
    path=package/name
    data=path.read_bytes().replace(b'\r\n',b'\n')
    assert b'\r' not in data and not data.startswith(b'\xef\xbb\xbf')
    if path.suffix=='.py':
        ast.parse(data.decode(),feature_version=(3,12))
    if path.read_bytes()!=data:
        path.write_bytes(data)
check=json.loads(Path(__file__).with_name('local_model_check.json').read_text())
assert check['status']=='passed' and check['optimizer_steps']==0 and check['baseline_matches_released_inference']
assert all(v['state_and_rng_unchanged'] for v in check['modes'].values())
protocol=json.loads((package/'protocol.json').read_text())
learning=root/'detection_learning_v1'
assert hashlib.sha256((learning/'SHA256SUMS').read_bytes()).hexdigest()==protocol['learning_package_checksums_sha256']
for name,expected in protocol['previous_result_files_sha256'].items():
    assert hashlib.sha256((root/'reports/detection_learning_v1/returned_records'/name).read_bytes()).hexdigest()==expected
validation={
    'status':'prepared_locally_ccast_diagnostic_pending','recorded_utc':datetime.now(timezone.utc).isoformat(),
    'unit_tests':{'command':'python -m unittest discover -s detection_diagnostics_v1/tests -p test_diagnose.py -v',
                  'passed':8,'skipped':0,'reported_seconds':6.430},
    'unit_test_scope':'Batch-stat versus running-stat functional behavior, unchanged state/RNG, mode restoration on exceptions, mutation rejection, raw candidate geometry versus class scores, empty-image assignments, input tampering, replay mismatch stops before intervention',
    'actual_pretrained_cpu_check':check,'python_3_12_grammar':'passed',
    'known_local_limit':'Local CPU torch 2.10.0 / torchvision 0.25.0 / pycocotools 2.0.11 differ from pinned CCAST. Ultralytics 8.4.182 is loaded from the isolated wheel.',
    'pending':['CCAST deployment and unit tests','Evaluation of the existing RDD step-100/200 checkpoints on the 15 training images','Returned diagnostic record review'],
    'actual_ccast_checkpoint_binaries_read_locally':0,'raw_rdd_images_read_locally':0}
(package/'LOCAL_VALIDATION.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
names.append('LOCAL_VALIDATION.json')
hashes={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in sorted(names)}
(package/'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name,value in hashes.items()),encoding='ascii',newline='\n')
names.append('SHA256SUMS')
with archive.open('xb') as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as bundle:
            for name in sorted(names):
                data=(package/name).read_bytes()
                info=tarfile.TarInfo(f'{package.name}/{name}')
                info.size,info.mode,info.mtime=len(data),0o644,0
                bundle.addfile(info,io.BytesIO(data))
with tarfile.open(archive) as bundle:
    assert set(bundle.getnames())=={f'{package.name}/{name}' for name in names}
    for name in names:
        data=bundle.extractfile(f'{package.name}/{name}').read()
        assert data==(package/name).read_bytes() and b'\r' not in data
        if name in hashes:
            assert hashlib.sha256(data).hexdigest()==hashes[name]
record={'status':'prepared_locally_ccast_transfer_pending','archive':archive.name,
        'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'archive_files':len(names),'checksum_entries':len(hashes),'line_endings':'LF, no BOM',
        'member_bytes_verified':True,'unit_tests_passed':8,'actual_pretrained_cpu_check':'passed',
        'optimizer_steps_permitted':0,'model_forward_budget':48,'actual_ccast_diagnostic':'pending'}
report_path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record,indent=2))
