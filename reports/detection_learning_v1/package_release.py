"""Freeze the validated learning package and verify every archive member."""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[2]
package = root/'detection_learning_v1'
archive = root/'detection_learning_v1.tar.gz'
report_path = Path(__file__).with_name('release.json')
for path in (archive,report_path,package/'LOCAL_VALIDATION.json',package/'SHA256SUMS'):
    if path.exists():
        raise FileExistsError(f'Preserve existing release: {path}')
names = ['README.md','protocol.json','learning.py','models.py','metrics.py','checkpoints.py',
         'integration.py','rdd_data.py','reference.json','selection.json','select_training_cases.py',
         'tests/test_learning.py']
for name in names:
    path = package/name
    data = path.read_bytes().replace(b'\r\n',b'\n')
    assert b'\r' not in data and not data.startswith(b'\xef\xbb\xbf'),name
    if path.suffix == '.py':
        ast.parse(data.decode(),feature_version=(3,12))
    if path.read_bytes() != data:
        path.write_bytes(data)
protocol = json.loads((package/'protocol.json').read_text())
for name,digest in protocol['loader_source_sha256'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest() == digest
    assert (package/name).read_bytes() == (root/'detection_integration_v1'/name).read_bytes()
assert (package/'integration.py').read_bytes() == (root/'detection_integration_v1/integration.py').read_bytes()
check = json.loads(Path(__file__).with_name('local_model_check.json').read_text())
assert check['status'] == 'passed' and set(check['models']) == set(protocol['models'])
for record in check['models'].values():
    assert record['status'] == 'passed' and record['all_empty_update']['classifier_max_absolute_update'] > 0
    assert all(record['restore'][k] for k in ('model_state_exact','optimizer_state_exact','rng_state_exact',
                                             'predictions_match_within_tolerance'))
validation = {
    'status':'prepared_locally_ccast_pending','recorded_utc':datetime.now(timezone.utc).isoformat(),
    'unit_tests':{'command':'python -m unittest discover -s detection_learning_v1/tests -p test_learning.py -v',
                  'passed':18,'skipped':0,'reported_seconds':5.967,
                  'evidence':'Final local unittest output: Ran 18 tests in 5.967s, OK, exit 0'},
    'unit_test_scope':'Known COCO metric cases, inverse coordinates, false positives on empty images, exact CPU stochastic checkpoint continuation, checkpoint tampering/no-overwrite, paired two-model orchestration with toy models, records-only archive',
    'actual_pretrained_cpu_models':check,
    'local_model_check_command':'python reports/detection_learning_v1/check_actual_models.py',
    'raw_rdd_images_read_locally':0,'synthetic_model_check_size':128,
    'python_3_12_grammar':'passed','unchanged_loader_files':4,'unchanged_integration_helpers':True,
    'pending':['CCAST transfer and checksum verification','Unit tests with pinned CCAST dependencies',
               'Both pretrained paths at 640 on the 15 actual training images',
               '200-update trajectories and scheduled GPU checkpoint reloads','Returned-record analysis'],
    'limits':'CPU tests use torch 2.10.0, torchvision 0.25.0 and pycocotools 2.0.11; CCAST pins 2.3.0, 0.18.0 and 2.0.10. Ultralytics 8.4.182 is loaded from an isolated wheel. No detector efficacy claim.'}
(package/'LOCAL_VALIDATION.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
names.append('LOCAL_VALIDATION.json')
hashes = {name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in sorted(names)}
(package/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in hashes.items()),encoding='ascii',newline='\n')
names.append('SHA256SUMS')
with archive.open('xb') as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as bundle:
            for name in sorted(names):
                data = (package/name).read_bytes()
                info = tarfile.TarInfo(f'{package.name}/{name}')
                info.size,info.mode,info.mtime = len(data),0o644,0
                bundle.addfile(info,io.BytesIO(data))
with tarfile.open(archive) as bundle:
    assert set(bundle.getnames()) == {f'{package.name}/{name}' for name in names}
    for name in names:
        data = bundle.extractfile(f'{package.name}/{name}').read()
        assert data == (package/name).read_bytes() and b'\r' not in data
        if name in hashes:
            assert hashlib.sha256(data).hexdigest() == hashes[name]
record = {'status':'prepared_locally_ccast_transfer_pending','archive':archive.name,
          'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
          'archive_files':len(names),'checksum_entries':len(hashes),'line_endings':'LF, no BOM',
          'member_bytes_verified':True,'unit_tests_passed':18,
          'actual_pretrained_cpu_checks':'Both models passed update, inference, checkpoint reload, continued update and all-empty update',
          'protocol_steps_per_model':200,'real_rdd_gpu_execution':'pending'}
report_path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record,indent=2))
