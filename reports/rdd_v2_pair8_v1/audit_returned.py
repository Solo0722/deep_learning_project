"""Verify and audit the returned paired epoch-8 comparison without changing released inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
RETURNED = REPORT/'returned_records'
PACKAGE = ROOT/'rdd_v2_pair8_v1'
ARCHIVE = ROOT/'rdd_v2_pair8_seed42_v1_records_87bdb2e326af.tar.gz'
EXPECTED_SHA = 'c99c49cf0cbfe043a53f6901d35ee251777dc048ee82d83751ce61935cf3972f'
PREFIX = 'rdd_v2_pair8_seed42_v1'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, value):
    (REPORT/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def verify_archive():
    assert sha(ARCHIVE) == EXPECTED_SHA, 'Wrong downloaded archive'
    count = total = 0
    seen = set()
    RETURNED.mkdir(exist_ok=True)
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        for member in archive:
            p = PurePosixPath(member.name)
            assert ':' not in member.name
            assert member.isfile() and p.parts[0] == PREFIX and len(p.parts) > 1
            assert '..' not in p.parts and not p.is_absolute() and '\\' not in member.name
            relative = PurePosixPath(*p.parts[1:]).as_posix()
            assert relative not in seen and p.suffix not in ('.pt', '.tmp')
            seen.add(relative)
            target = RETURNED.joinpath(*p.parts[1:])
            assert RETURNED.resolve() in target.resolve().parents
            data = archive.extractfile(member).read()
            assert len(data) == member.size
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): assert hashlib.sha256(data).hexdigest() == sha(target)
            else:
                with target.open('xb') as handle: handle.write(data)
            count += 1; total += len(data)
            if count % 100 == 0: print(f'Extracted and matched {count} files',flush=True)
    expected = load(RETURNED/'checksums.json')
    assert seen == set(expected) | {'checksums.json'}
    for name, digest in expected.items():
        assert sha(RETURNED/name) == digest, 'Changed returned record: '+name
    context = load(RETURNED/'context.json')
    for name, digest in context['sources'].items():
        assert sha(PACKAGE/name) == digest, 'Source mismatch: '+name
    assert context['configuration'] == load(PACKAGE/'protocol.json')
    cfg = context['configuration']
    for split in ('train', 'val'):
        assert sha(ROOT/'detection_splits_v1'/f'{split}.jsonl') == cfg['manifests'][split]
    assert sha(ROOT/'detection_splits_v1/config.json') == cfg['split_config_sha256']
    assert context['versions'] == cfg['versions']
    assert load(RETURNED/'status.json')['status'] == 'complete'
    assert load(RETURNED/'COMPLETE.json')['status'] == 'complete'
    result = {'archive_sha256': EXPECTED_SHA, 'archive_bytes': ARCHIVE.stat().st_size,
        'files': count, 'uncompressed_bytes': total, 'internal_checksums': len(expected),
        'source_checksums': len(context['sources']), 'configuration_matches': True,
        'train_val_and_split_config_match': True, 'versions_match': True,
        'remote_full_checkpoint_tensors_included': False}
    save('archive_audit.json', result)
    print(json.dumps(result), flush=True)


def overview():
    compact = {}
    for arm in ('geometry_only', 'class_adaptive'):
        out = RETURNED/arm
        run = load(out/'run.json')
        history = load(out/'history.json')
        compact[arm] = {'status': run['status'], 'epoch': run['completed_epoch'], 'step': run['step'],
            'train_seconds': run['training_seconds'], 'validation_seconds': run['validation_seconds'],
            'fork': load(out/'fork.json'),
            'history': [{k: h[k] for k in ('epoch', 'step', 'train_images', 'val_images', 'training_loss')} |
                {'metrics': {k: v for k, v in h['validation'].items() if k not in ('no_target_predicted_boxes',)},
                 'augmentation': h['augmentation'], 'active_update_epoch': h['active_update_epoch']} for h in history],
            'updates': [{'epoch': u['epoch'], 'file': u['file'], 'audit': u['audit'],
                         'seconds': u['elapsed_seconds'], 'images': u['detector_images'],
                         'pooled': u['policy']['pooled'], 'shared_allowed': u['policy']['shared_allowed'],
                         'routed': {k: {'allowed': v['allowed'], 'borrowed': v['borrowed_evidence'], 'vetoed': v['vetoed_actions']} for k,v in u['policy']['routed'].items()}}
                        for u in run['controller_updates']]}
    save('overview.json', compact)
    for arm, result in compact.items():
        print(arm, [(h['epoch'], round(h['metrics']['AP50_95']*100, 4), round(h['metrics']['AP50']*100, 4),
                    h['augmentation']['counts'].get('all/applied', 0), h['augmentation']['counts'].get('all/changed', 0)) for h in result['history']], flush=True)
        for u in result['updates']:
            print('UPDATE', u['epoch'], 'shared_actions', sum(u['shared_allowed']),
                  {c: (v['detected_objects'],v['detected_clusters'],sum(v['allowed'])) for c,v in u['pooled'].items()}, flush=True)
    print('ATTEMPT', load(RETURNED/'status.json'), flush=True)
    print('COMMON', {k:v for k,v in load(RETURNED/'common_warmup/epoch_001_validation.json')['metrics'].items() if k!='no_target_predicted_boxes'}, flush=True)


if __name__ == '__main__':
    verify_archive()
    overview()
