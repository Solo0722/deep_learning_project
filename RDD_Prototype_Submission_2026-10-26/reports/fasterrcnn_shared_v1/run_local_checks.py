"""Focused software checks, comparator parity and frozen metadata exposure plan."""
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT/'fasterrcnn_shared_v1'
sys.path.insert(0, str(PACKAGE))
import baseline as b
import shared_augmentation as aug


def save(name, value):
    with (OUT/name).open('xb') as handle:
        handle.write(b.json_bytes(value))


def main():
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(PACKAGE/'tests'), pattern='test_*.py')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    with (OUT/'unit_tests.txt').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(stream.getvalue())
    summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'seconds': time.monotonic()-started}
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit('Focused tests failed')
    save('unit_tests.json', summary)
    old_package = ROOT/'fasterrcnn_baseline_v1'
    unchanged_files = ['rdd_data.py','models.py','metrics.py','checkpoints.py','integration.py',
                       'select_training_cases.py','smoke_selection.json','tests/test_baseline.py']
    for name in unchanged_files:
        assert (old_package/name).read_bytes() == (PACKAGE/name).read_bytes(), name
    parse = lambda p: {n.name: ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef)}
    old, new = parse(old_package/'baseline.py'), parse(PACKAGE/'baseline.py')
    unchanged_functions = ['epoch_batches','learning_rate','update','validate','commit','restore','write_history','verify_partitions']
    for name in unchanged_functions:
        assert old[name] == new[name], name
    cfg = json.loads((PACKAGE/'protocol.json').read_text())
    aug.verify_pairing(cfg, json.loads((PACKAGE/'identity_protocol.json').read_text()))
    rows = {s: b.read_manifest(ROOT/'detection_splits_v1'/f'{s}.jsonl', cfg['manifests'][s], s) for s in ('train','val')}
    assert {s: len(v) for s, v in rows.items()} == cfg['expected_rows']
    b.verify_partitions(rows['train'], rows['val'])
    plan, total = [], Counter()
    for epoch in range(cfg['epochs']):
        counts = Counter()
        for row in rows['train']:
            selected = aug.decision(row, cfg['augmentation_policy'], cfg['seed'], epoch)
            counts['images'] += 1
            counts['target_images'] += int(bool(row['targets']))
            counts['operation/'+selected['operation']] += 1
            counts['reason/'+selected['reason']] += 1
            if selected['reason'] == 'applied':
                counts['applied_images'] += 1
                counts['family/'+selected['family']] += 1
        plan.append({'epoch': epoch+1, 'counts': dict(counts)})
        total.update(counts)
    assert plan[0]['counts'].get('applied_images', 0) == 0
    assert all(item['counts']['applied_images'] > 0 for item in plan[1:])
    selection = json.loads((PACKAGE/'smoke_selection.json').read_text())
    smoke_policy = {**cfg['augmentation_policy'], 'warmup_epochs': 0}
    smoke = Counter(aug.decision(r, smoke_policy, cfg['seed'], 0)['reason'] for r in rows['train'] if r['image'] in selection['train'])
    assert smoke['applied'] > 0
    save('expected_exposure.json', {'scope': 'Predicted assignments from frozen training metadata, not executed training or a guarantee of changed pixels',
                                  'policy_sha256': hashlib.sha256(aug.encoded(cfg['augmentation_policy'])).hexdigest(),
                                  'seed': cfg['seed'], 'train_manifest_sha256': cfg['manifests']['train'],
                                  'epochs': plan, 'total': dict(total), 'smoke_assignments': dict(smoke), 'test_data_read': False})
    save('comparator_checks.json', {'status': 'passed', 'unchanged_dependency_files': unchanged_files,
                                   'unchanged_functions': unchanged_functions, 'nonaugmentation_configuration_match': True,
                                   'frozen_train_val_manifest_and_separation_checks': 'passed', 'test_data_read': False})
    print(json.dumps({'tests': summary, 'comparison': 'passed', 'predicted_applied_images': total['applied_images'],
                      'smoke_assignments': dict(smoke)}, indent=2))


if __name__ == '__main__':
    main()
