"""Local invariants, unchanged training code, and paired training-metadata exposure."""
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT/'fasterrcnn_local_v1'
sys.path.insert(0, str(PACKAGE))
import baseline as b
import local_augmentation as aug
import shared_augmentation as shared


def save(name, value):
    with (OUT/name).open('xb') as handle:
        handle.write(b.json_bytes(value))


def main():
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(PACKAGE/'tests'), pattern='test_*.py')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'seconds': time.monotonic()-started}
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit('Focused tests failed')
    with (OUT/'unit_tests.txt').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(stream.getvalue())
    save('unit_tests.json', summary)
    print(json.dumps(summary), flush=True)
    old_package = ROOT/'fasterrcnn_baseline_v1'
    unchanged_files = ['rdd_data.py','models.py','metrics.py','checkpoints.py','integration.py',
                       'select_training_cases.py','smoke_selection.json','tests/test_baseline.py']
    for name in unchanged_files:
        assert (old_package/name).read_bytes() == (PACKAGE/name).read_bytes(), name
    assert (ROOT/'fasterrcnn_shared_v1/shared_augmentation.py').read_bytes() == (PACKAGE/'shared_augmentation.py').read_bytes()
    parse = lambda p: {n.name: ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef)}
    old, new = parse(old_package/'baseline.py'), parse(PACKAGE/'baseline.py')
    unchanged_functions = ['epoch_batches','learning_rate','update','validate','commit','restore','write_history','verify_partitions']
    for name in unchanged_functions:
        assert old[name] == new[name], name
    cfg = json.loads((PACKAGE/'protocol.json').read_text())
    aug.verify_pairing(cfg, json.loads((PACKAGE/'identity_protocol.json').read_text()), json.loads((PACKAGE/'shared_protocol.json').read_text()))
    for stem in ('identity_protocol', 'shared_protocol', 'shared_exposure'):
        assert b.checkpoints.file_hash(PACKAGE/f'{stem}.json') == cfg['comparator'][f'{stem}_sha256']
    rows = {s: b.read_manifest(ROOT/'detection_splits_v1'/f'{s}.jsonl', cfg['manifests'][s], s) for s in ('train','val')}
    assert {s: len(v) for s, v in rows.items()} == cfg['expected_rows']
    b.verify_partitions(rows['train'], rows['val'])
    geometry_counts, groups_per_image = Counter(), Counter()
    for row in rows['train']:
        gs = aug.groups(row, cfg['image_size'])
        groups_per_image[len(gs)] += 1
        geometry_counts['images'] += 1
        geometry_counts['groups'] += len(gs)
        geometry_counts['target_images'] += int(bool(gs))
        geometry_counts['multi_group_images'] += int(len(gs) > 1)
        geometry_counts['multi_box_groups'] += sum(len(g[0]) > 1 for g in gs)
        geometry_counts['mixed_class_groups'] += sum(len({t['class_index'] for t in row['targets'] if t['object_index'] in g[0]}) > 1 for g in gs)
        geometry_counts['single_pixel_groups'] += sum(g[2] == 1 for g in gs)
        assert sorted(i for g in gs for i in g[0]) == sorted(t['object_index'] for t in row['targets'])
    plan, total, fractions = [], Counter(), []
    for epoch in range(cfg['epochs']):
        trace = aug.empty_trace()
        for batch in b.epoch_batches(len(rows['train']), cfg['batch_size'], cfg['seed'], epoch):
            for index in batch:
                row = rows['train'][index]
                selected = aug.decision(row, cfg['augmentation_policy'], cfg['seed'], epoch, cfg['image_size'])
                aug.record_decision(trace, selected)
                if selected['local']:
                    local = selected['local']
                    fractions.append(local['union_pixels']/local['content_pixels'])
        plan.append({'epoch': epoch+1, 'augmentation': trace})
        total.update(trace['counts'])
        print(f"Planned epoch {epoch+1}: applied={trace['counts'].get('applied_images', 0)}", flush=True)
    aug.verify_shared_exposure(plan, json.loads((PACKAGE/'shared_exposure.json').read_text()))
    assert total['applied_images'] == 158151 and total['images'] == 537260
    selection = json.loads((PACKAGE/'smoke_selection.json').read_text())
    smoke_policy = {**cfg['augmentation_policy'], 'warmup_epochs': 0}
    smoke = Counter(aug.decision(r, smoke_policy, cfg['seed'], 0, cfg['image_size'])['reason'] for r in rows['train'] if r['image'] in selection['train'])
    assert smoke['applied'] > 0
    save('expected_exposure.json', {
        'scope': 'Prospective assignments from frozen train metadata; no raw pixels or training executed',
        'policy_sha256': hashlib.sha256(shared.encoded(cfg['augmentation_policy'])).hexdigest(),
        'local_policy_sha256': hashlib.sha256(shared.encoded(cfg['local_policy'])).hexdigest(),
        'seed': cfg['seed'], 'train_manifest_sha256': cfg['manifests']['train'],
        'epochs': plan, 'total': dict(total), 'smoke_assignments': dict(smoke),
        'geometry': dict(geometry_counts), 'groups_per_image': dict(groups_per_image),
        'union_fraction_of_content_per_applied_image': {
            'mean': float(np.mean(fractions)), 'median': float(np.median(fractions)),
            'p05': float(np.quantile(fractions, .05)), 'p95': float(np.quantile(fractions, .95))},
        'aggregate_union_fraction_of_applied_content': total['local_union_pixels']/total['local_reference_pixels'],
        'aggregate_alpha_weighted_fraction_of_applied_content': total['local_alpha_units']/(2*total['local_reference_pixels']),
        'appearance_decisions_match_all_20_completed_shared_epochs': True, 'test_data_read': False})
    save('comparator_checks.json', {'status': 'passed', 'unchanged_dependency_files': unchanged_files,
        'shared_augmentation_source_byte_identical': True, 'unchanged_functions': unchanged_functions,
        'nonaugmentation_configuration_match': True, 'frozen_train_val_manifest_and_separation_checks': 'passed',
        'appearance_decisions_match_all_20_completed_shared_epochs': True, 'test_data_read': False})
    print(json.dumps({'tests': summary, 'comparison': 'passed', 'predicted_applied_images': total['applied_images'],
                      'geometry': dict(geometry_counts), 'smoke_assignments': dict(smoke)}, indent=2))


if __name__ == '__main__':
    main()
