"""Final unit tests, comparator parity, and prospective training-panel metadata."""
import ast
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT/'fasterrcnn_online_shared_v1'
sys.path.insert(0, str(PACKAGE))
import baseline as b
import controller as c
import shared_augmentation as shared
import probe_runner as runner


def save(name, value):
    (OUT/name).write_bytes(b.json_bytes(value))


def main():
    started = time.monotonic()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        suite = unittest.defaultTestLoader.discover(str(PACKAGE/'tests'), pattern='test_*.py')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (OUT/'unit_tests.txt').write_text(log.getvalue(), encoding='utf-8', newline='\n')
    if not result.wasSuccessful():
        print(log.getvalue()); raise SystemExit('Focused tests failed')
    summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'seconds': time.monotonic()-started}
    save('unit_tests.json', summary); print(summary, flush=True)
    cfg = json.loads((PACKAGE/'protocol.json').read_text())
    refs = [json.loads((PACKAGE/f'{stem}_protocol.json').read_text()) for stem in ('identity', 'shared', 'local')]
    c.verify_pairing(cfg, *refs)
    unchanged = ['rdd_data.py', 'models.py', 'metrics.py', 'checkpoints.py', 'integration.py', 'select_training_cases.py',
                 'smoke_selection.json', 'local_augmentation.py', 'shared_augmentation.py', 'smoke_augmentation.py']
    for name in unchanged:
        assert (PACKAGE/name).read_bytes() == (ROOT/'fasterrcnn_local_v1'/name).read_bytes(), name
    parse = lambda path: {node.name: ast.dump(node) for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    old, new = parse(ROOT/'fasterrcnn_local_v1/baseline.py'), parse(PACKAGE/'baseline.py')
    functions = ['epoch_batches', 'learning_rate', 'update', 'validate', 'commit', 'restore', 'write_history', 'verify_partitions', 'archive_records']
    for name in functions: assert old[name] == new[name], name
    for stem in ('identity_protocol', 'shared_protocol', 'shared_exposure', 'local_protocol'):
        assert b.checkpoints.file_hash(PACKAGE/f'{stem}.json') == cfg['comparator'][f'{stem}_sha256']
    parity = {'status': 'passed', 'unchanged_dependency_files': unchanged, 'unchanged_baseline_functions': functions,
              'application_group_geometry_optimizer_validation_checkpoint_rules_match': True,
              'numerical_controller_settings': cfg['controller'], 'protocol_sha256': b.checkpoints.file_hash(PACKAGE/'protocol.json')}
    save('comparator_checks.json', parity)
    train = b.read_manifest(ROOT/'detection_splits_v1/train.jsonl', cfg['manifests']['train'], 'train')
    assert len(train) == cfg['expected_rows']['train']
    planned = []
    for epoch in range(cfg['epochs']):
        if not runner.due(cfg, epoch): continue
        panel = c.select_panel(list(enumerate(train)), cfg['image_size'], cfg['seed'], epoch, cfg['controller']['panel_images'])
        objects = Counter(); clusters = {str(i): set() for i in range(4)}
        for p in panel:
            for i in p['target_positions']:
                label = str(p['classes'][i]); objects[label] += 1; clusters[label].add(p['source_cluster_id'])
        planned.append({'epoch': epoch+1, 'panel_sha256': c.digest(panel), 'panel': panel, 'target_group_objects': dict(objects),
                        'target_group_clusters': {k: len(v) for k, v in clusters.items()}, 'images': len(panel),
                        'distinct_source_clusters': len({p['source_cluster_id'] for p in panel})})
    exposures = json.loads((PACKAGE/'shared_exposure.json').read_text())
    totals, counts = 0, []
    for epoch in range(cfg['epochs']):
        n = 0
        if epoch >= cfg['augmentation_policy']['warmup_epochs']:
            for row in train:
                if row['targets']:
                    key = shared.encoded([cfg['augmentation_policy']['version'], cfg['seed'], epoch, row['image']])
                    n += shared.random_word(key, 'application') < int(cfg['augmentation_policy']['probability']*2**64)
        assert n == exposures['epochs'][epoch]['counts'].get('applied_images', 0)
        counts.append({'epoch': epoch+1, 'opportunities': n}); totals += n
    assert totals == 158151
    plan = {'scope': 'Training metadata only; annotation counts are NOT detector support or retention evidence',
            'probe_image_forwards': sum(p['images'] for p in planned)*(1+len(c.variants(cfg['controller']))),
            'scheduled_updates': len(planned), 'epochs': planned, 'application_opportunities': counts,
            'total_application_opportunities': totals, 'all_opportunities_match_fixed_controls': True,
            'probe_settings_sha256': c.digest(cfg['controller']), 'test_data_read': False}
    assert plan['probe_image_forwards'] == 71680 and plan['scheduled_updates'] == 10
    save('panel_plan.json', plan)
    # Capture exact tested implementation files, excluding generated validation/checksum artifacts.
    sources = {p.relative_to(PACKAGE).as_posix(): b.checkpoints.file_hash(p) for p in PACKAGE.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json')}
    save('tested_sources.json', sources)
    print(json.dumps({'comparator_checks': 'passed', 'planned_probe_views': plan['probe_image_forwards'], 'opportunities': totals}), flush=True)


if __name__ == '__main__':
    main()
