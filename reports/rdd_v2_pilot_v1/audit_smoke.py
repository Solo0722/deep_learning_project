"""Audit archived GPU smoke evidence without detector inference or training.

Run from any directory with a Python environment containing numpy, torch,
torchvision, Pillow and pycocotools. Released packages are read, never edited.
"""
from collections import Counter, defaultdict
import contextlib
import csv
import gzip
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent / 'smoke_audit'
RETURNED = REPORT / 'returned_records'
PACKAGE = ROOT / 'rdd_v2_pilot_v1'
PREFIX = 'rdd_v2_pilot_smoke_797282.bright04'
ARCHIVE = ROOT / (PREFIX + '_records_v1.tar.gz')
EXPECTED_SHA = '12e8f65b129a55199f105c28bc7303fe9b1b852e950f3a062a76675aee5d0777'
EXPECTED_PASS_SHA = '2cb25bcaa1c377476fa103554094fdf745c70449d61cba3795fbb669148b1486'
ARMS = ('geometry_only', 'fixed_local', 'online_shared', 'class_adaptive')
SAME_POLICY = ('geometry_only', 'online_shared', 'class_adaptive')
CLASSES = ('D00', 'D10', 'D20', 'D40')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, obj):
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n',
                              encoding='utf-8', newline='\n')


def extract():
    assert sha(ARCHIVE) == EXPECTED_SHA, 'Downloaded archive differs from user-provided CCAST hash'
    seen, inventory = set(), {}
    RETURNED.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, 'r:gz') as source:
        for member in source:
            path = PurePosixPath(member.name)
            assert path.parts and not path.is_absolute() and '..' not in path.parts
            assert '\\' not in member.name and ':' not in member.name
            assert path.parts[0] in (PREFIX, 'pilot_job_logs')
            assert member.name not in seen, 'Duplicated archive path'
            seen.add(member.name)
            if member.isdir():
                continue
            assert member.isfile() and path.suffix not in ('.pt', '.tmp'), 'Unexpected archive entry'
            assert member.size <= 10_000_000
            target = RETURNED.joinpath(*path.parts)
            assert RETURNED.resolve() in target.resolve().parents
            data = source.extractfile(member).read()
            assert len(data) == member.size
            digest = hashlib.sha256(data).hexdigest()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == digest, 'Existing extraction changed'
            else:
                with target.open('xb') as handle:
                    handle.write(data)
            inventory[member.name] = {'bytes': len(data), 'sha256': digest}
    assert len(inventory) == 50
    folder = RETURNED / PREFIX
    assert sha(folder / 'PASS.json') == EXPECTED_PASS_SHA
    passed = read(folder / 'PASS.json')
    assert passed['status'] == 'pass'
    assert sha(PACKAGE / 'SHA256SUMS') == passed['package_manifest_sha256']
    source_count = 0
    for line in (PACKAGE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha(PACKAGE / name) == digest, 'Released source changed: ' + name
        source_count += 1
    cfg = read(PACKAGE / 'protocol.json')
    for split in ('train', 'val'):
        assert sha(ROOT / 'detection_splits_v1' / (split + '.jsonl')) == cfg['manifests'][split]
    provenance = {'archive': ARCHIVE.name, 'archive_sha256': EXPECTED_SHA,
                  'expected_hash_source': 'CCAST sha256sum output supplied by the user in chat, 2026-10-29',
                  'archive_bytes': ARCHIVE.stat().st_size, 'files': len(inventory),
                  'uncompressed_bytes': sum(v['bytes'] for v in inventory.values()),
                  'pass_sha256': EXPECTED_PASS_SHA, 'source_files_verified': source_count,
                  'train_and_validation_manifest_hashes_verified': True,
                  'checkpoint_tensors_included': False,
                  'inventory': inventory}
    save('archive_audit.json', provenance)
    return folder, cfg, passed


def replay_metrics(rows, predictions, saved):
    """Ground truth is assembled directly from frozen rows, retaining full-manifest IDs."""
    truth_data = {'info': {}, 'images': [], 'annotations': [],
                  'categories': [{'id': c+1, 'name': name} for c, name in enumerate(CLASSES)]}
    for image_id, row in rows.items():
        truth_data['images'].append({'id': image_id, 'width': row['width'], 'height': row['height']})
        for target in row['targets']:
            x1, y1, x2, y2 = target['box_xyxy']
            truth_data['annotations'].append({'id': len(truth_data['annotations'])+1,
                'image_id': image_id, 'category_id': target['class_index']+1,
                'bbox': [x1, y1, x2-x1, y2-y1], 'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
    counts = Counter()
    for p in predictions:
        assert type(p['image_id']) is int and p['image_id'] in rows
        assert p['category_id'] in (1, 2, 3, 4)
        assert np.isfinite(p['score']) and 0 <= p['score'] <= 1
        x, y, w, h = p['bbox']
        assert np.isfinite(p['bbox']).all() and min(w, h) > 0 and min(x, y) >= -1e-8
        row = rows[p['image_id']]
        assert x+w <= row['width']+1e-8 and y+h <= row['height']+1e-8
        counts[p['image_id']] += 1
    assert max(counts.values(), default=0) <= 100
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = truth_data; gt.createIndex()
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(rows)
        ev.params.catIds = [1, 2, 3, 4]
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate()
    def mean(a):
        a = a[a >= 0]
        return float(a.mean()) if a.size else None
    p = ev.eval['precision']
    values = {'AP50_95': mean(p[:, :, :, 0, -1]), 'AP50': mean(p[0, :, :, 0, -1]),
              'AP75': mean(p[5, :, :, 0, -1]), 'AP_small': mean(p[:, :, :, 1, -1]),
              'AP_medium': mean(p[:, :, :, 2, -1]), 'AP_large': mean(p[:, :, :, 3, -1]),
              'AR100': mean(ev.eval['recall'][:, :, 0, -1])}
    errors = [abs(value-saved[k]) for k, value in values.items() if value is not None]
    for c, label in enumerate(CLASSES):
        per_class = {'AP50_95': mean(p[:, :, c, 0, -1]), 'AP50': mean(p[0, :, c, 0, -1]),
                     'AP_small': mean(p[:, :, c, 1, -1])}
        for key, value in per_class.items():
            if value is None:
                assert saved['per_class'][label][key] is None
            else:
                errors.append(abs(value-saved['per_class'][label][key]))
        assert saved['per_class'][label]['target_count'] == sum(
            t['class_index'] == c for row in rows.values() for t in row['targets'])
    negative = {i for i, row in rows.items() if not row['targets']}
    ncounts = Counter(p['image_id'] for p in predictions if p['image_id'] in negative and p['score'] >= .25)
    assert saved['no_target_predicted_boxes'] == {str(i): ncounts[i] for i in sorted(negative)}
    assert saved['no_target_images_with_predictions'] == len(ncounts)
    assert saved['evaluation_images'] == len(rows) == 24
    assert saved['target_boxes'] == len(truth_data['annotations']) == 42
    assert saved['no_target_images'] == len(negative) == 7
    assert saved['prediction_boxes'] == len(predictions)
    assert max(errors) < 1e-12
    return {'metrics': values, 'max_metric_error': max(errors), 'prediction_boxes': len(predictions),
            'validation_images': len(rows), 'target_boxes': len(truth_data['annotations'])}


def prediction_comparison(left, right):
    def group(values):
        out = defaultdict(list)
        for row in values:
            out[row['image_id']].append(row)
        return out
    a, b = group(left), group(right)
    assert a.keys() == b.keys()
    # Index-wise differences are reported only as a numerical diagnostic, never
    # as matched-object distances. Reordered categories are counted separately.
    matched_order = [i for i in a if [p['category_id'] for p in a[i]] == [p['category_id'] for p in b[i]]]
    scores, boxes = [], []
    for i in matched_order:
        scores.extend(abs(x['score']-y['score']) for x, y in zip(a[i], b[i]))
        boxes.extend(abs(u-v) for x, y in zip(a[i], b[i]) for u, v in zip(x['bbox'], y['bbox']))
    return {'images': len(a), 'exactly_equal_images': sum(a[i] == b[i] for i in a),
            'images_with_same_category_sequence': len(matched_order),
            'same_index_max_score_difference': max(scores, default=None),
            'same_index_max_bbox_difference': max(boxes, default=None),
            'comparison_scope': 'Saved decoded COCO rows; index-wise differences are not object matches.'}


def main():
    folder, cfg, passed = extract()
    selection = read(PACKAGE / 'smoke_selection.json')
    raw_val = [json.loads(line) for line in (ROOT / 'detection_splits_v1/val.jsonl').read_text().splitlines()]
    val_rows = {i: row for i, row in enumerate(raw_val) if row['image'] in set(selection['val'])}
    histories, predictions, validations, results = {}, {}, {}, {}
    for arm in ARMS:
        histories[arm] = read(folder / arm / 'history.json')
        assert [h['epoch'] for h in histories[arm]] == [2, 3]
        for h in histories[arm]:
            epoch = h['epoch']; key = f'{arm}/epoch_{epoch}'
            for name, digest in h['validation_records'].items():
                assert sha(folder / arm / name) == digest
            v = read(folder / arm / f'epoch_{epoch:03d}_validation.json')
            assert h['validation'] == v['metrics']
            assert v['state_audit']['model_optimizer_rng_modes_unchanged']
            assert [c['image_id'] for c in v['clipping']] == sorted(val_rows)
            with gzip.open(folder / arm / f'epoch_{epoch:03d}_val_predictions.json.gz', 'rt') as handle:
                predictions[key] = json.load(handle)
            validations[key] = v
            results[key] = replay_metrics(val_rows, predictions[key], v['metrics'])
            results[key].update(state_audit=v['state_audit'], training_loss=h['training_loss'],
                                augmentation=h['augmentation'])
            cp = read((folder / arm / 'checkpoints' / h['checkpoint_file']).with_suffix('.json'))
            assert cp['completed_epochs'] == epoch and cp['step'] == h['step'] == epoch*7
            assert h['train_images'] == 26 and h['val_images'] == 24 and h['images_seen'] == epoch*26
            assert cp['file'] == h['checkpoint_file'] and cp['label'] == f'epoch_{epoch:03d}'
            print(f'METRICS {key}: PASS AP50:95={results[key]["metrics"]["AP50_95"]*100:.6f}', flush=True)
        last = read(folder / arm / 'last.json')
        assert last == read((folder / arm / 'checkpoints' / last['file']).with_suffix('.json'))
        assert last['completed_epochs'] == 3
        best = read(folder / arm / 'best.json')
        best_epoch = max(histories[arm], key=lambda h: h['validation']['AP50_95'])
        assert best['epoch'] == best_epoch['epoch'] and best['validation_AP50_95'] == best_epoch['validation']['AP50_95']
    comparisons = {}
    for epoch in (2, 3):
        h = {arm: histories[arm][epoch-2] for arm in ARMS}
        assert len({x['augmentation']['pairing_sha256'] for x in h.values()}) == 1
        assert h['online_shared']['augmentation']['decision_sha256'] == h['class_adaptive']['augmentation']['decision_sha256']
        for arm in SAME_POLICY:
            aug = h[arm]['augmentation']; counts = aug['counts']
            assert counts['all/operation/identity'] == aug['images'] == 26
            assert counts.get('all/applied', 0) == counts.get('all/changed', 0) == 0
            assert aug['pixel_changes_after_geometry'] == aug['pixel_changes_before_geometry'] == 0
        fixed = h['fixed_local']['augmentation']['counts']
        assert fixed['all/applied'] == fixed['all/changed'] == fixed['all/opportunities']
        state = {a: validations[f'{a}/epoch_{epoch}']['state_audit'] for a in SAME_POLICY}
        metric_values = [results[f'{a}/epoch_{epoch}']['metrics']['AP50_95'] for a in SAME_POLICY]
        comparisons[str(epoch)] = {
            'pairing_sha256': h['geometry_only']['augmentation']['pairing_sha256'],
            'online_decision_sha256': h['online_shared']['augmentation']['decision_sha256'],
            'same_policy_appearance_edits': 0,
            'fixed_local_appearance_edits': fixed['all/applied'],
            'identical_model_hashes': len({s['model_sha256'] for s in state.values()}) == 1,
            'identical_gradient_hashes': len({s['gradient_sha256'] for s in state.values()}) == 1,
            'identical_rng_hashes': len({s['rng_sha256'] for s in state.values()}) == 1,
            'same_policy_AP50_95_range_points': (max(metric_values)-min(metric_values))*100,
            'predictions_vs_geometry': {a: prediction_comparison(predictions[f'geometry_only/epoch_{epoch}'],
                predictions[f'{a}/epoch_{epoch}']) for a in SAME_POLICY[1:]}}

    # Reuse independently written geometric/matching/retention auditors. Only
    # the keyed candidate selection and production policy are shared components.
    import audit_controller as ac
    c = ac.c
    train_rows = {r['image']: r for r in map(json.loads, (ROOT / 'detection_splits_v1/train.jsonl').read_text().splitlines())}
    plan = read(PACKAGE / 'panel_plan.json')['2'][:2]
    panels, probes = {}, {}
    for arm in SAME_POLICY[1:]:
        attempts = sorted((folder / arm / 'controller').glob('*_attempt.json'))
        assert len(attempts) == 2
        for path in attempts:
            attempt = read(path); epoch = attempt['epoch']; key = f'{arm}/epoch_{epoch}'
            assert attempt['status'] == 'complete' and attempt['error'] is None
            evidence = path.parent / attempt['file']
            assert sha(evidence) == attempt['sha256']
            with gzip.open(evidence, 'rt', encoding='utf-8') as handle:
                panel = [json.loads(line) for line in handle]
            assert len(panel) == attempt['records_complete'] == 2
            assert attempt['detector_images'] == 62 and attempt['forward_calls'] == 32
            views = sum(ac.audit_row(p, train_rows[p['image']], selected, cfg, epoch) for p, selected in zip(panel, plan))
            assert views == 58
            units = c.units_from_panel(panel, train_rows)
            policy = c.build_policy(units, cfg['controller'])
            ac.old.audit_policy(units, policy)
            assert not any(policy['shared_allowed'])
            assert all(not any(v['allowed']) for v in policy['routed'].values())
            audit = attempt['audit']
            assert audit['model_optimizer_rng_modes_unchanged']
            if epoch == 3:
                assert audit == validations[f'{arm}/epoch_2']['state_audit']
            panels[key] = panel
            probes[key] = {'views_independently_checked': views, 'audit': audit,
                          'policy_recomputed_sha256': c.digest(policy),
                          'raw_panel_canonical_sha256': c.digest(panel),
                          'support': {cl: {k: v[k] for k in ('objects', 'detected_objects', 'detected_clusters', 'supported')}
                                      for cl, v in policy['pooled'].items()},
                          'all_routes_identity': True, 'attempt_seconds': attempt['elapsed_seconds']}
            print(f'PROBE {key}: PASS 58 views; all routes identity', flush=True)
    initial_equal = panels['online_shared/epoch_2'] == panels['class_adaptive/epoch_2']
    initial_audits_equal = probes['online_shared/epoch_2']['audit'] == probes['class_adaptive/epoch_2']['audit']
    log = (RETURNED / 'pilot_job_logs/797282.bright04.log').read_text()
    smoke_log = log.split('GPU FOUR-ARM SMOKE: PASS.')[0]
    result = {'scope': 'Archived smoke evidence only. No checkpoint tensors, pixels or new detector inference.',
              'versions': {name: version(name) for name in ('numpy', 'pycocotools', 'torch', 'torchvision', 'Pillow')},
              'original_versions': cfg['versions'], 'smoke_pass_record': passed,
              'metrics': results, 'same_policy_comparisons': comparisons, 'probes': probes,
              'initial_online_probe_records_exactly_identical': initial_equal,
              'initial_online_state_audits_exactly_identical': initial_audits_equal,
              'unit_test_log': re.findall(r'Ran \d+ tests in [^\n]+', smoke_log),
              'cudnn_plan_warning_present': 'CUDNN_STATUS_NOT_SUPPORTED' in smoke_log,
              'conclusion': 'Saved trajectories differ despite identical treatment. Root cause is unresolved; the smoke gate does not test training equivalence.',
              'limitations': ['Initial geometry fork is attested by PASS.json, not independently recovered from tensors.',
                  'Checkpoint files and per-step parameter/gradient records are absent.',
                  'Final RNG digests cannot reconstruct when or why random streams diverged.',
                  'The 24-image smoke AP range is not an error bar for the 5760-image pilot.',
                  'Recomputed policies verify raw evidence; adopted checkpoint policies were not included.']}
    save('audit.json', result)
    with (REPORT / 'endpoint_results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['arm', 'epoch', 'AP50_95_points', 'AP50_points', 'appearance_edits'])
        writer.writeheader()
        for arm in ARMS:
            for epoch in (2, 3):
                data = results[f'{arm}/epoch_{epoch}']
                writer.writerow({'arm': arm, 'epoch': epoch, 'AP50_95_points': data['metrics']['AP50_95']*100,
                    'AP50_points': data['metrics']['AP50']*100,
                    'appearance_edits': data['augmentation']['counts'].get('all/applied', 0)})
    print(json.dumps({'initial_online_records_identical': initial_equal,
                      'initial_online_state_audits_identical': initial_audits_equal,
                      'same_policy_comparisons': comparisons}, indent=2), flush=True)
    print('SMOKE RECORD AUDIT COMPLETE. Training-equivalence issue remains unresolved.', flush=True)


if __name__ == '__main__':
    main()
