"""Retrospective design diagnostics using preserved training-only evidence.

No modified controller, detector inference, training, or test data. Candidate
strength calculations on saved snapshots are not counterfactual accuracy.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/'fasterrcnn_class_adaptive_v1'))
import local_augmentation as local


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(name, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    path = OUT/name
    if path.exists():
        assert path.read_bytes() == data, f'Preserve prior result {name}'
    else:
        path.write_bytes(data)


def save_csv(name, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    save(name, stream.getvalue().encode())


def group_key(p):
    return ','.join(map(str, sorted({p['classes'][i] for i in p['target_positions']})))


def main():
    audit = json.loads((OUT/'ccast_verification.json').read_text())
    ctrl = json.loads((OUT/'controller_verification.json').read_text())
    config = json.loads((ROOT/'fasterrcnn_class_adaptive_v1/protocol.json').read_text())
    plan = json.loads((OUT/'panel_plan.json').read_text())
    assert audit['maximum_recomputed_metric_error'] == 0
    archive = ROOT/audit['archive']
    assert digest(archive) == audit['archive_sha256']
    entries = {r['epoch']: r for r in ctrl['rounds']}
    records = {}
    with tarfile.open(archive, 'r|gz') as tf:
        for member in tf:
            if '/controller/' in member.name and member.name.endswith('.json.gz'):
                epoch = int(Path(member.name).name.split('_')[1])
                raw = tf.extractfile(member).read()
                assert hashlib.sha256(raw).hexdigest() == entries[epoch]['policy_record_sha256']
                records[epoch] = json.loads(gzip.decompress(raw))
                if len(records) == 10:
                    break
    assert set(records) == set(range(2, 21, 2))

    # Actual applications (image presentations, not independent images) routed
    # through each particular update/class-set/family state.
    weights = Counter()
    for epoch in ctrl['epochs']:
        if epoch['epoch'] == 1:
            continue
        update = epoch['epoch'] if epoch['epoch'] % 2 == 0 else epoch['epoch']-1
        for key, count in epoch['counts'].items():
            if key.startswith('class_set/') and '/family/' in key:
                _, s, _, f = key.split('/')
                weights[(update, s, f)] += count
    assert sum(weights.values()) == ctrl['total']['applied_images']
    cells, raw_pairs, loss_zero_weights, near_uniform_weights = [], defaultdict(list), 0, 0
    assert settings_match(records, config)
    for epoch, record in sorted(records.items()):
        result, panel = record['result'], record['panel']
        for s, policy in result['policy']['class_sets'].items():
            subset = [p for p in panel if group_key(p) == s]
            for family in policy['families']:
                f, chosen = family['family'], family['level']
                measurements = [m for m in result['measurements'][s] if m['family'] == f]
                levels = {}
                for level in range(1, 4):
                    ms = [m for m in measurements if m['level'] == level]
                    levels[level] = {'eligible': all(c['passes'] for m in ms for c in m['classes'].values()),
                                     'difficulty': math.fsum(c['difficulty'] for m in ms for c in m['classes'].values())/
                                                   sum(len(m['classes']) for m in ms)}
                allowed = [level for level in levels if levels[level]['eligible']]
                assert chosen == max(allowed)
                weight = weights[(epoch, s, f)]
                zero = family['difficulty'] == 0
                loss_zero_weights += weight if zero else 0
                if len(policy['families']) == 2 and abs(family['probability']-.5) <= .05:
                    near_uniform_weights += weight
                ms = [m for m in measurements if m['level'] == chosen]
                min_retention = min(c['retention'] for m in ms for c in m['classes'].values())
                loss_best = max(allowed, key=lambda level: (levels[level]['difficulty'], -level))
                # Sensitivity of the selected gate to deleting one source cluster.
                failing_clusters = []
                clusters = sorted({p['source_cluster_id'] for p in subset})
                for cluster in clusters:
                    passed = True
                    for c in policy['classes']:
                        objects = [(p, i) for p in subset if p['source_cluster_id'] != cluster
                                   for i in p['target_positions'] if p['classes'][i] == c]
                        detected = [(p, i) for p, i in objects if p['reference']['detected'][i]]
                        if len(detected) < 8 or len({p['source_cluster_id'] for p, _ in detected}) < 4:
                            passed = False
                            break
                        for m in ms:
                            op = m['operation']
                            draws = range(2 if op == 'gaussian_noise' else 1)
                            retained = math.fsum(math.fsum(p['views'][f'{op}/{chosen}/{d}']['detected'][i] for d in draws)/len(draws)
                                                 for p, i in detected)/len(detected)
                            if retained < .8:
                                passed = False
                                break
                        if not passed:
                            break
                    if not passed:
                        failing_clusters.append(cluster)
                cells.append({'epoch': epoch, 'class_set': s, 'family': f, 'level': chosen,
                              'applications': weight, 'zero_difficulty': zero, 'difficulty': family['difficulty'],
                              'probability': family['probability'], 'eligible_levels': ','.join(map(str, allowed)),
                              'weakest_eligible': min(allowed), 'loss_maximizing_level_earliest_tie': loss_best,
                              'maximum_eligible_difficulty': levels[loss_best]['difficulty'],
                              'positive_alternative_to_zero': zero and levels[loss_best]['difficulty'] > 0,
                              'minimum_retention': min_retention,
                              'source_clusters': len(clusters), 'single_cluster_deletions_rejecting_selected_gate': len(failing_clusters),
                              'mean_selected_false_positive_change': float(np.mean([m['mean_false_positive_change'] for m in ms])),
                              'mean_selected_outside_recall_change': mean_present([m['outside_recall_change'] for m in ms])})
                for p in subset:
                    for i in p['target_positions']:
                        before = p['reference']
                        for m in ms:
                            op = m['operation']
                            views = [p['views'][f'{op}/{chosen}/{d}'] for d in range(2 if op == 'gaussian_noise' else 1)]
                            after_loss = np.mean([v['loss'][i] for v in views])
                            after_count = np.mean([v['roi_count'][i] for v in views])
                            row = (after_loss-before['loss'][i], after_count-before['roi_count'][i],
                                   before['roi_count'][i], after_count)
                            raw_pairs['all'].append(row)
                            raw_pairs['zero_score' if zero else 'positive_score'].append(row)

    # Training annotation availability: distinct images/clusters with each exact
    # overlap-group class set, using the frozen 640-pixel grouping implementation.
    manifest = ROOT/'detection_splits_v1/train.jsonl'
    assert digest(manifest) == config['manifests']['train']
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    inventory = {s: {'groups': 0, 'images': set(), 'source_clusters': set(), 'objects': Counter()} for s in entries[2]['policy']['class_sets']}
    for row in rows:
        objects = {t['object_index']: t['class_index'] for t in row['targets']}
        for group in local.groups(row, 640):
            ids = group[0]
            key = ','.join(map(str, sorted({objects[i] for i in ids})))
            inv = inventory[key]
            inv['groups'] += 1
            inv['images'].add(row['image'])
            inv['source_clusters'].add(row['source_cluster_id'])
            inv['objects'].update(str(objects[i]) for i in ids)
    inventory = {s: {**v, 'images': len(v['images']), 'source_clusters': len(v['source_clusters']), 'objects': dict(v['objects'])}
                 for s, v in inventory.items()}
    support = ctrl['class_set_exposure']
    mixed = [r for r in support if ',' in r['class_set']]
    def pair_summary(values):
        v = np.asarray(values)
        return {'object_operation_pairs': len(v), 'scope': 'Repeated correlated probe measurements, not independent samples',
                'loss_decreased_fraction': float(np.mean(v[:,0] < 0)),
                'roi_count_changed_fraction': float(np.mean(v[:,1] != 0)),
                'roi_count_decreased_fraction': float(np.mean(v[:,1] < 0)),
                'loss_and_roi_count_decreased_fraction': float(np.mean((v[:,0]<0)&(v[:,1]<0))),
                'mean_reference_roi_count': float(v[:,2].mean()), 'mean_transformed_roi_count': float(v[:,3].mean()),
                'loss_change_roi_count_change_correlation': float(np.corrcoef(v[:,0],v[:,1])[0,1])}
    applied = ctrl['total']['applied_images']
    summary = {
        'status': 'completed_retrospective_training_record_diagnostic',
        'scope': 'Frozen observed detector trajectories only; no performance predictions for altered controllers',
        'archive_sha256': audit['archive_sha256'],
        'strength_selection': {
            'active_family_states': len(cells), 'zero_difficulty_family_states': sum(c['zero_difficulty'] for c in cells),
            'applications': applied, 'applications_from_zero_difficulty_states': loss_zero_weights,
            'zero_difficulty_application_fraction': loss_zero_weights/applied,
            'applications_with_two_nearly_uniform_families': near_uniform_weights,
            'nearly_uniform_application_fraction': near_uniform_weights/applied,
            'zero_selected_score_with_positive_alternative_states': sum(c['positive_alternative_to_zero'] for c in cells),
            'zero_selected_score_with_positive_alternative_applications': sum(c['applications'] for c in cells if c['positive_alternative_to_zero']),
            'weakest_passing_different_applications': sum(c['applications'] for c in cells if c['weakest_eligible'] != c['level']),
            'states_sensitive_to_one_cluster_deletion': sum(c['single_cluster_deletions_rejecting_selected_gate'] > 0 for c in cells),
            'applications_routed_to_states_sensitive_to_one_cluster_deletion': sum(c['applications'] for c in cells if c['single_cluster_deletions_rejecting_selected_gate'] > 0),
        },
        'support': {
            'total_opportunities': ctrl['total']['augmentation_opportunities'],
            'total_fallbacks': ctrl['total']['controller_identity_images'],
            'mixed_opportunities': sum(r['opportunities'] for r in mixed),
            'mixed_fallbacks': sum(r['identity'] for r in mixed),
            'mixed_fallback_fraction': sum(r['identity'] for r in mixed)/sum(r['opportunities'] for r in mixed),
            'minimum_fallbacks_even_with_perfect_detection_on_old_panels': plan['minimum_identity_opportunities_due_to_annotation_sparsity'],
            'training_inventory_by_exact_set': inventory,
        },
        'changing_roi_diagnostic': {k: pair_summary(v) for k, v in raw_pairs.items()},
        'limits': [
            'Zero clipped net mean loss increase does not mean unchanged pixels, no object became harder, or no regularization benefit.',
            'ROI counts demonstrate changing measurement sets; saved records lack per-ROI boxes/logits, so fixed-ROI losses cannot be reconstructed.',
            'Correlation and cluster-deletion sensitivity are descriptive and cannot identify the cause of the AP result.',
            'Annotation availability is an upper bound on detector-supported samples. Group candidates can share images/source clusters across sets.',
            'New panels and new policies alter subsequent detector trajectories; replay cannot forecast their AP.'
        ],
        'analysis_source_sha256': digest(Path(__file__)),
        'sources': {p: digest(OUT/p) for p in ['ccast_verification.json','controller_verification.json','panel_plan.json']},
        'test_data_read': False, 'new_training_or_detector_inference_executed': False,
    }
    save('strength_support_diagnostics.json', summary)
    save_csv('strength_selection_diagnostic_cells.csv', cells)
    save_csv('class_set_training_inventory.csv', [{'class_set': s, 'groups':v['groups'],'images':v['images'],'source_clusters':v['source_clusters']}
                                               for s, v in inventory.items()])
    print(json.dumps(summary, indent=2), flush=True)


def mean_present(values):
    available = [v for v in values if v is not None]
    return float(np.mean(available)) if available else None


def settings_match(records, config):
    encoded = json.dumps(config['controller'], sort_keys=True, separators=(',',':'),allow_nan=False).encode()
    return all(r['settings_sha256'] == hashlib.sha256(encoded).hexdigest() for r in records.values())


if __name__ == '__main__':
    main()
