"""Recompute controller evidence and geometry from saved observations; no inference."""
from collections import Counter
import gc
import gzip
import importlib.util
import json
import sys
import time

import numpy as np
from audit_returned import ROOT, PACKAGE, RETURNED, load, save, sha

sys.path.insert(0, str(PACKAGE))
import controller as c
import diagnostic as d
import geometry as g
import local_augmentation as local


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = helper('independent_controller', ROOT/'reports/rdd_v2_controller_check_v1/audit_returned.py')
geo = helper('independent_geometry', ROOT/'reports/rdd_v2_geometry_check_v1/audit_returned.py')


def audit_row(p, row, selection, cfg, epoch):
    assert p['selection'] == selection
    assert p['image'] == row['image'] and row['split'] == 'train'
    labels = [t['class_index'] for t in row['targets']]
    ids = [t['object_index'] for t in row['targets']]
    assert p['classes'] == labels and p['object_indices'] == ids
    assert p['source_cluster_id'] == row['source_cluster_id'] == selection['source_cluster_id']
    assert p['target_positions'] == selection['target_positions']
    assert [ids[i] for i in p['target_positions']] == selection['object_indices']
    assert list(local.groups(row, 640)[selection['group_index']][0]) == selection['object_indices']
    assert sorted({labels[i] for i in p['target_positions']}) == selection['group_classes']
    annotations, points = geo.prepare(row, 640)
    clean = geo.rectangles(points)
    candidate = g.chosen_candidate(row, cfg['geometry'], cfg['seed'], epoch-1)
    scale = candidate['scale']
    affine = np.array([[-scale if candidate['flip'] else scale, 0,
                        candidate['left'] + (640*scale if candidate['flip'] else 0)],
                       [0, scale, candidate['top']], [0, 0, 1]], float)
    transformed = geo.rectangles(points @ affine.T)
    outside = ((transformed[:, :2] < 0).any(1) | (transformed[:, 2:] > 640).any(1))
    accepted = not bool(outside.any())
    mapped = transformed if accepted else clean
    reference = p['reference']['render']
    assert reference['candidate'] == candidate and reference['accepted'] == accepted
    assert reference['reason'] == ('accepted' if accepted else 'cropped_annotation')
    assert reference['cropped_ids'] == [a['id'] for a, bad in zip(annotations, outside) if bad]
    assert len(reference['annotations']) == len(annotations)
    for original, actual, box in zip(annotations, reference['annotations'], mapped):
        assert all(actual[k] == original[k] for k in ('kind', 'id', 'original'))
        np.testing.assert_allclose(actual['box_xyxy'], box, rtol=0, atol=1e-10)
    n = len(row['targets'])
    count = 0
    expected_views = {f'{op}/{level}/{draw}' for op, level in c.ACTIONS
                      for draw in range(2 if op == 'gaussian_noise' else 1)}
    assert set(p['views']) == expected_views
    for name, v in [('clean', p['clean']), ('reference', p['reference']), *p['views'].items()]:
        expected_boxes = clean[:n].astype(np.float32) if name == 'clean' else mapped[:n]
        # Clean loader tensors use float32; mapped annotation metadata uses float64.
        np.testing.assert_allclose(v['render']['boxes'], expected_boxes, rtol=0, atol=1e-4 if name == 'clean' else 1e-10)
        assert v['render']['labels'] == labels
        got = old.match(v['prediction'], v['render']['boxes'], labels)
        old.same({k: v[k] for k in got}, got)
        assert len(v['pixel_sha256']) == 64
        if name in p['views']:
            op, level, draw = name.split('/')
            chosen = v['render']['decision']
            assert chosen['image'] == row['image'] and chosen['operation'] == op
            assert chosen['parameters'] == d.parameters(op, int(level))
            assert chosen['local']['object_indices'] == selection['object_indices']
            expected_seed = int(c.digest(['v2_probe_noise', cfg['seed'], row['image'], int(level), int(draw)])[:16], 16)
            assert chosen['noise_seed'] == (expected_seed if op == 'gaussian_noise' else None)
        count += 1
    return count


def main():
    cfg, plans = load(PACKAGE/'protocol.json'), load(PACKAGE/'panel_plan.json')
    rows = {r['image']: r for r in map(json.loads, (ROOT/'detection_splits_v1/train.jsonl').read_text().splitlines())}
    result, first_digests = {}, {}
    for arm in ('online_shared', 'class_adaptive'):
        for entry in load(RETURNED/arm/'run.json')['controller_updates']:
            start = time.monotonic()
            epoch = entry['epoch']; path = RETURNED/arm/entry['file']
            assert sha(path) == entry['sha256']
            attempt = load(path.with_name(path.name.replace('.jsonl.gz', '_attempt.json')))
            assert attempt['status'] == 'complete' and attempt['error'] is None
            assert attempt['records_complete'] == 256
            assert attempt['detector_images'] == entry['detector_images'] == 7428
            assert attempt['forward_calls'] == entry['forward_calls'] == 3842
            assert entry['parity_max_abs'] <= cfg['parity_tolerance']
            assert entry['audit']['model_optimizer_rng_modes_unchanged']
            source = load(RETURNED/arm/'fork.json') if epoch == 2 else load(RETURNED/arm/'epoch_003_validation.json')['state_audit']
            assert all(entry['audit'][k] == source[k] for k in ('model_sha256', 'rng_sha256'))
            plan = plans[str(epoch)]
            assert c.digest(plan) == entry['plan_sha256']
            assert len({p['source_cluster_id'] for p in plan}) == 256
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                panel = [json.loads(line) for line in stream]
            assert len(panel) == len(plan) == 256
            views = 0
            for p, selection in zip(panel, plan):
                views += audit_row(p, rows[p['image']], selection, cfg, epoch)
            units = c.units_from_panel(panel, rows)
            policy = entry['policy']
            old.audit_policy(units, policy)
            counts = Counter(c.key(sorted({p['classes'][i] for i in p['target_positions']})) for p in panel)
            routed = {k: {'panel_groups': counts[k], 'allowed_actions': sum(cell['allowed']),
                           'direct_supported': cell['direct_supported'], 'borrowed': cell['borrowed_evidence'],
                           'vetoed_actions': cell['vetoed_actions'],
                           'tv_vs_own_shared': sum(abs(a-b) for a,b in zip(cell['distribution'],policy['shared_distribution']))/2}
                      for k, cell in policy['routed'].items()}
            # Leave-one-source-cluster-out sensitivity is a descriptive audit, not a new gate.
            sensitivity = {}
            for label, cell in policy['pooled'].items():
                found = [u for u in units if u['class'] == int(label) and u['reference_detected']]
                fragile = set()
                for cluster in {u['cluster'] for u in found}:
                    kept = [u for u in found if u['cluster'] != cluster]
                    gate = old.independent_gate(kept)
                    fragile.update(j for j,(a,b) in enumerate(zip(cell['allowed'],gate['allowed'])) if a != b)
                sensitivity[label] = [f'{c.ACTIONS[j][0]}/{c.ACTIONS[j][1]}' for j in sorted(fragile)]
            if epoch == 2:
                first_digests[arm] = c.digest(panel)
            result[f'{arm}/epoch_{epoch}'] = {
                'status': 'pass', 'panel_images': len(panel), 'views_recomputed': views,
                'detector_images_including_parity': entry['detector_images'],
                'model_state_matches_preceding_fork_or_validation': True,
                'pooled': {k: {key: v[key] for key in ('objects','detected_objects','detected_clusters','supported','reference_recall')} |
                             {'allowed_actions': sum(v['allowed'])} for k,v in policy['pooled'].items()},
                'global_shared_actions': sum(policy['shared_allowed']), 'routed': routed,
                'pooled_actions_sensitive_to_removing_one_cluster': sensitivity,
                'controller_elapsed_seconds': entry['elapsed_seconds'], 'audit_seconds': time.monotonic()-start}
            save('controller_audit.json', {'updates': result, 'initial_panel_records_identical': first_digests.get('online_shared') == first_digests.get('class_adaptive'),
                 'scope': 'Independent matching, affine boxes and retention policy replay. Pixel hashes checked as records; raw pixels and model tensors were not available for rerendering or inference.'})
            print(f'PASS controller {arm} epoch={epoch}: {views} views, all classes supported, {sum(policy["shared_allowed"])} shared actions', flush=True)
            del panel, units
            gc.collect()
    assert len(result) == 4 and first_digests['online_shared'] == first_digests['class_adaptive']
    print('CONTROLLER AUDIT: PASS', flush=True)


if __name__ == '__main__':
    main()
