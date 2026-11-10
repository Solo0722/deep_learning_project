"""Reconstruct exact-set policy arithmetic and replay saved training choices.

No raw pixels, detector inference, checkpoint tensors, or held-out test data are
used. Measurement arithmetic is independent; keyed decision replay deliberately
uses the checksum-matched released decision implementation.
"""
from collections import Counter
from itertools import combinations
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

from independent_feedback import (
    CLASSES, FAMILIES, OPERATIONS, require, canonical, sha, valid_hash,
    save, close, check_observation, independent_policy,
)


def exact_sets():
    return {','.join(map(str, group)): list(group)
            for size in range(1, 5) for group in combinations(range(4), size)}


def group_key(item):
    return ','.join(map(str, sorted({item['classes'][i] for i in item['target_positions']})))


def independent_table(panel, settings, epoch):
    states, support, measurements, tables = {}, {}, {}, {}
    for key, represented in exact_sets().items():
        subset = [item for item in panel if group_key(item) == key]
        result, levels = independent_policy(subset, settings, epoch, represented)
        families = result['policy']['families']
        reason = ('unobserved_class_set' if not subset else
                  'insufficient_support' if not all(s['sufficient'] for s in result['support'].values()) else
                  'no_admissible_family' if not families else 'active')
        states[key] = {'classes': represented, 'families': families, 'reason': reason}
        support[key], measurements[key], tables[key] = result['support'], result['measurements'], levels
    return {'policy': {'version': settings['version'], 'epoch': epoch, 'class_sets': states},
            'support': support, 'measurements': measurements}, tables


def write_csv(out, name, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    save(out, name, stream.getvalue().encode())


def audit_controller(rows, protocol, history, run, contents, hashes, context, root, out):
    sys.path.insert(0, str(root / 'fasterrcnn_class_adaptive_v1'))
    import controller as released
    import local_augmentation as local
    import torch

    refs = [json.loads((root / 'fasterrcnn_class_adaptive_v1' / f'{stem}_protocol.json').read_text())
            for stem in ('identity', 'shared', 'local', 'online_shared')]
    released.verify_pairing(protocol, *refs)
    settings, entries = protocol['controller'], run['controller_updates']
    require([e['epoch'] for e in entries] == list(range(2, 21, 2)), 'Missing/duplicate update')
    plan = json.loads((out / 'panel_plan.json').read_text())
    shared_plan = json.loads((root / 'reports/fasterrcnn_online_shared_v1/panel_plan.json').read_text())
    require(sha(canonical(settings)) == plan['probe_settings_sha256'], 'Changed prospective settings')
    expected_panels = {p['epoch']: p for p in plan['epochs']}
    shared_panels = {p['epoch']: p for p in shared_plan['epochs']}
    metadata_keys = ('image_id', 'image', 'source_cluster_id', 'group_index', 'target_positions', 'classes', 'object_indices')
    probe_names = {n for n in contents if n.startswith('controller/') and n.endswith('.json.gz')}
    cost_names = {n for n in contents if n.startswith('controller/') and n.endswith('_attempt.json')}
    require(probe_names == {e['file'] for e in entries} and len(cost_names) == len(entries),
            'Missing/orphan/interrupted probe record')
    totals, model_hashes, rounds, cells, constraints = Counter(), set(), [], [], []
    probe_counts, probe_pixels = Counter(), Counter()
    same_snapshot_shared = []
    for entry in entries:
        epoch = entry['epoch']
        step = (epoch-1) * math.ceil(len(rows)/protocol['batch_size'])
        require(hashes[entry['file']] == entry['sha256'], 'Adopted probe hash differs')
        record = json.loads(gzip.decompress(contents[entry['file']]))
        require(record['format'] == 1 and record['epoch'] == epoch and
                entry['step'] == record['optimizer_steps_at_start'] == step, 'Probe timing differs')
        require(record['context_sha256'] == sha(canonical(context)) and
                record['settings_sha256'] == sha(canonical(settings)), 'Probe context differs')
        panel = record['panel']
        meta = [{key: p[key] for key in metadata_keys} for p in panel]
        replay = released.select_panel(list(enumerate(rows)), protocol['image_size'], protocol['seed'], epoch-1, settings['panel_images'])
        require(meta == replay == expected_panels[epoch]['panel'] == shared_panels[epoch]['panel'], 'Panel/group mismatch')
        require(sha(canonical(meta)) == record['panel_sha256'] == expected_panels[epoch]['panel_sha256']
                == shared_panels[epoch]['panel_sha256'], 'Panel fingerprint differs')
        require(len(panel) == len({p['image'] for p in panel}) == len({p['source_cluster_id'] for p in panel}) == 256,
                'Incomplete/duplicate panel or source clusters')
        view_keys = {f'{op}/{level}/{draw}' for op in OPERATIONS for level in range(1, 4)
                     for draw in range(settings['noise_draws'] if op == 'gaussian_noise' else 1)}
        for p in panel:
            n = len(p['classes'])
            check_observation(p['reference'], n, settings)
            require(set(p['views']) == view_keys, 'Incomplete view grid')
            area = local.groups(rows[p['image_id']], protocol['image_size'])[p['group_index']][2]
            for key, obs in p['views'].items():
                check_observation(obs, n, settings)
                changed = obs['changed_pixels']
                require(type(changed) is int and 0 <= changed <= area, 'Invalid probe changed-pixel count')
                require((changed == 0) == (obs['prepared_rgb_sha256'] == p['reference']['prepared_rgb_sha256']),
                        'Probe fingerprints disagree with changed pixels')
                op, level, draw = key.split('/')
                label = f'{op}/{level}'
                probe_counts[label+'/views'] += 1
                probe_counts[label+'/unchanged'] += changed == 0
                probe_pixels[label] += changed
        calculated, level_tables = independent_table(panel, settings, epoch)
        actual = record['result']
        body = {k: v for k, v in actual['policy'].items() if k != 'id'}
        require(actual['policy']['id'] == sha(canonical(body)), 'Policy fingerprint differs')
        close(calculated, {**actual, 'policy': body}, f'epoch{epoch}')
        require(actual['policy'] == entry['policy'] and actual['support'] == entry['support'], 'Adopted table/support differs')
        inv = record['invariants']
        require(inv['eval_inference_only'] and inv['model_buffers_gradients_rng_unchanged'], 'Failed probe invariant')
        require(valid_hash(inv['model_state_sha256']) and valid_hash(inv['gradients_sha256']), 'Invalid state fingerprint')
        model_hashes.add(inv['model_state_sha256'])
        cost = json.loads(contents[entry['file'].removesuffix('.json.gz')+'_attempt.json'])
        require(cost['status'] == 'complete' and cost['record'] == Path(entry['file']).name and
                cost['record_sha256'] == entry['sha256'], 'Probe cost/result mismatch')
        require(cost['context_sha256'] == record['context_sha256'] and cost['panel_sha256'] == record['panel_sha256'], 'Cost context mismatch')
        require(cost['policy_id'] == actual['policy']['id'] and cost['optimizer_steps_at_start'] == step and
                cost['completed_panel_images'] == 256 and cost['epoch'] == epoch, 'Incomplete/mistimed probe')
        require(entry['probe_image_forwards'] == record['probe_image_forwards'] == cost['probe_image_forwards']
                == cost['expected_probe_image_forwards'] == 7168, 'Probe budget differs')
        require(record['parity_image_forwards'] == cost['parity_image_forwards'] == 0, 'Unexpected extra views')
        require(math.isfinite(cost['elapsed_seconds']) and cost['elapsed_seconds'] > 0, 'Invalid probe time')
        totals.update({'images': 256, 'image_forwards': 7168, 'seconds': cost['elapsed_seconds']})
        for key, cell in actual['policy']['class_sets'].items():
            subset = [p for p in panel if group_key(p) == key]
            annotation = expected_panels[epoch]['class_set_annotation_support'][key]
            require(len(subset) == annotation['panel_images'], 'Exact-set annotation panel count differs')
            for c, support in actual['support'][key].items():
                require(support['objects'] == annotation['annotated_objects'][c] and
                        support['detected_clusters'] <= annotation['source_clusters'][c], 'Exact-set support differs')
            active = {f['family']: f for f in cell['families']}
            cells.append({'epoch': epoch, 'class_set': key, 'labels': '+'.join(CLASSES[c] for c in cell['classes']),
                          'panel_images': len(subset), 'reason': cell['reason'],
                          'minimum_detected_objects': min(s['detected_objects'] for s in actual['support'][key].values()),
                          'minimum_detected_clusters': min(s['detected_clusters'] for s in actual['support'][key].values()),
                          **{f'{f}_level': active.get(f, {}).get('level', 0) for f in FAMILIES},
                          **{f'{f}_probability': active.get(f, {}).get('probability', 0.) for f in FAMILIES}})
            for family in FAMILIES:
                for level in range(1, 4):
                    failed = [{'class': CLASSES[int(c)], 'operation': m['operation'], 'retention': v['retention'],
                               'support_sufficient': actual['support'][key][c]['sufficient']}
                              for m in actual['measurements'][key] if m['family'] == family and m['level'] == level
                              for c, v in m['classes'].items() if not v['passes']]
                    constraints.append({'epoch': epoch, 'class_set': key, 'family': family, 'level': level,
                                        'admissible': level_tables[key][(family, level)]['admissible'], 'failed_checks': failed})
        pooled, _ = independent_policy(panel, settings, epoch)
        same_snapshot_shared.append({'epoch': epoch, 'families': pooled['policy']['families'], 'support': pooled['support'],
                                     'scope': 'Exploratory pooled arithmetic on this class-adaptive detector snapshot; no alternate training trajectory.'})
        rounds.append({'epoch': epoch, 'policy': entry['policy'], 'support': entry['support'],
                       'seconds': cost['elapsed_seconds'], 'images': 256, 'image_forwards': 7168,
                       'reason_counts': dict(Counter(c['reason'] for c in entry['policy']['class_sets'].values())),
                       'independent_measurement_and_policy_reconstruction': 'passed', 'policy_record_sha256': entry['sha256']})
        print(f'Controller epoch {epoch}: all 15 exact-set states and all operation/level measurements reconstructed', flush=True)
    require(len(model_hashes) == 10 and totals['image_forwards'] == run['committed_probe_image_forwards'] == 71680, 'Total probe budget differs')
    require(run['controller_evidence_recomputed'] and run['application_opportunities_match_fixed_controls'], 'Missing runtime audit flags')

    reference_exposure = json.loads((root/'fasterrcnn_class_adaptive_v1/shared_exposure.json').read_text())
    total_exposure, choices, pixels_by_set = Counter(), Counter(), {}
    changed_images = changed_pixels = 0
    current, epochs = None, []
    updates = {e['epoch']: e for e in entries}
    for index, h in enumerate(history):
        epoch = index+1
        if epoch in updates:
            current = updates[epoch]['policy']
        require(h['controller_policy'] == current, 'Wrong/stale/future policy in training')
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(protocol['seed']+1000003*index)).tolist()
        trace, runtime = released.empty_trace(), released.runtime_policy(protocol, current)
        for i in order:
            choice = released.decision(rows[i], runtime, protocol['seed'], index, protocol['image_size'])
            released.record_decision(trace, choice)
            if choice['operation'] != 'identity':
                choices[(choice['class_set'], choice['family'], choice['operation'], choice['strength_level'])] += 1
        actual = h['augmentation']
        for key in ('counts', 'decision_sha256', 'shared_decision_sha256'):
            require(actual[key] == trace[key], f'Training decisions/order/geometry/counts differ: epoch {epoch}, {key}')
        counts = actual['counts']
        applied, opportunity = counts.get('applied_images', 0), counts.get('augmentation_opportunities', 0)
        require(opportunity == reference_exposure['epochs'][index]['counts'].get('applied_images', 0), 'Fixed-control opportunity mismatch')
        require(applied+counts.get('controller_identity_images', 0) == opportunity, 'Opportunity accounting differs')
        require(0 <= actual['changed_images'] <= applied and actual['changed_images'] <= actual['local_changed_pixels']
                <= counts.get('local_union_pixels', 0), 'Changed-pixel accounting invalid')
        require((actual['changed_images'] == 0) == (actual['local_changed_pixels'] == 0), 'Image/pixel counters inconsistent')
        per_set = actual['class_set_pixel_stats']
        require(sum(v['changed_images'] for v in per_set.values()) == actual['changed_images'] and
                sum(v['changed_pixels'] for v in per_set.values()) == actual['local_changed_pixels'], 'Per-set pixels do not sum')
        for key, stats in per_set.items():
            require(key in exact_sets() and 0 <= stats['changed_images'] <= counts.get(f'class_set/{key}/applied', 0)
                    and stats['changed_pixels'] >= stats['changed_images'], 'Invalid per-set changed counters')
            pixels_by_set.setdefault(key, Counter()).update(stats)
        if index == 0:
            require(applied == opportunity == 0, 'Augmented warmup')
        changed_images += actual['changed_images']
        changed_pixels += actual['local_changed_pixels']
        total_exposure.update(counts)
        epochs.append({'epoch': epoch, **actual})
        print(f'Class-set decisions/order/groups/counts replayed: epoch {epoch:02d}', flush=True)
    require(total_exposure['augmentation_opportunities'] == 158151 == plan['total_application_opportunities'], 'Total opportunities differ')
    require(total_exposure['applied_images'] == run['augmentation_applied_images'] and
            total_exposure['controller_identity_images'] == run['controller_identity_images'], 'Total applied/fallback differs')
    require(changed_pixels == run['augmentation_changed_pixels'], 'Changed-pixel total differs')
    set_exposure = []
    for key, labels in exact_sets().items():
        prefix = f'class_set/{key}/'
        n = total_exposure.get(prefix+'opportunities', 0)
        n_applied = total_exposure.get(prefix+'applied', 0)
        n_identity = total_exposure.get(prefix+'identity', 0)
        require(n_applied+n_identity == n, 'Per-set applied/identity mismatch')
        reasons = {r: total_exposure.get(prefix+'fallback/'+r, 0)
                   for r in ('unobserved_class_set', 'insufficient_support', 'no_admissible_family')}
        require(sum(reasons.values()) == n_identity, 'Per-set fallback reason total differs')
        set_exposure.append({'class_set': key, 'labels': '+'.join(CLASSES[c] for c in labels),
                             'opportunities': n, 'applied': n_applied, 'identity': n_identity,
                             'applied_fraction': n_applied/n if n else None, **reasons,
                             'changed_images': pixels_by_set.get(key, {}).get('changed_images', 0),
                             **{f'level_{level}': total_exposure.get(prefix+f'strength/{level}', 0) for level in range(1, 4)}})
    no_ops = [{'operation': op, 'level': level, 'probe_views': probe_counts[f'{op}/{level}/views'],
               'unchanged_probe_views': probe_counts[f'{op}/{level}/unchanged'], 'changed_pixels': probe_pixels[f'{op}/{level}']}
              for op in OPERATIONS for level in range(1, 4)]
    exposure = [{'class_set': s, 'family': f, 'operation': op, 'level': level, 'images': count}
                for (s, f, op, level), count in sorted(choices.items())]
    report = {
        'status': 'passed_saved_observation_and_metadata_audit',
        'independent_policy_reconstruction': 'all_10_updates_all_15_exact_sets_all_24_operation_level_cells',
        'independent_feedback_source_sha256': sha((out/'independent_feedback.py').read_bytes()),
        'same_prospective_panels_and_group_choices_as_online_shared': True,
        'rounds': rounds, 'class_set_rounds': cells, 'compute': dict(totals), 'total': dict(total_exposure),
        'reported_changed_images': changed_images, 'reported_changed_pixels': changed_pixels,
        'applied_without_pixel_change': total_exposure['applied_images']-changed_images,
        'identity_opportunity_fraction': total_exposure['controller_identity_images']/158151,
        'class_set_exposure': set_exposure, 'epochs': epochs, 'constraints': constraints,
        'operation_strength_training_exposure': exposure, 'probe_no_change_observations': no_ops,
        'same_snapshot_pooled_policy_diagnostic': same_snapshot_shared,
        'limits': [
            'Independent arithmetic uses saved per-object observations; raw probe boxes/logits, raw pixels and checkpoint tensors are unavailable in this archive.',
            'Decision replay uses released keyed randomization and grouping code; it is not an independent implementation of those algorithms.',
            'Recorded unchanged model/buffer/gradient/RNG invariants are checked, not remeasured from live detector tensors.',
            'Changed-pixel statistics are recorded counters with consistency checks, not fresh image comparisons.',
            'Pooled policies on the class-adaptive snapshots are exploratory diagnostics, not results from a second training run.'
        ],
    }
    save(out, 'controller_verification.json', report)
    write_csv(out, 'controller_class_set_rounds.csv', cells)
    write_csv(out, 'controller_class_set_exposure.csv', set_exposure)
    write_csv(out, 'operation_strength_exposure.csv', exposure)
    write_csv(out, 'probe_unchanged_pixels.csv', no_ops)
    return {k: v for k, v in report.items() if k not in
            ('epochs', 'constraints', 'rounds', 'class_set_rounds', 'probe_no_change_observations', 'same_snapshot_pooled_policy_diagnostic')}
