"""Independent array-based policy reconstruction and complete metadata replay."""
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

CLASSES = ('D00', 'D10', 'D20', 'D40')
FAMILIES = ('photometric', 'blur_noise')
OPERATIONS = ('brightness_down', 'brightness_up', 'contrast_down', 'contrast_up', 'gamma_brighten', 'gamma_darken', 'gaussian_blur', 'gaussian_noise')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def save(out, name, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    path = out/name
    if path.exists():
        require(path.read_bytes() == data, f'Preserve different existing audit output: {name}')
    else:
        path.write_bytes(data)


def close(actual, expected, path='result'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f'{path}: keys differ')
        for k in expected: close(actual[k], expected[k], path+'.'+str(k))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{path}: length differs')
        for i, (a, b) in enumerate(zip(actual, expected)): close(a, b, path+f'[{i}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and abs(actual-expected) < 1e-12, f'{path}: {actual} != {expected}')
    else:
        require(actual == expected, f'{path}: value differs')


def check_observation(obs, n, settings):
    require(all(len(obs[key]) == n for key in ('detected', 'iou', 'loss', 'roi_count')), 'Incomplete object observations')
    require(all(type(v) is bool for v in obs['detected']), 'Nonbinary detected indicator')
    require(all(math.isfinite(v) and v >= 0 for v in obs['loss']), 'Invalid loss')
    require(all(type(v) is int and v >= 1 for v in obs['roi_count']), 'Object without positive ROI observation')
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in obs['iou']), 'Invalid IoU')
    for detected, iou in zip(obs['detected'], obs['iou']):
        require(iou >= settings['iou'] if detected else iou == 0, 'Detection indicator/IoU inconsistent')
    require(type(obs['false_positives']) is int and obs['false_positives'] >= 0, 'Invalid FP count')
    require(type(obs['confident_predictions']) is int and 0 <= obs['confident_predictions'] <= 100, 'Invalid prediction count')
    require(sum(obs['detected'])+obs['false_positives'] == obs['confident_predictions'], 'One-to-one matching count mismatch')
    require(valid_hash(obs['prepared_rgb_sha256']), 'Invalid image fingerprint')


def independent_policy(panel, settings, epoch):
    targets = [(p, i) for p in panel for i in p['target_positions']]
    outside = [(p, i) for p in panel for i in range(len(p['classes'])) if i not in p['target_positions']]
    labels = np.asarray([p['classes'][i] for p, i in targets])
    def vector(pairs, view, key):
        return np.asarray([p['reference' if view is None else 'views'][key][i] if view is None
                           else p['views'][view][key][i] for p, i in pairs], dtype=np.float64)
    reference = {k: vector(targets, None, k) for k in ('detected', 'loss', 'iou')}
    support = {}
    for c in range(4):
        mask = labels == c
        detected = [(p, i) for p, i in targets if p['classes'][i] == c and p['reference']['detected'][i]]
        clusters = len({p['source_cluster_id'] for p, _ in detected})
        support[str(c)] = {'objects': int(mask.sum()), 'detected_objects': len(detected), 'detected_clusters': clusters,
                           'sufficient': len(detected) >= settings['minimum_detected_objects'] and clusters >= settings['minimum_detected_clusters']}
    mean = lambda values: float(np.mean(values)) if len(values) else None
    measurements = []
    level_table = {}
    for family in FAMILIES:
        names = [n for n in OPERATIONS if ('blur_noise' if n.startswith('gaussian_') else 'photometric') == family]
        for level in range(1, 4):
            for name in names:
                keys = [f'{name}/{level}/{d}' for d in range(settings['noise_draws'] if name == 'gaussian_noise' else 1)]
                averaged = {k: np.mean(np.stack([vector(targets, key, k) for key in keys]), axis=0) for k in reference}
                cls = {}
                for c in range(4):
                    mask = labels == c
                    den = support[str(c)]['detected_objects']
                    retention = float(np.dot(reference['detected'][mask], averaged['detected'][mask])/den) if den else None
                    before, after = mean(reference['loss'][mask]), mean(averaged['loss'][mask])
                    score = float(np.clip((after-before)/(before+settings['loss_delta']), 0, settings['difficulty_cap'])) if before is not None else None
                    cls[str(c)] = {'retention': retention, 'reference_loss': before, 'transformed_loss': after, 'difficulty': score,
                                   'passes': bool(support[str(c)]['sufficient'] and retention >= settings['retention_threshold']),
                                   'reference_recall': den/int(mask.sum()) if mask.any() else None,
                                   'transformed_recall': mean(averaged['detected'][mask]),
                                   'reference_mean_matched_iou_including_misses': mean(reference['iou'][mask]),
                                   'transformed_mean_matched_iou_including_misses': mean(averaged['iou'][mask])}
                fp = np.mean([[p['views'][key]['false_positives'] for p in panel] for key in keys], axis=0)
                fp -= np.asarray([p['reference']['false_positives'] for p in panel])
                outside_change = {}
                for k in ('detected', 'loss'):
                    v = np.mean(np.stack([vector(outside, key, k) for key in keys]), axis=0) if outside else []
                    outside_change[k] = mean(v-vector(outside, None, k)) if outside else None
                measurements.append({'family': family, 'operation': name, 'level': level, 'classes': cls,
                                     'mean_false_positive_change': mean(fp), 'outside_objects': len(outside),
                                     'outside_recall_change': outside_change['detected'], 'outside_loss_change': outside_change['loss']})
            these = [m for m in measurements if m['family'] == family and m['level'] == level]
            level_table[(family, level)] = {'admissible': all(v['passes'] for m in these for v in m['classes'].values()),
                                           'difficulty': mean([v['difficulty'] for m in these for v in m['classes'].values() if v['difficulty'] is not None])}
    families = []
    for family in FAMILIES:
        allowed = [level for level in range(1, 4) if level_table[(family, level)]['admissible']]
        if allowed:
            level = max(allowed)
            families.append({'family': family, 'level': level, 'difficulty': level_table[(family, level)]['difficulty']})
    if families:
        scores = np.array([f['difficulty'] for f in families])/settings['temperature']
        probabilities = np.exp(scores-scores.max()); probabilities /= probabilities.sum()
        probabilities = (1-settings['exploration'])*probabilities + settings['exploration']/len(families)
        for f, probability in zip(families, probabilities): f['probability'] = float(probability)
    return {'support': support, 'measurements': measurements, 'policy': {'version': settings['version'], 'epoch': epoch, 'families': families}}, level_table


def audit_controller(rows, protocol, history, run, contents, hashes, context, root, out):
    sys.path.insert(0, str(root/'fasterrcnn_online_shared_v1'))
    import controller as released
    import local_augmentation as local
    import torch
    refs = [json.loads((root/'fasterrcnn_online_shared_v1'/f'{stem}_protocol.json').read_text()) for stem in ('identity', 'shared', 'local')]
    released.verify_pairing(protocol, *refs)
    settings = protocol['controller']
    entries = run['controller_updates']
    require([e['epoch'] for e in entries] == list(range(2, 21, 2)), 'Missing/duplicate controller update')
    plan = json.loads((out/'panel_plan.json').read_text())
    require(sha(canonical(settings)) == plan['probe_settings_sha256'], 'Prospective probe settings changed')
    panels_expected = {p['epoch']: p for p in plan['epochs']}
    metadata_keys = ('image_id', 'image', 'source_cluster_id', 'group_index', 'target_positions', 'classes', 'object_indices')
    probe_names = {n for n in contents if n.startswith('controller/') and n.endswith('.json.gz')}
    cost_names = {n for n in contents if n.startswith('controller/') and n.endswith('_attempt.json')}
    require(probe_names == {e['file'] for e in entries} and len(cost_names) == len(entries), 'Unexpected missing/orphan/interrupted probe records')
    totals = Counter(); rounds = []; constraints = []; probe_pixel_counts = Counter(); probe_pixel_sums = Counter(); class_options = []
    model_hashes = set()
    for entry in entries:
        epoch = entry['epoch']; expected_step = (epoch-1)*math.ceil(len(rows)/protocol['batch_size'])
        require(hashes[entry['file']] == entry['sha256'], 'Adopted probe file hash differs')
        record = json.loads(gzip.decompress(contents[entry['file']]))
        require(record['format'] == 1 and record['epoch'] == epoch and entry['step'] == record['optimizer_steps_at_start'] == expected_step, 'Probe timing differs')
        require(record['context_sha256'] == sha(canonical(context)) and record['settings_sha256'] == sha(canonical(settings)), 'Probe data/code/settings context differs')
        panel = record['panel']; meta = [{k: p[k] for k in metadata_keys} for p in panel]
        replay_panel = released.select_panel(list(enumerate(rows)), protocol['image_size'], protocol['seed'], epoch-1, settings['panel_images'])
        require(meta == replay_panel == panels_expected[epoch]['panel'], 'Panel image/group selection differs from frozen plan/replay')
        require(sha(canonical(meta)) == record['panel_sha256'] == panels_expected[epoch]['panel_sha256'], 'Panel fingerprint differs')
        require(len(panel) == len({p['image'] for p in panel}) == 256, 'Duplicate or missing probe image')
        require(len({p['source_cluster_id'] for p in panel}) == 256, 'Probe source-cluster repetition')
        expected_keys = {f'{name}/{level}/{draw}' for name in OPERATIONS for level in range(1, 4)
                         for draw in range(settings['noise_draws'] if name == 'gaussian_noise' else 1)}
        for p in panel:
            n = len(p['classes']); check_observation(p['reference'], n, settings)
            require(set(p['views']) == expected_keys, 'Incomplete operation/strength/draw grid')
            mask_pixels = local.groups(rows[p['image_id']], protocol['image_size'])[p['group_index']][2]
            for key, obs in p['views'].items():
                check_observation(obs, n, settings)
                changed = obs['changed_pixels']
                require(type(changed) is int and 0 <= changed <= mask_pixels, 'Invalid changed-pixel count')
                require((changed == 0) == (obs['prepared_rgb_sha256'] == p['reference']['prepared_rgb_sha256']), 'Changed pixels and image fingerprints disagree')
                name, level, draw = key.split('/')
                label = name+'/'+level
                probe_pixel_counts[label+'/views'] += 1
                probe_pixel_counts[label+'/unchanged'] += changed == 0
                probe_pixel_sums[label] += changed
        calculated, level_table = independent_policy(panel, settings, epoch)
        actual = record['result']
        policy_body = {k: v for k, v in actual['policy'].items() if k != 'id'}
        require(actual['policy']['id'] == sha(canonical(policy_body)), 'Policy fingerprint differs')
        close(calculated, {**actual, 'policy': policy_body})
        require(actual['policy'] == entry['policy'] and actual['support'] == entry['support'], 'Adopted policy/support differs from observations')
        require(all(actual['support'][str(c)]['sufficient'] for c in range(4)), 'Reference support missing in this run')
        inv = record['invariants']
        require(inv['eval_inference_only'] and inv['model_buffers_gradients_rng_unchanged'], 'Recorded probe-state invariant failed')
        require(valid_hash(inv['model_state_sha256']) and valid_hash(inv['gradients_sha256']), 'Invalid invariant fingerprint')
        model_hashes.add(inv['model_state_sha256'])
        cost_name = entry['file'].removesuffix('.json.gz')+'_attempt.json'
        cost = json.loads(contents[cost_name])
        require(cost['status'] == 'complete' and cost['record'] == Path(entry['file']).name and cost['record_sha256'] == entry['sha256'], 'Probe attempt/result mismatch')
        require(cost['context_sha256'] == record['context_sha256'] and cost['panel_sha256'] == record['panel_sha256'], 'Probe attempt context differs')
        require(cost['policy_id'] == actual['policy']['id'] and cost['optimizer_steps_at_start'] == expected_step, 'Probe attempt state differs')
        require(cost['completed_panel_images'] == 256 and cost['epoch'] == epoch, 'Incomplete probe attempt')
        require(entry['probe_image_forwards'] == record['probe_image_forwards'] == cost['probe_image_forwards'] == cost['expected_probe_image_forwards'] == 7168, 'Probe computation budget differs')
        require(record['parity_image_forwards'] == cost['parity_image_forwards'] == 0, 'Unexpected extra prediction forwards')
        require(math.isfinite(cost['elapsed_seconds']) and cost['elapsed_seconds'] > 0, 'Invalid probe timing')
        totals.update({'images': 256, 'image_forwards': 7168, 'seconds': cost['elapsed_seconds']})
        active = {f['family']: f for f in entry['policy']['families']}
        by_class = {}
        for family in FAMILIES:
            selected = active.get(family, {}).get('level', 0)
            min_selected = min((v['retention'] for m in actual['measurements'] if m['family'] == family and m['level'] == selected for v in m['classes'].values()), default=None)
            by_class[family] = {}
            for c in range(4):
                passing = [level for level in range(1, 4) if all(m['classes'][str(c)]['passes'] for m in actual['measurements'] if m['family'] == family and m['level'] == level)]
                by_class[family][CLASSES[c]] = max(passing, default=0)
            for level in range(1, 4):
                failed = [{'class': CLASSES[int(c)], 'operation': m['operation'], 'retention': v['retention']}
                          for m in actual['measurements'] if m['family'] == family and m['level'] == level
                          for c, v in m['classes'].items() if not v['passes']]
                constraints.append({'epoch': epoch, 'family': family, 'level': level, 'admissible': level_table[(family, level)]['admissible'],
                                    'selected_level': selected, 'selected_minimum_retention': min_selected, 'failed_checks': failed})
        class_options.append({'epoch': epoch, 'maximum_levels_from_pooled_class_feedback': by_class})
        round_summary = {'epoch': epoch, 'policy': entry['policy'], 'support': entry['support'], 'seconds': cost['elapsed_seconds'],
                         'images': 256, 'source_clusters': 256, 'image_forwards': 7168,
                         'independent_measurement_and_policy_reconstruction': 'passed',
                         'policy_record_sha256': entry['sha256']}
        rounds.append(round_summary)
        print(f'Controller epoch {epoch}: all 24 operation/level measurements and policy independently reconstructed', flush=True)
    require(len(model_hashes) == 10 and totals['image_forwards'] == run['committed_probe_image_forwards'] == 71680, 'Unexpected repeated model or total budget')
    require(run['controller_evidence_recomputed'] and run['application_opportunities_match_fixed_controls'], 'Missing runtime audit flags')
    total_exposure = Counter(); changed_images = changed_pixels = 0; epochs = []; choices = Counter()
    current = None; by_epoch = {e['epoch']: e for e in entries}
    reference_exposure = json.loads((root/'fasterrcnn_online_shared_v1/shared_exposure.json').read_text())
    for index, h in enumerate(history):
        epoch = index+1
        if epoch in by_epoch: current = by_epoch[epoch]['policy']
        require(h['controller_policy'] == current, 'Epoch adopted wrong/future controller policy')
        generator = torch.Generator().manual_seed(protocol['seed']+1000003*index)
        order = torch.randperm(len(rows), generator=generator).tolist()
        trace = released.empty_trace()
        for i in order:
            selected = released.decision(rows[i], released.runtime_policy(protocol, current), protocol['seed'], index, protocol['image_size'])
            released.record_decision(trace, selected)
            if selected['operation'] != 'identity':
                choices[(selected['family'], selected['operation'], selected['strength_level'])] += 1
        actual = h['augmentation']
        for key in ('counts', 'decision_sha256', 'shared_decision_sha256'):
            require(actual[key] == trace[key], f'Policy/choice/order trace differs at epoch {epoch}')
        counts = actual['counts']; applied = counts.get('applied_images', 0); opportunity = counts.get('augmentation_opportunities', 0)
        require(opportunity == reference_exposure['epochs'][index]['counts'].get('applied_images', 0), 'Fixed-control opportunities differ')
        require(applied+counts.get('controller_identity_images', 0) == opportunity, 'Missing applied/identity opportunity')
        require(0 <= actual['changed_images'] <= applied and actual['changed_images'] <= actual['local_changed_pixels'] <= counts.get('local_union_pixels', 0), 'Changed-pixel accounting invalid')
        require((actual['changed_images'] == 0) == (actual['local_changed_pixels'] == 0), 'Changed-pixel/image counters inconsistent')
        if index == 0: require(applied == opportunity == 0, 'Warmup changed')
        changed_images += actual['changed_images']; changed_pixels += actual['local_changed_pixels']
        total_exposure.update(counts); epochs.append({'epoch': epoch, **actual})
        print(f'Online policy choices/order/group areas replayed: epoch {epoch:02d}', flush=True)
    require(total_exposure['applied_images'] == run['augmentation_applied_images'] == total_exposure['augmentation_opportunities'] == 158151, 'Overall exposure differs')
    require(total_exposure.get('controller_identity_images', 0) == run['controller_identity_images'] == 0, 'Identity fallback total differs')
    require(changed_pixels == run['augmentation_changed_pixels'], 'Changed-pixel total differs')
    p_photo = [f['probability'] for r in rounds if len(r['policy']['families']) == 2 for f in r['policy']['families'] if f['family'] == 'photometric']
    no_ops = [{'operation': name, 'level': level, 'probe_views': probe_pixel_counts[f'{name}/{level}/views'],
               'unchanged_probe_views': probe_pixel_counts[f'{name}/{level}/unchanged'], 'changed_pixels': probe_pixel_sums[f'{name}/{level}']}
              for name in OPERATIONS for level in range(1, 4)]
    report = {'status': 'passed_saved_observation_and_metadata_audit', 'independent_policy_reconstruction': 'all_10_updates_all_24_operation_level_cells_all_4_classes',
              'rounds': rounds, 'compute': dict(totals), 'total': dict(total_exposure),
              'reported_changed_images': changed_images, 'applied_without_pixel_change': total_exposure['applied_images']-changed_images,
              'reported_changed_pixels': changed_pixels, 'epochs': epochs,
              'policy_strength_choices': [{'epoch': r['epoch'], **{f['family']: f['level'] for f in r['policy']['families']}} for r in rounds],
              'photometric_probability_when_both_families_eligible': {'min': min(p_photo), 'max': max(p_photo), 'mean_absolute_deviation_from_half': float(np.mean(np.abs(np.asarray(p_photo)-.5)))},
              'operation_strength_training_exposure': [{'family': f, 'operation': op, 'level': level, 'images': count} for (f, op, level), count in sorted(choices.items())],
              'probe_no_change_observations': no_ops, 'constraints': constraints, 'pooled_class_strength_diagnostic': class_options,
              'aggregate_union_fraction_of_applied_content': total_exposure['local_union_pixels']/total_exposure['local_reference_pixels'],
              'aggregate_alpha_weighted_fraction_of_applied_content': total_exposure['local_alpha_units']/(2*total_exposure['local_reference_pixels']),
              'reported_changed_fraction_of_applied_content': changed_pixels/total_exposure['local_reference_pixels'],
              'limits': ['Policy arithmetic independently reconstructed from saved per-object detections/IoUs/losses, not rerun from detector checkpoints or raw images.',
                         'Raw probe detection boxes/scores and per-ROI logits are not archived; one-to-one matching and loss computation cannot be independently re-executed from these records alone.',
                         'Unchanged model/buffer/gradient/RNG state is enforced by checksum-matched executed code and recorded as passed; checkpoint tensor bytes are not in this archive.',
                         'Training changed-pixel counts are aggregate records, not fresh pixel comparisons. Probe fingerprints and no-change counts are internally checked.',
                         'Per-class pooled strength diagnostics are exploratory; they are not class-set-specific policies or evidence of their downstream efficacy.']}
    save(out, 'controller_verification.json', report)
    for name, rows_csv in [('controller_rounds.csv', [{'epoch': r['epoch'], 'seconds': r['seconds'],
                             **{f'{f}_level': next((x['level'] for x in r['policy']['families'] if x['family'] == f), 0) for f in FAMILIES},
                             **{f'{f}_probability': next((x['probability'] for x in r['policy']['families'] if x['family'] == f), 0.) for f in FAMILIES}} for r in rounds]),
                           ('operation_strength_exposure.csv', report['operation_strength_training_exposure']), ('probe_unchanged_pixels.csv', no_ops)]:
        stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=list(rows_csv[0])); writer.writeheader(); writer.writerows(rows_csv)
        save(out, name, stream.getvalue().encode())
    return {k: v for k, v in report.items() if k not in ('epochs', 'constraints', 'rounds', 'probe_no_change_observations', 'pooled_class_strength_diagnostic')}
