"""Training-only shared policy; paired object feedback, joint gates, keyed draws."""
import copy
import hashlib
import math

import local_augmentation as local
import shared_augmentation as shared

OPERATIONS = shared.OPERATIONS
FAMILIES = shared.FAMILIES
apply_example = local.apply_example


def digest(value):
    return hashlib.sha256(shared.encoded(value)).hexdigest()


def family(operation):
    return 'blur_noise' if operation.startswith('gaussian_') else 'photometric'


def variants(settings):
    return [(name, level, draw) for name in OPERATIONS
            for level in range(1, len(settings['levels'])+1)
            for draw in range(settings['noise_draws'] if name == 'gaussian_noise' else 1)]


def view_key(name, level, draw):
    return f'{name}/{level}/{draw}'


def group_index(row, size, seed, epoch):
    options = local.groups(row, size)
    if row['split'] != 'train' or not options:
        raise ValueError('Probe requires target-bearing training rows')
    key = shared.encoded([local.LOCAL_VERSION, seed, epoch, row['image']])
    return shared.choice_index(key, 'group', len(options))


def select_panel(rows, size, seed, epoch, count):
    if any(row['split'] != 'train' for _, row in rows):
        raise ValueError('Held-out data cannot enter probe selection')
    eligible = [(i, row) for i, row in rows if row['targets']]
    if len({r['image'] for _, r in rows}) != len(rows):
        raise ValueError('Duplicate probe source image')
    ranked = sorted(eligible, key=lambda item: (digest(['online_probe_panel_v1', seed, epoch, item[1]['image']]), item[1]['image']))
    primary, extra, clusters = [], [], set()
    for item in ranked:
        cluster = item[1]['source_cluster_id']
        (extra if cluster in clusters else primary).append(item)
        clusters.add(cluster)
    selected = (primary + extra)[:count]
    if len(selected) != count:
        raise ValueError('Insufficient positive training images for declared panel')
    result = []
    for image_id, row in selected:
        index = group_index(row, size, seed, epoch)
        ids = local.groups(row, size)[index][0]
        result.append({'image_id': image_id, 'image': row['image'], 'source_cluster_id': row['source_cluster_id'],
                       'group_index': index, 'target_positions': [j for j, t in enumerate(row['targets']) if t['object_index'] in ids],
                       'classes': [t['class_index'] for t in row['targets']],
                       'object_indices': [t['object_index'] for t in row['targets']]})
    return result


def probe_choice(row, group, settings, seed, epoch, size, name, level, draw):
    # Intentionally omit strength from this seed: identical standard-normal fields across levels.
    noise_seed = shared.random_word(shared.encoded(['online_probe_noise_v1', seed, epoch, row['image'], group, name, draw]), 'noise') if name == 'gaussian_noise' else None
    selected = {'image': row['image'], 'epoch': epoch+1, 'target_bearing': True,
                'operation': name, 'family': family(name), 'parameters': copy.deepcopy(settings['levels'][level-1][name]),
                'noise_seed': noise_seed, 'reason': 'probe', 'strength_level': level}
    return local.attach_local(row, selected, seed, epoch, size, group)


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def validate_observation(obs, count):
    for key in ('detected', 'iou', 'loss', 'roi_count'):
        if len(obs[key]) != count:
            raise ValueError('Object observation omitted/reordered targets')
    if any(type(v) is not bool for v in obs['detected']):
        raise ValueError('Invalid detection indicator')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in obs['iou']):
        raise ValueError('Invalid matched IoU')
    if any(not math.isfinite(v) or v < 0 for v in obs['loss']):
        raise ValueError('Nonfinite or negative object loss')
    if any(type(v) is not int or v < 1 for v in obs['roi_count']):
        raise ValueError('Every annotated object needs an assigned ROI')
    if type(obs['false_positives']) is not int or obs['false_positives'] < 0:
        raise ValueError('Invalid false-positive count')


def aggregate(panel, settings, epoch):
    """Recomputable from saved object observations; never takes validation metrics."""
    if not panel or len({p['image'] for p in panel}) != len(panel):
        raise ValueError('Empty/duplicate panel')
    keys = {view_key(*v) for v in variants(settings)}
    class_objects = {c: [] for c in settings['classes']}
    for p in panel:
        n = len(p['classes'])
        if len(set(p['object_indices'])) != n or len(p['object_indices']) != n:
            raise ValueError('Duplicate/missing object identity')
        positions = p['target_positions']
        if not positions or len(set(positions)) != len(positions) or any(type(i) is not int or not 0 <= i < n for i in positions):
            raise ValueError('Invalid selected group positions')
        if any(c not in class_objects for c in p['classes']) or set(p['views']) != keys:
            raise ValueError('Unexpected class or incomplete paired probe views')
        validate_observation(p['reference'], n)
        for obs in p['views'].values():
            validate_observation(obs, n)
        for index in positions:
            class_objects[p['classes'][index]].append((p, index))
    support, reference_means = {}, {}
    for c, objects in class_objects.items():
        detected = [(p, i) for p, i in objects if p['reference']['detected'][i]]
        clusters = len({p['source_cluster_id'] for p, _ in detected})
        support[str(c)] = {'objects': len(objects), 'detected_objects': len(detected), 'detected_clusters': clusters,
                           'sufficient': len(detected) >= settings['minimum_detected_objects'] and clusters >= settings['minimum_detected_clusters']}
        reference_means[c] = _mean([p['reference']['loss'][i] for p, i in objects])
    measurements, eligible = [], []
    for f in FAMILIES:
        names = [name for name in OPERATIONS if family(name) == f]
        levels = []
        for level in range(1, len(settings['levels'])+1):
            admissible, scores = True, []
            for name in names:
                draws = range(settings['noise_draws'] if name == 'gaussian_noise' else 1)
                class_values = {}
                for c, objects in class_objects.items():
                    averaged = [(p, i, {k: _mean([p['views'][view_key(name, level, d)][k][i] for d in draws])
                                             for k in ('detected', 'loss', 'iou')}) for p, i in objects]
                    denom = support[str(c)]['detected_objects']
                    retained = math.fsum(v['detected'] for p, i, v in averaged if p['reference']['detected'][i])
                    retention = retained/denom if denom else None
                    before = reference_means[c]
                    after = _mean([v['loss'] for _, _, v in averaged])
                    score = min(settings['difficulty_cap'], max(0., (after-before)/(before+settings['loss_delta']))) if before is not None else None
                    allowed = support[str(c)]['sufficient'] and retention >= settings['retention_threshold']
                    admissible = admissible and allowed
                    if score is not None:
                        scores.append(score)
                    class_values[str(c)] = {'retention': retention, 'reference_loss': before, 'transformed_loss': after,
                                           'difficulty': score, 'passes': allowed,
                                           'reference_recall': denom/len(objects) if objects else None,
                                           'transformed_recall': _mean([v['detected'] for _, _, v in averaged]),
                                           'reference_mean_matched_iou_including_misses': _mean([p['reference']['iou'][i] for p, i in objects]),
                                           'transformed_mean_matched_iou_including_misses': _mean([v['iou'] for _, _, v in averaged])}
                outside = [(p, i) for p in panel for i in range(len(p['classes'])) if i not in p['target_positions']]
                measurements.append({'family': f, 'operation': name, 'level': level, 'classes': class_values,
                                     'mean_false_positive_change': _mean([_mean([p['views'][view_key(name, level, d)]['false_positives'] for d in draws])-p['reference']['false_positives'] for p in panel]),
                                     'outside_objects': len(outside),
                                     'outside_recall_change': _mean([_mean([p['views'][view_key(name, level, d)]['detected'][i] for d in draws])-p['reference']['detected'][i] for p, i in outside]),
                                     'outside_loss_change': _mean([_mean([p['views'][view_key(name, level, d)]['loss'][i] for d in draws])-p['reference']['loss'][i] for p, i in outside])})
            levels.append({'level': level, 'admissible': admissible, 'difficulty': _mean(scores)})
        allowed = [v for v in levels if v['admissible']]
        if allowed:
            chosen = max(allowed, key=lambda v: v['level'])
            eligible.append({'family': f, 'level': chosen['level'], 'difficulty': chosen['difficulty']})
    if eligible:
        top = max(v['difficulty'] for v in eligible)
        weights = [math.exp((v['difficulty']-top)/settings['temperature']) for v in eligible]
        total = math.fsum(weights)
        for item, weight in zip(eligible, weights):
            item['probability'] = (1-settings['exploration'])*weight/total + settings['exploration']/len(eligible)
    policy = {'version': settings['version'], 'epoch': epoch+1, 'families': eligible}
    policy['id'] = digest(policy)
    return {'policy': policy, 'support': support, 'measurements': measurements}


def decision(row, policy, seed, epoch, size=640):
    # Original fixtures and fixed-bank preview use the preserved fixed-policy path.
    if policy is None or '_online' not in policy:
        return local.decision(row, policy, seed, epoch, size)
    result = shared.decision(row, policy, seed, epoch)
    result['local'] = None
    result['policy_id'] = None
    result['opportunity'] = None
    if result['operation'] == 'identity':
        return result
    bundle = policy['_online']
    current = bundle['state']
    if current is None or current['epoch'] > epoch+1:
        raise ValueError('Missing or future online policy')
    result['policy_id'] = current['id']
    index = group_index(row, size, seed, epoch)
    result['opportunity'] = {'group_index': index, 'object_indices': list(local.groups(row, size)[index][0])}
    entries = current['families']
    if not entries:
        result.update(operation='identity', family='identity', parameters={}, noise_seed=None, reason='controller_identity', strength_level=0)
        return result
    key = shared.encoded(['online_shared_training_v1', seed, epoch, row['image']])
    value = shared.random_word(key, 'family')/2**64
    cumulative = 0.
    chosen = entries[-1]
    for item in entries:
        cumulative += item['probability']
        if value < cumulative:
            chosen = item
            break
    names = [name for name in OPERATIONS if family(name) == chosen['family']]
    name = names[shared.choice_index(key, 'operation', len(names))]
    result.update(operation=name, family=chosen['family'], strength_level=chosen['level'], reason='applied',
                  parameters=copy.deepcopy(bundle['settings']['levels'][chosen['level']-1][name]),
                  noise_seed=shared.random_word(key, 'noise') if name == 'gaussian_noise' else None)
    return local.attach_local(row, result, seed, epoch, size, index)


def runtime_policy(cfg, state):
    if 'controller' not in cfg:
        return cfg.get('augmentation_policy')
    return {**cfg['augmentation_policy'], '_online': {'settings': cfg['controller'], 'state': state}}


def empty_trace():
    return local.empty_trace()


def record_decision(trace, selected, changed=False, pixel_stats=None):
    local.record_decision(trace, selected, changed, pixel_stats)
    if selected.get('opportunity'):
        counts = trace['counts']
        counts['augmentation_opportunities'] = counts.get('augmentation_opportunities', 0)+1
        if selected['operation'] == 'identity':
            counts['controller_identity_images'] = counts.get('controller_identity_images', 0)+1
        else:
            key = f"strength_level_{selected['strength_level']}"
            counts[key] = counts.get(key, 0)+1


def verify_pairing(cfg, identity, image_level, fixed_local):
    allowed = {'augmentation', 'augmentation_policy', 'local_policy', 'version', 'purpose', 'recipe_status', 'comparator', 'sources', 'controller'}
    strip = lambda c: {k: v for k, v in c.items() if k not in allowed}
    if any(strip(cfg) != strip(c) for c in (identity, image_level, fixed_local)):
        raise ValueError('Detector training/evaluation budget changed')
    # These two descriptions change because the fixed reference bank is now selected online.
    comparable = lambda p: {k: v for k, v in p.items() if k not in ('schedule', 'family_sampling')}
    if comparable(cfg['augmentation_policy']) != comparable(fixed_local['augmentation_policy']) or cfg['local_policy'] != fixed_local['local_policy']:
        raise ValueError('Application/group/mask or fixed strength reference changed')
    s = cfg['controller']
    if cfg['augmentation'] != 'online_shared_local' or s['classes'] != [0, 1, 2, 3] or s['condition'] != 'shared_across_all_group_class_sets':
        raise ValueError('Wrong shared controller condition')
    for key in ('panel_images', 'probe_batch_size', 'noise_draws', 'minimum_detected_objects', 'minimum_detected_clusters', 'update_interval_epochs'):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError('Invalid controller count')
    for key in ('confidence', 'iou', 'retention_threshold', 'exploration'):
        if not 0 < s[key] <= 1:
            raise ValueError('Invalid controller threshold')
    if any(not math.isfinite(s[k]) or s[k] <= 0 for k in ('loss_delta', 'difficulty_cap', 'temperature')):
        raise ValueError('Invalid difficulty scale')
    if len(s['levels']) != 3:
        raise ValueError('Expected prospective three-level grid')
    for level in s['levels']:
        p = copy.deepcopy(cfg['augmentation_policy'])
        p['operations'] = {name: {'family': family(name), **params} for name, params in level.items()}
        shared.validate_policy(p)
    if s['levels'][1] != {name: {k: v for k, v in p.items() if k != 'family'} for name, p in fixed_local['augmentation_policy']['operations'].items()}:
        raise ValueError('Middle strength must equal fixed comparator')
