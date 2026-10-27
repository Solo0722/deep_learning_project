"""Shared numerical measurement rules applied within one exact class set."""
import math
from shared_controller import OPERATIONS, FAMILIES, variants, view_key, family, _mean, validate_observation, digest


def _aggregate_set(panel, settings, epoch, represented):
    """Recomputable from saved object observations; never takes validation metrics."""
    if len({p['image'] for p in panel}) != len(panel):
        raise ValueError('Empty/duplicate panel')
    keys = {view_key(*v) for v in variants(settings)}
    class_objects = {c: [] for c in represented}
    for p in panel:
        n = len(p['classes'])
        if len(set(p['object_indices'])) != n or len(p['object_indices']) != n:
            raise ValueError('Duplicate/missing object identity')
        positions = p['target_positions']
        if not positions or len(set(positions)) != len(positions) or any(type(i) is not int or not 0 <= i < n for i in positions):
            raise ValueError('Invalid selected group positions')
        if any(c not in settings['classes'] for c in p['classes']) or set(p['views']) != keys:
            raise ValueError('Unexpected class or incomplete paired probe views')
        validate_observation(p['reference'], n)
        for obs in p['views'].values():
            validate_observation(obs, n)
        if tuple(sorted({p['classes'][i] for i in positions})) != tuple(represented):
            raise ValueError('Feedback belongs to a different exact class set')
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

