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


def independent_policy(panel, settings, epoch, represented=range(4)):
    targets = [(p, i) for p in panel for i in p['target_positions']]
    outside = [(p, i) for p in panel for i in range(len(p['classes'])) if i not in p['target_positions']]
    labels = np.asarray([p['classes'][i] for p, i in targets])
    def vector(pairs, view, key):
        return np.asarray([p['reference' if view is None else 'views'][key][i] if view is None
                           else p['views'][view][key][i] for p, i in pairs], dtype=np.float64)
    reference = {k: vector(targets, None, k) for k in ('detected', 'loss', 'iou')}
    support = {}
    for c in represented:
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
                for c in represented:
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

