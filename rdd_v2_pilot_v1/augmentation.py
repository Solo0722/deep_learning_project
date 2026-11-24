"""Keyed, paired training transforms; never use or advance model RNG streams."""
from dataclasses import replace
import hashlib
import numpy as np

import controller as c
import diagnostic as d
import geometry as g
import local_augmentation as local
import shared_augmentation as shared

ARMS = ('geometry_only', 'fixed_local', 'online_shared', 'class_adaptive')


def decision(row, cfg, epoch, arm, policy):
    d.require(arm in ARMS and row['split'] == 'train', 'Training arm/partition mismatch')
    key = shared.encoded(['v2_pilot_opportunity', cfg['seed'], epoch, row['image']])
    options = local.groups(row, cfg['image_size'])
    group = shared.choice_index(key, 'group', len(options)) if options else None
    ids = list(options[group][0]) if group is not None else []
    classes = sorted({t['class_index'] for t in row['targets'] if t['object_index'] in ids})
    opportunity = bool(options) and shared.random_word(key, 'application') < int(cfg['application_probability']*2**64)
    result = {'image': row['image'], 'epoch': epoch+1, 'group_index': group, 'group_ids': ids,
              'classes': classes, 'opportunity': opportunity, 'operation': 'identity',
              'parameters': {}, 'noise_seed': None, 'borrowed': False, 'reason': 'probability' if options else 'no_target'}
    if arm == 'geometry_only':
        result['reason'] = 'geometry_control'
    elif opportunity:
        if arm == 'fixed_local':
            allowed = [True]*24
        else:
            d.require(policy is not None, 'Online arm has no current measured policy')
            route = policy['routed'][c.key(classes)]
            result['borrowed'] = route['borrowed_evidence']
            allowed = policy['shared_allowed'] if arm == 'online_shared' else route['allowed']
        action = c.draw_action(allowed, cfg['seed'], epoch, row['image'], group)
        if action is None:
            result['reason'] = 'no_admissible_action'
        else:
            op, level = action
            result.update(operation=op, level=level, parameters=d.parameters(op, level), reason='applied',
                          noise_seed=shared.random_word(key, 'noise') if op == 'gaussian_noise' else None)
    return local.attach_local(row, result, cfg['seed'], epoch, cfg['image_size'], group)


def apply(example, cfg, epoch, arm, policy):
    row = example.record
    selected = decision(row, cfg, epoch, arm, policy)
    edited = local.apply_example(example, selected)
    candidate = g.chosen_candidate(row, cfg['geometry'], cfg['seed'], epoch)
    try:
        annotations = g.prepared_annotations(row, cfg['image_size'])
    except g.InvalidAnnotation as error:
        # The loader already rejects invalid target boxes. An invalid non-target
        # disables whole-image geometry, exactly as in the audited preflight.
        rgb, reference, boxes = edited.rgb, example.rgb, example.boxes.copy()
        verdict = {'accepted': False, 'reason': 'invalid_annotation', 'cropped_ids': [], 'detail': str(error)}
    else:
        rgb, mapped, verdict = g.apply(edited.rgb, annotations, candidate)
        reference = g.apply(example.rgb, annotations, candidate)[0] if selected['operation'] != 'identity' else rgb
        boxes = np.asarray([a['box_xyxy'] for a in mapped if a['kind'] == 'target'], dtype=np.float32).reshape(-1, 4)
    d.require(boxes.shape == example.boxes.shape and np.isfinite(boxes).all()
              and (boxes[:, 2:] > boxes[:, :2]).all(), 'Lost or collapsed transformed target')
    d.require(((boxes >= 0) & (boxes <= cfg['image_size'])).all(), 'Transformed box outside canvas')
    trace = {'image': row['image'], 'source': row['source'], 'targets': len(row['targets']),
        'target_objects_per_class': {str(cl): sum(t['class_index'] == cl for t in row['targets']) for cl in range(4)},
        'group_objects_per_class': {str(cl): sum(t['class_index'] == cl and t['object_index'] in selected['group_ids'] for t in row['targets']) for cl in range(4)},
        'opportunity': selected['opportunity'], 'group_index': selected['group_index'],
        'group_ids': selected['group_ids'], 'classes': selected['classes'],
        'candidate': candidate, 'geometry_accepted': verdict['accepted'], 'geometry_reason': verdict['reason'],
        'cropped_ids': verdict['cropped_ids'], 'boxes': boxes.tolist(),
        'operation': selected['operation'], 'level': selected.get('level'), 'reason': selected['reason'],
        'borrowed': selected['borrowed'], 'pixel_changes_before_geometry': edited.augmentation_pixel_stats['changed_pixels'],
        'pixel_changes_after_geometry': int(np.any(rgb != reference, axis=2).sum()),
        'geometry_changed': not np.array_equal(reference, example.rgb),
        'input_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(reference.tobytes()).hexdigest()}
    result = replace(example, rgb=rgb, boxes=boxes)
    result.pilot_trace = trace
    return result


def empty_trace():
    return {'images': 0, 'counts': {}, 'pairing_sha256': '0'*64, 'decision_sha256': '0'*64,
            'pixel_changes_before_geometry': 0, 'pixel_changes_after_geometry': 0}


def record_trace(trace, event):
    trace['images'] += 1
    for scope in ('all', 'source/'+event['source'], *('group_class/'+str(c) for c in event['classes'])):
        keys = ['images', 'positive' if event['targets'] else 'negative', 'operation/'+event['operation'],
                'reason/'+event['reason'], 'geometry/'+event['geometry_reason']]
        if event['opportunity']: keys.append('opportunities')
        if event['operation'] != 'identity': keys.append('applied')
        if event['pixel_changes_after_geometry']: keys.append('changed')
        if event['borrowed'] and event['opportunity']: keys.append('borrowed_opportunities')
        if event['geometry_changed']: keys.append('geometry_changed')
        for key in keys:
            name = scope+'/'+key
            trace['counts'][name] = trace['counts'].get(name, 0)+1
    for key in ('pixel_changes_before_geometry', 'pixel_changes_after_geometry'):
        trace[key] += event[key]
    for cl in range(4):
        for field in ('target_objects_per_class', 'group_objects_per_class'):
            name = field+'/'+str(cl)
            trace['counts'][name] = trace['counts'].get(name, 0)+event[field][str(cl)]
        if event['operation'] != 'identity':
            name = 'edited_group_objects/'+str(cl)
            trace['counts'][name] = trace['counts'].get(name, 0)+event['group_objects_per_class'][str(cl)]
    paired = {k: event[k] for k in ('image', 'opportunity', 'group_index', 'group_ids', 'candidate',
              'geometry_accepted', 'geometry_reason', 'cropped_ids', 'boxes', 'reference_sha256')}
    for key, value in (('pairing_sha256', paired), ('decision_sha256', event)):
        trace[key] = hashlib.sha256(bytes.fromhex(trace[key])+shared.encoded(value)).hexdigest()
