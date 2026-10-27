"""Training-only exact-class-set policies using the matched shared probe budget."""
import copy
from itertools import combinations
import math

import shared_controller as common
import local_augmentation as local
import shared_augmentation as shared
from class_set_feedback import _aggregate_set

OPERATIONS, FAMILIES = common.OPERATIONS, common.FAMILIES
digest, family, variants, view_key = common.digest, common.family, common.variants, common.view_key
group_index, select_panel, probe_choice = common.group_index, common.select_panel, common.probe_choice
apply_example = common.apply_example


def class_set_key(classes):
    labels = tuple(sorted(set(classes)))
    if not labels or any(type(c) is not int or c not in (0, 1, 2, 3) for c in labels):
        raise ValueError('Invalid group class set')
    return ','.join(map(str, labels))


def all_class_sets(settings):
    if settings['classes'] != [0, 1, 2, 3]:
        raise ValueError('Expected the four frozen RDD classes')
    return {class_set_key(s): list(s) for n in range(1, 5) for s in combinations(settings['classes'], n)}


def selected_classes(row, size, index):
    ids = set(local.groups(row, size)[index][0])
    return sorted({t['class_index'] for t in row['targets'] if t['object_index'] in ids})


def panel_class_set(item):
    positions = item['target_positions']
    labels = item['classes']
    if not positions or len(positions) != len(set(positions)) or any(type(i) is not int or not 0 <= i < len(labels) for i in positions):
        raise ValueError('Invalid selected group positions')
    return class_set_key(labels[i] for i in positions)


def aggregate(panel, settings, epoch):
    """Pool feedback only within exact class sets; rebuild all states each update."""
    if not panel or len({p['image'] for p in panel}) != len(panel):
        raise ValueError('Empty/duplicate panel')
    sets = all_class_sets(settings)
    buckets = {key: [] for key in sets}
    for item in panel:
        buckets[panel_class_set(item)].append(item)
    states, support, measurements = {}, {}, {}
    for key, labels in sets.items():
        result = _aggregate_set(buckets[key], settings, epoch, labels)
        families = result['policy']['families']
        if not buckets[key]:
            reason = 'unobserved_class_set'
        elif not all(s['sufficient'] for s in result['support'].values()):
            reason = 'insufficient_support'
        elif not families:
            reason = 'no_admissible_family'
        else:
            reason = 'active'
        states[key] = {'classes': labels, 'families': families, 'reason': reason}
        support[key] = result['support']
        measurements[key] = result['measurements']
    policy = {'version': settings['version'], 'epoch': epoch+1, 'class_sets': states}
    policy['id'] = digest(policy)
    validate_policy(policy, settings)
    return {'policy': policy, 'support': support, 'measurements': measurements}


def validate_policy(state, settings):
    if state is None:
        return
    if state.get('version') != settings['version'] or type(state.get('epoch')) is not int or state['epoch'] < 1:
        raise ValueError('Policy version/epoch differs')
    if set(state) != {'version', 'epoch', 'class_sets', 'id'} or state['id'] != digest({k: v for k, v in state.items() if k != 'id'}):
        raise ValueError('Policy fingerprint/schema differs')
    expected = all_class_sets(settings)
    if set(state['class_sets']) != set(expected):
        raise ValueError('Incomplete class-set policy table')
    reasons = {'active', 'unobserved_class_set', 'insufficient_support', 'no_admissible_family'}
    for key, cell in state['class_sets'].items():
        if set(cell) != {'classes', 'families', 'reason'} or cell['classes'] != expected[key] or cell['reason'] not in reasons:
            raise ValueError('Wrong class-set policy routing')
        families = cell['families']
        if bool(families) != (cell['reason'] == 'active'):
            raise ValueError('Active/identity reason disagrees with families')
        if len(families) != len({f['family'] for f in families}) or any(f['family'] not in FAMILIES for f in families):
            raise ValueError('Invalid eligible families')
        for f in families:
            if type(f['level']) is not int or not 1 <= f['level'] <= len(settings['levels']):
                raise ValueError('Invalid strength level')
            if not math.isfinite(f['difficulty']) or not 0 <= f['difficulty'] <= settings['difficulty_cap']:
                raise ValueError('Invalid family difficulty')
            if not math.isfinite(f['probability']) or not 0 < f['probability'] <= 1:
                raise ValueError('Invalid family probability')
        if families and abs(math.fsum(f['probability'] for f in families)-1) > 1e-12:
            raise ValueError('Family probabilities do not sum to one')


def runtime_policy(cfg, state):
    if 'controller' not in cfg:
        return cfg.get('augmentation_policy')
    validate_policy(state, cfg['controller'])
    return common.runtime_policy(cfg, state)


def decision(row, policy, seed, epoch, size=640):
    if policy is None or '_online' not in policy:
        return local.decision(row, policy, seed, epoch, size)
    base = shared.decision(row, policy, seed, epoch)
    if base['operation'] == 'identity':
        base.update(local=None, policy_id=None, opportunity=None, class_set=None, fallback_reason=None)
        return base
    current = policy['_online']['state']
    if current is None or current['epoch'] > epoch+1:
        raise ValueError('Missing or future class-set policy')
    index = group_index(row, size, seed, epoch)
    key = class_set_key(selected_classes(row, size, index))
    cell = current['class_sets'][key]
    # Identical application/group/family/operation/noise random keys to the shared run.
    # Only the selected set's eligible families/strengths/probabilities differ.
    projected = {**policy, '_online': {**policy['_online'], 'state': {
        'id': current['id'], 'epoch': current['epoch'], 'families': cell['families']}}}
    result = common.decision(row, projected, seed, epoch, size)
    result['class_set'] = key
    result['fallback_reason'] = cell['reason'] if result['operation'] == 'identity' else None
    result['opportunity']['classes'] = cell['classes'].copy()
    return result


def empty_trace():
    return {**common.empty_trace(), 'class_set_pixel_stats': {}}


def record_decision(trace, selected, changed=False, pixel_stats=None):
    common.record_decision(trace, selected, changed, pixel_stats)
    if not selected.get('opportunity'):
        return
    key = selected['class_set']
    counts = trace['counts']
    prefix = f'class_set/{key}/'
    def add(name): counts[prefix+name] = counts.get(prefix+name, 0)+1
    add('opportunities')
    if selected['operation'] == 'identity':
        add('identity'); add('fallback/'+selected['fallback_reason'])
    else:
        add('applied'); add('family/'+selected['family']); add(f"strength/{selected['strength_level']}")
        if pixel_stats is not None:
            stats = trace['class_set_pixel_stats'].setdefault(key, {'changed_images': 0, 'changed_pixels': 0})
            stats['changed_images'] += int(changed)
            stats['changed_pixels'] += pixel_stats['changed_pixels']


def policy_summary(state):
    return {key: {'reason': cell['reason'], 'families': {
        f['family']: {'level': f['level'], 'p': round(f['probability'], 6)} for f in cell['families']}}
        for key, cell in state['class_sets'].items()}


def verify_pairing(cfg, identity, image_level, fixed_local, online_shared):
    # Reuse all original checks through a shared-shaped view, then forbid extra changes.
    if (cfg['augmentation'] != 'online_class_adaptive_local' or cfg['controller']['condition'] != 'exact_group_class_set'
            or cfg['controller']['version'] != 'online_class_adaptive_local_v1'):
        raise ValueError('Wrong class-adaptive condition')
    comparable = copy.deepcopy(cfg)
    comparable['augmentation'] = online_shared['augmentation']
    comparable['controller']['condition'] = online_shared['controller']['condition']
    common.verify_pairing(comparable, identity, image_level, fixed_local)
    ignored = {'augmentation', 'augmentation_policy', 'version', 'purpose', 'recipe_status', 'comparator', 'sources', 'controller'}
    strip = lambda v: {k: x for k, x in v.items() if k not in ignored}
    if strip(cfg) != strip(online_shared):
        raise ValueError('Non-controller settings differ from online shared')
    descriptive = {'version', 'condition', 'interpretation', 'support'}
    strip_settings = lambda s: {k: v for k, v in s.items() if k not in descriptive}
    if strip_settings(cfg['controller']) != strip_settings(online_shared['controller']):
        raise ValueError('Probe observer/grid/budget/thresholds differ from online shared')
    strip_policy = lambda s: {k: v for k, v in s.items() if k not in {'schedule','family_sampling'}}
    if strip_policy(cfg['augmentation_policy']) != strip_policy(online_shared['augmentation_policy']):
        raise ValueError('Application policy differs from online shared')
