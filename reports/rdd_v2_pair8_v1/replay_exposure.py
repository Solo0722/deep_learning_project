"""Metadata-only replay of actual keyed training actions and common geometry."""
from collections import Counter
import hashlib
import json
import sys
import time

import numpy as np
from audit_returned import ROOT, PACKAGE, RETURNED, load, save, sha
sys.path.insert(0, str(PACKAGE))
import controller as c
import geometry as g
import local_augmentation as local
import shared_augmentation as shared
from audit_controller import old, geo

ARMS = ('geometry_only','class_adaptive')


def draw(distribution, epoch, image, group):
    value = int(c.digest(['v2_action',42,epoch,image,group])[:16],16)
    uniform = (value >> 11)/2**53
    total = 0.
    for j,p in enumerate(distribution):
        total += p
        if uniform < total:
            return ('identity',None) if j == 24 else c.ACTIONS[j]
    raise AssertionError('Probability mass error')


def main():
    start = time.monotonic()
    cfg = load(PACKAGE/'protocol.json')
    rows = [json.loads(x) for x in (ROOT/'detection_splits_v1/train.jsonl').read_text().splitlines()]
    histories = {a: load(RETURNED/a/'history.json') for a in ARMS}
    runs = {a: load(RETURNED/a/'run.json') for a in ARMS}
    forks = [load(RETURNED/a/'fork.json') for a in ARMS]
    assert all(f == forks[0] for f in forks) and forks[0]['model_optimizer_rng_exact']
    expected = load(PACKAGE/'exposure_plan.json')['epochs']
    policies = {a: {u['epoch']: u['policy'] for u in runs[a]['controller_updates']} for a in ARMS}
    prepared = []
    for row in rows:
        anns, points = geo.prepare(row,640)
        prepared.append((anns,points,local.groups(row,640)))
    bank = g.bank(cfg['geometry'])
    fixed_dist = old.distribution([True]*24)
    outputs = {}
    for epoch in range(2,9):
        index, zero, update = epoch-2, epoch-1, max(e for e in cfg['update_epochs'] if e <= epoch)
        pair_hashes = {histories[a][index]['augmentation']['pairing_sha256'] for a in ARMS}
        assert len(pair_hashes) == 1
        counters = {a: Counter() for a in ARMS}
        distributions = {a: Counter() for a in ARMS}
        set_counts, tv_sums, different = Counter(), Counter(), Counter()
        geometry = Counter()
        for row,(anns,points,options) in zip(rows,prepared):
            key = shared.encoded(['v2_pilot_opportunity',42,zero,row['image']])
            group = shared.choice_index(key,'group',len(options)) if options else None
            ids = list(options[group][0]) if options else []
            labels = sorted({t['class_index'] for t in row['targets'] if t['object_index'] in ids})
            opportunity = bool(options) and shared.random_word(key,'application') < 2**63
            class_key = c.key(labels)
            for i in range(1024):
                word = int(c.digest(['v2_geometry',42,zero,row['image'],i]),16)
                if word < 2**256-(2**256 % len(bank)):
                    candidate = bank[word % len(bank)]; break
            else: raise AssertionError('Keyed candidate failure')
            scale = candidate['scale']
            affine = np.array([[-scale if candidate['flip'] else scale,0,candidate['left']+(640*scale if candidate['flip'] else 0)],
                               [0,scale,candidate['top']],[0,0,1]],float)
            transformed = geo.rectangles(points @ affine.T)
            outside = ((transformed[:,:2]<0).any(1) | (transformed[:,2:]>640).any(1))
            reason = 'cropped_annotation' if outside.any() else 'accepted'
            geometry[reason] += 1
            geometry['images'] += 1
            geometry['positive_images'] += bool(row['targets'])
            geometry['target_boxes_preserved'] += len(row['targets'])
            geometry['opportunities'] += opportunity
            actions = {}
            for arm in ARMS:
                op, level, borrowed = 'identity', None, False
                why = 'probability' if options else 'no_target'
                if arm == 'geometry_only': why='geometry_control'
                elif opportunity:
                    if arm == 'fixed_local': dist = fixed_dist
                    else:
                        policy = policies[arm][update]
                        route = policy['routed'][class_key]
                        borrowed = route['borrowed_evidence']
                        dist = policy['shared_distribution'] if arm == 'online_shared' else route['distribution']
                    op,level = draw(dist,zero,row['image'],group)
                    why = 'applied' if op != 'identity' else 'no_admissible_action'
                actions[arm] = (op,level)
                counts = counters[arm]
                for scope in ('all','source/'+row['source'],*('group_class/'+str(cl) for cl in labels)):
                    keys = ['images','positive' if row['targets'] else 'negative', 'operation/'+op,'reason/'+why,'geometry/'+reason]
                    if opportunity: keys.append('opportunities')
                    if op != 'identity': keys.append('applied')
                    if borrowed and opportunity: keys.append('borrowed_opportunities')
                    counts.update(scope+'/'+k for k in keys)
                for cl in range(4):
                    n = sum(t['class_index']==cl for t in row['targets'])
                    chosen = sum(t['class_index']==cl and t['object_index'] in ids for t in row['targets'])
                    counts['target_objects_per_class/'+str(cl)] += n
                    counts['group_objects_per_class/'+str(cl)] += chosen
                    if op != 'identity': counts['edited_group_objects/'+str(cl)] += chosen
                if opportunity: distributions[arm][op+'/'+str(level)] += 1
            if opportunity:
                set_counts[class_key] += 1
                cp = policies['class_adaptive'][update]
                route = cp['routed'][class_key]
                tv_sums['own_global_shared'] += sum(abs(a-b) for a,b in zip(route['distribution'],cp['shared_distribution']))/2
                tv_sums['fixed_local'] += sum(abs(a-b) for a,b in zip(route['distribution'],fixed_dist))/2
                different['vs_own_global_shared'] += actions['class_adaptive'] != draw(cp['shared_distribution'],zero,row['image'],group)
                different['vs_fixed_local'] += actions['class_adaptive'] != draw(fixed_dist,zero,row['image'],group)
        assert dict(geometry) == expected[str(epoch)], (geometry,expected[str(epoch)])
        for arm in ARMS:
            record = histories[arm][index]
            assert record['epoch']==epoch and record['step']==epoch*6716 and record['images_seen']==epoch*26863
            assert record['train_images']==26863 and record['val_images']==5760
            if arm in ('online_shared','class_adaptive'): assert record['active_update_epoch']==update
            actual=record['augmentation']['counts']
            # Changed pixels and geometry_changed require raw images, absent in this archive.
            selected={k:v for k,v in actual.items() if not k.endswith('/changed') and not k.endswith('/geometry_changed')}
            assert dict(counters[arm]) == selected, (arm,epoch,{k:(counters[arm].get(k),v) for k,v in selected.items() if counters[arm].get(k)!=v})
            for file,digest in record['validation_records'].items(): assert sha(RETURNED/arm/file)==digest
            metric=load(RETURNED/arm/f'epoch_{epoch:03d}_validation.json')['metrics']
            assert metric==record['validation']
        outputs[str(epoch)] = {'status':'pass','geometry_and_opportunities':dict(geometry),
            'logged_pairing_sha256':next(iter(pair_hashes)), 'all_metadata_counts_match':True,
            'sampled_action_counts':{a:dict(v) for a,v in distributions.items()},'opportunities_by_class_set':dict(set_counts),
            'class_adaptive_action_disagreements':dict(different),
            'class_adaptive_opportunity_weighted_TV':{k:v/geometry['opportunities'] for k,v in tv_sums.items()},
            'borrowed_opportunities':{a:counters[a]['all/borrowed_opportunities'] for a in ARMS},
            'applied_opportunities':{a:counters[a]['all/applied'] for a in ARMS}}
        save('exposure_replay.json',{'status':'partial' if epoch<8 else 'pass','epochs':outputs,
            'identical_fork_metadata':True,'audit_seconds':time.monotonic()-start,
            'scope':'Independent keyed action sampling and all-annotation affine replay, checked against every non-pixel exposure counter. Logged pairing hashes agree. Raw pixels unavailable: pixel hashes and changed-pixel totals not rerendered.'})
        print(f'PASS exposure epoch={epoch}: {geometry["images"]} images, {geometry["opportunities"]} opportunities, both arms matched',flush=True)
    assert all(r['completed_epoch']==8 and r['step']==53728 and r['status']=='complete' for r in runs.values())
    print('EXPOSURE REPLAY: PASS',flush=True)


if __name__ == '__main__': main()
