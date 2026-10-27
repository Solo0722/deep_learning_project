import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as b
import checkpoints
import controller as c
import local_augmentation as local
import shared_controller as common
import test_baseline as fixtures


def config():
    return json.loads((b.PACKAGE/'protocol.json').read_text())


def observations(settings, labels=(0,), count=2, prefix='a'):
    result = []
    for i in range(count):
        n = len(labels)
        obs = {'detected': [True]*n, 'iou': [.9]*n, 'loss': [1.]*n,
               'roi_count': [1]*n, 'false_positives': 0}
        result.append({'image': f'{prefix}{i}', 'source_cluster_id': f'{prefix}{i}',
                       'target_positions': list(range(n)), 'object_indices': list(range(n)),
                       'classes': list(labels), 'reference': copy.deepcopy(obs),
                       'views': {c.view_key(*v): copy.deepcopy(obs) for v in c.variants(settings)}})
    return result


def resign(state):
    state['id'] = c.digest({k: v for k, v in state.items() if k != 'id'})
    return state


class ClassFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.s = copy.deepcopy(self.cfg['controller'])
        self.s.update(minimum_detected_objects=2, minimum_detected_clusters=2)

    def test_singleton_support_is_not_blocked_by_absent_unrelated_classes(self):
        result = c.aggregate(observations(self.s), self.s, 1)
        self.assertEqual(len(result['policy']['class_sets']), 15)
        self.assertEqual(result['policy']['class_sets']['0']['reason'], 'active')
        self.assertEqual([f['level'] for f in result['policy']['class_sets']['0']['families']], [3, 3])
        self.assertEqual(result['policy']['class_sets']['3']['reason'], 'unobserved_class_set')
        self.assertFalse(common.aggregate(observations(self.s), self.s, 1)['policy']['families'])

    def test_mixed_group_every_class_and_operation_constrains_strength(self):
        panel = observations(self.s, (0, 3))
        panel[0]['views']['brightness_down/3/0']['detected'][1] = False
        result = c.aggregate(panel, self.s, 1)
        families = result['policy']['class_sets']['0,3']['families']
        self.assertEqual([f['level'] for f in families], [2, 3])
        self.assertEqual([f['probability'] for f in families], [.5, .5])
        self.assertEqual(result['support']['0,3']['3']['detected_objects'], 2)

    def test_support_is_never_borrowed_between_singleton_and_mixed_sets(self):
        panel = observations(self.s, (0,), 8, 'single')+observations(self.s, (0, 3), 1, 'mixed')
        result = c.aggregate(panel, self.s, 1)
        self.assertEqual(result['policy']['class_sets']['0']['reason'], 'active')
        self.assertEqual(result['policy']['class_sets']['0,3']['reason'], 'insufficient_support')
        self.assertEqual(result['support']['0,3']['0']['detected_objects'], 1)

    def test_unrelated_set_loss_does_not_change_other_set_policy(self):
        panel = observations(self.s, (0,), prefix='crack')+observations(self.s, (3,), prefix='pothole')
        before = c.aggregate(panel, self.s, 1)['policy']['class_sets']
        for p in panel[2:]:
            for key, obs in p['views'].items():
                if not key.startswith('gaussian'): obs['loss'] = [100.]
        after = c.aggregate(panel, self.s, 1)['policy']['class_sets']
        self.assertEqual(before['0'], after['0'])
        self.assertNotEqual(before['3'], after['3'])
        self.assertGreater(after['3']['families'][0]['probability'], .89)

    def test_nonmonotone_strengths_do_not_trigger_early_stop(self):
        panel = observations(self.s)
        for p in panel:
            p['views']['brightness_down/1/0']['detected'][0] = False
            p['views']['brightness_down/2/0']['detected'][0] = False
        self.assertEqual(c.aggregate(panel, self.s, 1)['policy']['class_sets']['0']['families'][0]['level'], 3)

    def test_noise_draws_average_without_inflating_class_set_support(self):
        panel = observations(self.s, (0, 3))
        for p in panel:
            p['views']['gaussian_noise/3/0']['detected'][1] = False
            p['views']['gaussian_noise/3/0']['loss'][1] = 3.
        result = c.aggregate(panel, self.s, 1)
        m = next(m for m in result['measurements']['0,3'] if m['operation']=='gaussian_noise' and m['level']==3)
        self.assertEqual(m['classes']['3']['retention'], .5)
        self.assertEqual(m['classes']['3']['transformed_loss'], 2.)
        self.assertEqual(result['support']['0,3']['3']['detected_objects'], 2)
        panel[1]['source_cluster_id'] = panel[0]['source_cluster_id']
        result = c.aggregate(panel, self.s, 1)
        self.assertEqual(result['policy']['class_sets']['0,3']['reason'], 'insufficient_support')

    def test_class_mean_before_clipping_and_equal_class_operation_weights(self):
        panel = observations(self.s, (0, 0, 0, 3))
        panel[0]['views']['brightness_down/3/0']['loss'] = [3., 3., 3., 1.]
        panel[1]['views']['brightness_down/3/0']['loss'] = [0., 0., 0., 1.]
        result = c.aggregate(panel, self.s, 1)
        f = result['policy']['class_sets']['0,3']['families'][0]
        self.assertAlmostEqual(f['difficulty'], (.5/1.1)/(2*6))

    def test_outside_group_class_does_not_route_or_drive_policy(self):
        panel = observations(self.s, (0, 3))
        for p in panel:
            p['target_positions'] = [0]
            for obs in p['views'].values(): obs['loss'][1] = 1e8
        result = c.aggregate(panel, self.s, 1)
        self.assertEqual(result['policy']['class_sets']['0']['families'][0]['difficulty'], 0.)
        self.assertEqual(result['policy']['class_sets']['0,3']['reason'], 'unobserved_class_set')
        self.assertEqual(result['measurements']['0'][0]['outside_objects'], 2)
        self.assertEqual(result['measurements']['0'][0]['outside_loss_change'], 1e8-1)

    def test_missing_support_or_failed_gates_replace_previous_eligibility(self):
        panel = observations(self.s)
        first = c.aggregate(panel, self.s, 1)['policy']
        for p in panel:
            for obs in p['views'].values(): obs['detected'] = [False]
        second = c.aggregate(panel, self.s, 3)['policy']
        self.assertEqual(first['class_sets']['0']['reason'], 'active')
        self.assertEqual(second['class_sets']['0']['reason'], 'no_admissible_family')
        third = c.aggregate(observations(self.s, (3,)), self.s, 5)['policy']
        self.assertEqual(third['class_sets']['0']['reason'], 'unobserved_class_set')
        self.assertFalse(third['class_sets']['0']['families'])

    def test_all_four_class_set_exactly_matches_shared_numerics(self):
        panel = observations(self.s, (0, 1, 2, 3))
        panel[0]['views']['contrast_up/3/0']['loss'][1] = 2.5
        panel[0]['views']['gaussian_noise/3/1']['detected'][2] = False
        shared = common.aggregate(panel, self.s, 1)
        actual = c.aggregate(panel, self.s, 1)
        self.assertEqual(actual['policy']['class_sets']['0,1,2,3']['families'], shared['policy']['families'])
        self.assertEqual(actual['support']['0,1,2,3'], shared['support'])
        self.assertEqual(actual['measurements']['0,1,2,3'], shared['measurements'])

    def test_bad_pairing_labels_objects_and_views_are_rejected(self):
        for what in ('duplicate_image', 'duplicate_position', 'bad_label', 'missing_view', 'nan_loss', 'duplicate_object'):
            panel = observations(self.s)
            if what=='duplicate_image': panel[1]['image']=panel[0]['image']
            if what=='duplicate_position': panel[0]['target_positions']=[0, 0]
            if what=='bad_label': panel[0]['classes']=[4]
            if what=='missing_view': panel[0]['views'].pop('gaussian_noise/3/1')
            if what=='nan_loss': panel[0]['reference']['loss'][0]=float('nan')
            if what=='duplicate_object': panel[0]['object_indices']=[0,0]
            with self.assertRaises(ValueError): c.aggregate(panel, self.s, 1)


class ClassRoutingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BaselineTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.cfg = config()
        self.cfg['controller'].update(minimum_detected_objects=2, minimum_detected_clusters=2)
        self.cfg['augmentation_policy']['probability'] = 1.
        self.row = self.fixture.train_rows[0]
        self.state = c.aggregate(observations(self.cfg['controller']), self.cfg['controller'], 1)['policy']

    def test_exact_shared_draw_and_pixel_parity_when_policy_entries_match(self):
        runtime = c.runtime_policy(self.cfg, self.state)
        common_state = {'id': self.state['id'], 'epoch': self.state['epoch'], 'families': self.state['class_sets']['0']['families']}
        reference = common.runtime_policy(self.cfg, common_state)
        before = checkpoints.rng_state()
        for epoch in range(1, 11):
            a, z = c.decision(self.row, runtime, 42, epoch, 32), common.decision(self.row, reference, 42, epoch, 32)
            self.assertEqual(a['class_set'], '0')
            self.assertEqual({k:v for k,v in a.items() if k not in ('class_set','fallback_reason','opportunity')},
                             {k:v for k,v in z.items() if k!='opportunity'})
            self.assertEqual(a['opportunity']['object_indices'],z['opportunity']['object_indices'])
            np.testing.assert_array_equal(c.apply_example(self.fixture.train[0], a).rgb, common.apply_example(self.fixture.train[0], z).rgb)
        checkpoints.assert_same(checkpoints.rng_state(), before)

    def test_group_class_set_uses_full_overlap_set_not_image_majority(self):
        row = copy.deepcopy(self.row)
        other = copy.deepcopy(row['targets'][0]); other.update(object_index=100, class_index=3)
        row['targets'].append(other)
        self.assertEqual(c.selected_classes(row,32,0), [0,3])
        p = c.aggregate(observations(self.cfg['controller'], (0,3)), self.cfg['controller'], 1)['policy']
        selected = c.decision(row,c.runtime_policy(self.cfg,p),42,1,32)
        self.assertEqual(selected['class_set'],'0,3')
        self.assertEqual(selected['local']['object_indices'], [row['targets'][0]['object_index'],100])
        self.assertEqual(c.class_set_key([3,0,3]), '0,3')

    def test_fallback_preserves_group_and_counts_reason_without_resampling(self):
        state = c.aggregate(observations(self.cfg['controller'],(3,)),self.cfg['controller'],1)['policy']
        selected=c.decision(self.row,c.runtime_policy(self.cfg,state),42,1,32)
        self.assertEqual(selected['fallback_reason'],'unobserved_class_set')
        self.assertEqual(selected['opportunity']['group_index'], c.group_index(self.row,32,42,1))
        self.assertIsNone(selected['local'])
        out=c.apply_example(self.fixture.train[0],selected)
        np.testing.assert_array_equal(out.rgb,self.fixture.train[0].rgb)
        trace=c.empty_trace();c.record_decision(trace,selected,out.augmentation_changed,out.augmentation_pixel_stats)
        self.assertEqual(trace['counts']['class_set/0/fallback/unobserved_class_set'],1)
        self.assertEqual(trace['counts']['controller_identity_images'],1)

    def test_pixel_counts_are_separate_from_replayable_choice_counts(self):
        selected=c.decision(self.row,c.runtime_policy(self.cfg,self.state),42,1,32)
        out=c.apply_example(self.fixture.train[0],selected)
        a,z=c.empty_trace(),c.empty_trace()
        c.record_decision(a,selected,out.augmentation_changed,out.augmentation_pixel_stats)
        c.record_decision(z,selected)
        self.assertEqual(a['counts'],z['counts'])
        self.assertEqual(a['decision_sha256'],z['decision_sha256'])
        self.assertEqual(a['class_set_pixel_stats']['0']['changed_pixels'],out.augmentation_pixel_stats['changed_pixels'])

    def test_validation_negative_and_warmup_have_no_policy_opportunity(self):
        runtime=c.runtime_policy(self.cfg,self.state)
        for row,epoch,reason in [(self.fixture.val_rows[0],1,'validation'),(self.fixture.train_rows[-1],1,'no_target'),(self.row,0,'warmup')]:
            choice=c.decision(row,runtime,42,epoch,32)
            self.assertEqual(choice['reason'],reason)
            self.assertIsNone(choice['opportunity'])
        with self.assertRaises(ValueError): c.decision(self.row,c.runtime_policy(self.cfg,None),42,1,32)

    def test_changed_missing_future_and_invalid_policy_states_fail(self):
        for mode in ('hash','missing_set','wrong_classes','probability'):
            state=copy.deepcopy(self.state)
            if mode=='hash':state['epoch']=3
            if mode=='missing_set':state['class_sets'].pop('0,3');resign(state)
            if mode=='wrong_classes':state['class_sets']['0']['classes']=[3];resign(state)
            if mode=='probability':state['class_sets']['0']['families'][0]['probability']=float('inf')
            with self.assertRaises(ValueError):c.runtime_policy(self.cfg,state)
        state=copy.deepcopy(self.state);state['epoch']=5;resign(state)
        with self.assertRaises(ValueError):c.decision(self.row,c.runtime_policy(self.cfg,state),42,1,32)

    def test_all_matched_numeric_settings_and_panels_are_guarded(self):
        cfg=config()
        refs=[json.loads((b.PACKAGE/f'{stem}_protocol.json').read_text()) for stem in ('identity','shared','local','online_shared')]
        c.verify_pairing(cfg,*refs)
        for location,key,val in [('controller','minimum_detected_objects',7),('controller','panel_images',512),
                                 ('controller','retention_threshold',.7),('controller','noise_draws',3),
                                 ('controller','temperature',.5),('local_policy','taper_pixels',3),
                                 ('augmentation_policy','probability',.6)]:
            bad=copy.deepcopy(cfg);bad[location][key]=val
            with self.assertRaises(ValueError):c.verify_pairing(bad,*refs)
        rows=list(enumerate(self.fixture.train_rows))
        self.assertEqual(c.select_panel(rows,32,42,1,4),common.select_panel(rows,32,42,1,4))
        with self.assertRaises(ValueError):c.select_panel(list(enumerate(self.fixture.val_rows)),32,42,1,4)


if __name__=='__main__':unittest.main()
