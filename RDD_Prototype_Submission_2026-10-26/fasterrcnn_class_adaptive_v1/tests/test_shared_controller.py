import copy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection._utils import Matcher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as b
import checkpoints
import shared_controller as c
import local_augmentation as local
import probe_detector as p
import shared_augmentation as shared
import test_baseline as fixtures


def configuration():
    return json.loads((b.PACKAGE/'online_shared_protocol.json').read_text())


def observations(settings, images=2):
    panel = []
    for i in range(images):
        obs = {'detected': [True]*4, 'iou': [.8]*4, 'loss': [1.]*4, 'roi_count': [1]*4, 'false_positives': 0}
        panel.append({'image': str(i), 'source_cluster_id': str(i), 'target_positions': [0, 1, 2, 3],
                      'object_indices': [0, 1, 2, 3], 'classes': [0, 1, 2, 3], 'reference': copy.deepcopy(obs),
                      'views': {c.view_key(*v): copy.deepcopy(obs) for v in c.variants(settings)}})
    return panel


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration()
        self.s = copy.deepcopy(self.cfg['controller'])
        self.s.update(minimum_detected_objects=2, minimum_detected_clusters=2)
        self.panel = observations(self.s)

    def test_each_class_and_operation_constrains_strongest_level(self):
        self.panel[0]['views']['brightness_down/3/0']['detected'][3] = False
        result = c.aggregate(self.panel, self.s, 1)
        families = {f['family']: f for f in result['policy']['families']}
        self.assertEqual(families['photometric']['level'], 2)
        self.assertEqual(families['blur_noise']['level'], 3)
        self.assertEqual(families['photometric']['probability'], .5)

    def test_nonmonotone_levels_are_tested_individually(self):
        for p0 in self.panel:
            p0['views']['brightness_down/1/0']['detected'][0] = False
            p0['views']['brightness_down/2/0']['detected'][0] = False
        self.assertEqual(c.aggregate(self.panel, self.s, 1)['policy']['families'][0]['level'], 3)

    def test_support_counts_detected_objects_and_distinct_clusters_not_draws(self):
        self.s['minimum_detected_objects'] = 3
        result = c.aggregate(self.panel, self.s, 1)
        self.assertFalse(result['policy']['families'])
        self.assertEqual(result['support']['0']['detected_objects'], 2)
        self.s['minimum_detected_objects'] = 2
        self.panel[1]['source_cluster_id'] = '0'
        result = c.aggregate(self.panel, self.s, 1)
        self.assertFalse(result['policy']['families'])
        self.assertEqual(result['support']['0']['detected_clusters'], 1)

    def test_absent_class_and_zero_reference_detections_fall_back_to_identity(self):
        for p0 in self.panel:
            p0['reference']['detected'] = [False]*4
        result = c.aggregate(self.panel, self.s, 1)
        self.assertFalse(result['policy']['families'])
        self.assertIsNone(result['measurements'][0]['classes']['0']['retention'])
        for p0 in self.panel:
            p0['reference']['detected'] = [True]*4
            p0['target_positions'] = [0, 1, 2]
        self.assertFalse(c.aggregate(self.panel, self.s, 1)['policy']['families'])

    def test_noise_draws_average_per_object_before_retention_and_loss(self):
        for p0 in self.panel:
            p0['views']['gaussian_noise/3/0']['detected'][0] = False
            p0['views']['gaussian_noise/3/0']['loss'][0] = 3.
        result = c.aggregate(self.panel, self.s, 1)
        measurement = next(r for r in result['measurements'] if r['operation'] == 'gaussian_noise' and r['level'] == 3)
        self.assertEqual(measurement['classes']['0']['retention'], .5)
        self.assertEqual(measurement['classes']['0']['transformed_loss'], 2.)
        self.assertEqual(result['policy']['families'][1]['level'], 2)

    def test_net_class_mean_precedes_clipping_and_classes_have_equal_weight(self):
        # One positive and one negative change cancel partly; clipping per object would be wrong.
        self.panel[0]['views']['brightness_down/3/0']['loss'][0] = 3.
        self.panel[1]['views']['brightness_down/3/0']['loss'][0] = 0.
        result = c.aggregate(self.panel, self.s, 1)
        measurement = next(r for r in result['measurements'] if r['operation'] == 'brightness_down' and r['level'] == 3)
        score = .5/1.1
        self.assertAlmostEqual(measurement['classes']['0']['difficulty'], score)
        self.assertAlmostEqual(result['policy']['families'][0]['difficulty'], score/(4*6))

    def test_exploration_probability_and_single_family(self):
        for p0 in self.panel:
            for key, obs in p0['views'].items():
                if key.startswith('brightness') or key.startswith('contrast') or key.startswith('gamma'):
                    obs['loss'] = [100.]*4
        policy = c.aggregate(self.panel, self.s, 1)['policy']
        self.assertAlmostEqual(sum(f['probability'] for f in policy['families']), 1.)
        self.assertGreaterEqual(policy['families'][1]['probability'], .1)
        self.assertGreater(policy['families'][0]['probability'], .89)
        for p0 in self.panel:
            for key, obs in p0['views'].items():
                if key.startswith('gaussian'):
                    obs['detected'] = [False]*4
        policy = c.aggregate(self.panel, self.s, 1)['policy']
        self.assertEqual(len(policy['families']), 1)
        self.assertEqual(policy['families'][0]['probability'], 1.)

    def test_duplicating_one_class_does_not_reweight_family_score(self):
        for p0 in self.panel:
            p0['views']['brightness_down/3/0']['loss'][0] = 3.
        before = c.aggregate(self.panel, self.s, 1)['policy']
        for p0 in self.panel:
            p0['classes'].extend([0]*5); p0['object_indices'].extend(range(4, 9)); p0['target_positions'].extend(range(4, 9))
            for obs in [p0['reference'], *p0['views'].values()]:
                for key in ('detected', 'iou', 'loss', 'roi_count'):
                    obs[key].extend([obs[key][0]]*5)
        self.assertEqual(c.aggregate(self.panel, self.s, 1)['policy'], before)

    def test_outside_group_feedback_does_not_drive_family_difficulty(self):
        for p0 in self.panel:
            p0['classes'].append(0); p0['object_indices'].append(9)
            for obs in [p0['reference'], *p0['views'].values()]:
                obs['detected'].append(True); obs['iou'].append(.9); obs['roi_count'].append(1); obs['loss'].append(1000.)
            p0['views']['brightness_down/3/0']['loss'][-1] = 1e8
        result = c.aggregate(self.panel, self.s, 1)
        self.assertEqual(result['policy']['families'][0]['difficulty'], 0.)
        measurement = next(r for r in result['measurements'] if r['operation'] == 'brightness_down' and r['level'] == 3)
        self.assertEqual(measurement['outside_objects'], 2)
        self.assertEqual(measurement['outside_loss_change'], 1e8-1000.)

    def test_corrupt_or_unpaired_observations_are_rejected(self):
        for change in ('missing_view', 'duplicate_object', 'nan', 'empty_assignment'):
            bad = copy.deepcopy(self.panel)
            if change == 'missing_view': bad[0]['views'].pop('gaussian_noise/3/1')
            if change == 'duplicate_object': bad[0]['object_indices'][1] = 0
            if change == 'nan': bad[0]['reference']['loss'][0] = float('nan')
            if change == 'empty_assignment': bad[0]['reference']['roi_count'][0] = 0
            with self.assertRaises(ValueError): c.aggregate(bad, self.s, 1)

    def test_configuration_guard_preserves_controls_and_middle_strength(self):
        refs = [json.loads((b.PACKAGE/f'{stem}_protocol.json').read_text()) for stem in ('identity', 'shared', 'local')]
        c.verify_pairing(self.cfg, *refs)
        for where, key, value in [(None, 'batch_size', 8), ('local_policy', 'taper_pixels', 3),
                                  ('augmentation_policy', 'probability', .6), ('controller', 'classes', [0, 1, 2])]:
            bad = copy.deepcopy(self.cfg)
            (bad[where] if where else bad)[key] = value
            with self.assertRaises(ValueError): c.verify_pairing(bad, *refs)


class DecisionsAndProbes(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BaselineTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.cfg = configuration()
        self.s = self.cfg['controller']
        self.row = self.fixture.train_rows[0]

    def test_panel_is_training_only_order_independent_and_cluster_diverse(self):
        rows = list(enumerate(self.fixture.train_rows))
        state = checkpoints.rng_state()
        a = c.select_panel(rows, 32, 42, 1, 4)
        self.assertEqual(a, c.select_panel(list(reversed(rows)), 32, 42, 1, 4))
        self.assertEqual(len({p['source_cluster_id'] for p in a}), 4)
        for entry in a:
            row = dict(rows)[entry['image_id']]
            self.assertEqual(entry['group_index'], c.group_index(row, 32, 42, 1))
        checkpoints.assert_same(checkpoints.rng_state(), state)
        with self.assertRaises(ValueError): c.select_panel(list(enumerate(self.fixture.val_rows)), 32, 42, 1, 4)
        with self.assertRaises(ValueError): c.select_panel(rows, 32, 42, 1, 5)

    def test_training_opportunities_groups_and_no_feedback_rng_match_fixed(self):
        policy = {'version': self.s['version'], 'epoch': 2, 'families': [{'family': 'blur_noise', 'level': 2, 'difficulty': 0., 'probability': 1.}]}
        policy['id'] = c.digest(policy)
        config = copy.deepcopy(self.cfg)
        runtime = c.runtime_policy(config, policy)
        state = checkpoints.rng_state()
        applied = 0
        for epoch in range(1, 100):
            choice = c.decision(self.row, runtime, 42, epoch, 32)
            fixed = local.decision(self.row, config['augmentation_policy'], 42, epoch, 32)
            self.assertEqual(choice['operation'] != 'identity', fixed['operation'] != 'identity')
            if choice['operation'] != 'identity':
                self.assertEqual(choice['local'], fixed['local'])
                self.assertEqual(choice['strength_level'], 2)
                self.assertIn(choice['operation'], ('gaussian_blur', 'gaussian_noise'))
                transformed = c.apply_example(self.fixture.train[0], choice)
                np.testing.assert_array_equal(transformed.boxes, self.fixture.train[0].boxes)
                applied += 1
        self.assertGreater(applied, 0)
        checkpoints.assert_same(checkpoints.rng_state(), state)

    def test_identity_fallback_retains_opportunity_and_validation_never_augments(self):
        cfg = copy.deepcopy(self.cfg); cfg['augmentation_policy']['probability'] = 1.
        policy = {'version': self.s['version'], 'epoch': 2, 'families': []}
        policy['id'] = c.digest(policy)
        runtime = c.runtime_policy(cfg, policy)
        choice = c.decision(self.row, runtime, 42, 1, 32)
        self.assertEqual(choice['reason'], 'controller_identity')
        self.assertIsNone(choice['local']); self.assertIsNotNone(choice['opportunity'])
        result = c.apply_example(self.fixture.train[0], choice)
        np.testing.assert_array_equal(result.rgb, self.fixture.train[0].rgb)
        trace = c.empty_trace(); c.record_decision(trace, choice)
        self.assertEqual(trace['counts']['controller_identity_images'], 1)
        val_choice = c.decision(self.fixture.val_rows[0], runtime, 42, 1, 32)
        self.assertEqual(val_choice['reason'], 'validation')
        self.assertIsNone(val_choice['opportunity'])
        with self.assertRaises(ValueError): c.decision(self.row, c.runtime_policy(cfg, None), 42, 1, 32)

    def test_probe_noise_field_is_fixed_across_strengths_and_draws_are_distinct(self):
        choices = [c.probe_choice(self.row, 0, self.s, 42, 1, 32, 'gaussian_noise', level, 0) for level in (1, 2, 3)]
        self.assertEqual(len({d['noise_seed'] for d in choices}), 1)
        other = c.probe_choice(self.row, 0, self.s, 42, 1, 32, 'gaussian_noise', 2, 1)
        self.assertNotEqual(other['noise_seed'], choices[0]['noise_seed'])

    def test_one_to_one_matching_wrong_class_duplicates_and_thresholds(self):
        pred = {'boxes': torch.tensor([[0., 0., 10., 10.]]*4), 'labels': torch.tensor([2, 1, 1, 1]), 'scores': torch.tensor([.99, .9, .8, .1])}
        result = p.match_detections(pred, torch.tensor([[0., 0., 10., 10.]]), torch.tensor([1]), .25, .5)
        self.assertEqual(result['detected'], [True]); self.assertEqual(result['false_positives'], 2)
        self.assertEqual(result['iou'], [1.])

    def test_positive_roi_assignment_includes_missed_and_duplicate_gt(self):
        class Head:
            proposal_matcher = Matcher(.5, .5, allow_low_quality_matches=False)
            box_similarity = staticmethod(__import__('torchvision').ops.box_iou)
            assign_targets_to_proposals = RoIHeads.assign_targets_to_proposals
        gt = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.], [20., 20., 30., 30.]])
        rois, assignments = p.positive_rois(Head(), [torch.tensor([[0., 0., 10., 10.], [40., 40., 50., 50.]])], [{'boxes': gt, 'labels': torch.tensor([1, 1, 2])}])
        self.assertEqual(len(rois[0]), 4)
        self.assertEqual(assignments[0].tolist(), [0, 0, 1, 2])
        rois, assignments = p.positive_rois(Head(), [torch.empty(0, 4)], [{'boxes': gt, 'labels': torch.tensor([1, 1, 2])}])
        self.assertTrue(torch.equal(rois[0], gt))
        self.assertEqual(assignments[0].tolist(), [0, 1, 2])

    def test_object_losses_average_assigned_rois_not_background_or_boxes_per_class(self):
        logits = torch.zeros(3, 5)
        regression = torch.zeros(3, 20)
        regression[0, 4:8] = 1.
        result = p.object_loss_values(logits, regression, [torch.zeros(3, 4)], [torch.tensor([1, 1, 2])], [torch.tensor([0, 0, 1])], [2])[0]
        self.assertEqual(result['roi_count'], [2, 1])
        self.assertAlmostEqual(result['loss'][0], math.log(5)+4*(1-1/18)/2, places=5)
        self.assertAlmostEqual(result['loss'][1], math.log(5), places=5)

    def test_frozen_context_restores_modes_and_detects_rng_or_parameter_change(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
        model.train(); model[0].eval()
        modes = [m.training for m in model.modules()]
        rng = checkpoints.rng_state()
        with p.frozen_probe(model):
            self.assertFalse(model.training); self.assertFalse(torch.is_grad_enabled())
            model(torch.ones(1, 2))
        self.assertEqual([m.training for m in model.modules()], modes)
        checkpoints.assert_same(checkpoints.rng_state(), rng)
        with self.assertRaises(ValueError):
            with p.frozen_probe(model): torch.rand(1)
        checkpoints.assert_same(checkpoints.rng_state(), rng)
        with self.assertRaises(ValueError):
            with p.frozen_probe(model): model[0].weight.add_(1.)


if __name__ == '__main__':
    unittest.main()
