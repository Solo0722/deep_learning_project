import copy
import gzip
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import augmentation as a
import baseline as b
import checkpoints
import controller as c
import diagnostic as d
import geometry as g
import observer
import pilot
import test_baseline as fixtures


def policy():
    units = [{'image': str(i), 'object_index': i, 'cluster': str(i), 'source': 'Fixture',
              'class': cl, 'class_set': (cl,), 'reference_detected': True, 'retained': [1.]*24}
             for cl in range(4) for i in range(20)]
    return c.build_policy(units, json.loads((pilot.PACKAGE/'protocol.json').read_text())['controller'])


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BaselineTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.cfg = copy.deepcopy(self.fixture.cfg)
        self.cfg.update(epochs=4, warmup_epoch=1, update_epochs=[2, 4], checkpoint_every_steps=2,
                        warmup_steps=0, lr_decay_epochs=[], application_probability=1.)
        self.cfg['geometry'].update(size=32, resized_canvas_sizes=[30, 32, 34], translations_pixels=[-1, 0, 1])
        self.cfg['controller']['size'] = 32
        self.cfg['controller'].update(minimum_detected_objects=1, minimum_detected_clusters=1)
        self.rows = self.fixture.train_rows
        self.context = {'test': True, 'cfg': self.cfg}
        self.plan = []
        for row in self.rows[:4]:
            cl = row['targets'][0]['class_index']
            self.plan.append({'image': row['image'], 'source': 'Fixture', 'source_cluster_id': row['source_cluster_id'],
                'group_index': 0, 'object_indices': [0], 'target_positions': [0], 'anchor_class': cl,
                'group_classes': [cl], 'slot': 0, 'panel': 'test'})
        self.plans = {'2': self.plan, '4': self.plan}

    def train(self, arm='class_adaptive'):
        return pilot.Dataset(self.fixture.root, self.rows, self.cfg, arm)

    def fresh(self):
        model, optimizer = self.fixture.fresh()
        # A real momentum buffer is essential to testing a warm-start fork.
        optimizer.zero_grad(); model.weight.square().backward(); optimizer.step()
        return model, optimizer

    def run_fit(self, out, model, opt, p, stop=lambda: False, arm='class_adaptive'):
        return pilot.fit(model, opt, self.train(arm), self.fixture.val, self.cfg, 'cpu', out, self.context, p, self.plans, stop)

    def reference(self):
        model, opt = self.fresh(); p = pilot.new_progress(self.cfg, len(self.rows))
        self.assertEqual(self.run_fit(self.fixture.output('full'), model, opt, p), 'complete')
        return checkpoints.cpu_tree(model.state_dict()), checkpoints.cpu_tree(opt.state_dict()), checkpoints.rng_state(), p

    def assert_final(self, model, opt, p, ref):
        checkpoints.assert_same(model.state_dict(), ref[0]); checkpoints.assert_same(opt.state_dict(), ref[1])
        checkpoints.assert_same(checkpoints.rng_state(), ref[2])
        self.assertEqual(p['step'], 12); self.assertEqual(p['images_seen'], 20)
        self.assertEqual([r['augmentation'] for r in p['history']], [r['augmentation'] for r in ref[3]['history']])
        self.assertEqual([e['policy'] for e in p['controller_updates']], [e['policy'] for e in ref[3]['controller_updates']])

    def test_four_arms_match_geometry_opportunities_and_target_labels(self):
        for i, row in enumerate(self.rows):
            original = self.fixture.train[i]
            transforms = [a.apply(original, self.cfg, 1, arm, policy()) for arm in a.ARMS]
            for x in transforms:
                self.assertTrue(np.array_equal(x.labels, original.labels))
                self.assertEqual(x.record, original.record)
                self.assertEqual(len(x.boxes), len(original.boxes))
            traces = [a.empty_trace() for _ in transforms]
            for trace, ex in zip(traces, transforms): a.record_trace(trace, ex.pilot_trace)
            self.assertEqual(len({t['pairing_sha256'] for t in traces}), 1)
        self.assertFalse(np.array_equal(transforms[0].rgb, transforms[0].rgb[::-1]))

    def test_class_policy_retains_differences_and_unsupported_routes_identity(self):
        pol = policy()
        pol['routed']['0']['allowed'] = [False]*24
        row = self.rows[0]
        self.assertEqual(a.decision(row, self.cfg, 1, 'class_adaptive', pol)['reason'], 'no_admissible_action')
        self.assertNotEqual(a.decision(row, self.cfg, 1, 'online_shared', pol)['operation'], 'identity')
        self.assertNotEqual(a.decision(row, self.cfg, 1, 'fixed_local', None)['operation'], 'identity')

    def test_keyed_transforms_do_not_consume_global_rng(self):
        before = checkpoints.rng_state()
        for arm in a.ARMS: a.apply(self.fixture.train[0], self.cfg, 2, arm, policy())
        checkpoints.assert_same(checkpoints.rng_state(), before)

    def test_negative_geometry_retains_empty_targets_and_no_appearance(self):
        ex = a.apply(self.fixture.train[4], self.cfg, 2, 'fixed_local', None)
        self.assertEqual(ex.boxes.shape, (0, 4)); self.assertEqual(ex.pilot_trace['operation'], 'identity')
        self.assertFalse(ex.pilot_trace['opportunity'])

    def test_invalid_other_annotation_disables_geometry_without_dropping_target(self):
        ex = copy.deepcopy(self.fixture.train[0])
        ex.record['other_annotations'] = [{'box': [0, 0, 0, 4], 'index': 7}]
        transformed = a.apply(ex, self.cfg, 1, 'geometry_only', None)
        self.assertEqual(transformed.pilot_trace['geometry_reason'], 'invalid_annotation')
        self.assertTrue(np.array_equal(ex.boxes, transformed.boxes))

    def test_non_target_crop_rejects_whole_geometry_and_thin_boxes_survive(self):
        ex = copy.deepcopy(self.fixture.train[0])
        ex.record['targets'][0]['box_xyxy'] = [4, 4, 20, 4.4]
        ex.boxes = np.asarray([[4, 4, 20, 4.4]], dtype=np.float32)
        ex.record['other_annotations'] = [{'box': [0, 0, 2, 3], 'index': 7}]
        candidate = dict(flip=False, resized_size=32, tx=-1, ty=0, scale=1., left=-1, top=0)
        with patch.object(g, 'chosen_candidate', return_value=candidate):
            transformed = a.apply(ex, self.cfg, 1, 'geometry_only', None)
        self.assertEqual(transformed.pilot_trace['geometry_reason'], 'cropped_annotation')
        self.assertTrue(np.array_equal(ex.rgb, transformed.rgb))
        self.assertGreater(float(transformed.boxes[0, 3]-transformed.boxes[0, 1]), 0)

    def test_fork_restores_momentum_and_all_rng_streams(self):
        model, opt = self.fresh()
        payload = {'format': 1, 'model': checkpoints.cpu_tree(model.state_dict()),
                   'optimizer': checkpoints.cpu_tree(opt.state_dict()), 'rng': checkpoints.rng_state()}
        first = pilot.fork_state(model, opt, payload)
        random.random(); np.random.rand(); torch.rand(2)
        other, optimizer = self.fresh()
        self.assertEqual(first, pilot.fork_state(other, optimizer, payload))
        self.assertTrue(optimizer.state_dict()['state'])

    def test_mid_epoch_resume_equals_uninterrupted_cpu_trajectory(self):
        ref = self.reference()
        model, opt = self.fresh(); p = pilot.new_progress(self.cfg, 5); out = self.fixture.output('resume')
        self.assertEqual(self.run_fit(out, model, opt, p, lambda: p['step'] >= 4), 'budget_stopped')
        self.assertEqual(p['next_batch'], 1)
        model, opt = self.fresh(); p = b.restore(out, model, opt, self.context)
        observer.verify_updates(out, p['controller_updates'], self.rows, self.plans, self.cfg)
        self.assertEqual(self.run_fit(out, model, opt, p), 'complete')
        self.assert_final(model, opt, p, ref)

    def test_committed_controller_is_not_measured_twice_on_resume(self):
        model, opt = self.fresh(); p = pilot.new_progress(self.cfg, 5); out = self.fixture.output('boundary')
        self.assertEqual(self.run_fit(out, model, opt, p, lambda: bool(p['controller_updates'])), 'budget_stopped')
        self.assertEqual(p['step'], 3)
        model, opt = self.fresh(); p = b.restore(out, model, opt, self.context)
        self.assertEqual(self.run_fit(out, model, opt, p), 'complete')
        self.assertEqual(len(list((out/'controller').glob('*_attempt.json'))), 2)

    def test_prevalidation_resume_does_not_repeat_training(self):
        ref = self.reference()
        model, opt = self.fresh(); p = pilot.new_progress(self.cfg, 5); out = self.fixture.output('preval')
        self.assertEqual(self.run_fit(out, model, opt, p, lambda: p['step'] >= 6), 'budget_stopped')
        model, opt = self.fresh(); p = b.restore(out, model, opt, self.context)
        self.assertEqual(self.run_fit(out, model, opt, p), 'complete')
        self.assert_final(model, opt, p, ref)

    def test_interrupted_probe_keeps_partial_evidence_and_restarts(self):
        model, opt = self.fresh(); p = pilot.new_progress(self.cfg, 5); out = self.fixture.output('partial')
        calls = [0]
        real_predict = observer.CurrentDetector.predict
        def predict(det, images):
            calls[0] += 1
            return real_predict(det, images)
        with patch.object(observer.CurrentDetector, 'predict', predict):
            self.assertEqual(self.run_fit(out, model, opt, p, lambda: calls[0] >= 4), 'budget_stopped')
        self.assertFalse(p['controller_updates']); self.assertEqual(p['step'], 3)
        self.assertEqual(json.loads(next((out/'controller').glob('*_attempt.json')).read_text())['status'], 'budget_stopped')
        model, opt = self.fresh(); p = b.restore(out, model, opt, self.context)
        self.assertEqual(self.run_fit(out, model, opt, p), 'complete')
        self.assertEqual(len(list((out/'controller').glob('*_attempt.json'))), 3)

    def test_evidence_and_policy_tampering_rejected(self):
        model, opt = self.fresh(); p = pilot.new_progress(self.cfg, 5); out = self.fixture.output('tamper')
        self.run_fit(out, model, opt, p, lambda: bool(p['controller_updates']))
        observer.verify_updates(out, p['controller_updates'], self.rows, self.plans, self.cfg)
        bad = copy.deepcopy(p['controller_updates']); bad[0]['policy']['shared_allowed'][0] = not bad[0]['policy']['shared_allowed'][0]
        with self.assertRaisesRegex(ValueError, 'policy'):
            observer.verify_updates(out, bad, self.rows, self.plans, self.cfg)
        (out/p['controller_updates'][0]['file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            observer.verify_updates(out, p['controller_updates'], self.rows, self.plans, self.cfg)

    def test_observer_restores_modes_even_on_error_and_detects_rng_use(self):
        model, opt = self.fresh(); model.train()
        state = checkpoints.rng_state()
        with self.assertRaisesRegex(ValueError, 'consumed training RNG'):
            with observer.frozen(model, opt): torch.rand(1)
        self.assertTrue(model.training); checkpoints.assert_same(checkpoints.rng_state(), state)

    def test_saved_validation_corruption_is_rejected(self):
        model, opt = self.fresh(); p = pilot.new_progress(self.cfg, 5); out = self.fixture.output('validation_tamper')
        self.run_fit(out, model, opt, p, lambda: p['epoch'] >= 2, arm='geometry_only')
        pilot.verify_validations(out, p['history'])
        (out/'epoch_002_val_predictions.json.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Validation evidence changed'):
            pilot.verify_validations(out, p['history'])

    def test_empty_admissible_set_is_identity_not_a_redrawn_action(self):
        pol = policy()
        pol['shared_allowed'] = [False]*24
        selected = a.decision(self.rows[0], self.cfg, 1, 'online_shared', pol)
        self.assertTrue(selected['opportunity'])
        self.assertEqual(selected['operation'], 'identity')
        self.assertEqual(selected['reason'], 'no_admissible_action')
        self.assertIsNone(selected['local'])

    def test_exposure_guard_rejects_all_arms_sharing_a_wrong_draw(self):
        trace = a.empty_trace()
        for ex in [a.apply(self.fixture.train[i], self.cfg, 1, 'geometry_only', None) for i in range(5)]:
            a.record_trace(trace, ex.pilot_trace)
        expected = {'images': 5, 'positive_images': 4, 'opportunities': 4,
                    'accepted': trace['counts'].get('all/geometry/accepted', 0),
                    'cropped_annotation': trace['counts'].get('all/geometry/cropped_annotation', 0), 'target_boxes_preserved': 4}
        pilot.verify_exposure(trace, expected)
        wrong = copy.deepcopy(trace); wrong['counts']['all/opportunities'] -= 1
        with self.assertRaisesRegex(ValueError, 'exposure differs'):
            pilot.verify_exposure(wrong, expected)

    def test_mismatched_pairing_and_output_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            pilot.check_output(self.fixture.root, [self.fixture.root/'input'])
        h = {arm: [{'epoch': e, 'augmentation': {'pairing_sha256': arm}} for e in [2, 3, 4]] for arm in a.ARMS}
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            pilot.paired_summary(h, self.cfg)

    def test_failed_controller_commit_leaves_preprobe_checkpoint_authoritative(self):
        model, opt = self.fresh(); p = pilot.new_progress(self.cfg, 5); out = self.fixture.output('commit')
        commit = b.commit
        def fail(*args):
            if args[-1] == 'controller': raise OSError('injected checkpoint failure')
            return commit(*args)
        with patch.object(b, 'commit', fail), self.assertRaises(OSError): self.run_fit(out, model, opt, p)
        model, opt = self.fresh(); restored = b.restore(out, model, opt, self.context)
        self.assertFalse(restored['controller_updates']); self.assertEqual(restored['step'], 3)


if __name__ == '__main__': unittest.main()
