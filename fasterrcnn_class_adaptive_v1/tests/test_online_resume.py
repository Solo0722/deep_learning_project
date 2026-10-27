import copy
import gzip
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as b
import checkpoints
import controller as c
import probe_runner as r
import test_baseline as fixtures


class OnlineResumeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BaselineTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.cfg = copy.deepcopy(self.fixture.cfg)
        production = json.loads((b.PACKAGE/'protocol.json').read_text())
        self.cfg['controller'] = production['controller']
        self.cfg['controller'].update(panel_images=4, minimum_detected_objects=1, minimum_detected_clusters=1, update_interval_epochs=1)
        self.cfg['epochs'] = 3
        self.context = {'synthetic_online': True, 'configuration': self.cfg}
        self.calls = 0
        self.observer = patch.object(r.probe_detector, 'observe', side_effect=self.observe)
        self.observer.start(); self.addCleanup(self.observer.stop)

    def observe(self, model, examples, settings, device, parity=False):
        self.calls += 1
        self.assertFalse(model.training)
        return [{'detected': [True]*len(ex.boxes), 'iou': [.8]*len(ex.boxes),
                 'loss': [1.+float(ex.rgb.mean())/255.+float(model.weight)]*len(ex.boxes),
                 'roi_count': [1]*len(ex.boxes), 'false_positives': 0} for ex in examples]

    def fit(self, out, model, optimizer, progress, stop=lambda: False):
        return b.fit(model, optimizer, self.fixture.train, self.fixture.val, self.cfg, 'cpu', out, self.context, progress, stop)

    def full(self):
        model, optimizer = self.fixture.fresh()
        out, progress = self.fixture.output('reference'), b.initial_progress()
        self.assertEqual(self.fit(out, model, optimizer, progress), 'complete')
        return checkpoints.cpu_tree(model.state_dict()), checkpoints.cpu_tree(optimizer.state_dict()), checkpoints.rng_state(), progress

    def assert_final(self, model, optimizer, progress, reference):
        checkpoints.assert_same(model.state_dict(), reference[0])
        checkpoints.assert_same(optimizer.state_dict(), reference[1])
        checkpoints.assert_same(checkpoints.rng_state(), reference[2])
        self.assertEqual([e['policy'] for e in progress['controller_updates']], [e['policy'] for e in reference[3]['controller_updates']])
        self.assertEqual([e['augmentation'] for e in progress['history']], [e['augmentation'] for e in reference[3]['history']])
        self.assertEqual(progress['step'], 9)
        self.assertEqual(progress['images_seen'], 15)

    def test_mid_epoch_resume_replays_active_policy_without_reprobing(self):
        reference = self.full()
        model, optimizer = self.fixture.fresh()
        out, progress = self.fixture.output('resume'), b.initial_progress()
        self.assertEqual(self.fit(out, model, optimizer, progress, lambda: progress['step'] >= 4), 'budget_stopped')
        self.assertEqual(len(progress['controller_updates']), 1)
        calls = self.calls
        model, optimizer = self.fixture.fresh(); progress = b.restore(out, model, optimizer, self.context)
        self.assertEqual(self.fit(out, model, optimizer, progress), 'complete')
        # Exactly one remaining update (two 2-image chunks, 28 views each).
        self.assertEqual(self.calls-calls, 56)
        self.assert_final(model, optimizer, progress, reference)

    def test_interrupted_probe_restarts_without_training_repeat_and_records_partial_cost(self):
        reference = self.full()
        model, optimizer = self.fixture.fresh()
        out, progress = self.fixture.output('probe_restart'), b.initial_progress()
        self.calls = 0
        self.assertEqual(self.fit(out, model, optimizer, progress, lambda: self.calls >= 5), 'budget_stopped')
        self.assertEqual(progress['step'], 3); self.assertFalse(progress['controller_updates'])
        attempt = json.loads(next((out/'controller').glob('*_attempt.json')).read_text())
        self.assertEqual(attempt['status'], 'budget_stopped'); self.assertEqual(attempt['probe_image_forwards'], 10)
        model, optimizer = self.fixture.fresh(); progress = b.restore(out, model, optimizer, self.context)
        self.assertEqual(self.fit(out, model, optimizer, progress), 'complete')
        self.assert_final(model, optimizer, progress, reference)

    def test_committed_boundary_policy_survives_stop_before_first_training_batch(self):
        model, optimizer = self.fixture.fresh()
        out, progress = self.fixture.output('boundary'), b.initial_progress()
        self.assertEqual(self.fit(out, model, optimizer, progress, lambda: len(progress['controller_updates']) > 0), 'budget_stopped')
        self.assertEqual(progress['next_batch'], 0); self.assertEqual(progress['step'], 3)
        calls = self.calls
        model, optimizer = self.fixture.fresh(); progress = b.restore(out, model, optimizer, self.context)
        self.assertEqual(self.fit(out, model, optimizer, progress), 'complete')
        self.assertEqual(self.calls-calls, 56)

    def test_changed_probe_records_or_policy_cannot_resume(self):
        model, optimizer = self.fixture.fresh()
        out, progress = self.fixture.output('evidence'), b.initial_progress()
        self.fit(out, model, optimizer, progress, lambda: progress['step'] >= 4)
        entries = progress['controller_updates']
        r.validate_saved_updates(out, entries, self.cfg, self.context)
        bad = copy.deepcopy(entries); bad[0]['policy']['class_sets']['0']['families'][0]['level'] = 1
        with self.assertRaises(ValueError): r.validate_saved_updates(out, bad, self.cfg, self.context)
        path = out/entries[0]['file']; path.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'evidence changed'): r.validate_saved_updates(out, entries, self.cfg, self.context)

    def test_policy_checkpoint_failure_keeps_preupdate_checkpoint_authoritative(self):
        model, optimizer = self.fixture.fresh()
        out, progress = self.fixture.output('atomic_policy'), b.initial_progress()
        real_commit = b.commit
        def commit(*args, **kwargs):
            if args[-1] == 'controller':
                raise OSError('simulated failure before controller checkpoint commit')
            return real_commit(*args, **kwargs)
        with patch.object(b, 'commit', side_effect=commit):
            with self.assertRaises(OSError): self.fit(out, model, optimizer, progress)
        model, optimizer = self.fixture.fresh(); restored = b.restore(out, model, optimizer, self.context)
        self.assertEqual(restored['epoch'], 1); self.assertEqual(restored['step'], 3)
        self.assertFalse(restored['controller_updates'])
        self.assertEqual(self.fit(out, model, optimizer, restored), 'complete')
        # Failed commit's complete evidence stays present but is not counted as an adopted update.
        self.assertEqual(len(list((out/'controller').glob('*.json.gz'))), 3)
        self.assertEqual(len(restored['controller_updates']), 2)


if __name__ == '__main__':
    unittest.main()
