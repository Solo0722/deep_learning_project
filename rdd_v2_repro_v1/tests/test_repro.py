import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repro as r
import trace_state as ts


class SmallDetector(torch.nn.Module):
    """Stochastic differentiable fixture with a changing training buffer."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.25, -.1]))
        self.register_buffer('batches_seen', torch.tensor(0))

    def forward(self, images, targets=None):
        if not self.training:
            return self.weight.square().detach()
        self.batches_seen.add_(1)
        jitter = torch.rand(()) + random.random() + float(np.random.rand())
        value = sum(im.mean() for im in images) + sum(t['boxes'].sum() for t in targets)/100
        return {'loss_classifier': (self.weight-jitter-value).square().sum() + .01}


class StateTests(unittest.TestCase):
    def test_compare_reports_known_difference_and_values(self):
        left = {'x': torch.tensor([1., 2.])}
        right = {'x': torch.tensor([1., 3.])}
        result, pair = ts.compare(left, right)
        self.assertFalse(result['equal'])
        item = result['changes'][0]
        self.assertEqual(item['max_abs'], 1.)
        self.assertAlmostEqual(item['relative_l2'], 1/np.sqrt(5))
        self.assertEqual(item['numerically_changed_elements'], 1)
        self.assertEqual(item['values'][0], {'flat_index': 1, 'expected': 2., 'actual': 3.})
        self.assertTrue(torch.equal(pair[1], left['x']))

    def test_structure_dtype_empty_and_signed_zero_are_distinguished(self):
        for left, right in [({'x': torch.ones(2)}, {'x': torch.ones(1)}),
                            ({'x': torch.ones(2)}, {'x': torch.ones(2).double()}),
                            ({'x': []}, {'x': ()}), ({'x': 1}, {'y': 1}),
                            (torch.tensor([0.]), torch.tensor([-0.]))]:
            self.assertFalse(ts.compare(left, right)[0]['equal'])
        self.assertTrue(ts.compare(torch.empty(0), torch.empty(0))[0]['equal'])

    def test_nonfinite_values_are_rejected(self):
        for value in [torch.tensor(float('nan')), {'x': float('inf')}, torch.tensor([1., float('inf')])]:
            with self.assertRaises(ValueError):
                ts.finite(value)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix='rdd_repro_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = r.read(r.pilot.PACKAGE/'protocol.json')
        self.cfg.update(batch_size=2, workers=0, warmup_steps=0)
        self.protocol = r.read(r.PACKAGE/'protocol.json')
        self.protocol.update(updates_per_condition=3)
        self.batches = []
        for i in range(3):
            images = [torch.full((3, 8, 8), .05*(i+1)), torch.full((3, 8, 8), .01)]
            targets = [{'boxes': torch.tensor([[1., 1., 5., 5.]]), 'labels': torch.tensor([1])},
                       {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]
            self.batches.append({'images': images, 'targets': targets, 'sha256': r.batch_digest(images, targets), 'metadata': {}})
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model, optimizer = self.factory()
        self.payload = {'format': 1, 'model': ts.cpu(model.state_dict()),
                        'optimizer': ts.cpu(optimizer.state_dict()), 'rng': r.cp.rng_state()}

    def factory(self):
        model = SmallDetector()
        return model, torch.optim.SGD(model.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)

    def probe(self, model, optimizer, folder):
        with r.observer.frozen(model, optimizer):
            model([torch.zeros(3, 8, 8)])

    def run_suite(self, **kwargs):
        options = dict(out=self.root/'suite', cfg=self.cfg, protocol=self.protocol, payload=self.payload,
                       batches=self.batches, factory=self.factory, device='cpu', probe=self.probe)
        options.update(kwargs)
        return r.run_suite(**options)

    def test_four_conditions_match_on_stochastic_cpu_fixture(self):
        result = self.run_suite()
        self.assertEqual(result['status'], 'complete_exact_on_this_diagnostic')
        self.assertEqual([v['updates_complete'] for v in result['conditions'].values()], [3]*4)
        self.assertTrue(result['conditions']['probe_only']['probe_boundary_exact'])
        self.assertTrue(result['conditions']['resume_only']['resume_boundary_exact'])
        self.assertTrue(all(v['first_difference'] is None for v in result['conditions'].values()))

    def test_instrumented_update_matches_original_update(self):
        a, opt_a = self.factory(); b, opt_b = self.factory()
        r.pilot.fork_state(a, opt_a, self.payload)
        losses, norm = r.b.update(a, opt_a, self.batches[0]['images'], self.batches[0]['targets'], 'cpu')
        expected = {'model': ts.cpu(a.state_dict()), 'optimizer': ts.cpu(opt_a.state_dict()), 'rng': r.cp.rng_state()}
        r.pilot.fork_state(b, opt_b, self.payload)
        got = r.traced_update(b, opt_b, self.batches[0], 'cpu')
        self.assertEqual({k: float(v) for k,v in got['forward']['losses'].items()}, losses)
        self.assertEqual(got['backward']['grad_norm'], norm)
        self.assertTrue(ts.compare(expected, {k: got['update'][k] for k in expected})[0]['equal'])

    def test_changed_gradient_is_localized_to_backward(self):
        calls = [0]
        def factory():
            calls[0] += 1
            model, optimizer = self.factory()
            if calls[0] == 2:
                model.weight.register_hook(lambda grad: grad + .01)
            return model, optimizer
        result = self.run_suite(factory=factory)
        first = result['conditions']['control_b']['first_difference']
        self.assertEqual((first['step'], first['phase']), (1, 'backward'))
        evidence = list((self.root/'suite/control_b/tensor_evidence').glob('*.npz'))
        self.assertTrue(evidence)
        with np.load(evidence[0]) as data:
            self.assertFalse(np.array_equal(data['expected'], data['actual']))
        self.assertEqual(result['status'], 'complete_with_divergence')

    def test_probe_rng_leak_is_rejected(self):
        def leaking_probe(*_):
            torch.rand(1)
        with self.assertRaisesRegex(ValueError, 'Probe changed'):
            self.run_suite(probe=leaking_probe)
        result = r.read(self.root/'suite/probe_only/probe_after.json')
        self.assertFalse(result['comparison']['equal'])
        self.assertEqual(r.read(self.root/'suite/comparison.json')['status'], 'incomplete')

    def test_probe_buffer_mutation_is_rejected(self):
        def bad_probe(model, *_):
            model.batches_seen.add_(1)
        with self.assertRaisesRegex(ValueError, 'Probe changed'):
            self.run_suite(probe=bad_probe)

    def test_resume_state_mutation_is_rejected(self):
        original = r.b.restore
        def corrupt_restore(out, model, optimizer, context):
            result = original(out, model, optimizer, context)
            with torch.no_grad():
                model.weight.add_(1.)
            return result
        with patch.object(r.b, 'restore', corrupt_restore):
            with self.assertRaisesRegex(ValueError, 'Resume changed'):
                self.run_suite()
        record = r.read(self.root/'suite/resume_only/resume_boundary.json')
        self.assertFalse(record['comparison']['equal'])

    def test_materialized_input_mutation_is_rejected(self):
        self.batches[0]['images'][0].add_(1)
        model, optimizer = self.factory()
        with self.assertRaisesRegex(ValueError, 'Materialized batch changed'):
            r.traced_update(model, optimizer, self.batches[0], 'cpu')

    def test_forward_input_mutation_is_rejected(self):
        model, optimizer = self.factory()
        original = model.forward
        def mutate(images, targets=None):
            images[0].add_(1)
            return original(images, targets)
        model.forward = mutate
        with self.assertRaisesRegex(ValueError, 'Model mutated input'):
            r.traced_update(model, optimizer, self.batches[0], 'cpu')
        self.assertEqual(r.batch_digest(self.batches[0]['images'], self.batches[0]['targets']), self.batches[0]['sha256'])

    def test_nonfinite_loss_does_not_update_parameters(self):
        model, optimizer = self.factory()
        before = model.weight.detach().clone()
        model.forward = lambda *_: {'loss': model.weight.sum()*float('nan')}
        with self.assertRaisesRegex(ValueError, 'Nonfinite/malformed loss'):
            r.traced_update(model, optimizer, self.batches[0], 'cpu')
        self.assertTrue(torch.equal(before, model.weight))

    def test_nonfinite_gradient_does_not_update_parameters(self):
        model, optimizer = self.factory()
        before = model.weight.detach().clone()
        model.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
        with self.assertRaisesRegex(ValueError, 'Nonfinite training gradient'):
            r.traced_update(model, optimizer, self.batches[0], 'cpu')
        self.assertTrue(torch.equal(before, model.weight))

    def test_budget_stop_preserves_partial_records(self):
        calls = [0]
        def stop():
            calls[0] += 1
            return calls[0] >= 3
        with self.assertRaises(r.BudgetStop):
            self.run_suite(stop=stop)
        record = r.read(self.root/'suite/comparison.json')
        self.assertEqual(record['status'], 'incomplete')
        self.assertEqual(record['conditions']['control_a']['updates_complete'], 1)
        self.assertTrue((self.root/'suite/control_a/steps/01_update.json').exists())

    def test_existing_suite_refused(self):
        (self.root/'suite').mkdir()
        sentinel = self.root/'suite/sentinel.txt'; sentinel.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.run_suite()
        self.assertEqual(sentinel.read_text(), 'preserve')

    def test_archive_preserves_evidence_excludes_checkpoint_and_refuses_overwrite(self):
        out = self.root/'records'; out.mkdir()
        r.write(out/'status.json', {'status': 'failed'})
        (out/'checkpoint.pt').write_bytes(b'private-heavy-state')
        (out/'interrupted.tmp').write_bytes(b'partial')
        with (out/'values.npz').open('wb') as stream:
            np.savez_compressed(stream, expected=np.array([1.]), actual=np.array([2.]))
        archive = r.archive_records(out)
        before = hashlib.sha256(archive.read_bytes()).hexdigest()
        with tarfile.open(archive) as source:
            names = [m.name for m in source]
            self.assertEqual(len(names), 3)
            self.assertTrue(all(not name.endswith(('.pt', '.tmp')) for name in names))
        with self.assertRaises(FileExistsError):
            r.archive_records(out)
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
