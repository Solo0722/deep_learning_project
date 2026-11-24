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
    def test_reported_zero_stride_failure_is_reproduced_by_legacy_conversion(self):
        value = torch.as_strided(torch.empty(0, dtype=torch.int64), (0,), (0,))
        self.assertTrue(value.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, 'stride'):
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        self.assertEqual(ts.raw_tensor(value), b'')

    def test_empty_shapes_and_dtypes_hash_without_storage_access(self):
        for dtype in (torch.int64, torch.int32, torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.uint8, torch.bool):
            for shape, strides in (((0,), (0,)), ((0,4), (0,0)), ((2,0,3), (0,0,0))):
                with self.subTest(dtype=dtype, shape=shape):
                    value = torch.as_strided(torch.empty(0,dtype=dtype), shape, strides)
                    self.assertEqual(ts.raw_tensor(value), b'')
                    self.assertEqual(ts.digest(value), ts.digest(torch.empty(shape,dtype=dtype)))
        self.assertNotEqual(ts.digest(torch.empty(0,dtype=torch.long)), ts.digest(torch.empty(0)))
        self.assertNotEqual(ts.digest(torch.empty(0)), ts.digest(torch.empty(0,4)))

    def test_nonempty_strides_match_independent_numpy_bytes(self):
        for dtype in (torch.int64, torch.float32, torch.float64, torch.float16, torch.uint8, torch.bool):
            base = torch.tensor([1,0,1,1],dtype=dtype)
            cases = [base[::2], base.as_strided((1,),(0,)), base.as_strided((1,),(3,)),
                     base[:1].expand(3), base.reshape(2,2).T, base[0]]
            for value in cases:
                with self.subTest(dtype=dtype, shape=tuple(value.shape), stride=value.stride()):
                    before = value.clone()
                    self.assertEqual(ts.raw_tensor(value), value.numpy().tobytes(order='C'))
                    self.assertTrue(torch.equal(value,before))
                    dense = torch.empty(value.shape,dtype=dtype).copy_(value)
                    self.assertEqual(ts.digest(value), ts.digest(dense))

    def test_bfloat16_singleton_uses_dense_byte_payload(self):
        dense = torch.tensor([1.5],dtype=torch.bfloat16)
        broadcast = dense.as_strided((1,),(0,))
        expected = dense.view(torch.uint8).numpy().tobytes()
        self.assertEqual(ts.raw_tensor(broadcast),expected)
        self.assertEqual(ts.digest(broadcast),ts.digest(dense))

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
            examples = []
            for j in range(2):
                boxes = np.array([[1.,1.,5.,5.]],dtype=np.float32) if j==0 else np.empty((0,4),dtype=np.float32)
                labels = np.array([0],dtype=np.int64) if j==0 else np.empty(0,dtype=np.int64)
                ex = r.rdd_data.Example({'image':f'synthetic_{i}_{j}.png'}, i*2+j,
                    np.full((8,8,3),13*(i+1) if j==0 else 3,dtype=np.uint8), boxes, labels)
                examples.append(r.rdd_data.prepare_example(ex,32))
            # Use every production target field, including NumPy-derived empty
            # labels. Handwritten torch.empty labels missed the reported fault.
            images, targets, _ = r.rdd_data.torchvision_collate(examples)
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

    def test_production_collation_includes_all_fields_and_negative_targets(self):
        batch = self.batches[0]
        expected = {'boxes','labels','image_id','area','iscrowd'}
        self.assertTrue(all(set(t)==expected for t in batch['targets']))
        self.assertEqual(batch['targets'][1]['labels'].numel(),0)
        for target in batch['targets']:
            for value in target.values():
                self.assertEqual(ts.raw_tensor(value), value.numpy().tobytes(order='C'))
        dense_targets = [{k:torch.empty(v.shape,dtype=v.dtype).copy_(v) for k,v in t.items()} for t in batch['targets']]
        self.assertEqual(r.batch_digest(batch['images'],dense_targets),batch['sha256'])

    def test_all_negative_production_batch_can_train_and_hash(self):
        batch = self.batches[0]
        negative = {'images':[batch['images'][1].clone() for _ in range(2)],
                    'targets':[ts.cpu(batch['targets'][1]) for _ in range(2)]}
        negative['sha256'] = r.batch_digest(negative['images'],negative['targets'])
        model, optimizer = self.factory()
        before = model.weight.detach().clone()
        r.traced_update(model,optimizer,negative,'cpu')
        self.assertFalse(torch.equal(before,model.weight))

    def test_materialize_exercises_real_file_loader_geometry_and_collation(self):
        sys.path.insert(0,str(r.pilot.PACKAGE/'tests'))
        import test_baseline as fixtures
        fixture = fixtures.BaselineTests(); fixture.setUp()
        try:
            cfg = copy.deepcopy(fixture.cfg)
            cfg.update(workers=0,warmup_steps=0)
            cfg['geometry'].update(size=32,resized_canvas_sizes=[30,32,34],translations_pixels=[-1,0,1])
            rows = fixture.train_rows
            names = [row['image'] for row in rows]
            data = r.pilot.Dataset(fixture.root,rows,cfg,'geometry_only',names)
            data.epoch = 1
            expected = r.aug.empty_trace()
            order = r.b.epoch_batches(len(data),cfg['batch_size'],cfg['seed'],1)
            for indices in order:
                for i in indices:
                    r.aug.record_trace(expected,data[i].pilot_trace)
            protocol = {**self.protocol,'train_images':len(data),'updates_per_condition':len(order),
                        'expected_pairing_sha256':expected['pairing_sha256'],
                        'expected_decision_sha256':expected['decision_sha256']}
            batches, record = r.materialize(fixture.root,rows,names,cfg,protocol)
            self.assertEqual([len(x['images']) for x in batches],[2,2,1])
            self.assertEqual(record['augmentation'],expected)
            self.assertEqual(sum(t['labels'].numel()==0 for batch in batches for t in batch['targets']),1)
            self.assertTrue(record['materialization_rng_unchanged'])
        finally:
            fixture.temp.cleanup()

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
