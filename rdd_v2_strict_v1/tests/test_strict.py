import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strict_check as s
import torch

spec = importlib.util.spec_from_file_location('repro_test_fixtures', s.PACKAGE/'source_repro/tests/test_repro.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class StrictTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DiagnosticTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.cfg = self.fixture.cfg
        self.protocol = s.r.read(s.PACKAGE/'protocol.json')
        self.protocol['initial_smoke_step'] = 3
        self.protocol['maximum_optimizer_updates'] = 14
        self.protocol['updates_per_condition'] = 3
        self.protocol['stages'][1]['updates'] = 3
        self.original = s.backend_state()
        self.env = patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.restore_backend)
        s.configure_strict(self.protocol, fresh_cuda=False)

    def restore_backend(self):
        torch.use_deterministic_algorithms(self.original['deterministic_algorithms'], warn_only=self.original['warn_only'])
        torch.backends.cudnn.deterministic = self.original['cudnn_deterministic']
        torch.backends.cudnn.benchmark = self.original['cudnn_benchmark']
        torch.backends.cuda.matmul.allow_tf32 = self.original['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = self.original['cudnn_tf32']
        torch.utils.deterministic.fill_uninitialized_memory = self.original['fill_uninitialized_memory']

    def setup_run(self):
        out = self.root/'strict'; out.mkdir()
        meter = s.Meter(out, self.protocol, 'cpu')
        return out, meter

    def run_check(self, **changes):
        out, meter = self.setup_run()
        args = dict(out=out, cfg=self.cfg, protocol=self.protocol, payload=self.fixture.payload,
            batches=self.fixture.batches, factory=self.fixture.factory, device='cpu',
            probe=self.fixture.probe, meter=meter)
        args.update(changes)
        result = s.run_stages(**args)
        return out, meter, result

    def test_strict_settings_reject_wrong_workspace_before_cuda(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}):
            with self.assertRaisesRegex(ValueError, 'before starting Python'):
                s.configure_strict(self.protocol)

    def test_strict_settings_reject_late_configuration(self):
        with patch.object(torch.cuda, 'is_initialized', return_value=True):
            with self.assertRaisesRegex(ValueError, 'CUDA initialized'):
                s.configure_strict(self.protocol)

    def test_warn_only_or_disabled_strict_mode_is_rejected(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            s.assert_strict(self.protocol)
        torch.use_deterministic_algorithms(False)
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            s.assert_strict(self.protocol)

    def test_gated_stages_replay_identically_and_restart_from_original_state(self):
        out, meter, result = self.run_check()
        self.assertEqual(result['status'], 'complete_exact_on_this_diagnostic')
        self.assertEqual(meter.execution['updates_observed'], 14)
        initial = s.r.read(out/'first_update_gate/control_a/steps/01_before.json')
        full = s.r.read(out/'seven_update_suite/control_a/steps/01_before.json')
        self.assertEqual(initial, full)
        complete = result['stages']['seven_update_suite']['conditions']
        self.assertTrue(complete['probe_only']['probe_boundary_exact'])
        self.assertTrue(complete['resume_only']['resume_boundary_exact'])
        self.assertEqual(s.r.read(out/'execution.json')['updates_observed'], 14)

    def test_first_gate_gradient_mismatch_stops_before_suite(self):
        calls = [0]
        def bad_factory():
            calls[0] += 1
            model, opt = self.fixture.factory()
            if calls[0] == 2:
                model.weight.register_hook(lambda grad: grad+.01)
            return model, opt
        with self.assertRaisesRegex(s.CheckStop, 'backward'):
            self.run_check(factory=bad_factory)
        self.assertEqual(calls[0], 2)
        out = self.root/'strict'
        self.assertFalse((out/'seven_update_suite').exists())
        self.assertEqual(s.r.read(out/'execution.json')['updates_observed'], 2)
        self.assertTrue(list((out/'first_update_gate/control_b/tensor_evidence').glob('*.npz')))

    def test_later_mismatch_stops_before_probe_or_resume(self):
        calls = [0]
        def bad_factory():
            calls[0] += 1
            model, opt = self.fixture.factory()
            if calls[0] == 4:
                model.weight.register_hook(lambda grad: grad+.01)
            return model, opt
        with self.assertRaises(s.CheckStop):
            self.run_check(factory=bad_factory)
        out = self.root/'strict'
        self.assertEqual(calls[0], 4)
        self.assertEqual(s.r.read(out/'execution.json')['updates_observed'], 6)
        self.assertFalse((out/'seven_update_suite/probe_only').exists())

    def test_resource_stop_preserves_completed_update_and_skips_second_run(self):
        self.protocol['resource_limits']['maximum_update_seconds'] = 0
        with self.assertRaisesRegex(s.CheckStop, 'declared limit'):
            self.run_check()
        out = self.root/'strict'
        self.assertEqual(s.r.read(out/'execution.json')['updates_observed'], 1)
        resource = s.r.read(out/'first_update_gate/control_a/steps/01_resources.json')
        self.assertEqual(resource['state'], 'complete')
        self.assertGreater(resource['elapsed_seconds'], 0)
        self.assertFalse((out/'first_update_gate/control_b').exists())

    def test_failed_resource_observation_does_not_hide_original_error(self):
        out, meter = self.setup_run()
        model, optimizer = self.fixture.factory()
        folder = out/'failed_steps'
        with patch.object(s.r, 'traced_update', side_effect=RuntimeError('primary computation error')):
            with patch.object(meter, 'finish', side_effect=RuntimeError('secondary CUDA observation error')):
                with self.assertRaisesRegex(RuntimeError, 'primary computation error'):
                    meter.update(model,optimizer,self.fixture.batches[0],folder,'gate','control_a',1)
        self.assertIn('secondary CUDA', s.r.read(folder/'01_resource_record_failure.json')['message'])

    def test_gpu_reserved_fraction_is_an_explicit_resource_limit(self):
        with self.assertRaisesRegex(s.CheckStop, 'reserved'):
            s.enforce_resources(.1, 91, 100, self.protocol['resource_limits'])
        s.enforce_resources(.1, 89, 100, self.protocol['resource_limits'])

    def test_probe_rng_leak_is_rejected_in_strict_pipeline(self):
        with self.assertRaisesRegex(ValueError, 'Probe changed'):
            self.run_check(probe=lambda *_: torch.rand(1))

    def test_resume_state_leak_is_rejected_in_strict_pipeline(self):
        original = s.r.b.restore
        def bad_restore(out, model, optimizer, context):
            value = original(out, model, optimizer, context)
            with torch.no_grad(): model.weight.add_(1)
            return value
        with patch.object(s.r.b, 'restore', bad_restore):
            with self.assertRaisesRegex(ValueError, 'Resume changed'):
                self.run_check()

    def test_runtime_flag_change_is_caught_at_phase_boundary(self):
        def factory():
            model, opt = self.fixture.factory()
            original = model.forward
            def bad_forward(*args):
                torch.use_deterministic_algorithms(False)
                return original(*args)
            model.forward = bad_forward
            return model, opt
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            self.run_check(factory=factory)
        self.assertFalse((self.root/'strict/first_update_gate/control_a/steps/01_update.json').exists())

    def test_roi_observer_preserves_outputs_gradients_and_restores_on_error(self):
        module = s.importlib.import_module('torchvision.ops.roi_align')
        def tiny(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
            return input.square()
        with patch.object(module, '_roi_align', tiny):
            x = torch.tensor([[[[2.]]]], requires_grad=True)
            observer = s.RoiObserver()
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                with observer.installed():
                    output = module._roi_align(x, torch.zeros(1,5), 1., 1, 1, 2, False)
                    output.sum().backward()
                    self.assertEqual(x.grad.item(), 4.)
                    self.assertEqual(output.item(), 4.)
                    raise RuntimeError('deliberate')
            self.assertIs(module._roi_align, tiny)
            self.assertEqual(len(observer.calls), 1)

    def test_error_status_keeps_unsupported_and_oom_distinct(self):
        self.assertEqual(s.classify_error(RuntimeError('foo does not have a deterministic implementation')), 'unsupported_deterministic_operation')
        self.assertEqual(s.classify_error(torch.cuda.OutOfMemoryError('test')), 'cuda_out_of_memory')
        self.assertEqual(s.classify_error(RuntimeError('ordinary bug')), 'failed')

    def test_abrupt_worker_exit_is_archived_without_completion(self):
        out = self.root/'child'; out.mkdir()
        result = s.monitor_process([sys.executable,'-c','print("child evidence", flush=True);raise SystemExit(7)'],
            out, self.protocol['resource_limits'], 10, poll_seconds=.01)
        status, archive = s.finalize(out, self.protocol, result)
        self.assertEqual(status, 'worker_failed')
        self.assertEqual(result['returncode'], 7)
        self.assertIn('child evidence', (out/'worker.log').read_text())
        self.assertTrue(archive.exists())
        self.assertFalse((out/'COMPLETE.json').exists())

    def test_supervisor_enforces_hard_operation_timeout(self):
        out = self.root/'child'; out.mkdir()
        s.r.write(out/'active_operation.json', {'state':'running','started_unix':time.time()-2,'hard_timeout_seconds':1})
        limits = {**self.protocol['resource_limits'],'terminate_grace_seconds':.1}
        result = s.monitor_process([sys.executable,'-c','import time;time.sleep(5)'], out, limits, 10, poll_seconds=.01)
        self.assertEqual(result['stop_reason'], 'operation_wall_limit')
        self.assertEqual(s.finalize(out,self.protocol,result)[0], 'budget_stopped')

    def test_supervisor_enforces_total_wall_timeout(self):
        out = self.root/'child'; out.mkdir()
        limits = {**self.protocol['resource_limits'],'terminate_grace_seconds':.1}
        result = s.monitor_process([sys.executable,'-c','import time;time.sleep(5)'], out, limits, .1, poll_seconds=.01)
        self.assertEqual(result['stop_reason'], 'worker_wall_limit')

    def test_incomplete_success_record_cannot_approve_check(self):
        out = self.root/'partial'; out.mkdir()
        s.r.write(out/'worker_status.json', {'status':'complete_exact_on_this_diagnostic','exit_code':0,'updates_observed':1})
        status, _ = s.finalize(out,self.protocol,{'returncode':0,'stop_reason':None})
        self.assertEqual(status, 'inconsistent_completion')
        self.assertFalse((out/'COMPLETE.json').exists())

    def test_complete_fixture_and_archive_preservation(self):
        out, meter, result = self.run_check()
        s.r.write(out/'worker_status.json', {'status':result['status'],'exit_code':0,'updates_observed':14})
        status, archive = s.finalize(out,self.protocol,{'returncode':0,'stop_reason':None})
        self.assertEqual(status, 'complete_exact_on_this_diagnostic')
        before = s.r.d.file_hash(archive)
        with self.assertRaises(FileExistsError):
            s.r.archive_records(out)
        self.assertEqual(before, s.r.d.file_hash(archive))


if __name__ == '__main__':
    unittest.main()
