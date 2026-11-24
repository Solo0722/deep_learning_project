"""Bounded strict-determinism check, reusing the frozen reproduction helpers."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'source_repro'))
import repro as r
import torch
import torch.utils.deterministic

ts = r.ts


class CheckStop(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def verify_package():
    sums = {}
    for line in (PACKAGE / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        target = r.rdd_data.inside(PACKAGE, name)
        r.d.require(name not in sums and r.d.file_hash(target) == expected, 'Changed strict-package file: ' + name)
        sums[name] = expected
    r.d.require({'strict_check.py', 'protocol.json', 'job.pbs', 'run_job.sh',
                 'tests/test_strict.py', 'source_repro/SHA256SUMS'} <= sums.keys(), 'Incomplete strict package')
    protocol = r.read(PACKAGE / 'protocol.json')
    r.d.require(r.d.file_hash(r.PACKAGE / 'SHA256SUMS') == protocol['source_repro_manifest_sha256'],
                'Frozen diagnostic manifest changed')
    r.verify_package()
    base = r.read(r.PACKAGE / 'protocol.json')
    for key in ('source_checkpoint', 'source_manifest_sha256', 'expected_pairing_sha256',
                'expected_decision_sha256', 'train_images', 'batch_size', 'epoch_index',
                'initial_smoke_step', 'resume_after_update', 'conditions', 'phase_order'):
        r.d.require(protocol[key] == base[key], 'Scientific input changed: ' + key)
    r.d.require(protocol['maximum_optimizer_updates'] == 30 and protocol['updates_per_condition'] == 7,
                'Unexpected update budget')
    return sums


def backend_state():
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'fill_uninitialized_memory': torch.utils.deterministic.fill_uninitialized_memory}


def configure_strict(protocol, fresh_cuda=True):
    r.d.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
                'Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
    if fresh_cuda:
        r.d.require(not torch.cuda.is_initialized(), 'CUDA initialized before strict configuration')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.utils.deterministic.fill_uninitialized_memory = True
    assert_strict(protocol)


def assert_strict(protocol):
    r.d.require(backend_state() == protocol['strict_settings'], 'Strict execution settings changed')


def enforce_resources(elapsed, peak_reserved, total, limits, kind='update'):
    cap = limits['maximum_update_seconds' if kind == 'update' else 'maximum_probe_seconds']
    if elapsed > cap:
        raise CheckStop('resource_limit', f'{kind} took {elapsed:.3f}s; declared limit is {cap}s')
    if total and peak_reserved > total * limits['peak_reserved_fraction_limit']:
        raise CheckStop('resource_limit', f'{kind} reserved {peak_reserved} of {total} GPU bytes; declared fraction exceeded')


class RoiObserver:
    """Count the library-selected Python RoIAlign calls without changing tensors."""
    def __init__(self):
        self.calls = []

    @contextmanager
    def installed(self):
        module = importlib.import_module('torchvision.ops.roi_align')
        original = module._roi_align
        def observed(*args, **kwargs):
            names = ('input', 'rois', 'spatial_scale', 'pooled_height', 'pooled_width', 'sampling_ratio', 'aligned')
            values = {**dict(zip(names, args)), **kwargs}
            self.calls.append({'input_shape': list(values['input'].shape),
                               'rois_shape': list(values['rois'].shape),
                               'dtype': str(values['input'].dtype),
                               'device': str(values['input'].device),
                               'sampling_ratio': values['sampling_ratio'],
                               'pooled_size': [values['pooled_height'], values['pooled_width']],
                               'aligned': values['aligned']})
            return original(*args, **kwargs)
        module._roi_align = observed
        try:
            yield self
        finally:
            module._roi_align = original


class Meter:
    def __init__(self, out, protocol, device, roi=None, clock=time.monotonic):
        self.out, self.protocol, self.device = Path(out), protocol, str(device)
        self.roi, self.clock = roi, clock
        self.cuda = self.device.startswith('cuda')
        self.total = torch.cuda.get_device_properties(0).total_memory if self.cuda else 0
        self.execution = {'updates_observed': 0, 'events': [], 'reference_bytes': 0}
        self.started = clock()
        self.active = None

    def stop(self):
        return self.clock() - self.started >= self.protocol['resource_limits']['worker_wall_seconds']

    def begin(self, kind, stage, condition, step=None):
        r.check_stop(self.stop)
        assert_strict(self.protocol)
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.begin_clock = self.clock()
        self.roi_start = len(self.roi.calls) if self.roi else 0
        self.active = {'kind': kind, 'stage': stage, 'condition': condition, 'step': step,
                       'started_unix': time.time(), 'state': 'running',
                       'hard_timeout_seconds': self.protocol['resource_limits']['hard_update_seconds' if kind == 'update' else 'hard_probe_seconds']}
        r.write(self.out / 'active_operation.json', self.active)

    def snapshot(self):
        return {'peak_allocated_bytes': torch.cuda.max_memory_allocated() if self.cuda else 0,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved() if self.cuda else 0,
                'total_gpu_bytes': self.total}

    def finish(self, success, folder):
        if self.cuda and success:
            torch.cuda.synchronize()
        record = {**self.active, **self.snapshot(), 'state': 'complete' if success else 'failed',
                  'elapsed_seconds': self.clock()-self.begin_clock,
                  'python_roi_align_calls': (self.roi.calls[self.roi_start:] if self.roi else []),
                  'timing_scope': 'Instrumented wall time including CPU copies/hashing/phase-record writes; not pure kernel throughput.'}
        r.write(Path(folder) / (f"{record['step']:02d}_resources.json" if record['step'] else 'probe_resources.json'), record)
        r.write(self.out / 'active_operation.json', record)
        self.active = None
        if success:
            assert_strict(self.protocol)
            enforce_resources(record['elapsed_seconds'], record['peak_reserved_bytes'], self.total,
                              self.protocol['resource_limits'], record['kind'])
            if self.cuda:
                r.d.require(record['python_roi_align_calls'], 'Expected strict CUDA Python RoIAlign path was not observed')
        return record

    def update(self, model, optimizer, batch, folder, stage, condition, step):
        self.begin('update', stage, condition, step)
        previous = self.begin_clock
        phases = {}
        def emit(phase, value):
            nonlocal previous
            assert_strict(self.protocol)
            now = self.clock()
            phases[phase] = {'wall_seconds_since_previous_phase': now-previous, **self.snapshot()}
            r.write(folder / f'{step:02d}_{phase}.json', r.phase_record(value))
            r.write(folder / f'{step:02d}_phase_resources.json', phases)
            if phase == 'update':
                self.execution['updates_observed'] += 1
                self.execution['events'].append({'stage': stage, 'condition': condition, 'step': step})
                r.d.require(self.execution['updates_observed'] <= self.protocol['maximum_optimizer_updates'], 'Optimizer budget exceeded')
                r.write(self.out / 'execution.json', self.execution)
            previous = self.clock()
        try:
            result = r.traced_update(model, optimizer, batch, self.device, emit)
        except BaseException:
            try:
                self.finish(False, folder)
            except Exception as recovery_error:
                r.write(folder / f'{step:02d}_resource_record_failure.json', {'message': repr(recovery_error)})
            raise
        resource = self.finish(True, folder)
        print(f"STRICT UPDATE stage={stage} condition={condition} update={step} seconds={resource['elapsed_seconds']:.2f} peak_reserved_GiB={resource['peak_reserved_bytes']/2**30:.3f}", flush=True)
        return result

    def reference(self, path, state):
        meta = r.save_reference(path, state)
        self.execution['reference_bytes'] += meta['bytes']
        r.write(self.out / 'execution.json', self.execution)
        if self.execution['reference_bytes'] > self.protocol['resource_limits']['reference_storage_limit_bytes']:
            raise CheckStop('resource_limit', 'Reference snapshot disk cap exceeded')
        r.write(path.with_suffix('.json'), meta)
        return meta


def release_model(device):
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()


def run_stage(out, stage, cfg, protocol, payload, batches, factory, device, probe, meter):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    count, conditions = stage['updates'], stage['conditions']
    r.d.require(conditions in (['control_a','control_b'], protocol['conditions']), 'Unexpected stage conditions')
    r.d.require(1 <= count <= len(batches), 'Invalid stage horizon')
    references, cases, expected_fork = [], {}, None
    for condition in conditions:
        r.check_stop(meter.stop)
        folder = out / condition
        (folder / 'steps').mkdir(parents=True)
        model, optimizer = factory()
        fork = r.pilot.fork_state(model, optimizer, payload)
        model.train()
        expected_fork = fork if expected_fork is None else expected_fork
        r.d.require(fork == expected_fork, 'Initial state differs between strict conditions')
        r.write(folder / 'fork.json', fork)
        progress = r.pilot.new_progress(cfg, sum(len(x['images']) for x in batches))
        r.d.require(progress['step'] == protocol['initial_smoke_step'], 'Wrong initial smoke counter')
        context = {'strict_protocol': protocol, 'stage': stage['name'], 'condition': condition, 'fork': fork}
        case = {'updates_complete': 0, 'fork': fork, 'first_difference': None, 'comparisons': []}
        cases[condition] = case
        r.write(out / 'comparison.json', r.make_summary(cases, False))
        if condition == 'probe_only':
            before = r.boundary(model, optimizer)
            r.write(folder / 'probe_before.json', {**before, 'rng': r.json_rng(before['rng'])})
            meter.begin('probe', stage['name'], condition)
            try:
                probe(model, optimizer, folder)
            except BaseException:
                try:
                    meter.finish(False, folder)
                except Exception as recovery_error:
                    r.write(folder / 'probe_resource_record_failure.json', {'message': repr(recovery_error)})
                raise
            meter.finish(True, folder)
            after = r.boundary(model, optimizer)
            comparison, _ = ts.compare(before, after)
            r.write(folder / 'probe_after.json', {'state': {**after, 'rng': r.json_rng(after['rng'])}, 'comparison': comparison})
            r.d.require(comparison['equal'], 'Probe changed training state')
            case['probe_boundary_exact'] = True
        exported = set()
        for index, batch in enumerate(batches[:count]):
            r.check_stop(meter.stop)
            step = index+1
            for group in optimizer.param_groups:
                group['lr'] = r.b.learning_rate(cfg, progress['step'], protocol['epoch_index'])
            actual = meter.update(model, optimizer, batch, folder/'steps', stage['name'], condition, step)
            progress['step'] += 1
            progress['next_batch'] = step
            progress['epoch_images'] += len(batch['images'])
            progress['images_seen'] += len(batch['images'])
            case['updates_complete'] = step
            if condition == 'control_a':
                destination = folder / 'reference_steps' / f'step_{step:02d}.pt'
                destination.parent.mkdir(exist_ok=True)
                metadata = meter.reference(destination, actual)
                references.append((destination, metadata))
            else:
                path, meta = references[index]
                r.d.require(r.d.file_hash(path) == meta['sha256'], 'Strict reference changed')
                reference = torch.load(path, map_location='cpu', weights_only=True)
                comparisons, first = r.compare_step(reference, actual, condition, step, folder, protocol, exported)
                r.write(folder / 'steps' / f'{step:02d}_comparison.json', comparisons)
                case['comparisons'].append({'step': step, 'exact': first is None,
                    'phase_equal': {k:v['equal'] for k,v in comparisons.items()}})
                case['first_difference'] = first
                del reference
                if first:
                    result = r.make_summary(cases, False)
                    result['status'] = 'stopped_on_strict_mismatch'
                    r.write(out / 'comparison.json', result)
                    raise CheckStop('strict_mismatch', f"{stage['name']} {condition}: first difference {first}")
                print(f"STRICT EXACT stage={stage['name']} condition={condition} update={step}/{count}", flush=True)
            r.write(out / 'comparison.json', r.make_summary(cases, False))
            del actual
            if condition == 'resume_only' and step == protocol['resume_after_update']:
                before = r.boundary(model, optimizer)
                resume = folder / 'resume_checkpoint'
                (resume / 'checkpoints').mkdir(parents=True)
                meta = r.b.commit(resume, model, optimizer, progress, context, 'intermediate')
                del model, optimizer
                release_model(device)
                model, optimizer = factory()
                restored = r.b.restore(resume, model, optimizer, context)
                r.cp.assert_same(restored, progress, 'strict resume progress')
                model.train()
                after = r.boundary(model, optimizer)
                comparison, _ = ts.compare(before, after)
                r.write(folder / 'resume_boundary.json', {'checkpoint': meta, 'comparison': comparison,
                    'before': {**before, 'rng': r.json_rng(before['rng'])},
                    'after': {**after, 'rng': r.json_rng(after['rng'])},
                    'gradient_handling': 'All conditions zero gradients before each forward; gradients are not checkpointed.'})
                r.d.require(comparison['equal'], 'Resume changed training state')
                case['resume_boundary_exact'] = True
                progress = restored
        del model, optimizer
        release_model(device)
    r.d.require(all(v['updates_complete'] == count for v in cases.values()), 'Incomplete strict stage')
    result = r.make_summary(cases, True)
    r.write(out / 'comparison.json', result)
    return result


def run_stages(out, cfg, protocol, payload, batches, factory, device, probe, meter):
    result = {'status': 'running', 'stages': {}, 'no_accuracy_evaluation': True}
    for stage in protocol['stages']:
        value = run_stage(Path(out)/stage['name'], stage, cfg, protocol, payload, batches,
                          factory, device, probe, meter)
        r.d.require(value['status'] == 'complete_exact_on_this_diagnostic', 'Stage did not pass exactness gate')
        result['stages'][stage['name']] = value
        r.write(Path(out)/'summary.json', result)
        print(f"STRICT STAGE COMPLETE: {stage['name']}", flush=True)
    r.d.require(meter.execution['updates_observed'] == protocol['maximum_optimizer_updates'], 'Incomplete strict update count')
    result.update(status='complete_exact_on_this_diagnostic', updates=protocol['maximum_optimizer_updates'],
                  decision='Review resource cost and scope before further training; no automatic continuation.')
    r.write(Path(out)/'summary.json', result)
    return result


def classify_error(exc):
    if isinstance(exc, CheckStop):
        return exc.status
    if isinstance(exc, (r.BudgetStop, KeyboardInterrupt)):
        return 'budget_stopped'
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return 'cuda_out_of_memory'
    message = str(exc).lower()
    if isinstance(exc, RuntimeError) and ('deterministic implementation' in message or 'does not have a deterministic' in message):
        return 'unsupported_deterministic_operation'
    return 'failed'


def worker(args, protocol):
    meter = None
    started = time.monotonic()
    status, exit_code = 'failed', 1
    try:
        request = r.read(args.out/'supervisor_request.json')
        r.d.require(request['token'] == args.token and request['protocol'] == protocol, 'Worker must be started by the supervisor')
        sources = verify_package()
        configure_strict(protocol)
        cfg = r.read(r.pilot.PACKAGE/'protocol.json')
        versions = r.pilot.runtime(cfg)
        cfg.update(workers=0, warmup_steps=0)
        assert_strict(protocol)
        torch.cuda.set_per_process_memory_fraction(protocol['resource_limits']['gpu_allocator_fraction'], 0)
        props = torch.cuda.get_device_properties(0)
        source_module = importlib.import_module('torchvision.ops.roi_align')
        library_path = Path(inspect.getsourcefile(source_module))
        library_dir = args.out/'library_sources'; library_dir.mkdir()
        shutil.copyfile(library_path, library_dir/'torchvision_roi_align.py')
        r.write(args.out/'context.json', {'protocol': protocol, 'sources': sources, 'effective_configuration': cfg,
            'versions': versions, 'python': sys.version, 'executable': sys.executable,
            'job': os.environ['PBS_JOBID'], 'host': socket.gethostname(), 'device': props.name,
            'device_total_memory': props.total_memory, 'cuda_build': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(), 'backends': backend_state(),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'roi_align_source_sha256': r.d.file_hash(library_path),
            'roi_observer': 'Transparent call counter around library _roi_align; no replacement numerical implementation.',
            'started_utc': datetime.now(timezone.utc).isoformat()})
        print('PBS job:', os.environ['PBS_JOBID'], 'Assigned GPU:', props.name, flush=True)
        print('STRICT SETTINGS:', json.dumps(backend_state(), sort_keys=True), flush=True)
        def interrupted(*_):
            raise r.BudgetStop('Supervisor requested stop')
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        rows, payload, plan, names = r.load_inputs(args.root, args.splits, args.checkpoint, cfg, protocol)
        batches, materialized = r.materialize(args.root, rows, names, cfg, protocol)
        r.write(args.out/'materialized_inputs.json', materialized)
        def factory():
            model, _, _ = r.b.build_model(r.b.MODEL, cfg['image_size'], 'cuda', cfg['evaluation'])
            return model, torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'],
                momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        roi = RoiObserver()
        meter = Meter(args.out, protocol, 'cuda', roi)
        def probe(model, optimizer, folder):
            entry = r.observer.collect(model, optimizer, args.root, rows, plan, cfg, 2, 'cuda', folder, meter.stop)
            if entry is None:
                raise r.BudgetStop('Strict probe did not complete')
            r.observer.verify_updates(folder, [entry], rows, {'2': plan}, cfg)
            r.write(folder/'probe_entry.json', entry)
        with roi.installed():
            result = run_stages(args.out, cfg, protocol, payload, batches, factory, 'cuda', probe, meter)
        status, exit_code = result['status'], 0
    except BaseException as exc:
        status = classify_error(exc)
        exit_code = 3 if status in ('resource_limit','budget_stopped','cuda_out_of_memory') else 1
        r.write(args.out/'failure.json', {'status': status, 'exception_type': type(exc).__name__,
            'message': str(exc), 'traceback': traceback.format_exc(),
            'updates_observed': meter.execution['updates_observed'] if meter else 0})
        traceback.print_exc()
    finally:
        r.write(args.out/'worker_status.json', {'status': status, 'exit_code': exit_code,
            'elapsed_seconds': time.monotonic()-started,
            'updates_observed': meter.execution['updates_observed'] if meter else 0})
    return exit_code


def monitor_process(command, out, limits, wall_seconds, env=None, poll_seconds=0.5):
    """Parent remains outside CUDA and can archive even after an OOM/hung worker."""
    out = Path(out)
    start = time.monotonic()
    reason = None
    with (out/'worker.log').open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        with (out/'worker.log').open('rb') as reader:
            def echo():
                data = reader.read()
                if data:
                    print(data.decode('utf-8', errors='replace'), end='', flush=True)
            try:
                while process.poll() is None:
                    echo()
                    if time.monotonic()-start >= wall_seconds:
                        reason = 'worker_wall_limit'
                    active_file = out/'active_operation.json'
                    if active_file.exists():
                        active = r.read(active_file)
                        if active['state'] == 'running' and time.time()-active['started_unix'] > active['hard_timeout_seconds']:
                            reason = 'operation_wall_limit'
                    if reason:
                        process.terminate()
                        try:
                            process.wait(timeout=limits['terminate_grace_seconds'])
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                        break
                    time.sleep(poll_seconds)
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
                echo()
    return {'returncode': process.returncode, 'stop_reason': reason, 'elapsed_seconds': time.monotonic()-start}


def finalize(out, protocol, process_result):
    out = Path(out)
    worker_record = r.read(out/'worker_status.json') if (out/'worker_status.json').exists() else None
    status = 'budget_stopped' if process_result['stop_reason'] else (worker_record['status'] if worker_record else 'worker_failed')
    if status == 'complete_exact_on_this_diagnostic':
        summary = r.read(out/'summary.json') if (out/'summary.json').exists() else {}
        valid = (process_result['returncode'] == 0 and worker_record['exit_code'] == 0
                 and worker_record['updates_observed'] == protocol['maximum_optimizer_updates']
                 and summary.get('updates') == protocol['maximum_optimizer_updates']
                 and summary.get('status') == status
                 and set(summary.get('stages', {})) == {s['name'] for s in protocol['stages']})
        if valid:
            for stage in protocol['stages']:
                result = summary['stages'][stage['name']]
                valid = valid and result['status'] == status and set(result['conditions']) == set(stage['conditions'])
                valid = valid and all(c['updates_complete'] == stage['updates'] and c['first_difference'] is None
                                      for c in result['conditions'].values())
        if not valid:
            status = 'inconsistent_completion'
    r.write(out/'supervisor_status.json', process_result)
    r.write(out/'status.json', {'status': status, 'worker': worker_record, 'supervisor': process_result,
             'no_accuracy_evaluation': True, 'no_longer_training_started': True})
    if status == 'complete_exact_on_this_diagnostic':
        r.write(out/'COMPLETE.json', {'status': status, 'updates': protocol['maximum_optimizer_updates'],
            'scope': 'Short strict-determinism and resource check only; no automatic training approval.'})
    archive = r.archive_records(out)
    print('STRICT DETERMINISM CHECK:', status.upper(), '- download and review records.', flush=True)
    return status, archive


def main():
    parser = argparse.ArgumentParser()
    for name in ('root','splits','checkpoint','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--token', default='', help=argparse.SUPPRESS)
    args = parser.parse_args()
    protocol = r.read(PACKAGE/'protocol.json')
    if args.worker:
        return worker(args, protocol)
    verify_package()
    r.d.require(os.environ.get('PBS_JOBID'), 'Submit through the PBS job script')
    r.d.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Use the provided job script for strict settings')
    out = r.pilot.check_output(args.out, [args.root,args.splits,args.checkpoint,PACKAGE])
    r.d.require(not out.exists() and not out.with_name(out.name+'_records.tar.gz').exists(), 'Existing strict output; inspect, do not overwrite')
    out.parent.mkdir(parents=True, exist_ok=True)
    limits = protocol['resource_limits']
    r.d.require(shutil.disk_usage(out.parent).free >= limits['minimum_free_disk_bytes'], 'Insufficient free disk: need 10 GiB')
    wall = min(limits['worker_wall_seconds'], float(os.environ.get('RDD_STRICT_DEADLINE','inf'))-time.time())
    r.d.require(wall > 0, 'No administrative budget remains')
    out.mkdir()
    token = uuid.uuid4().hex
    r.write(out/'supervisor_request.json', {'token': token, 'protocol': protocol,
        'command_inputs': {k:str(getattr(args,k)) for k in ('root','splits','checkpoint','out')},
        'started_utc': datetime.now(timezone.utc).isoformat()})
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--token', token]
    for name in ('root','splits','checkpoint','out'):
        command += ['--'+name, str(getattr(args,name))]
    try:
        process_result = monitor_process(command, out, limits, wall, dict(os.environ))
    except BaseException as exc:
        r.write(out/'supervisor_failure.json', {'message': repr(exc), 'traceback': traceback.format_exc()})
        process_result = {'returncode': None, 'stop_reason': 'supervisor_interrupted' if isinstance(exc, KeyboardInterrupt) else None,
                          'exception': repr(exc)}
    status, _ = finalize(out, protocol, process_result)
    return 0 if status == 'complete_exact_on_this_diagnostic' else 3


if __name__ == '__main__':
    raise SystemExit(main())
