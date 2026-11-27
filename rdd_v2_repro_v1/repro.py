"""Seven-update CUDA diagnostic. Does not train a new experimental endpoint."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import tarfile
import time
import traceback

import torch

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'source_pilot'))
import augmentation as aug
import baseline as b
import checkpoints as cp
import controller
import diagnostic as d
import observer
import pilot
import rdd_data
import trace_state as ts


class BudgetStop(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    b.write_json(Path(path), value)


def check_stop(stop):
    if stop():
        raise BudgetStop('Declared diagnostic time limit or stop signal reached')


def verify_package():
    sums = {}
    for line in (PACKAGE / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        target = rdd_data.inside(PACKAGE, name)
        d.require(name not in sums and d.file_hash(target) == expected, 'Changed package source: ' + name)
        sums[name] = expected
    d.require({'repro.py', 'trace_state.py', 'protocol.json', 'source_pilot/SHA256SUMS',
               'tests/test_repro.py', 'run_job.sh', 'job.pbs'} <= sums.keys(), 'Incomplete package')
    protocol = read(PACKAGE / 'protocol.json')
    d.require(d.file_hash(PACKAGE / 'source_pilot/SHA256SUMS') == protocol['source_manifest_sha256'],
              'Original pilot manifest changed')
    pilot.verify_package()
    return sums


def load_inputs(root, splits, checkpoint, cfg, protocol):
    # Deliberately read only the training manifest and its split-policy hash.
    rows = rdd_data.read_manifest(splits / 'train.jsonl', cfg['manifests']['train'])
    d.require(len(rows) == cfg['expected_rows']['train'], 'Wrong training manifest size')
    d.require(d.file_hash(splits / 'config.json') == cfg['split_config_sha256'], 'Split policy changed')
    pin = cfg['checkpoint']
    d.require(pin == protocol['source_checkpoint'], 'Diagnostic and source checkpoint pins differ')
    d.require(checkpoint.name == pin['file'] and checkpoint.stat().st_size == pin['bytes']
              and d.file_hash(checkpoint) == pin['sha256'], 'Wrong source checkpoint')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    p = payload['progress']
    d.require(payload['format'] == 1 and p['epoch'] == 1 and p['step'] == 6716
              and p['images_seen'] == len(rows) and p['next_batch'] == p['epoch_images'] == 0,
              'Source is not the pinned completed epoch-1 checkpoint')
    plans = read(pilot.PACKAGE / 'panel_plan.json')
    plan = plans['2'][:2]
    names = sorted(set(read(pilot.PACKAGE / 'smoke_selection.json')['train']) | {s['image'] for s in plan})
    d.require(len(names) == protocol['train_images'] and len(plan) == 2, 'Smoke selection changed')
    return rows, payload, plan, names


def batch_digest(images, targets):
    return ts.digest({'images': images, 'targets': targets})


def materialize(root, rows, names, cfg, protocol):
    before_rng = cp.rng_state()
    dataset = pilot.Dataset(root, rows, cfg, 'geometry_only', names)
    dataset.epoch = protocol['epoch_index']
    batches = b.epoch_batches(len(dataset), cfg['batch_size'], cfg['seed'], dataset.epoch)
    trace = aug.empty_trace()
    result, records = [], []
    for images, targets, examples in b.loader(dataset, batches, cfg, dataset.epoch):
        for ex in examples:
            aug.record_trace(trace, ex.pilot_trace)
        metadata = {'images': [ex.record['image'] for ex in examples],
                    'image_ids': [ex.image_id for ex in examples],
                    'target_counts': [len(ex.boxes) for ex in examples],
                    'traces': [ex.pilot_trace for ex in examples]}
        signature = batch_digest(images, targets)
        result.append({'images': images, 'targets': targets, 'sha256': signature, 'metadata': metadata})
        records.append({'sha256': signature, **metadata})
    cp.assert_same(cp.rng_state(), before_rng, 'materialization RNG')
    d.require(len(result) == protocol['updates_per_condition'], 'Wrong number of materialized batches')
    d.require(trace['pairing_sha256'] == protocol['expected_pairing_sha256'], 'Inputs differ from archived smoke pairing')
    d.require(trace['decision_sha256'] == protocol['expected_decision_sha256'], 'Geometry-only decisions differ from smoke')
    d.require(trace['counts']['all/operation/identity'] == protocol['train_images']
              and trace['counts'].get('all/applied', 0) == 0, 'Unexpected appearance augmentation')
    return result, {'batches': records, 'augmentation': trace, 'materialization_rng_unchanged': True}


def modes(model):
    return {name: module.training for name, module in model.named_modules()}


def boundary(model, optimizer):
    return {'model': ts.digest(model.state_dict()), 'optimizer': ts.digest(optimizer.state_dict()),
            'rng': cp.rng_state(), 'modes': modes(model)}


def json_rng(value):
    if torch.is_tensor(value):
        return value.cpu().tolist()
    if isinstance(value, dict):
        return {k: json_rng(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_rng(v) for v in value]
    return value


def phase_record(phase):
    record = {'state': ts.overview(phase),
              'rng_sha256': {k: ts.digest(v) for k, v in phase['rng'].items()},
              'rng_state': json_rng(phase['rng'])}
    for key in ('model_sha256', 'optimizer_sha256', 'modes', 'inputs_sha256', 'learning_rates', 'gradients_cleared'):
        if key in phase:
            record[key] = phase[key]
    if 'losses' in phase:
        record['losses'] = {k: float(v) for k, v in phase['losses'].items()}
    if 'grad_norm' in phase:
        record['grad_norm'] = phase['grad_norm']
    return record


def traced_update(model, optimizer, batch, device, emit=lambda phase, value: None):
    """The same operations/order as source_pilot.baseline.update, with observations.

    CPU copies synchronize CUDA and may affect scheduling. That limitation is
    explicit; observations neither draw random numbers nor modify training state.
    """
    d.require(batch_digest(batch['images'], batch['targets']) == batch['sha256'], 'Materialized batch changed')
    images = [im.clone().to(device) for im in batch['images']]
    targets = [{k: v.clone().to(device) for k, v in t.items()} for t in batch['targets']]
    d.require(batch_digest(images, targets) == batch['sha256'], 'Device inputs differ')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before_model = ts.cpu(model.state_dict())
    d.require(all(p.grad is None for p in model.parameters()), 'Gradients not cleared')
    before = {'inputs_sha256': batch['sha256'], 'model_sha256': ts.digest(before_model),
              'optimizer_sha256': ts.digest(optimizer.state_dict()), 'rng': cp.rng_state(),
              'modes': modes(model), 'gradients_cleared': True,
              'learning_rates': [g['lr'] for g in optimizer.param_groups]}
    emit('before', before)
    losses = model(images, targets)
    d.require(losses and all(v.numel() == 1 and torch.isfinite(v).item() for v in losses.values()),
              'Nonfinite/malformed loss')
    forward = {'losses': ts.cpu(losses), 'buffers': ts.cpu(dict(model.named_buffers())), 'rng': cp.rng_state()}
    emit('forward', forward)
    d.require(batch_digest(images, targets) == batch['sha256'], 'Model mutated input pixels or targets')
    sum(losses.values()).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    d.require(grads, 'No model gradients')
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads]))
    d.require(torch.isfinite(norm).item(), 'Nonfinite training gradient')
    backward = {'gradients': ts.cpu({n: p.grad for n, p in model.named_parameters()}),
                'grad_norm': float(norm), 'rng': cp.rng_state()}
    ts.finite(backward)
    emit('backward', backward)
    optimizer.step()
    after_model = ts.cpu(model.state_dict())
    updates = {n: after_model[n]-before_model[n] for n, _ in model.named_parameters()}
    after = {'model': after_model, 'optimizer': ts.cpu(optimizer.state_dict()),
             'parameter_updates': updates, 'rng': cp.rng_state()}
    ts.finite(after)
    emit('update', after)
    d.require(batch_digest(batch['images'], batch['targets']) == batch['sha256'], 'Source batch mutated')
    return {'before': before, 'forward': forward, 'backward': backward, 'update': after}


def save_reference(path, value):
    with path.open('xb') as stream:
        torch.save(value, stream)
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': d.file_hash(path)}


def compare_step(reference, actual, condition, step, folder, protocol, exported):
    comparisons = {}
    first = None
    for phase in protocol['phase_order']:
        comparison, pair = ts.compare(reference[phase], actual[phase])
        if not comparison['equal']:
            if first is None:
                first = {'step': step, 'phase': phase, 'changed_paths': comparison['changed_paths']}
            if phase not in exported:
                exported.add(phase)
                comparison['first_tensor_pair'] = ts.export_pair(folder / 'tensor_evidence',
                    f'{condition}_step_{step:02d}_{phase}', pair, protocol['tensor_pair_export_limit_bytes'])
        comparisons[phase] = comparison
    return comparisons, first


def make_summary(cases, complete):
    if not complete:
        status = 'incomplete'
    else:
        status = 'complete_with_divergence' if any(v.get('first_difference') for v in cases.values()) else 'complete_exact_on_this_diagnostic'
    return {'status': status, 'conditions': cases,
            'scope': 'Execution/reproducibility diagnostic only; no AP, generalization or method-superiority claim.',
            'decision': 'Review first differences and numeric magnitudes before any longer run. No automatic continuation.',
            'interpretation': [
                'A control_b difference means probing and resume are not necessary for divergence; it does not exclude their additional effects.',
                'A probe_only or resume_only difference with an exact control repeat points to that path for further investigation, not proof from repeated trials.',
                'If every comparison is exact, the original discrepancy was not reproduced under this instrumentation; its cause remains unestablished.']}


def run_suite(out, cfg, protocol, payload, batches, factory, device, probe, stop=lambda: False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    d.require(len(batches) == protocol['updates_per_condition'], 'Incorrect diagnostic update budget')
    d.require(protocol['conditions'] == ['control_a', 'control_b', 'probe_only', 'resume_only'], 'Condition order changed')
    references, cases = [], {}
    expected_fork = None
    storage = 0
    for condition in protocol['conditions']:
        check_stop(stop)
        folder = out / condition
        (folder / 'steps').mkdir(parents=True)
        model, optimizer = factory()
        fork = pilot.fork_state(model, optimizer, payload)
        model.train()
        expected_fork = fork if expected_fork is None else expected_fork
        d.require(fork == expected_fork, 'Model/optimizer/RNG forks differ')
        write(folder / 'fork.json', fork)
        progress = pilot.new_progress(cfg, sum(len(batch['images']) for batch in batches))
        context = {'diagnostic': protocol, 'condition': condition, 'initial_fork': fork}
        case = {'updates_complete': 0, 'fork': fork, 'first_difference': None, 'comparisons': []}
        cases[condition] = case
        write(out / 'comparison.json', make_summary(cases, False))
        if condition == 'probe_only':
            before = boundary(model, optimizer)
            write(folder / 'probe_before.json', {k: (json_rng(v) if k == 'rng' else v) for k, v in before.items()})
            probe(model, optimizer, folder)
            after = boundary(model, optimizer)
            comparison, _ = ts.compare(before, after)
            write(folder / 'probe_after.json', {'state': {k: (json_rng(v) if k == 'rng' else v) for k, v in after.items()},
                                              'comparison': comparison})
            d.require(comparison['equal'], 'Probe changed observable model/optimizer/RNG/mode state')
            case['probe_boundary_exact'] = True
        exported = set()
        for index, batch in enumerate(batches):
            check_stop(stop)
            step = index+1
            lr = b.learning_rate(cfg, progress['step'], protocol['epoch_index'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            def emit(phase, value):
                write(folder / 'steps' / f'{step:02d}_{phase}.json', phase_record(value))
            started = time.monotonic()
            actual = traced_update(model, optimizer, batch, device, emit)
            elapsed = time.monotonic()-started
            progress['step'] += 1; progress['next_batch'] = step
            progress['epoch_images'] += len(batch['images']); progress['images_seen'] += len(batch['images'])
            if condition == 'control_a':
                ref_folder = folder / 'reference_steps'; ref_folder.mkdir(exist_ok=True)
                path = ref_folder / f'step_{step:02d}.pt'
                metadata = save_reference(path, actual)
                storage += metadata['bytes']
                d.require(storage <= protocol['reference_storage_limit_bytes'], 'Reference storage cap exceeded')
                write(path.with_suffix('.json'), metadata)
                references.append((path, metadata))
                first = None
            else:
                path, metadata = references[index]
                d.require(d.file_hash(path) == metadata['sha256'], 'Reference snapshot changed')
                reference = torch.load(path, map_location='cpu', weights_only=True)
                comparisons, first = compare_step(reference, actual, condition, step, folder, protocol, exported)
                write(folder / 'steps' / f'{step:02d}_comparison.json', comparisons)
                case['comparisons'].append({'step': step, 'exact': first is None,
                    'phase_equal': {phase: value['equal'] for phase, value in comparisons.items()}})
                del reference
            if first is not None and case['first_difference'] is None:
                case['first_difference'] = first
            case['updates_complete'] = step
            write(out / 'comparison.json', make_summary(cases, False))
            print(f'DIAGNOSTIC condition={condition} update={step}/{len(batches)} loss={sum(float(x) for x in actual["forward"]["losses"].values()):.7f} '
                  f'first_difference={case["first_difference"]} seconds={elapsed:.2f}', flush=True)
            del actual
            if condition == 'resume_only' and step == protocol['resume_after_update']:
                check_stop(stop)
                resume_folder = folder / 'resume_checkpoint'; (resume_folder / 'checkpoints').mkdir(parents=True)
                before = boundary(model, optimizer)
                metadata = b.commit(resume_folder, model, optimizer, progress, context, 'intermediate')
                del model, optimizer
                gc.collect()
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
                model, optimizer = factory()
                restored = b.restore(resume_folder, model, optimizer, context)
                cp.assert_same(restored, progress, 'resume progress')
                model.train()
                after = boundary(model, optimizer)
                comparison, _ = ts.compare(before, after)
                write(folder / 'resume_boundary.json', {'checkpoint': metadata, 'comparison': comparison,
                    'before': {k: (json_rng(v) if k == 'rng' else v) for k, v in before.items()},
                    'after': {k: (json_rng(v) if k == 'rng' else v) for k, v in after.items()},
                    'gradient_handling': 'Gradients are not restored; all conditions clear them before every next forward pass.'})
                d.require(comparison['equal'], 'Resume changed model/optimizer/RNG/mode state')
                progress = restored
                case['resume_boundary_exact'] = True
        del model, optimizer
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    d.require(all(v['updates_complete'] == protocol['updates_per_condition'] for v in cases.values()), 'Incomplete suite')
    summary = make_summary(cases, True)
    summary['reference_snapshot_bytes_retained_on_ccast'] = storage
    write(out / 'comparison.json', summary)
    return summary


def archive_records(out):
    """Archive bounded records/evidence, never full checkpoints/reference tensors."""
    paths = sorted(p for p in out.rglob('*') if p.is_file() and p.suffix not in ('.pt', '.tmp')
                   and '__pycache__' not in p.parts and p.name != 'checksums.json')
    sums = {p.relative_to(out).as_posix(): d.file_hash(p) for p in paths}
    write(out / 'checksums.json', sums)
    archive = out.with_name(out.name + '_records.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        for path in paths + [out / 'checksums.json']:
            tar.add(path, arcname=out.name+'/'+path.relative_to(out).as_posix(), recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        seen = {}
        for member in tar:
            d.require(member.isfile() and member.name.startswith(out.name+'/'), 'Bad output archive member')
            relative = member.name[len(out.name)+1:]
            seen[relative] = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
        d.require(seen == {**sums, 'checksums.json': d.file_hash(out / 'checksums.json')}, 'Archive verification failed')
    digest = d.file_hash(archive)
    archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n', encoding='utf-8')
    print('Records archive:', archive, flush=True)
    print('Archive SHA-256:', digest, flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'splits', 'checkpoint', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    protocol, source_cfg = read(PACKAGE / 'protocol.json'), read(pilot.PACKAGE / 'protocol.json')
    sources = verify_package()
    versions = pilot.runtime(source_cfg)
    # Preserve the archived setting, and record it. Strict determinism is a
    # separate potential intervention, not silently enabled for this comparison.
    d.require(not torch.are_deterministic_algorithms_enabled(), 'Global deterministic algorithms unexpectedly enabled')
    out = pilot.check_output(args.out, [args.root, args.splits, args.checkpoint, PACKAGE])
    d.require(not out.with_name(out.name+'_records.tar.gz').exists(), 'Previous records archive exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    d.require(shutil.disk_usage(out.parent).free >= protocol['minimum_free_disk_bytes'], 'Less than required 10 GiB free for retained diagnostic references')
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    limit = started + protocol['diagnostic_seconds']
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    stop = lambda: stopping[0] or time.monotonic() >= limit or time.time() >= float(os.environ.get('RDD_REPRO_DEADLINE', 'inf'))
    cfg = copy.deepcopy(source_cfg)
    cfg.update(workers=0, warmup_steps=0)
    context = {'protocol': protocol, 'source_configuration': source_cfg, 'effective_configuration': cfg,
        'sources': sources, 'versions': versions, 'python': sys.version, 'executable': sys.executable,
        'job': os.environ['PBS_JOBID'], 'host': socket.gethostname(), 'device': torch.cuda.get_device_name(0),
        'cuda_build': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'backends': {'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}}
    write(out / 'context.json', context)
    status, exit_code = 'running', 0
    try:
        rows, payload, plan, names = load_inputs(args.root, args.splits, args.checkpoint, cfg, protocol)
        check_stop(stop)
        batches, materialized = materialize(args.root, rows, names, cfg, protocol)
        write(out / 'materialized_inputs.json', materialized)
        def factory():
            model, _, _ = b.build_model(b.MODEL, cfg['image_size'], 'cuda', cfg['evaluation'])
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
            return model, optimizer
        def probe(model, optimizer, folder):
            entry = observer.collect(model, optimizer, args.root, rows, plan, cfg, 2, 'cuda', folder, stop)
            if entry is None:
                raise BudgetStop('Probe incomplete')
            observer.verify_updates(folder, [entry], rows, {'2': plan}, cfg)
            write(folder / 'probe_entry.json', entry)
        summary = run_suite(out / 'suite', cfg, protocol, payload, batches, factory, 'cuda', probe, stop)
        status = summary['status']
        write(out / 'COMPLETE.json', {'status': status, 'updates': 4*protocol['updates_per_condition'],
                                   'no_accuracy_claim': True, 'no_longer_training_started': True})
    except BudgetStop as exc:
        status, exit_code = 'budget_stopped', 3
        write(out / 'failure.json', {'kind': status, 'message': str(exc)})
    except Exception as exc:
        status, exit_code = 'failed', 1
        write(out / 'failure.json', {'kind': status, 'message': repr(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        write(out / 'status.json', {'status': status, 'elapsed_seconds': time.monotonic()-started,
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': exit_code})
        archive_records(out)
    print('REPRODUCIBILITY DIAGNOSTIC:', status.upper(), '- review records; no automatic continuation.', flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
