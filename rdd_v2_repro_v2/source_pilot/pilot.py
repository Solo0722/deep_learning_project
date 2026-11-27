"""Three additional epochs in four paired arms. No test-set access or auto-scale."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import gzip
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
import uuid

import numpy as np
import torch

import augmentation as aug
import baseline as b
import checkpoints
import controller
import diagnostic as d
from detector import tensor_digest, rng_digest
import observer
import rdd_data

PACKAGE = Path(__file__).resolve().parent


def tree_digest(value):
    result = hashlib.sha256()
    def walk(item):
        result.update(type(item).__name__.encode())
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str): walk(key); walk(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item: walk(part)
        else: result.update(repr(item).encode())
    walk(value)
    return result.hexdigest()


class Dataset(b.Dataset):
    def __init__(self, root, rows, cfg, arm, selection=None):
        super().__init__(root, rows, 'train', cfg['image_size'], selection)
        self.cfg, self.arm, self.epoch, self.policy = cfg, arm, cfg['warmup_epoch'], None

    def __getitem__(self, index):
        return aug.apply(super().__getitem__(index), self.cfg, self.epoch, self.arm, self.policy)


def new_progress(cfg, length):
    p = b.initial_progress()
    p.update(epoch=cfg['warmup_epoch'], step=cfg['warmup_epoch']*math.ceil(length/cfg['batch_size']),
             images_seen=cfg['warmup_epoch']*length, controller_updates=[], augmentation=aug.empty_trace(),
             training_seconds=0., validation_seconds=0.)
    return p


def fork_state(model, optimizer, payload):
    d.require(payload['format'] == 1, 'Unknown warmup checkpoint format')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    checkpoints.assert_same(model.state_dict(), payload['model'], 'fork model')
    checkpoints.assert_same(optimizer.state_dict(), payload['optimizer'], 'fork optimizer')
    checkpoints.restore_rng(payload['rng'])
    checkpoints.assert_same(checkpoints.rng_state(), payload['rng'], 'fork RNG')
    return {'model_sha256': tensor_digest(model.state_dict().items()),
            'optimizer_sha256': tree_digest(optimizer.state_dict()), 'rng_sha256': rng_digest(),
            'model_optimizer_rng_exact': True}


def validate(model, optimizer, val, cfg, device, stop):
    with observer.frozen(model, optimizer) as audit:
        result = b.validate(model, val, cfg, device, stop)
    if result is not None: result['state_audit'] = audit
    return result


def save_validation(out, epoch, result):
    predictions = result.pop('predictions')
    raw = json.dumps(predictions, separators=(',', ':'), allow_nan=False).encode()
    b.atomic_bytes(out/f'epoch_{epoch:03d}_val_predictions.json.gz', gzip.compress(raw, mtime=0))
    b.write_json(out/f'epoch_{epoch:03d}_validation.json', result)
    names = [f'epoch_{epoch:03d}_val_predictions.json.gz', f'epoch_{epoch:03d}_validation.json']
    return {name: d.file_hash(out/name) for name in names}


def verify_validations(out, history):
    for row in history:
        for name, expected in row['validation_records'].items():
            d.require(d.file_hash(rdd_data.inside(out, name)) == expected, 'Validation evidence changed')


def verify_exposure(trace, expected):
    counts = trace['counts']
    for actual, planned in [('all/images', 'images'), ('all/positive', 'positive_images'),
                            ('all/opportunities', 'opportunities'), ('all/geometry/accepted', 'accepted'),
                            ('all/geometry/cropped_annotation', 'cropped_annotation')]:
        d.require(counts.get(actual, 0) == expected.get(planned, 0), 'Training exposure differs from frozen metadata plan: '+actual)
    d.require(sum(counts.get('target_objects_per_class/'+str(c), 0) for c in range(4)) == expected['target_boxes_preserved'],
              'Target-object exposure changed')
    d.require(counts.get('all/geometry/invalid_annotation', 0) == expected.get('invalid_annotation_fallback', 0), 'Unexpected invalid annotation')


def fit(model, optimizer, train, val, cfg, device, out, context, progress, plans, should_stop=lambda: False,
        collect=observer.collect):
    arm = train.arm
    while progress['epoch'] < cfg['epochs']:
        epoch = progress['epoch']
        if should_stop():
            b.commit(out, model, optimizer, progress, context, 'intermediate')
            return 'budget_stopped'
        if arm.startswith('online') or arm == 'class_adaptive':
            if epoch+1 in cfg['update_epochs'] and not any(e['epoch'] == epoch+1 for e in progress['controller_updates']):
                d.require(progress['next_batch'] == 0, 'Missing policy at a mid-epoch resume')
                b.commit(out, model, optimizer, progress, context, 'preprobe')
                entry = collect(model, optimizer, train.root, [r for _, r in train.rows], plans[str(epoch+1)],
                                cfg, epoch+1, device, out, should_stop)
                if entry is None: return 'budget_stopped'
                progress['controller_updates'].append(entry)
                b.commit(out, model, optimizer, progress, context, 'controller')
            d.require(progress['controller_updates'], 'No adopted controller update')
            train.policy = progress['controller_updates'][-1]['policy']
        train.epoch = epoch
        batches = b.epoch_batches(len(train), cfg['batch_size'], cfg['seed'], epoch, progress['next_batch'])
        start = progress['next_batch']
        clock = time.monotonic()
        for batch_index, (images, targets, examples) in enumerate(b.loader(train, batches, cfg, epoch), start):
            if should_stop():
                progress['training_seconds'] += time.monotonic()-clock
                b.commit(out, model, optimizer, progress, context, 'intermediate')
                return 'budget_stopped'
            lr = b.learning_rate(cfg, progress['step'], epoch)
            for group in optimizer.param_groups: group['lr'] = lr
            losses, grad_norm = b.update(model, optimizer, images, targets, device)
            progress['step'] += 1
            progress['next_batch'] = batch_index+1
            progress['epoch_images'] += len(images)
            progress['images_seen'] += len(images)
            for key, value in losses.items():
                progress['loss_sums'][key] = progress['loss_sums'].get(key, 0.)+value*len(images)
            for example in examples: aug.record_trace(progress['augmentation'], example.pilot_trace)
            if progress['step'] % cfg['log_every_steps'] == 0:
                print(f'TRAIN arm={arm} epoch={epoch+1}/{cfg["epochs"]} batch={batch_index+1} step={progress["step"]} loss={sum(losses.values()):.5f} grad_norm={grad_norm:.4f}', flush=True)
            if progress['step'] % cfg['checkpoint_every_steps'] == 0:
                progress['training_seconds'] += time.monotonic()-clock
                b.commit(out, model, optimizer, progress, context, 'intermediate')
                clock = time.monotonic()
        progress['training_seconds'] += time.monotonic()-clock
        d.require(progress['epoch_images'] == progress['augmentation']['images'] == len(train), 'Epoch coverage mismatch')
        if len(train) == cfg['expected_rows']['train']:
            planned = json.loads((PACKAGE/'exposure_plan.json').read_text())['epochs'][str(epoch+1)]
            verify_exposure(progress['augmentation'], planned)
        b.commit(out, model, optimizer, progress, context, 'prevalidation')
        clock = time.monotonic()
        result = validate(model, optimizer, val, cfg, device, should_stop)
        progress['validation_seconds'] += time.monotonic()-clock
        if result is None:
            b.commit(out, model, optimizer, progress, context, 'prevalidation')
            return 'budget_stopped'
        completed = epoch+1
        validation_records = save_validation(out, completed, result)
        progress['history'].append({'epoch': completed, 'step': progress['step'], 'images_seen': progress['images_seen'],
            'train_images': len(train), 'val_images': len(val),
            'training_loss': {k: v/len(train) for k, v in progress['loss_sums'].items()},
            'validation': result['metrics'], 'validation_records': validation_records, 'augmentation': copy.deepcopy(progress['augmentation']),
            'active_update_epoch': progress['controller_updates'][-1]['epoch'] if progress['controller_updates'] else None})
        # Retain best metadata for provenance, but it is not the pilot endpoint.
        score = result['metrics']['AP50_95']
        if progress['best'] is None or score > progress['best']['validation_AP50_95']:
            progress['best'] = {'epoch': completed, 'validation_AP50_95': score}
        progress.update(epoch=completed, next_batch=0, epoch_images=0, loss_sums={}, augmentation=aug.empty_trace())
        b.commit(out, model, optimizer, progress, context, f'epoch_{completed:03d}')
        print(f'VALIDATION arm={arm} epoch={completed} AP50:95={score:.6f} AP50={result["metrics"]["AP50"]:.6f}', flush=True)
    d.require(progress['step'] == cfg['epochs']*math.ceil(len(train)/cfg['batch_size'])
              and progress['images_seen'] == cfg['epochs']*len(train), 'Completed budget differs')
    expected_updates = cfg['update_epochs'] if arm in ('online_shared', 'class_adaptive') else []
    d.require([e['epoch'] for e in progress['controller_updates']] == expected_updates, 'Missing/extra controller updates')
    return 'complete'


def verify_package():
    sums = {}
    for line in (PACKAGE/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        d.require(name not in sums and d.file_hash(rdd_data.inside(PACKAGE, name)) == expected, f'Package changed: {name}')
        sums[name] = expected
    required = {'pilot.py', 'augmentation.py', 'observer.py', 'smoke.py', 'protocol.json', 'panel_plan.json',
                'baseline.py', 'controller.py', 'geometry.py', 'rdd_data.py', 'models.py', 'checkpoints.py', 'metrics.py', 'exposure_plan.json'}
    d.require(required <= sums.keys(), 'Incomplete package checksum list')
    return sums


def load_inputs(root, splits, checkpoint, cfg):
    train = rdd_data.read_manifest(splits/'train.jsonl', cfg['manifests']['train'])
    val = rdd_data.read_manifest(splits/'val.jsonl', cfg['manifests']['val'], 'val')
    b.verify_partitions(train, val)
    d.require(len(train) == cfg['expected_rows']['train'] and len(val) == cfg['expected_rows']['val'], 'Wrong dataset size')
    d.require(d.file_hash(splits/'config.json') == cfg['split_config_sha256'], 'Split policy changed')
    pin = cfg['checkpoint']
    d.require(checkpoint.name == pin['file'] and checkpoint.stat().st_size == pin['bytes']
              and d.file_hash(checkpoint) == pin['sha256'], 'Wrong warmup checkpoint')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    p = payload['progress']
    d.require(p['epoch'] == 1 and p['step'] == 6716 and p['images_seen'] == len(train)
              and p['next_batch'] == p['epoch_images'] == 0, 'Not a completed first-epoch checkpoint')
    plans = json.loads((PACKAGE/'panel_plan.json').read_text())
    for epoch in cfg['update_epochs']:
        planning = copy.deepcopy(cfg)
        planning['seed'] = cfg['panel_plan_seeds'][str(epoch)]
        expected = d.plan_panels(train, planning)['panels']['A']
        d.require(plans[str(epoch)] == expected and len(expected) == 256, 'Frozen panel plan mismatch')
    return train, val, payload, plans


def runtime(cfg):
    d.require(Path(sys.prefix).resolve() == Path(cfg['expected_environment']).resolve(), 'Wrong Python environment')
    d.require(os.environ.get('PBS_JOBID') and torch.cuda.is_available() and torch.cuda.device_count() == 1,
              'One assigned CUDA GPU in PBS is required')
    versions = {name: version(name) for name in cfg['versions']}
    d.require(versions == cfg['versions'], f'Package versions changed: {versions}')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return versions


def check_output(out, protected):
    out = out.resolve()
    for p in protected:
        p = p.resolve()
        d.require(out != p and out not in p.parents and p not in out.parents, 'Output overlaps an input')
    return out


def paired_summary(histories, cfg):
    d.require(set(histories) == set(cfg['arms']), 'Missing arm history')
    epochs = list(range(cfg['warmup_epoch']+1, cfg['epochs']+1))
    for arm, history in histories.items():
        d.require([r['epoch'] for r in history] == epochs, f'Incomplete history for {arm}')
    for i in range(len(epochs)):
        d.require(len({h[i]['augmentation']['pairing_sha256'] for h in histories.values()}) == 1,
                  f'Unpaired geometry/data/opportunities at epoch {epochs[i]}')
    final = {arm: h[-1]['validation'] for arm, h in histories.items()}
    comparisons = {}
    for arm, control in [('fixed_local', 'geometry_only'), ('online_shared', 'fixed_local'),
                         ('class_adaptive', 'online_shared'), ('class_adaptive', 'geometry_only')]:
        comparisons[arm+' minus '+control] = {'AP50_95_points': 100*(final[arm]['AP50_95']-final[control]['AP50_95']),
            'per_class_AP50_95_points': {c: None if final[arm]['per_class'][c]['AP50_95'] is None or final[control]['per_class'][c]['AP50_95'] is None
                else 100*(final[arm]['per_class'][c]['AP50_95']-final[control]['per_class'][c]['AP50_95']) for c in rdd_data.CLASSES}}
    return {'scope': 'Single-seed development pilot; final fixed epoch, not selected best epoch; no test evaluation',
            'epoch': cfg['epochs'], 'paired_geometry_order_opportunities': True, 'final_metrics': final, 'comparisons': comparisons}


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'splits', 'checkpoint', 'out'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((PACKAGE/'protocol.json').read_text())
    sources, versions = verify_package(), runtime(cfg)
    smoke_path = Path(os.environ.get('RDD_PILOT_SMOKE_RECORD', 'MISSING_SMOKE_RECORD'))
    d.require(smoke_path.is_file(), 'Run the GPU smoke gate through run_job.sh before training')
    smoke = json.loads(smoke_path.read_text())
    d.require(smoke['status'] == 'pass' and smoke['package_manifest_sha256'] == d.file_hash(PACKAGE/'SHA256SUMS'), 'GPU smoke gate is stale or unsuccessful')
    out = check_output(args.out, [args.root, args.splits, args.checkpoint, PACKAGE])
    d.require(args.resume == out.exists(), 'Use start for a new output, explicit resume for existing output')
    out.mkdir(parents=True, exist_ok=args.resume)
    attempt = uuid.uuid4().hex[:12]
    started = time.monotonic()
    deadline = min(time.time()+cfg['administrative_seconds'], float(os.environ.get('RDD_PILOT_DEADLINE', 'inf')))
    cancelled = [False]
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    stop = lambda: cancelled[0] or time.time() >= deadline
    record = {'attempt': attempt, 'job': os.environ.get('PBS_JOBID'), 'hostname': socket.gethostname(),
              'gpu': torch.cuda.get_device_name(0), 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
    with b.exclusive_run(out):
        try:
            d.require(not (out/'COMPLETE.json').exists(), 'Pilot already complete; do not rerun')
            b.write_json(out/f'smoke_gate_{attempt}.json', {'source': str(smoke_path), 'sha256': d.file_hash(smoke_path), 'result': smoke})
            context = {'configuration': cfg, 'sources': sources, 'versions': versions,
                       'raw_root': str(args.root.resolve()), 'splits': str(args.splits.resolve())}
            if (out/'context.json').exists():
                d.require(json.loads((out/'context.json').read_text()) == context, 'Resume context changed')
            else: b.write_json(out/'context.json', context)
            train_rows, val_rows, payload, plans = load_inputs(args.root, args.splits, args.checkpoint, cfg)
            val = b.Dataset(args.root, val_rows, 'val', cfg['image_size'])
            histories, forks = {}, []
            for arm in cfg['arms']:
                arm_out = out/arm
                arm_out.mkdir(exist_ok=True)
                (arm_out/'checkpoints').mkdir(exist_ok=True)
                arm_context = {**context, 'arm': arm}
                model, _, effective = b.build_model(b.MODEL, cfg['image_size'], 'cuda', cfg['evaluation'])
                optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
                fork = fork_state(model, optimizer, payload)
                forks.append(fork)
                d.require(all(f == fork for f in forks), 'Arms do not start from identical states')
                if (arm_out/'fork.json').exists(): d.require(json.loads((arm_out/'fork.json').read_text()) == fork, 'Fork changed on resume')
                else: b.write_json(arm_out/'fork.json', fork)
                if (arm_out/'last.json').exists():
                    progress = b.restore(arm_out, model, optimizer, arm_context)
                    observer.verify_updates(arm_out, progress['controller_updates'], train_rows, plans, cfg)
                    verify_validations(arm_out, progress['history'])
                else:
                    progress = new_progress(cfg, len(train_rows))
                    b.commit(arm_out, model, optimizer, progress, arm_context, 'fork')
                common = out/'common_warmup'
                if arm == cfg['arms'][0] and not (common/'index.json').exists():
                    # This runs only from the unchanged warmup, never a resumed trained model.
                    d.require(progress['epoch'] == cfg['warmup_epoch'] and progress['next_batch'] == 0, 'Missing common warmup validation after training began')
                    result = validate(model, optimizer, val, cfg, 'cuda', stop)
                    if result is None:
                        record['status'] = 'budget_stopped'; break
                    common.mkdir(exist_ok=True)
                    b.write_json(common/'index.json', save_validation(common, 1, result))
                if arm == cfg['arms'][0] and (common/'index.json').exists():
                    verify_validations(common, [{'validation_records': json.loads((common/'index.json').read_text())}])
                train = Dataset(args.root, train_rows, cfg, arm)
                print(f'ARM {arm}: start completed_epoch={progress["epoch"]} next_batch={progress["next_batch"]}', flush=True)
                status = fit(model, optimizer, train, val, cfg, 'cuda', arm_out, arm_context, progress, plans, stop)
                b.write_json(arm_out/'run.json', {'status': status, 'arm': arm, 'effective_model': effective,
                    'completed_epoch': progress['epoch'], 'step': progress['step'], 'training_seconds': progress['training_seconds'],
                    'validation_seconds': progress['validation_seconds'], 'controller_updates': progress['controller_updates']})
                histories[arm] = progress['history']
                if status == 'complete':
                    anchor = histories[cfg['arms'][0]]
                    d.require([h['augmentation']['pairing_sha256'] for h in progress['history']] ==
                              [h['augmentation']['pairing_sha256'] for h in anchor], 'Cross-arm pairing failed')
                del model, optimizer, train
                gc.collect(); torch.cuda.empty_cache()
                if status != 'complete': record['status'] = status; break
            else:
                summary = paired_summary(histories, cfg)
                b.write_json(out/'comparison.json', summary)
                b.write_json(out/'COMPLETE.json', {'status': 'complete', 'arms': cfg['arms'], 'additional_epochs_each': 3,
                             'full_experiment_authorized': False})
                record['status'] = 'complete'
        except Exception:
            record.update(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic()-started
            record['finished_utc'] = datetime.now(timezone.utc).isoformat()
            b.write_json(out/f'attempt_{attempt}.json', record)
            b.write_json(out/'status.json', record)
            if record['status'] in ('complete', 'budget_stopped'):
                b.archive_records(out, attempt)
            print(f'FOUR-ARM PILOT: {record["status"].upper()}; inspect comparison and controller records before scaling.', flush=True)
    return 0 if record['status'] == 'complete' else 3


if __name__ == '__main__':
    raise SystemExit(main())
