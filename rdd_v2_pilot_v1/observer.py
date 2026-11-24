"""Ordinary detections from the current arm; transactional training-only evidence."""
from contextlib import contextmanager
import copy
import gzip
import json
import time
import uuid

import numpy as np
import torch

import baseline as b
import checkpoints
import controller
import diagnostic as d
from detector import tensor_digest, rng_digest
import rdd_data


@contextmanager
def frozen(model, optimizer=None):
    modes = [(m, m.training) for m in model.modules()]
    state = tensor_digest(model.state_dict().items())
    grads = tensor_digest((n, p.grad) for n, p in model.named_parameters())
    rng = checkpoints.rng_state()
    rng_hash = rng_digest()
    opt = checkpoints.cpu_tree(optimizer.state_dict()) if optimizer is not None else None
    audit = {}
    model.eval()
    try:
        with torch.inference_mode():
            yield audit
    finally:
        after_rng = rng_digest()
        checkpoints.restore_rng(rng)
        for module, training in modes:
            module.training = training
        d.require(tensor_digest(model.state_dict().items()) == state, 'Probe/validation changed model state')
        d.require(tensor_digest((n, p.grad) for n, p in model.named_parameters()) == grads, 'Probe/validation changed gradients')
        d.require(after_rng == rng_hash, 'Probe/validation consumed training RNG')
        if opt is not None:
            checkpoints.assert_same(optimizer.state_dict(), opt, 'observer optimizer')
        audit.update(model_sha256=state, gradient_sha256=grads, rng_sha256=rng_hash,
                     model_optimizer_rng_modes_unchanged=True)


class CurrentDetector:
    def __init__(self, model, device, limit):
        self.model, self.device, self.limit = model, device, limit
        self.images_forwarded, self.forward_calls, self.forward_seconds = 0, 0, 0.

    def predict(self, images):
        d.require(not torch.is_grad_enabled() and not self.model.training, 'Observer must be frozen')
        d.require(self.images_forwarded+len(images) <= self.limit, 'Probe forward cap exceeded')
        tensors = [torch.from_numpy(im.transpose(2, 0, 1).copy()).to(self.device, dtype=torch.float32)/255. for im in images]
        if str(self.device).startswith('cuda'): torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = self.model(tensors)  # No GT/target argument, proposal injection or loss.
        if str(self.device).startswith('cuda'): torch.cuda.synchronize()
        self.forward_seconds += time.perf_counter()-start
        self.images_forwarded += len(images)
        self.forward_calls += 1
        return [{'boxes': p['boxes'].detach().cpu().tolist(), 'scores': p['scores'].detach().cpu().tolist(),
                 'labels': (p['labels']-1).detach().cpu().tolist()} for p in outputs]


def collect(model, optimizer, root, rows, plan, cfg, epoch, device, out, should_stop):
    """A partial probe is never adopted. Its raw evidence and cost remain saved."""
    folder = out/'controller'
    folder.mkdir(exist_ok=True)
    stem = f'epoch_{epoch:03d}_{uuid.uuid4().hex[:12]}'
    path = folder/(stem+'.jsonl.gz')
    report_path = folder/(stem+'_attempt.json')
    start = time.monotonic()
    detector = CurrentDetector(model, device, len(plan)*cfg['views_per_image']+4)
    probe_cfg = {**cfg, 'geometry_epoch_index': epoch-1}
    lookup = {r['image']: r for r in rows}
    records, parity, audit = [], None, {}
    status, error = 'running', None
    try:
        with frozen(model, optimizer) as audit, gzip.open(path, 'xt', encoding='utf-8') as handle:
            for number, selection in enumerate(plan, 1):
                if should_stop():
                    status = 'budget_stopped'
                    return None
                row = lookup[selection['image']]
                example = rdd_data.prepare_example(rdd_data.load_example(root, row), cfg['image_size'])
                p = {'image': row['image'], 'source_cluster_id': row['source_cluster_id'],
                     'classes': [t['class_index'] for t in row['targets']],
                     'object_indices': [t['object_index'] for t in row['targets']],
                     'target_positions': selection['target_positions'], 'selection': selection, 'views': {}}
                views = list(d.action_views(example, selection, probe_cfg))
                d.require(len(views) == cfg['views_per_image'], 'Probe view count mismatch')
                if parity is None:
                    ref = views[1][1]
                    parity = d.check_prediction_parity(detector.predict([ref, ref]), detector.predict([ref, ref]), cfg['parity_tolerance'])
                for offset in range(0, len(views), cfg['probe_batch_size']):
                    if should_stop():
                        status = 'budget_stopped'
                        return None
                    chunk = views[offset:offset+cfg['probe_batch_size']]
                    outputs = detector.predict([pixels for _, pixels, _ in chunk])
                    d.require(len(outputs) == len(chunk), 'Missing probe prediction')
                    for (name, pixels, meta), pred in zip(chunk, outputs):
                        value = d.match_prediction(pred, meta['boxes'], meta['labels'], cfg['controller'])
                        value.update(prediction=pred, pixel_sha256=rdd_data.sha256(pixels.tobytes()), render=meta)
                        if name in ('clean', 'reference'): p[name] = value
                        else: p['views'][name] = value
                handle.write(json.dumps(p, separators=(',', ':'), allow_nan=False)+'\n')
                handle.flush()
                records.append(p)
                if number == 1 or number % 32 == 0:
                    print(f'PROBE before_epoch={epoch} images={number}/{len(plan)} forwards={detector.images_forwarded}', flush=True)
        d.require(detector.images_forwarded == len(plan)*cfg['views_per_image']+4, 'Incomplete probe budget')
        units = controller.units_from_panel(records, lookup)
        policy = controller.build_policy(units, cfg['controller'])
        status = 'complete'
        entry = {'epoch': epoch, 'file': path.relative_to(out).as_posix(), 'sha256': d.file_hash(path),
                 'plan_sha256': controller.digest(plan), 'policy': policy, 'audit': audit,
                 'parity_max_abs': parity, 'detector_images': detector.images_forwarded,
                 'forward_calls': detector.forward_calls, 'forward_seconds': detector.forward_seconds,
                 'elapsed_seconds': time.monotonic()-start}
        for label, cell in policy['pooled'].items():
            print(f"SUPPORT epoch={epoch} class={label} detected={cell['detected_objects']} clusters={cell['detected_clusters']} supported={cell['supported']} allowed={sum(cell['allowed'])}", flush=True)
        return entry
    except Exception as exc:
        status, error = 'failed', repr(exc)
        raise
    finally:
        b.write_json(report_path, {'status': status, 'error': error, 'epoch': epoch,
            'records_complete': len(records), 'detector_images': detector.images_forwarded,
            'forward_calls': detector.forward_calls, 'forward_seconds': detector.forward_seconds,
            'elapsed_seconds': time.monotonic()-start, 'audit': audit,
            'file': path.name, 'sha256': d.file_hash(path) if path.exists() else None})


def verify_updates(out, entries, rows, plans, cfg):
    lookup = {r['image']: r for r in rows}
    epochs = [entry['epoch'] for entry in entries]
    d.require(epochs == sorted(set(epochs)) and set(epochs) <= set(cfg['update_epochs']), 'Invalid adopted update sequence')
    for entry in entries:
        path = rdd_data.inside(out, entry['file'])
        d.require(d.file_hash(path) == entry['sha256'], 'Controller evidence changed')
        plan = plans[str(entry['epoch'])]
        d.require(entry['plan_sha256'] == controller.digest(plan), 'Controller panel changed')
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle]
        d.require([p['selection'] for p in records] == plan, 'Controller record coverage changed')
        policy = controller.build_policy(controller.units_from_panel(records, lookup), cfg['controller'])
        d.require(policy == entry['policy'], 'Saved policy does not match recorded evidence')
