"""Bounded forward probes, durable evidence, and policy checkpoint validation."""
import copy
import gzip
import json
import os
from pathlib import Path
import time
import uuid

import controller as c
from integration import require
import probe_detector
from rdd_data import inside, load_example, prepare_example, sha256


def atomic(path, data):
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def json_data(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def due(cfg, epoch):
    warmup = cfg['augmentation_policy']['warmup_epochs']
    return epoch >= warmup and (epoch-warmup) % cfg['controller']['update_interval_epochs'] == 0


def run_update(model, dataset, cfg, device, out, epoch, step, context, should_stop=lambda: False):
    settings = cfg['controller']
    panel = c.select_panel(dataset.rows, cfg['image_size'], cfg['seed'], epoch, settings['panel_images'])
    rows = {r['image']: (i, r) for i, r in dataset.rows}
    folder = Path(out)/'controller'
    folder.mkdir(exist_ok=True)
    attempt = f"epoch_{epoch+1:03d}_{uuid.uuid4().hex[:12]}"
    cost_path = folder/f'{attempt}_attempt.json'
    started = time.monotonic()
    cost = {'epoch': epoch+1, 'status': 'running', 'probe_image_forwards': 0, 'parity_image_forwards': 0,
            'completed_panel_images': 0, 'expected_probe_image_forwards': len(panel)*(1+len(c.variants(settings))),
            'optimizer_steps_at_start': step, 'context_sha256': c.digest(context), 'panel_sha256': c.digest(panel)}
    atomic(cost_path, json_data(cost))
    def save_cost():
        cost['elapsed_seconds'] = time.monotonic()-started
        atomic(cost_path, json_data(cost))
    complete = []
    try:
        with probe_detector.frozen_probe(model) as invariants:
            for start in range(0, len(panel), settings['probe_batch_size']):
                if should_stop():
                    cost['status'] = 'budget_stopped'
                    return None
                chunk = copy.deepcopy(panel[start:start+settings['probe_batch_size']])
                examples = []
                for item in chunk:
                    image_id, row = rows[item['image']]
                    require(image_id == item['image_id'], 'Probe row identity changed')
                    examples.append(prepare_example(load_example(dataset.root, row, image_id, 'train'), cfg['image_size']))
                parity = bool(cfg.get('probe_prediction_parity', False) and start == 0)
                observations = probe_detector.observe(model, examples, settings, device, parity)
                cost['probe_image_forwards'] += len(chunk)
                cost['parity_image_forwards'] += len(chunk) if parity else 0
                for item, example, obs in zip(chunk, examples, observations):
                    item.update(reference={**obs, 'prepared_rgb_sha256': sha256(example.rgb.tobytes())}, views={})
                for name, level, draw in c.variants(settings):
                    if should_stop():
                        cost['status'] = 'budget_stopped'
                        return None
                    transformed = [c.apply_example(ex, c.probe_choice(ex.record, p['group_index'], settings, cfg['seed'], epoch,
                                                                  cfg['image_size'], name, level, draw)) for ex, p in zip(examples, chunk)]
                    observations = probe_detector.observe(model, transformed, settings, device)
                    cost['probe_image_forwards'] += len(chunk)
                    for item, example, obs in zip(chunk, transformed, observations):
                        item['views'][c.view_key(name, level, draw)] = {**obs, 'prepared_rgb_sha256': sha256(example.rgb.tobytes()),
                                                                     'changed_pixels': example.augmentation_pixel_stats['changed_pixels']}
                complete.extend(chunk)
                cost['completed_panel_images'] = len(complete)
                save_cost()
                if len(complete) % 32 == 0 or len(complete) == len(panel):
                    print(f"PROBE epoch={epoch+1} images={len(complete)}/{len(panel)} forwards={cost['probe_image_forwards']}/{cost['expected_probe_image_forwards']}", flush=True)
        require(cost['probe_image_forwards'] == cost['expected_probe_image_forwards'], 'Incomplete controller views')
        result = c.aggregate(complete, settings, epoch)
        record = {'format': 1, 'epoch': epoch+1, 'optimizer_steps_at_start': step, 'settings_sha256': c.digest(settings),
                  'context_sha256': c.digest(context), 'panel_sha256': cost['panel_sha256'], 'panel': complete,
                  'result': result, 'invariants': {**invariants, 'model_buffers_gradients_rng_unchanged': True, 'eval_inference_only': True},
                  'probe_image_forwards': cost['probe_image_forwards'], 'parity_image_forwards': cost['parity_image_forwards']}
        path = folder/f'{attempt}.json.gz'
        data = gzip.compress(json_data(record), mtime=0)
        with path.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        cost.update(status='complete', record=path.name, record_sha256=sha256(data), policy_id=result['policy']['id'])
        entry = {'epoch': epoch+1, 'step': step, 'file': path.relative_to(out).as_posix(), 'sha256': sha256(data),
                 'policy': result['policy'], 'support': result['support'], 'probe_image_forwards': cost['probe_image_forwards']}
        summary = c.policy_summary(result['policy'])
        print(f"CONTROLLER epoch={epoch+1} policy={result['policy']['id'][:12]} class_sets={summary} support={result['support']}", flush=True)
        return entry
    except BaseException as error:
        cost.update(status='failed', error=str(error))
        raise
    finally:
        save_cost()


def validate_saved_updates(out, entries, cfg, context):
    previous = 0
    for entry in entries:
        require(entry['epoch'] > previous and due(cfg, entry['epoch']-1), 'Invalid controller update schedule')
        previous = entry['epoch']
        path = inside(Path(out), entry['file'])
        data = path.read_bytes()
        require(sha256(data) == entry['sha256'], 'Saved probe evidence changed')
        record = json.loads(gzip.decompress(data))
        require(record['context_sha256'] == c.digest(context) and record['settings_sha256'] == c.digest(cfg['controller']), 'Probe context/settings changed')
        require(record['epoch'] == entry['epoch'] and record['optimizer_steps_at_start'] == entry['step'], 'Probe step/epoch changed')
        result = c.aggregate(record['panel'], cfg['controller'], record['epoch']-1)
        require(result == record['result'] and result['policy'] == entry['policy'] and result['support'] == entry['support'], 'Policy cannot be recomputed from saved observations')
        require(record['probe_image_forwards'] == entry['probe_image_forwards'], 'Probe compute count changed')
    return entries[-1]['policy'] if entries else None
