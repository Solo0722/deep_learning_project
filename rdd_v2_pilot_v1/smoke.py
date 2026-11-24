"""GPU gate: four tiny arms, real transforms/probes/validation and mid-run restore."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

import torch

import baseline as b
import checkpoints
import diagnostic as d
import observer
import pilot


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'splits', 'checkpoint', 'out'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    full = json.loads((pilot.PACKAGE/'protocol.json').read_text())
    pilot.verify_package(); versions = pilot.runtime(full)
    out = pilot.check_output(args.out, [args.root, args.splits, args.checkpoint, pilot.PACKAGE])
    out.mkdir(parents=True, exist_ok=False)
    rows, val_rows, payload, full_plans = pilot.load_inputs(args.root, args.splits, args.checkpoint, full)
    chosen = json.loads((pilot.PACKAGE/'smoke_selection.json').read_text())
    # Include the exact panel images in the smoke train selection. Test identities
    # are training-only; no relabeling or use of validation images in the probe.
    plan = full_plans['2'][:2]
    names = sorted(set(chosen['train']) | {s['image'] for s in plan})
    cfg = copy.deepcopy(full)
    cfg.update(epochs=3, update_epochs=[2, 3], workers=0, checkpoint_every_steps=1000,
               log_every_steps=1000, warmup_steps=0)
    plans = {'2': plan, '3': plan}
    val = b.Dataset(args.root, val_rows, 'val', cfg['image_size'], chosen['val'])
    histories, forks, report = {}, [], {}
    deadline = time.monotonic()+1800
    stop = lambda: time.monotonic() >= deadline or time.time() >= float(os.environ.get('RDD_PILOT_DEADLINE', 'inf'))
    for arm in cfg['arms']:
        target = out/arm
        target.mkdir(); (target/'checkpoints').mkdir()
        train = pilot.Dataset(args.root, rows, cfg, arm, names)
        model, _, _ = b.build_model(b.MODEL, cfg['image_size'], 'cuda', cfg['evaluation'])
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        forks.append(pilot.fork_state(model, optimizer, payload))
        d.require(all(f == forks[0] for f in forks), 'Smoke warmup forks differ')
        progress = pilot.new_progress(cfg, len(train))
        context = {'mode': 'GPU smoke only; subset counters; no smoke checkpoint used by full pilot', 'cfg': cfg, 'arm': arm, 'versions': versions}
        b.commit(target, model, optimizer, progress, context, 'fork')
        if arm == 'class_adaptive':
            initial_step = progress['step']
            status = pilot.fit(model, optimizer, train, val, cfg, 'cuda', target, context, progress, plans,
                               lambda: stop() or progress['step'] >= initial_step+1)
            d.require(status == 'budget_stopped' and progress['next_batch'] == 1, 'Smoke interruption was not mid-epoch')
            image = b.Dataset(args.root, rows, 'train', cfg['image_size'], [names[0]])[0]
            images = b.collate([image])[0]
            with observer.frozen(model, optimizer): before = b.infer(model, b.MODEL, images, cfg['evaluation'], 'cuda')
            del model, optimizer
            gc.collect(); torch.cuda.empty_cache()
            model, _, _ = b.build_model(b.MODEL, cfg['image_size'], 'cuda', cfg['evaluation'])
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
            progress = b.restore(target, model, optimizer, context)
            observer.verify_updates(target, progress['controller_updates'], rows, plans, cfg)
            with observer.frozen(model, optimizer): after = b.infer(model, b.MODEL, images, cfg['evaluation'], 'cuda')
            for left, right in zip(before, after):
                d.require(torch.equal(left['labels'], right['labels']), 'Reload inference labels differ')
                for key in ('boxes', 'scores'):
                    d.require(left[key].shape == right[key].shape and torch.allclose(left[key], right[key], rtol=0, atol=1e-5), 'Reload inference differs')
            report['restore'] = {'model_optimizer_rng_exact': True, 'prediction_parity_atol': 1e-5,
                                 'mid_epoch_resume': True, 'CUDA_trajectory_bitwise_claim': False}
        d.require(pilot.fit(model, optimizer, train, val, cfg, 'cuda', target, context, progress, plans, stop) == 'complete', 'GPU smoke incomplete')
        observer.verify_updates(target, progress['controller_updates'], rows, plans, cfg)
        histories[arm] = progress['history']
        del model, optimizer, train
        gc.collect(); torch.cuda.empty_cache()
    pilot.paired_summary(histories, cfg)
    b.write_json(out/'PASS.json', {'status': 'pass', 'scope': 'Actual CUDA four-arm pipeline on training/validation subsets, not learning evidence',
        'forks_identical': True, 'geometry_order_opportunities_paired': True, 'report': report,
        'subset_train_images': len(names), 'subset_val_images': len(val), 'additional_epochs': 2,
        'probe_images_each_update': len(plan), 'thresholds_unchanged': True,
        'online_support_warning': 'Tiny smoke panels may fall back to identity; full pilot uses 256 images.',
        'package_manifest_sha256': d.file_hash(pilot.PACKAGE/'SHA256SUMS')})
    print('GPU FOUR-ARM SMOKE: PASS. Full pilot will restart from the original warmup file.', flush=True)


if __name__ == '__main__': main()
