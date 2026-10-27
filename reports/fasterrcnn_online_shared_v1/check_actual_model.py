"""Synthetic CPU check of the actual pretrained detector and full probe sweep."""
import copy
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'fasterrcnn_online_shared_v1'
sys.path.insert(0, str(PACKAGE/'tests'))
import test_baseline as fixtures
import baseline as b
import checkpoints
import controller as c
import probe_detector as p
import probe_runner as r
import torch


def main():
    started = time.monotonic()
    fixture = fixtures.BaselineTests(); fixture.setUp()
    try:
        torch.set_num_threads(2)
        cfg = json.loads((PACKAGE/'protocol.json').read_text())
        cfg.update(image_size=128, workers=0, epochs=1, probe_prediction_parity=True)
        cfg['augmentation_policy']['warmup_epochs'] = 0
        cfg['controller']['panel_images'] = 4
        train = b.Dataset(fixture.root, fixture.train_rows, 'train', 128)
        val = b.Dataset(fixture.root, fixture.val_rows, 'val', 128)
        weights = ROOT/'tmp/detection_learning_v1_weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        assert checkpoints.file_hash(weights) == cfg['weight']['sha256']
        torch.manual_seed(42)
        model, head, effective = b.build_model(b.MODEL, 128, 'cpu', cfg['evaluation'], weights)
        initial = p.tensor_digest(model.state_dict().items())
        spec = importlib.util.spec_from_file_location('identity_models', ROOT/'fasterrcnn_baseline_v1/models.py')
        reference_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference_module)
        torch.manual_seed(42)
        reference, _, _ = reference_module.build_model(b.MODEL, 128, 'cpu', cfg['evaluation'], weights)
        assert p.tensor_digest(reference.state_dict().items()) == initial
        del reference
        initial_head = head.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)
        images, targets, _ = b.collate([train[0], train[1], train[4]])
        losses, norm = b.update(model, optimizer, images, targets, 'cpu')
        assert not torch.equal(head, initial_head)
        before = b.infer(model, b.MODEL, images, cfg['evaluation'], 'cpu')
        out = fixture.output('actual_online')
        context = {'synthetic_cpu': True, 'configuration': cfg}
        entry = r.run_update(model, train, cfg, 'cpu', out, 0, 1, context)
        assert entry['probe_image_forwards'] == 112
        assert r.validate_saved_updates(out, [entry], cfg, context) == entry['policy']
        after = b.infer(model, b.MODEL, images, cfg['evaluation'], 'cpu')
        for a, z in zip(before, after):
            for key in a: assert torch.equal(a[key], z[key])
        validation = b.validate(model, val, cfg, 'cpu')
        progress = b.initial_progress()
        progress.update(step=1, next_batch=1, images_seen=3, epoch_images=3, controller_updates=[entry])
        record = b.commit(out, model, optimizer, progress, context, 'initial')
        restored, _, _ = b.build_model(b.MODEL, 128, 'cpu', cfg['evaluation'], None)
        opt = torch.optim.SGD(restored.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)
        replay = b.restore(out, restored, opt, context)
        assert replay == progress
        r.validate_saved_updates(out, replay['controller_updates'], cfg, context)
        assert b.validate(restored, val, cfg, 'cpu') == validation
        # Duplicate GT boxes and a missed box retain separate supervised loss observations.
        duplicate = copy.deepcopy(train[0])
        import numpy as np
        duplicate.boxes = np.concatenate((duplicate.boxes, duplicate.boxes, np.array([[100., 100., 120., 120.]])))
        duplicate.labels = np.array([0, 0, 1], dtype=np.int64)
        with p.frozen_probe(restored):
            observed = p.observe(restored, [duplicate], cfg['controller'], 'cpu', True)[0]
        assert len(observed['loss']) == 3 and all(n > 0 for n in observed['roi_count'])
        report = {'status': 'passed', 'scope': 'Synthetic 128-pixel CPU data; no RDD pixels, CUDA or efficacy measurement',
                  'seconds': time.monotonic()-started, 'initialization_matches_identity_exactly': True,
                  'standard_predictions_unchanged_before_after_probe': True, 'hook_prediction_parity': 'exact',
                  'complete_probe_views': 112, 'parity_extra_views': 2, 'operations': list(c.OPERATIONS), 'strength_levels': 3,
                  'noise_draws_per_strength': 2, 'model_buffers_gradients_rng_unchanged': True,
                  'policy_recomputed_from_saved_evidence': True, 'duplicate_and_missed_gt_losses_present': True,
                  'fresh_model_checkpoint_prediction_policy_replay': 'exact',
                  'training_update': {'losses': losses, 'gradient_norm': norm, 'includes_negative_image': True},
                  'support': entry['support'], 'active_families': entry['policy']['families'],
                  'checkpoint_sha256': record['sha256'], 'versions': {k: version(k) for k in cfg['versions']}}
        (Path(__file__).parent/'actual_model_check.json').write_bytes(b.json_bytes(report))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        fixture.temp.cleanup()


if __name__ == '__main__':
    main()
