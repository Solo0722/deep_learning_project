"""Actual pretrained CPU model with all fixed operations and exact reload replay."""
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'fasterrcnn_local_v1/tests'))
import test_baseline as fixtures
import baseline as b
import checkpoints
import local_augmentation as aug
import torch


def main():
    fixture = fixtures.BaselineTests()
    fixture.setUp()
    try:
        torch.set_num_threads(2)
        cfg = {**fixture.cfg, 'image_size': 128}
        train = b.Dataset(fixture.root, fixture.train_rows, 'train', 128)
        val = b.Dataset(fixture.root, fixture.val_rows, 'val', 128)
        weight = ROOT/'tmp/detection_learning_v1_weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
        assert checkpoints.file_hash(weight) == cfg['weight']['sha256']
        torch.manual_seed(42)
        model, head, effective = b.build_model(b.MODEL, 128, 'cpu', cfg['evaluation'], weight)
        initial_head = head.detach().clone()
        spec = importlib.util.spec_from_file_location('identity_models', ROOT/'fasterrcnn_baseline_v1/models.py')
        identity_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(identity_module)
        torch.manual_seed(42)
        reference, _, _ = identity_module.build_model(b.MODEL, 128, 'cpu', cfg['evaluation'], weight)
        checkpoints.assert_same(model.state_dict(), reference.state_dict(), 'paired pretrained initialization')
        del reference
        prepared = []
        for index, operation in enumerate(aug.OPERATIONS):
            example = train[index % 4]
            params = cfg['augmentation_policy']['operations'][operation]
            choice = {'image': example.record['image'], 'epoch': 2, 'target_bearing': True,
                      'operation': operation, 'family': params['family'], 'reason': 'forced_model_check',
                      'parameters': {k: v for k, v in params.items() if k != 'family'},
                      'noise_seed': 42 if operation == 'gaussian_noise' else None}
            choice = aug.attach_local(example.record, choice, 42, 1, 128)
            transformed = aug.apply_example(example, choice)
            assert transformed.augmentation_changed
            prepared.append(transformed)
        optimizer = torch.optim.SGD(model.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)
        updates = []
        for examples in (prepared[:4], prepared[4:], [train[4], train[4]]):
            images, targets, _ = b.collate(examples)
            losses, norm = b.update(model, optimizer, images, targets, 'cpu')
            updates.append({'targets': [len(t['boxes']) for t in targets], 'losses': losses, 'gradient_norm': norm})
        assert not torch.equal(initial_head, head)
        before = checkpoints.cpu_tree(model.state_dict())
        result = b.validate(model, val, cfg, 'cpu')
        checkpoints.assert_same(model.state_dict(), before)
        del before
        out = fixture.output('actual_local_model')
        context = {'synthetic': True, 'condition': 'all_fixed_local_operations', 'size': 128}
        progress = b.initial_progress()
        progress.update(step=3, next_batch=3, images_seen=10, epoch_images=10)
        record = b.commit(out, model, optimizer, progress, context, 'initial')
        restored, _, _ = b.build_model(b.MODEL, 128, 'cpu', cfg['evaluation'], None)
        resumed_optimizer = torch.optim.SGD(restored.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)
        replay = b.restore(out, restored, resumed_optimizer, context)
        assert replay == progress
        assert b.validate(restored, val, cfg, 'cpu') == result
        report = {'status': 'passed', 'scope': 'Synthetic 128-pixel CPU data; no RDD pixels or CCAST execution',
                  'operations_exercised': list(aug.OPERATIONS), 'updates': updates,
                  'initialization_matches_identity_model_exactly': True,
                  'all_negative_batch_included': True, 'classifier_updated': True,
                  'validation_unchanged_and_without_parameter_update': True,
                  'fresh_model_checkpoint_prediction_replay': 'exact',
                  'versions': {k: version(k) for k in cfg['versions']},
                  'checkpoint_sha256': record['sha256'], 'model_parameters': effective['parameters']}
        path = Path(__file__).with_name('local_model_check.json')
        with path.open('xb') as handle:
            handle.write(b.json_bytes(report))
        print(json.dumps(report, indent=2))
    finally:
        fixture.doCleanups()


if __name__ == '__main__':
    main()
