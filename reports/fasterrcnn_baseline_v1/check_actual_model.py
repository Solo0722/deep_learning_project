"""Local synthetic CPU check of the actual pretrained model and new validation path."""
import json
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'fasterrcnn_baseline_v1/tests'))
from test_baseline import BaselineTests
import baseline as b
import checkpoints
import torch

fixture = BaselineTests()
fixture.setUp()
try:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = {**fixture.cfg, 'image_size': 128}
    train = b.Dataset(fixture.root, fixture.train_rows, 'train', 128)
    val = b.Dataset(fixture.root, fixture.val_rows, 'val', 128)
    weight = ROOT / 'tmp/detection_learning_v1_weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    assert checkpoints.file_hash(weight) == cfg['weight']['sha256']
    model, head, effective = b.build_model(b.MODEL, 128, 'cpu', cfg['evaluation'], weight)
    initial_head = head.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)
    updates = []
    for indices in ([0, 4], [4, 4]):
        images, targets, _ = b.collate([train[i] for i in indices])
        losses, grad_norm = b.update(model, optimizer, images, targets, 'cpu')
        updates.append({'target_counts': [len(t['boxes']) for t in targets], 'losses': losses, 'gradient_norm': grad_norm})
    assert not torch.equal(initial_head, head)
    before = checkpoints.cpu_tree(model.state_dict())
    result = b.validate(model, val, cfg, 'cpu')
    checkpoints.assert_same(model.state_dict(), before)
    out = fixture.output('real_checkpoint')
    context = {'synthetic': True, 'image_size': 128}
    record = b.commit(out, model, optimizer, b.initial_progress(), context, 'initial')
    restored, _, _ = b.build_model(b.MODEL, 128, 'cpu', cfg['evaluation'], None)
    resumed_optimizer = torch.optim.SGD(restored.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)
    b.restore(out, restored, resumed_optimizer, context)
    repeated = b.validate(restored, val, cfg, 'cpu')
    assert repeated == result
    report = {'status': 'passed', 'scope': 'Synthetic 128-pixel CPU inputs; no RDD pixels, CCAST execution or efficacy claim',
              'versions': {k: version(k) for k in cfg['versions']}, 'updates': updates,
              'all_empty_batch_included': True, 'classifier_updated': True, 'validation_images': len(val),
              'validation_did_not_update_model': True, 'fresh_model_checkpoint_prediction_replay': 'exact',
              'checkpoint_sha256': record['sha256'], 'model_parameters': effective['parameters']}
    path = Path(__file__).with_name('local_model_check.json')
    path.write_bytes(b.json_bytes(report))
    print(json.dumps(report, indent=2))
finally:
    fixture.doCleanups()
