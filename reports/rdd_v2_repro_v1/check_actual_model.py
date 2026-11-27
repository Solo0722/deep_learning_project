"""Actual Faster R-CNN CPU comparison against the unchanged baseline update."""
import gc
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'rdd_v2_repro_v1'))
import repro as r
import trace_state as ts


def main():
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    cfg = r.read(r.pilot.PACKAGE/'protocol.json')
    model, _, effective = r.b.build_model(r.b.MODEL, 640, 'cpu', cfg['evaluation'])
    optimizer = torch.optim.SGD(model.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)
    images = [torch.rand(3, 640, 640) for _ in range(2)]
    targets = [{'boxes': torch.tensor([[60.,70.,110.,490.],[100.,200.,510.,240.],[260.,300.,390.,420.],[420.,100.,510.,160.]]),
                'labels': torch.tensor([1,2,3,4])},
               {'boxes': torch.empty(0,4), 'labels': torch.empty(0,dtype=torch.long)}]
    payload = {'format': 1, 'model': ts.cpu(model.state_dict()),
               'optimizer': ts.cpu(optimizer.state_dict()), 'rng': r.cp.rng_state()}
    batch = {'images': images, 'targets': targets, 'sha256': r.batch_digest(images, targets)}
    started = time.monotonic()
    losses, norm = r.b.update(model, optimizer, images, targets, 'cpu')
    expected = {'model': ts.cpu(model.state_dict()), 'optimizer': ts.cpu(optimizer.state_dict()),
                'rng': r.cp.rng_state(), 'gradients': ts.cpu({n:p.grad for n,p in model.named_parameters()})}
    original_seconds = time.monotonic()-started
    r.pilot.fork_state(model, optimizer, payload)
    started = time.monotonic()
    observed = r.traced_update(model, optimizer, batch, 'cpu')
    actual = {k: observed['update'][k] for k in ('model','optimizer','rng')}
    actual['gradients'] = observed['backward']['gradients']
    comparison, _ = ts.compare(expected, actual)
    loss_equal = {k:float(v) for k,v in observed['forward']['losses'].items()} == losses
    norm_equal = observed['backward']['grad_norm'] == norm
    result = {'status':'pass' if comparison['equal'] and loss_equal and norm_equal else 'fail',
        'scope':'Actual randomly initialized Faster R-CNN, 640 pixels, two synthetic images with four classes and an empty target; one original update versus one instrumented update from the same state. CPU only, not CCAST CUDA evidence.',
        'versions':{k:version(k) for k in ('torch','torchvision','numpy','Pillow')},
        'losses':losses,'gradient_norm':norm,'losses_exact':loss_equal,'gradient_norm_exact':norm_equal,
        'model_optimizer_rng_gradients_exact':comparison['equal'],
        'comparison':comparison,'original_seconds':original_seconds,
        'instrumented_and_comparison_seconds':time.monotonic()-started,'effective_model':effective}
    r.write(Path(__file__).with_name('actual_model_check.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('effective_model','comparison')},indent=2),flush=True)
    assert result['status']=='pass'


if __name__=='__main__':
    main()
