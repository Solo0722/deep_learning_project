"""Actual Faster R-CNN CPU gate and independent baseline-update comparison."""
import gc
from importlib.metadata import version
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'rdd_v2_strict_v1'))
import strict_check as s


def main():
    torch.set_num_threads(2)
    protocol = s.r.read(s.PACKAGE/'protocol.json')
    s.configure_strict(protocol, fresh_cuda=False)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    cfg = s.r.read(s.r.pilot.PACKAGE/'protocol.json')
    cfg.update(workers=0, warmup_steps=0)
    def factory():
        model, _, _ = s.r.b.build_model(s.r.b.MODEL, 640, 'cpu', cfg['evaluation'])
        return model, torch.optim.SGD(model.parameters(),lr=.0025,momentum=.9,weight_decay=.0005)
    model, optimizer = factory()
    examples = []
    generator = np.random.default_rng(42)
    for i in range(4):
        boxes = np.array([[60,70,110,490],[100,200,510,240],[260,300,390,420],[420,100,510,160]],dtype=np.float32) if i==0 else np.empty((0,4),dtype=np.float32)
        labels = np.arange(4,dtype=np.int64) if i==0 else np.empty(0,dtype=np.int64)
        example = s.r.rdd_data.Example({'image':f'synthetic_{i}.png'},i,
            generator.integers(0,256,(640,640,3),dtype=np.uint8),boxes,labels)
        examples.append(s.r.rdd_data.prepare_example(example,640))
    images, targets, _ = s.r.rdd_data.torchvision_collate(examples)
    payload = {'format':1,'model':s.ts.cpu(model.state_dict()),'optimizer':s.ts.cpu(optimizer.state_dict()),'rng':s.r.cp.rng_state()}
    batch = {'images':images,'targets':targets,'sha256':s.r.batch_digest(images,targets)}
    # Reuse seven references for accounting only: the gate consumes just the first.
    # Three size-4 pairs plus the final size-2 negative batch reproduce the 26-image counter.
    small = {'images':images[2:],'targets':targets[2:]}
    small['sha256'] = s.r.batch_digest(small['images'],small['targets'])
    batches = [batch]*6+[small]
    temporary_parent = ROOT/'tmp/rdd_strict_actual_model'
    temporary_parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_parent,prefix='actual_') as temporary:
        folder = Path(temporary).resolve()
        assert temporary_parent.resolve() in folder.parents
        # The baseline update runs independently from the same original state.
        s.r.pilot.fork_state(model,optimizer,payload)
        started = time.monotonic()
        losses, norm = s.r.b.update(model,optimizer,images,targets,'cpu')
        original_seconds = time.monotonic()-started
        reference = {'model':s.ts.cpu(model.state_dict()),'optimizer':s.ts.cpu(optimizer.state_dict()),
            'rng':s.r.cp.rng_state(),'gradients':s.ts.cpu({n:p.grad for n,p in model.named_parameters()})}
        torch.save(reference,folder/'baseline.pt')
        del reference,model,optimizer
        gc.collect()
        meter = s.Meter(folder,protocol,'cpu')
        result = s.run_stage(folder/'gate',protocol['stages'][0],cfg,protocol,payload,batches,
                             factory,'cpu',lambda *_:None,meter)
        observed = torch.load(folder/'gate/control_a/reference_steps/step_01.pt',map_location='cpu',weights_only=True)
        expected = torch.load(folder/'baseline.pt',map_location='cpu',weights_only=True)
        actual = {k:observed['update'][k] for k in ('model','optimizer','rng')}
        actual['gradients'] = observed['backward']['gradients']
        same, _ = s.ts.compare(expected,actual)
        loss_equal = losses == {k:float(v) for k,v in observed['forward']['losses'].items()}
        norm_equal = norm == observed['backward']['grad_norm']
        assert same['equal'] and loss_equal and norm_equal and result['status']=='complete_exact_on_this_diagnostic'
        resources = {c:s.r.read(folder/f'gate/{c}/steps/01_resources.json') for c in ['control_a','control_b']}
        output = {'status':'pass','scope':'Randomly initialized real Faster R-CNN, four synthetic 640x640 RGB images, four classes and three negative images through production preparation/collation. Original baseline update versus new strict instrumentation and two independent gate repeats. CPU only.',
            'versions':{k:version(k) for k in ('torch','torchvision','numpy','Pillow')},
            'strict_flags':s.backend_state(),'original_baseline_update_seconds':original_seconds,
            'model_optimizer_rng_gradients_exact':same['equal'],'losses_exact':loss_equal,'gradient_norm_exact':norm_equal,
            'gate':result,'resources':resources,'losses':losses,'gradient_norm':norm,
            'target_counts':[len(t['boxes']) for t in targets],
            'target_fields':[sorted(t) for t in targets],
            'cuda_tested':False,'deterministic_cuda_roi_align_path_tested':False}
        s.r.write(Path(__file__).with_name('actual_model_check.json'),output)
        print('ACTUAL FASTER R-CNN CPU GATE AND ORIGINAL-UPDATE EQUIVALENCE: PASS',flush=True)


if __name__ == '__main__':
    main()
