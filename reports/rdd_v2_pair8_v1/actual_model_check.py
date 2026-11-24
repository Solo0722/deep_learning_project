"""Real Faster R-CNN CPU integration check with synthetic images; no RDD accuracy claim."""
import copy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'rdd_v2_pair8_v1'
sys.path[:0]=[str(PKG),str(PKG/'tests')]
import numpy as np
import torch
import torchvision
import baseline as b
import checkpoints
import gate
import strict_runtime as strict
import test_baseline


def main():
    cfg=json.loads((PKG/'protocol.json').read_text())
    strict.configure(cfg)
    torch.set_num_threads(2)
    fixture=test_baseline.BaselineTests();fixture.setUp()
    torch.set_num_threads(2)
    assert ROOT.resolve() in fixture.root.resolve().parents
    cfg.update(epochs=2,update_epochs=[2],workers=0,warmup_steps=0,
               checkpoint_every_steps=1000,log_every_steps=1000)
    rows=fixture.train_rows
    val=b.Dataset(fixture.root,fixture.val_rows,'val',640)
    plan=[]
    for i,row in enumerate(rows[:2]):
        plan.append({'image':row['image'],'source':row['source'],'source_cluster_id':row['source_cluster_id'],
            'group_index':0,'object_indices':[0],'target_positions':[0],
            'anchor_class':i,'group_classes':[i],'slot':0,'panel':'synthetic_cpu_check'})
    def factory():
        model,_,_=b.build_model(b.MODEL,640,'cpu',cfg['evaluation'])
        return model,torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model,opt=factory()
    payload={'format':1,'model':checkpoints.cpu_tree(model.state_dict()),
        'optimizer':checkpoints.cpu_tree(opt.state_dict()),'rng':checkpoints.rng_state()}
    del model,opt;gc.collect()
    started=time.monotonic()
    result=gate.compare_run(fixture.root,rows,val,payload,{'2':plan},sorted(r['image'] for r in rows),cfg,
        fixture.root/'actual_gate',factory,'cpu',lambda:time.monotonic()-started>1200)
    result.update(status='pass',elapsed_seconds=time.monotonic()-started,
        torch=torch.__version__,torchvision=torchvision.__version__,python=sys.version,
        device='cpu',image_size=640,batch_size=4,actual_train_batches=[4,1],
        limitations=['Synthetic images, random weights, local CPU versions; no CCAST GPU execution or RDD learning conclusion.',
                    'The CCAST entry gate uses pinned epoch-1 weights, real selected train/val images and production workers=2.'])
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    fixture.temp.cleanup()


if __name__=='__main__':main()
