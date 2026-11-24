"""Exact short check through the same fit/loader/update path used by training."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import torch

import augmentation
import baseline as b
import checkpoints
import diagnostic as d
import observer
import pilot
import strict_runtime as strict
import trace_state as ts


def signature(model,optimizer,images,targets,result=None):
    value={'inputs':ts.digest({'images':images,'targets':targets}),
        'model':ts.digest(model.state_dict()),'optimizer':ts.digest(optimizer.state_dict()),
        'rng':ts.digest(checkpoints.rng_state()),
        'modes':{n:m.training for n,m in model.named_modules()}}
    if result is not None:
        value.update(losses=result[0],grad_norm=result[1],
            gradients=ts.digest({n:p.grad for n,p in model.named_parameters()}))
    return value


def compare_run(root,rows,val,payload,plans,names,cfg,out,factory,device,stop):
    """Two paired trajectories; class branch includes a real probe and mid-epoch restore."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    reference=[];histories={};forks=[];trace_counts={};restore_exact=False
    payload_digest=ts.digest(payload)
    for arm in cfg['arms']:
        folder=out/arm;folder.mkdir();(folder/'checkpoints').mkdir()
        train=pilot.Dataset(root,rows,cfg,arm,names)
        model,optimizer=factory()
        forks.append(pilot.fork_state(model,optimizer,payload))
        d.require(all(x==forks[0] for x in forks),'Production gate forks differ')
        progress=pilot.new_progress(cfg,len(train))
        context={'scope':'Production-loop software gate only','cfg':cfg,'arm':arm}
        traces=[]
        def record(phase,step,model,opt,images,targets,result):
            value={'phase':phase,'step':step,**signature(model,opt,images,targets,result)}
            index=len(traces)
            if phase=='after':d.require(value['inputs']==traces[-1]['inputs'],'Training update mutated input tensors')
            traces.append(value)
            b.write_json(folder/'update_trace.json',traces)
            if arm==cfg['arms'][0]:reference.append(value)
            else:
                d.require(index<len(reference) and value==reference[index],
                          f'Production-loop exactness failed at {arm} step={step} phase={phase}')
        if arm=='class_adaptive':
            first=progress['step']+1
            status=pilot.fit(model,optimizer,train,val,cfg,device,folder,context,progress,plans,
                lambda:stop() or progress['step']>=first,update_observer=record)
            d.require(status=='budget_stopped' and progress['next_batch']==1,'Gate did not stop after first update')
            d.require(len(progress['controller_updates'])==1,'Real production probe was not adopted')
            policy=progress['controller_updates'][0]['policy']
            d.require(not any(policy['shared_allowed']) and all(not any(c['allowed']) for c in policy['routed'].values()),
                'Tiny gate probe must produce unsupported identity; do not reinterpret changed treatment as a numerical failure')
            before={'model':ts.digest(model.state_dict()),'optimizer':ts.digest(optimizer.state_dict()),
                'rng':ts.digest(checkpoints.rng_state()),'modes':{n:m.training for n,m in model.named_modules()}}
            saved_progress=copy.deepcopy(progress)
            del model,optimizer
            gc.collect()
            if str(device).startswith('cuda'):torch.cuda.empty_cache()
            model,optimizer=factory()
            progress=b.restore(folder,model,optimizer,context)
            after={'model':ts.digest(model.state_dict()),'optimizer':ts.digest(optimizer.state_dict()),
                'rng':ts.digest(checkpoints.rng_state()),'modes':{n:m.training for n,m in model.named_modules()}}
            checkpoints.assert_same(progress,saved_progress,'production gate resume progress')
            d.require(before==after,'Production gate resume state differs')
            observer.verify_updates(folder,progress['controller_updates'],rows,plans,cfg)
            b.write_json(folder/'resume_boundary.json',{'before':before,'after':after,'exact':True})
            restore_exact=True
        status=pilot.fit(model,optimizer,train,val,cfg,device,folder,context,progress,plans,stop,update_observer=record)
        d.require(status=='complete','Production gate did not complete within its budget')
        histories[arm]=progress['history'];trace_counts[arm]=len(traces)
        observer.verify_updates(folder,progress['controller_updates'],rows,plans,cfg)
        d.require(ts.digest(payload)==payload_digest,'Training mutated the source warmup payload')
        del model,optimizer,train
        gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    d.require(len(set(trace_counts.values()))==1 and reference,'Missing production gate trace')
    pilot.paired_summary(histories,cfg)
    left,right=(histories[arm][-1] for arm in cfg['arms'])
    d.require(left['validation']==right['validation'],'Identical gate models give different validation metrics')
    # Compare raw saved prediction values, not gzip hashes or wall-time metadata.
    import gzip
    predictions=[]
    for arm in cfg['arms']:
        with gzip.open(out/arm/f"epoch_{cfg['epochs']:03d}_val_predictions.json.gz",'rt') as f:
            predictions.append(json.load(f))
    d.require(predictions[0]==predictions[1],'Identical gate models give different validation predictions')
    return {'production_loop_exact':True,'resume_exact':restore_exact,'validation_predictions_exact':True,
        'trace_records_per_arm':trace_counts,'optimizer_updates_each':len(reference)//2,
        'train_images':len(names),'val_images':len(val),'forks':forks,
        'scope':'Actual fit/loader/baseline.update on tiny train/validation subsets; identity appearance only; no accuracy evidence'}


def main():
    parser=argparse.ArgumentParser()
    for key in ('root','splits','checkpoint','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((pilot.PACKAGE/'protocol.json').read_text())
    pilot.verify_package();pilot.runtime(cfg)
    out=pilot.check_output(args.out,[args.root,args.splits,args.checkpoint,pilot.PACKAGE])
    d.require(not out.exists(),'Gate output already exists')
    started=time.monotonic();report=None
    try:
        rows,val_rows,payload,full_plans=pilot.load_inputs(args.root,args.splits,args.checkpoint,cfg)
        chosen=json.loads((pilot.PACKAGE/'smoke_selection.json').read_text())
        plan=full_plans['2'][:2]
        names=sorted(set(chosen['train'])|{p['image'] for p in plan})
        small=copy.deepcopy(cfg)
        small.update(epochs=2,update_epochs=[2],warmup_steps=0,checkpoint_every_steps=1000,log_every_steps=1000)
        val=b.Dataset(args.root,val_rows,'val',cfg['image_size'],chosen['val'])
        def factory():
            model,_,_=b.build_model(b.MODEL,cfg['image_size'],'cuda',cfg['evaluation'])
            return model,torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
        stop=lambda:time.monotonic()-started>=cfg['production_gate']['maximum_seconds'] or time.time()>=float(os.environ.get('RDD_PILOT_DEADLINE','inf'))
        report=compare_run(args.root,rows,val,payload,{'2':plan},names,small,out,factory,'cuda',stop)
        strict.check(cfg)
        report.update(status='pass',job=os.environ['PBS_JOBID'],strict_settings=strict.state(),
            elapsed_seconds=time.monotonic()-started,package_manifest_sha256=d.file_hash(pilot.PACKAGE/'SHA256SUMS'))
        b.write_json(out/'PASS.json',report)
        print('STRICT PRODUCTION-LOOP GATE: PASS. Full comparison restarts from original epoch-1 checkpoint.',flush=True)
    except BaseException:
        out.mkdir(parents=True,exist_ok=True)
        b.write_json(out/'FAILURE.json',{'status':'failed','traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        if out.exists():b.archive_records(out,uuid.uuid4().hex[:12])


if __name__=='__main__':main()
