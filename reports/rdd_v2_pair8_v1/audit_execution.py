"""Audit saved gates, paired budgets, checkpoints, validation and resource records."""
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import math
import sys

from audit_returned import ROOT, REPORT, PACKAGE, RETURNED, load, save, sha


def finite(value):
    if isinstance(value,float): assert math.isfinite(value)
    elif isinstance(value,dict):
        for v in value.values(): finite(v)
    elif isinstance(value,list):
        for v in value: finite(v)


def main():
    cfg=load(PACKAGE/'protocol.json')
    attempts=sorted([load(p) for p in RETURNED.glob('attempt_*.json')],key=lambda r:r['started_utc'])
    assert len(attempts)==3
    assert [r['status'] for r in attempts]==['budget_stopped','budget_stopped','complete']
    assert load(RETURNED/'status.json')==attempts[-1]
    assert all(r['strict_settings']==cfg['strict_settings'] for r in attempts)
    assert all(r['gpu']=='NVIDIA A10' and r['cuda_build']=='12.1' and r['cudnn_version']==8902 for r in attempts)
    for r in attempts:
        elapsed=(datetime.fromisoformat(r['finished_utc'])-datetime.fromisoformat(r['started_utc'])).total_seconds()
        assert abs(elapsed-r['elapsed_seconds'])<1
    gates=[];reference=None;phase_comparisons=0;cross_job_comparisons=0
    for attempt in attempts:
        folder=RETURNED/'production_gates'/attempt['attempt']
        proof=load(folder/'PASS.json')
        assert proof==load(RETURNED/f"smoke_gate_{attempt['attempt']}.json")['result']
        assert proof['job']==attempt['job'] and proof['status']=='pass'
        assert proof['strict_settings']==cfg['strict_settings']
        assert proof['package_manifest_sha256']==sha(PACKAGE/'SHA256SUMS')
        assert proof['production_loop_exact'] and proof['resume_exact'] and proof['validation_predictions_exact']
        assert proof['optimizer_updates_each']==7 and proof['train_images']==26 and proof['val_images']==24
        copied=load(folder/'copied_checksums.json')
        missing=[name for name in copied if not (folder/name).is_file()]
        assert missing==['checksums.json']
        # The archive writer omits nested checksum manifests by basename. Their
        # payloads are present; reconstruct each redundant manifest and check it.
        reconstructed=(json.dumps({n:h for n,h in copied.items() if n!='checksums.json'},
            indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        assert hashlib.sha256(reconstructed).hexdigest()==copied['checksums.json']
        for name,digest in copied.items():
            if name!='checksums.json': assert sha(folder/name)==digest
        a=load(folder/'geometry_only/update_trace.json');b=load(folder/'class_adaptive/update_trace.json')
        assert a==b and len(a)==14
        finite(a)
        assert [(x['phase'],x['step']) for x in a]==[(p,s) for s in range(8,15) for p in ['before','after']]
        assert all(a[i]['inputs']==a[i+1]['inputs'] for i in range(0,14,2))
        boundary=load(folder/'class_adaptive/resume_boundary.json')
        assert boundary['exact'] and boundary['before']==boundary['after']
        with gzip.open(folder/'geometry_only/epoch_002_val_predictions.json.gz','rt') as f: pa=json.load(f)
        with gzip.open(folder/'class_adaptive/epoch_002_val_predictions.json.gz','rt') as f: pb=json.load(f)
        assert pa==pb
        phase_comparisons+=len(a)
        cross_same=reference is None or a==reference
        if reference is not None: cross_job_comparisons+=len(a)
        else: reference=a
        # Cross-device equality is observed here, not a universal guarantee.
        assert cross_same, 'Cross-job gate traces differ; inspect before interpretation'
        gates.append({'job':proof['job'],'scope':'26 training and 24 validation images, identity appearance',
            'updates_each':7,'phase_records_each':14,'paired_phases_exact':True,'cross_job_reference_exact':cross_same,
            'save_restore_boundary_exact':True,'validation_predictions_exact':True,
            'omitted_nested_manifest_reconstructed_and_verified':True,'elapsed_seconds':proof['elapsed_seconds']})
    histories={arm:load(RETURNED/arm/'history.json') for arm in cfg['arms']}
    runs={arm:load(RETURNED/arm/'run.json') for arm in cfg['arms']}
    forks=[load(RETURNED/arm/'fork.json') for arm in cfg['arms']]
    assert forks[0]==forks[1] and forks[0]['model_optimizer_rng_exact']
    assert all(r['completed_epoch']==8 and r['step']==53728 and r['status']=='complete' for r in runs.values())
    validated=0
    for arm,h in histories.items():
        assert [r['epoch'] for r in h]==list(range(2,9))
        for row in h:
            finite(row)
            assert row['step']==row['epoch']*6716 and row['images_seen']==row['epoch']*26863
            assert row['train_images']==26863 and row['val_images']==5760
            for name,digest in row['validation_records'].items(): assert sha(RETURNED/arm/name)==digest
            val=load(RETURNED/arm/f"epoch_{row['epoch']:03d}_validation.json")
            assert val['metrics']==row['validation']
            assert val['state_audit']['model_optimizer_rng_modes_unchanged']
            assert len(val['clipping'])==5760 and sorted(x['image_id'] for x in val['clipping'])==list(range(5760))
            assert sum(x['after_clipping'] for x in val['clipping'])==val['metrics']['prediction_boxes']
            ck=load((RETURNED/arm/'checkpoints'/row['checkpoint_file']).with_suffix('.json'))
            assert ck['completed_epochs']==row['epoch'] and ck['step']==row['step'] and ck['file']==row['checkpoint_file']
            validated+=1
        last=load(RETURNED/arm/'last.json')
        assert last==load((RETURNED/arm/'checkpoints'/h[-1]['checkpoint_file']).with_suffix('.json'))
    for i in range(7):
        assert histories['geometry_only'][i]['augmentation']['pairing_sha256']==histories['class_adaptive'][i]['augmentation']['pairing_sha256']
    for name,digest in load(RETURNED/'common_warmup/index.json').items(): assert sha(RETURNED/'common_warmup'/name)==digest
    assert load(RETURNED/'common_warmup/epoch_001_validation.json')['state_audit']['model_optimizer_rng_modes_unchanged']
    comparison=load(RETURNED/'comparison.json')
    assert comparison['epoch']==8
    assert comparison['final_metrics']=={arm:h[-1]['validation'] for arm,h in histories.items()}
    delta=100*(comparison['final_metrics']['class_adaptive']['AP50_95']-comparison['final_metrics']['geometry_only']['AP50_95'])
    assert delta==comparison['comparisons']['class_adaptive minus geometry_only']['AP50_95_points']
    assert [u['epoch'] for u in runs['class_adaptive']['controller_updates']]==[2,4,6,8]
    assert not runs['geometry_only']['controller_updates']
    resources={}
    for arm in cfg['arms']:
        records=[load(f) for f in (RETURNED/arm/'resources').glob('*.json')]
        assert all(v['strict_settings']==cfg['strict_settings'] for v in records)
        assert all(v['peak_allocated_bytes']<=v['peak_reserved_bytes']<=v['device_total_bytes'] for v in records)
        probes=[v for v in records if v.get('kind')=='controller_probe']
        assert all(v['status']=='complete' and v['elapsed_seconds']<v['maximum_seconds']==1800 for v in probes)
        windows=sorted([v for v in records if 'start_step' in v],key=lambda v:v['start_step'])
        assert all(v['updates']==len(v['seconds'])==128 for v in windows)
        resources[arm]={'training_seconds':runs[arm]['training_seconds'],'validation_seconds':runs[arm]['validation_seconds'],
            'probe_seconds':sum(v['elapsed_seconds'] for v in probes),
            'probe_epochs':[v['epoch'] for v in probes],
            'training_windows':[{k:v[k] for k in ['start_step','updates','median_update_seconds','peak_allocated_bytes','peak_reserved_bytes']} for v in windows],
            'peak_reserved_bytes_recorded':max(v['peak_reserved_bytes'] for v in records)}
    assert [v['start_step'] for v in resources['class_adaptive']['training_windows']]==[6716,43072]
    assert [v['start_step'] for v in resources['geometry_only']['training_windows']]==[6716,33580]
    train=[json.loads(s) for s in (ROOT/'detection_splits_v1/train.jsonl').read_text().splitlines()]
    val=[json.loads(s) for s in (ROOT/'detection_splits_v1/val.jsonl').read_text().splitlines()]
    for key in ['image','source_cluster_id','rgb_sha256']:
        assert not ({r[key] for r in train}&{r[key] for r in val})
    report={'status':'pass','attempts':attempts,'gates':gates,
        'within_job_exact_phase_comparisons':phase_comparisons,'additional_cross_job_reference_phase_comparisons':cross_job_comparisons,
        'paired_epoch_hashes':7,'identical_fork_metadata':True,'validation_records_checked':validated,
        'training_images_each_epoch':26863,'validation_images':5760,'training_updates_each_after_warmup':47012,
        'training_presentations_each_after_warmup':188041,'main_invocation_seconds':sum(a['elapsed_seconds'] for a in attempts),
        'gpu_gate_seconds':sum(g['elapsed_seconds'] for g in gates),'resources':resources,
        'limitations':['No full checkpoint tensors or raw source images in archive; no local detector re-inference.',
            'Full-run resume consistency checked through counts, paired hashes, unchanged context and available checkpoint metadata; not a second uninterrupted full-training reproduction.',
            'Exact short gate records agree across two physical A10 UUIDs; no universal cross-device determinism claim.',
            'Recorded main/gate timings exclude queue time, some startup/unit-test work and archive creation; not a complete scheduler billing record.',
            'One development seed, reused validation, no held-out test evaluation or seed-variance estimate.']}
    save('execution_audit.json',report)
    print(json.dumps({'status':report['status'],'gates':len(gates),'phase_comparisons':phase_comparisons,
        'cross_job_reference_phases':cross_job_comparisons,'validation_records_checked':validated,
        'main_and_gate_hours':(report['main_invocation_seconds']+report['gpu_gate_seconds'])/3600},indent=2))


if __name__=='__main__':main()
