"""Prepare an immutable two-arm extension, preserving released experiments."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'rdd_v2_pair8_v1'
OLD = ROOT / 'rdd_v2_pilot_v1'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n',
                    encoding='utf-8', newline='\n')


def main():
    OUT.mkdir(exist_ok=False)
    provenance = {}
    for line in (OLD/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ',1)
        data = (OLD/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==expected
        target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        provenance[name]={'source':str((OLD/name).relative_to(ROOT)),'original_sha256':expected}
    shutil.copyfile(ROOT/'rdd_v2_repro_v2/trace_state.py',OUT/'trace_state.py')
    cfg=copy.deepcopy(json.loads((OLD/'protocol.json').read_text()))
    cfg.update(version='rdd_v2_pair8_v1',epochs=8,arms=['geometry_only','class_adaptive'],
        purpose='Single-seed development comparison to fixed epoch 8; assess whether class-adaptive local augmentation beats geometry-only after a longer common horizon.',
        augmentation='Common geometry, with no local appearance versus unchanged class-adaptive local appearance',
        selection_metric='Final prescribed epoch-8 clean-validation AP50:95; no best-epoch selection',
        initialization='Both arms fork the same fixed epoch-1 identity checkpoint; epochs 2 through 8 use strict execution. No epoch-4 pilot checkpoint is resumed.',
        update_epochs=[2,4,6,8],panel_plan_seeds={str(e):4200001+e*1000 for e in (2,4,6,8)},
        controller_updates='Before epochs 2, 4, 6 and 8 on the current class-adaptive detector; frozen 256-image training panels, no threshold changes or adaptive resampling.',
        budget='One A10 per 12-hour PBS allocation; 11-hour administrative cap including tests and gate. Seven further epochs per arm, sequential arms; checkpoints and explicit resume only. No automatic resubmission.',
        precision='Float32, AMP/TF32 off, strict deterministic algorithms with warn_only=False, CUBLAS_WORKSPACE_CONFIG=:4096:8; exact short production-loop gate required.',
        continuation={'technical':'Production-loop equality gate, strict flags, finite updates, unchanged probe state, complete paired budgets and immutable evidence.',
            'performance':'Primary difference is class_adaptive minus geometry_only epoch-8 AP50:95. No interim accuracy stopping, winner/epoch selection or threshold tuning.',
            'review':'Assess fixed endpoint, per-class metrics, false positives, exposure, support and added cost. A favorable one-seed development result motivates independent paired seeds and online-shared comparison. A persisting deficit without a declared compensating benefit stops scaling this recipe.'})
    cfg['panels']['selection']='Panel A from the existing deterministic planner: seeds 4202001, 4204001, 4206001, 4208001. First two panels unchanged; new panels fixed before this experiment.'
    cfg['limitations']=[
        'One development seed and epoch-8 horizon; no variance, significance, convergence or generalization claim.',
        'Validation has informed project development; test remains closed.',
        'Both arms have geometry; this comparison does not estimate geometry benefit or isolate class conditioning against online-shared control.',
        'Tolerance is not training utility; mixed-set support can be borrowed. Actual edit rates may differ from opportunities.',
        'Strict and legacy pilot results are separate execution regimes; the primary comparison is within this new run.',
        'Training steps are matched; controller compute is additional. Sequential timings are descriptive.']
    cfg['strict_settings']=json.loads((ROOT/'rdd_v2_strict_v1/protocol.json').read_text())['strict_settings']
    cfg['strict_prerequisite']={'job':'799047.bright04','archive_sha256':'4912aec513e3f3b64f89234bdc65e2b3bc7d7d4f9b609bc486670af6bf0806ee',
        'source':'Audited 30-update strict diagnostic; no accuracy measurement.'}
    cfg['resource_checks']={'allocator_fraction':0.9,'first_window_updates':128,
        'maximum_probe_seconds':1800,'minimum_free_disk_bytes':20*2**30,
        'scope':'Record normal training time and allocator peaks for each arm first window. Each full controller probe has a 30-minute operational cap, including incomplete attempts. No throughput-to-accuracy claim.'}
    cfg['production_gate']={'train_selection':'Existing smoke selection union first two epoch-2 panel rows',
        'epochs':[2],'conditions':['geometry_only','class_adaptive'],
        'comparison':'Exact input/model/optimizer/gradient/RNG fingerprints after each actual baseline.update; class branch has a real unsupported two-image probe and save/rebuild/restore after first update.',
        'maximum_seconds':1800,'policy':'Two-image unsupported policy must remain identity; full scientific policy and thresholds unchanged.'}
    save(OUT/'protocol.json',cfg)
    sys.path.insert(0,str(OUT))
    import diagnostic
    import rdd_data
    rows=rdd_data.read_manifest(ROOT/'detection_splits_v1/train.jsonl',cfg['manifests']['train'])
    old_plans=json.loads((OLD/'panel_plan.json').read_text())
    plans={}
    for epoch in cfg['update_epochs']:
        planning=copy.deepcopy(cfg);planning['seed']=cfg['panel_plan_seeds'][str(epoch)]
        plans[str(epoch)]=diagnostic.plan_panels(rows,planning)['panels']['A']
        if str(epoch) in old_plans: assert plans[str(epoch)]==old_plans[str(epoch)]
    save(OUT/'panel_plan.json',plans)
    save(Path(__file__).with_name('source_provenance.json'),provenance)
    metadata=(ROOT/'reports/rdd_v2_pilot_v1/metadata_preflight.py').read_text()
    metadata=metadata.replace('rdd_v2_pilot_v1','rdd_v2_pair8_v1').replace('range(1, 4)','range(1, cfg[\'epochs\'])')
    Path(__file__).with_name('metadata_preflight.py').write_text(metadata,encoding='utf-8',newline='\n')
    print('Created rdd_v2_pair8_v1 and four frozen 256-image training panels.')


if __name__=='__main__':main()
