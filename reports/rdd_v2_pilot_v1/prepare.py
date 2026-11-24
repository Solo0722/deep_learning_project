"""Assemble the bounded pilot without modifying any released experiment."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'rdd_v2_pilot_v1'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def main():
    PKG.mkdir(exist_ok=True)
    (PKG/'tests').mkdir(exist_ok=True)
    sources = {}
    for folder, names in {
        'fasterrcnn_baseline_v1': ['baseline.py', 'checkpoints.py', 'models.py', 'metrics.py',
                                  'integration.py', 'select_training_cases.py', 'smoke_selection.json', 'tests/test_baseline.py'],
        'rdd_v2_controller_check_v1': ['controller.py', 'geometry.py', 'diagnostic.py', 'detector.py',
                                     'rdd_data.py', 'local_augmentation.py', 'shared_augmentation.py'],
    }.items():
        for name in names:
            shutil.copyfile(ROOT/folder/name, PKG/name)
            sources[name] = {'source': f'{folder}/{name}', 'sha256': hashlib.sha256((PKG/name).read_bytes()).hexdigest()}
    cfg = json.loads((ROOT/'fasterrcnn_baseline_v1/protocol.json').read_text())
    diag = json.loads((ROOT/'rdd_v2_controller_check_v1/protocol.json').read_text())
    for key in ('geometry', 'controller', 'checkpoint', 'panels', 'parity_tolerance'):
        cfg[key] = copy.deepcopy(diag[key])
    cfg.update(version='rdd_v2_pilot_v1', epochs=4, warmup_epoch=1,
        arms=['geometry_only', 'fixed_local', 'online_shared', 'class_adaptive'],
        purpose='Bounded development comparison: three additional full training epochs from one fixed epoch-1 state; not a final efficacy experiment',
        augmentation='common whole-image geometry with four appearance-policy arms',
        application_probability=.5, update_epochs=[2, 4], panel_plan_seeds={'2': 4202001, '4': 4204001},
        probe_batch_size=2, views_per_image=29, max_probe_images_per_update=7428,
        administrative_seconds=39600, cpu_threads=2,
        budget='One A10 allocation, 12-hour PBS ceiling, 11-hour in-process cap across smoke and pilot. Three additional epochs per arm, sequential arms; explicit resume only.',
        selection_metric='final prescribed epoch-4 validation AP50:95, not best epoch',
        initialization='Exact identity epoch-1 model, SGD momentum, step and Python/NumPy/Torch/CUDA RNG; fork all four arms from the same file',
        precision='float32; AMP/TF32 off; cudnn benchmark off and deterministic on; exact state restore checked, CUDA trajectory bitwise equality not promised',
        geometry_scope='All training images including negatives; same keyed candidate and annotation fallback in every arm; validation remains unaugmented',
        fixed_policy='All 24 actions eligible; uniform family, then operation, then level; same 0.5 positive-image opportunity and same uniformly selected overlap group as online arms',
        controller_updates='Before epochs 2 and 4 on each online arm current model; same preselected panel at a given epoch. No threshold relaxation, stale policy, or adaptive panel replacement.',
        pipeline_order=diag['pipeline_order'],
        resume_policy='Immutable code/config/data/environment; checkpoint model, optimizer, RNG, batch, counters and adopted policies. Validate hashed evidence. Re-run incomplete probes, retaining their attempts. No automatic resubmission.',
        limitations=['One development seed and short horizon; no significance or convergence claim.',
                     'Validation has been used for development; test remains unopened.',
                     'All four arms include geometry; this pilot does not isolate the benefit of geometry against identity.',
                     'Tolerance is not training utility or semantic correctness. Mixed-set support can remain borrowed.',
                     'Equal opportunities need not give equal actual edit rates; record identity fallbacks and pixel changes.',
                     'Sequential-arm runtime is descriptive and includes queue/node/IO variation.'],
        continuation={'technical': 'Finite updates, unchanged probe state, complete matched budgets, immutable evidence and valid boxes are required.',
            'review': 'Compare all final-epoch and per-class curves, support, policies, actual exposure, geometry fallback, negatives and runtime. If class routing collapses or stalls, or accuracy is worse without an interpretable benefit, do not scale this recipe. No automatic 20-epoch continuation.',
            'performance': 'No post-hoc winner selection, threshold tuning, or unvalidated numeric go/no-go cutoff. A favorable pilot motivates a separately frozen longer comparison, not a claim of efficacy.'})
    cfg['panels'].update(count=1, images_each=256,
        selection='Use panel A of the audited paired planner with fresh predeclared planning seeds 4202001 and 4204001. A is a fixed label, not chosen from results; B is not evaluated. Geometry and training seeds remain 42.',
        limitations='256 distinct clusters at each update, 64 anchor groups/class. Panels may overlap across updates and with training; training-only controller evidence.')
    save(PKG/'protocol.json', cfg)
    sys.path.insert(0, str(PKG))
    import diagnostic
    from rdd_data import read_manifest
    rows = read_manifest(ROOT/'detection_splits_v1/train.jsonl', cfg['manifests']['train'])
    plans = {}
    for epoch in cfg['update_epochs']:
        planning = copy.deepcopy(cfg)
        planning['seed'] = cfg['panel_plan_seeds'][str(epoch)]
        plans[str(epoch)] = diagnostic.plan_panels(rows, planning)['panels']['A']
    save(PKG/'panel_plan.json', plans)
    save(Path(__file__).with_name('provenance.json'), sources)
    print('Prepared inherited sources, protocol and two frozen 256-image panels.')


if __name__ == '__main__':
    main()
