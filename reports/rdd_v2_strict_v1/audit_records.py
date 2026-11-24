"""Audit downloaded strict-check records without model inference or training.

Full model/gradient tensors remain on CCAST. This independently reconciles
their saved fingerprints, reconstructs raw RNG/boundary hashes, and replays
the controller from its saved observations.
"""
from dataclasses import dataclass
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'rdd_v2_strict_v1'
PILOT = PACKAGE / 'source_repro/source_pilot'
PREFIX = 'rdd_v2_strict_check_v1'
ARCHIVE = ROOT / f'{PREFIX}_records.tar.gz'
EXPECTED = '4912aec513e3f3b64f89234bdc65e2b3bc7d7d4f9b609bc486670af6bf0806ee'
RECORDS = HERE / 'returned_records' / PREFIX
PHASES = ('before', 'forward', 'backward', 'update')
STATUS = 'complete_exact_on_this_diagnostic'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def manifest(path):
    pairs = [line.split('  ', 1) for line in Path(path).read_text().splitlines()]
    result = {name: digest for digest, name in pairs}
    assert len(result) == len(pairs)
    return result


def extract_verified():
    assert sha(ARCHIVE) == EXPECTED
    inventory = {}
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        for member in archive:
            p = PurePosixPath(member.name)
            assert member.isfile() and p.parts[0] == PREFIX and not p.is_absolute()
            assert '..' not in p.parts and '\\' not in member.name and ':' not in member.name
            relative = '/'.join(p.parts[1:])
            assert relative and relative not in inventory and p.suffix not in ('.pt', '.tmp')
            assert 0 <= member.size < 10_000_000
            data = archive.extractfile(member).read()
            assert len(data) == member.size
            digest = hashlib.sha256(data).hexdigest()
            target = RECORDS.joinpath(*p.parts[1:])
            assert RECORDS.resolve() in target.resolve().parents
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == digest, f'Existing extracted file changed: {relative}'
            else:
                with target.open('xb') as f:
                    f.write(data)
            inventory[relative] = {'sha256': digest, 'bytes': member.size}
    sums = read(RECORDS / 'checksums.json')
    assert set(sums) == set(inventory) - {'checksums.json'}
    assert all(inventory[n]['sha256'] == h for n, h in sums.items())
    assert not any('failure' in n or n.startswith('tensor_evidence/') for n in inventory)
    return {'archive': ARCHIVE.name, 'sha256': EXPECTED, 'bytes': ARCHIVE.stat().st_size,
            'files': len(inventory), 'checksummed_files_verified': len(sums),
            'uncompressed_bytes': sum(v['bytes'] for v in inventory.values()),
            'inventory': inventory}


@dataclass
class ByteTensor:
    """Independent representation of the raw uint8 RNG tensors in JSON."""
    data: bytes


def leaves(value, path='$'):
    if isinstance(value, dict):
        yield path + '/@container', 'dict'
        for k in sorted(value, key=lambda x: (type(x).__name__, repr(x))):
            yield from leaves(value[k], path + '/' + type(k).__name__ + ':' + repr(k))
    elif isinstance(value, (tuple, list)):
        yield path + '/@container', type(value).__name__
        for i, item in enumerate(value):
            yield from leaves(item, path + '/' + str(i))
    else:
        yield path, value


def digest(value):
    h = hashlib.sha256()
    for path, item in leaves(value):
        key = path.encode()
        if isinstance(item, ByteTensor):
            header = json.dumps(['torch.uint8', [len(item.data)]], separators=(',', ':')).encode()
            payload = header + b'\0' + item.data
        else:
            payload = (type(item).__name__ + ':' + repr(item)).encode()
        h.update(len(key).to_bytes(8, 'little')); h.update(key)
        h.update(len(payload).to_bytes(8, 'little')); h.update(payload)
    return h.hexdigest()


def rng(raw):
    return {'python': (raw['python'][0], tuple(raw['python'][1]), raw['python'][2]),
            'numpy': raw['numpy'], 'torch': ByteTensor(bytes(raw['torch'])),
            'cuda': [ByteTensor(bytes(x)) for x in raw['cuda']]}


def exact_comparison(c, left, right):
    assert c['equal'] and not c['changes'] and not c['missing_paths'] and not c['unexpected_paths']
    assert c['expected_sha256'] == left == right == c['actual_sha256']


def main():
    provenance = extract_verified()
    context = read(RECORDS / 'context.json')
    protocol = read(PACKAGE / 'protocol.json')
    assert context['protocol'] == protocol
    assert context['backends'] == protocol['strict_settings']
    sources = manifest(PACKAGE / 'SHA256SUMS')
    assert context['sources'] == sources
    for name, expected in sources.items():
        assert sha(PACKAGE / name) == expected, name
    originals = 0
    for path in (PACKAGE / 'source_repro').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            relative = path.relative_to(PACKAGE / 'source_repro')
            assert sha(path) == sha(ROOT / 'rdd_v2_repro_v2' / relative)
            originals += 1
    assert sha(PACKAGE / 'source_repro/SHA256SUMS') == protocol['source_repro_manifest_sha256']
    assert sha(PILOT / 'SHA256SUMS') == protocol['source_manifest_sha256']
    assert sha(RECORDS / 'library_sources/torchvision_roi_align.py') == context['roi_align_source_sha256']
    cfg = read(PILOT / 'protocol.json')
    assert context['effective_configuration'] == {**cfg, 'workers': 0, 'warmup_steps': 0}
    assert cfg['checkpoint'] == protocol['source_checkpoint']
    train = ROOT / 'detection_splits_v1/train.jsonl'
    assert sha(train) == cfg['manifests']['train']
    assert sha(train.with_name('config.json')) == cfg['split_config_sha256']
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    assert len(rows) == cfg['expected_rows']['train'] and all(r['split'] == 'train' for r in rows)
    lookup, row_ids = {r['image']: r for r in rows}, {r['image']: i for i, r in enumerate(rows)}
    provenance.update(package_sources_verified=len(sources), unchanged_source_repro_files=originals,
                      training_manifest_sha256=sha(train), train_rows=len(rows))

    inputs = read(RECORDS / 'materialized_inputs.json')
    assert inputs['materialization_rng_unchanged']
    batches = inputs['batches']
    assert [len(b['images']) for b in batches] == [4,4,4,4,4,4,2]
    names = [n for b in batches for n in b['images']]
    plan = read(PILOT / 'panel_plan.json')['2'][:2]
    assert len(names) == len(set(names)) == 26
    assert sorted(names) == sorted(set(read(PILOT / 'smoke_selection.json')['train']) | {p['image'] for p in plan})
    pairing = decision = '0' * 64
    for batch in batches:
        assert len(batch['images']) == len(batch['image_ids']) == len(batch['target_counts']) == len(batch['traces'])
        for name, iid, count, event in zip(batch['images'], batch['image_ids'], batch['target_counts'], batch['traces']):
            assert name == event['image'] and iid == row_ids[name]
            assert count == len(lookup[name]['targets']) == event['targets']
            assert event['operation'] == 'identity' and event['input_sha256'] == event['reference_sha256']
            assert event['pixel_changes_before_geometry'] == event['pixel_changes_after_geometry'] == 0
            paired = {k:event[k] for k in ('image','opportunity','group_index','group_ids','candidate',
                       'geometry_accepted','geometry_reason','cropped_ids','boxes','reference_sha256')}
            encode = lambda o: json.dumps(o, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            pairing = hashlib.sha256(bytes.fromhex(pairing) + encode(paired)).hexdigest()
            decision = hashlib.sha256(bytes.fromhex(decision) + encode(event)).hexdigest()
    assert pairing == protocol['expected_pairing_sha256'] == inputs['augmentation']['pairing_sha256']
    assert decision == protocol['expected_decision_sha256'] == inputs['augmentation']['decision_sha256']
    assert not any(batches[-1]['target_counts'])

    status, summary = read(RECORDS / 'status.json'), read(RECORDS / 'summary.json')
    complete, execution = read(RECORDS / 'COMPLETE.json'), read(RECORDS / 'execution.json')
    assert status['status'] == summary['status'] == complete['status'] == STATUS
    assert status['worker'] == read(RECORDS / 'worker_status.json')
    assert status['supervisor'] == read(RECORDS / 'supervisor_status.json')
    assert status['worker']['exit_code'] == status['supervisor']['returncode'] == 0
    assert status['supervisor']['stop_reason'] is None
    assert complete['updates'] == summary['updates'] == execution['updates_observed'] == status['worker']['updates_observed'] == 30
    assert status['no_accuracy_evaluation'] and status['no_longer_training_started']
    limits = protocol['resource_limits']
    assert 0 < status['worker']['elapsed_seconds'] < limits['worker_wall_seconds']
    expected_events = [{'stage':s['name'], 'condition':c, 'step':i}
                       for s in protocol['stages'] for c in s['conditions'] for i in range(1,s['updates']+1)]
    assert execution['events'] == expected_events

    phases, resources, comparisons = {}, [], []
    all_forks, reference_bytes = [], 0
    for stage in protocol['stages']:
        sn = stage['name']; stage_summary = read(RECORDS / sn / 'comparison.json')
        assert stage_summary == summary['stages'][sn] and stage_summary['status'] == STATUS
        assert set(stage_summary['conditions']) == set(stage['conditions'])
        for condition in stage['conditions']:
            folder = RECORDS / sn / condition
            case = stage_summary['conditions'][condition]
            fork = read(folder / 'fork.json'); all_forks.append(fork)
            assert fork == case['fork'] and fork['model_optimizer_rng_exact']
            assert case['first_difference'] is None and case['updates_complete'] == stage['updates']
            for step in range(1, stage['updates']+1):
                step_dir = folder / 'steps'
                for phase in PHASES:
                    value = read(step_dir / f'{step:02d}_{phase}.json')
                    phases[sn, condition, step, phase] = value
                    restored_rng = rng(value['rng_state'])
                    assert {k:digest(v) for k,v in restored_rng.items()} == value['rng_sha256']
                    if phase == 'before':
                        state = {k:value[k] for k in ('inputs_sha256','model_sha256','optimizer_sha256','modes','gradients_cleared','learning_rates')}
                        assert digest({**state, 'rng': restored_rng}) == value['state']['sha256']
                        assert value['inputs_sha256'] == batches[step-1]['sha256']
                        assert value['gradients_cleared'] and value['learning_rates'] == [cfg['learning_rate']]
                    elif phase == 'forward':
                        assert value['losses'] and all(math.isfinite(x) for x in value['losses'].values())
                    elif phase == 'backward':
                        assert math.isfinite(value['grad_norm']) and value['grad_norm'] > 0
                if condition == 'control_a':
                    reference_bytes += read(folder / 'reference_steps' / f'step_{step:02d}.json')['bytes']
                else:
                    cp = read(step_dir / f'{step:02d}_comparison.json')
                    for phase in PHASES:
                        reference = phases[sn, 'control_a', step, phase]
                        actual = phases[sn, condition, step, phase]
                        assert actual == reference, (sn, condition, step, phase)
                        exact_comparison(cp[phase], reference['state']['sha256'], actual['state']['sha256'])
                    assert case['comparisons'][step-1] == {'step':step,'exact':True,'phase_equal':dict.fromkeys(PHASES,True)}
                    comparisons.append({'stage':sn,'condition':condition,'step':step,'phases_exact':4})
                resource = read(step_dir / f'{step:02d}_resources.json')
                assert (resource['stage'], resource['condition'], resource['step'], resource['kind'],resource['state']) == (sn,condition,step,'update','complete')
                assert 0 < resource['elapsed_seconds'] < limits['maximum_update_seconds']
                assert resource['hard_timeout_seconds'] == limits['hard_update_seconds']
                assert resource['total_gpu_bytes'] == context['device_total_memory']
                assert 0 < resource['peak_allocated_bytes'] <= resource['peak_reserved_bytes'] < resource['total_gpu_bytes']*limits['peak_reserved_fraction_limit']
                assert resource['python_roi_align_calls']
                assert all(c['device']=='cuda:0' and c['dtype']=='torch.float32' and c['pooled_size']==[7,7] and c['sampling_ratio']==2 for c in resource['python_roi_align_calls'])
                phase_resources = read(step_dir / f'{step:02d}_phase_resources.json')
                assert set(phase_resources) == set(PHASES)
                assert all(0 <= p['wall_seconds_since_previous_phase'] < resource['elapsed_seconds'] for p in phase_resources.values())
                assert all(p['peak_allocated_bytes'] <= resource['peak_allocated_bytes'] and p['peak_reserved_bytes'] <= resource['peak_reserved_bytes'] for p in phase_resources.values())
                resources.append({'stage':sn,'condition':condition,'step':step,
                    'elapsed_seconds':resource['elapsed_seconds'],
                    'peak_allocated_GiB':resource['peak_allocated_bytes']/2**30,
                    'peak_reserved_GiB':resource['peak_reserved_bytes']/2**30,
                    'python_roi_align_calls':len(resource['python_roi_align_calls'])})
    assert len(phases)==120 and len(comparisons)==22 and len(resources)==30
    assert all(f==all_forks[0] for f in all_forks)
    assert reference_bytes == execution['reference_bytes'] < limits['reference_storage_limit_bytes']
    for condition in ('control_a','control_b'):
        for phase in PHASES:
            assert phases['first_update_gate',condition,1,phase] == phases['seven_update_suite',condition,1,phase]
    for condition in protocol['conditions']:
        assert len({phases['seven_update_suite',condition,i,'before']['model_sha256'] for i in range(1,8)}) == 7
        for step in range(1,8):
            reference_roi=read(RECORDS/'seven_update_suite/control_a/steps'/f'{step:02d}_resources.json')['python_roi_align_calls']
            actual_roi=read(RECORDS/'seven_update_suite'/condition/'steps'/f'{step:02d}_resources.json')['python_roi_align_calls']
            assert actual_roi==reference_roi

    probe_dir = RECORDS / 'seven_update_suite/probe_only'
    before, after = read(probe_dir / 'probe_before.json'), read(probe_dir / 'probe_after.json')
    assert before == after['state']
    boundary_sha = digest({**before, 'rng':rng(before['rng'])})
    exact_comparison(after['comparison'],boundary_sha,boundary_sha)
    resume_dir = RECORDS / 'seven_update_suite/resume_only'
    resume = read(resume_dir / 'resume_boundary.json')
    assert resume['before']==resume['after']
    boundary_sha = digest({**resume['before'], 'rng':rng(resume['before']['rng'])})
    exact_comparison(resume['comparison'],boundary_sha,boundary_sha)
    meta = resume['checkpoint']
    assert meta['step'] == protocol['initial_smoke_step'] + 1
    assert read(resume_dir / 'resume_checkpoint/last.json') == meta
    assert read(resume_dir / 'resume_checkpoint/checkpoints' / meta['file'].replace('.pt','.json')) == meta

    spec=importlib.util.spec_from_file_location('strict_audit_controller',PILOT/'controller.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    entry=read(probe_dir/'probe_entry.json')
    panel_file=probe_dir/entry['file']
    assert sha(panel_file)==entry['sha256'] and entry['plan_sha256']==controller.digest(plan)
    with gzip.open(panel_file,'rt',encoding='utf-8') as f:
        panel=[json.loads(line) for line in f]
    assert [p['selection'] for p in panel]==plan
    policy=controller.build_policy(controller.units_from_panel(panel,lookup),cfg['controller'])
    assert policy==entry['policy']
    assert not any(policy['shared_allowed']) and all(not any(x['allowed']) for x in policy['routed'].values())
    support={}
    for cl in range(4):
        hits=[(p['image'],p['object_indices'][i],p['source_cluster_id']) for p in panel for i in p['target_positions']
              if p['classes'][i]==cl and p['reference']['detected'][i]]
        support[str(cl)]={'detected_objects':len(hits),'detected_clusters':len({h[2] for h in hits})}
        assert all(policy['pooled'][str(cl)][k]==v for k,v in support[str(cl)].items())
    probe_resource=read(probe_dir/'probe_resources.json')
    assert probe_resource['state']=='complete' and 0 < probe_resource['elapsed_seconds'] < limits['maximum_probe_seconds']
    assert probe_resource['peak_reserved_bytes'] < context['device_total_memory']*limits['peak_reserved_fraction_limit']
    assert probe_resource['python_roi_align_calls']
    assert entry['audit']['model_optimizer_rng_modes_unchanged'] and entry['parity_max_abs']==0

    prior_root=ROOT/'reports/rdd_v2_repro_v2/returned_records/rdd_v2_repro_check_v2'
    assert sha(ROOT/'rdd_v2_repro_check_v2_records.tar.gz')==protocol['prior_diagnostic']['archive_sha256']
    prior_sums=read(prior_root/'checksums.json')
    assert all(sha(prior_root/name)==expected for name,expected in prior_sums.items())
    prior_context,prior_inputs=read(prior_root/'context.json'),read(prior_root/'materialized_inputs.json')
    assert inputs==prior_inputs
    assert context['effective_configuration']==prior_context['effective_configuration']
    assert all_forks[0]==read(prior_root/'suite/control_a/fork.json')
    prior_status=read(prior_root/'status.json')
    assert prior_status['status']=='complete_with_divergence'
    prior_audit=read(ROOT/'reports/rdd_v2_repro_v2/audit.json')
    prior_code_match=all(context['sources']['source_repro/'+n]==h for n,h in prior_context['sources'].items())
    assert prior_code_match
    time_values=[r['elapsed_seconds'] for r in resources]
    result={'status':'audit_complete','audit_date':'2026-11-02','job':context['job'],
        'versions':context['versions'],'device':context['device'],'host':context['host'],
        'strict_settings':context['backends'],'updates':30,'stages':2,'images':26,
        'positive_images':sum(bool(lookup[n]['targets']) for n in names),
        'negative_images':sum(not lookup[n]['targets'] for n in names),
        'batch_sizes':[len(b['images']) for b in batches],
        'phase_records_verified':len(phases),'phase_comparisons_exact':4*len(comparisons),
        'cross_stage_phase_comparisons_exact':8,'raw_rng_fingerprints_recomputed':len(phases)*4,
        'before_state_fingerprints_recomputed':30,'probe_boundary_exact':True,'resume_boundary_exact':True,
        'probe_policy_replayed_exactly':True,'probe_support':support,'appearance_edits':0,
        'same_materialized_inputs_as_previous_diagnostic':True,
        'same_initial_model_optimizer_rng_fork':True,'same_effective_training_configuration':True,
        'previous_diagnostic_first_differences':prior_audit['first_differences'],
        'worker_elapsed_seconds':status['worker']['elapsed_seconds'],
        'supervisor_elapsed_seconds':status['supervisor']['elapsed_seconds'],
        'update_seconds_min':min(time_values),'update_seconds_median':statistics.median(time_values),
        'update_seconds_max':max(time_values),'update_seconds_sum':sum(time_values),
        'peak_update_allocated_GiB':max(r['peak_allocated_GiB'] for r in resources),
        'peak_update_reserved_GiB':max(r['peak_reserved_GiB'] for r in resources),
        'gpu_total_GiB':context['device_total_memory']/2**30,
        'probe_seconds':probe_resource['elapsed_seconds'],
        'probe_peak_allocated_GiB':probe_resource['peak_allocated_bytes']/2**30,
        'probe_peak_reserved_GiB':probe_resource['peak_reserved_bytes']/2**30,
        'probe_forward_calls':entry['forward_calls'],'probe_detector_images':entry['detector_images'],
        'python_roi_align_update_calls':sum(r['python_roi_align_calls'] for r in resources),
        'python_roi_align_probe_calls':len(probe_resource['python_roi_align_calls']),
        'reference_snapshot_bytes_on_ccast':reference_bytes,
        'limitations':[
            'Full model, gradient, optimizer and reference tensors were not downloaded; equality is supported by returned fingerprints and GPU comparison records, not local recomputation of those tensors.',
            'Raw training pixels were not downloaded; the materialization, geometry and target evidence is the saved trace and pinned input hashes.',
            'No fresh training or detector inference was performed by this audit.',
            'One checkpoint, seed, device and short instrumented trajectory; not a full-run or cross-machine determinism guarantee.',
            'The two-image probe has insufficient support and returns identity; it tests probe side effects, not active policy efficacy.',
            'Strict settings change several operations together; this does not isolate the unique cause of prior divergence.',
            'Resource timings include instrumentation and exclude queue time and preceding unit tests; the full-panel probe cost is unmeasured under strict mode.',
            'No accuracy evaluation; the prior class-adaptive AP deficit cannot be attributed to nondeterminism.'
        ]}
    save('archive_audit.json',provenance)
    save('audit.json',result)
    for filename,table in [('update_resources.csv',resources),('exact_comparisons.csv',comparisons)]:
        with (HERE/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write_report(result,provenance)
    save('analysis_checksums.json',{name:sha(HERE/name) for name in
        ('audit_records.py','audit.json','archive_audit.json','update_resources.csv','exact_comparisons.csv','RESULTS.md')})
    print(json.dumps(result,indent=2))


def write_report(a,p):
    text=f'''# Audit of the strict determinism diagnostic

Audit date: {a['audit_date']}. CCAST PBS job: {a['job']}. This audit reads saved records and launches no experiment.

The strict diagnostic passes its declared exactness and resource gates. Plain repeated training, training after a controller probe, and training after a save/rebuild/restore all have exactly matching recorded states over the seven-update comparison. The first-update gate also passes. The earlier divergence is absent from this strict execution of the same materialized inputs and initial state.

Recommendation: close this specific repeatability investigation and prepare the previously proposed bounded, two-arm comparison through epoch 8, using strict execution for both geometry-only and class-adaptive local augmentation. Start both from the common original epoch-1 checkpoint in a new run. Keep the method fixed. This is a development experiment, not evidence that the proposed method will win.

## Integrity and protocol

Archive: `{p['archive']}` ({p['bytes']:,} bytes).

SHA-256: `{p['sha256']}`.

All {p['checksummed_files_verified']} manifest entries in the {p['files']}-file archive match. Safe extraction rejects links, traversal and duplicate members; any existing extracted files must match. All {p['package_sources_verified']} released package payloads match the source hashes recorded by the job. The {p['unchanged_source_repro_files']} vendored reproduction files are unchanged from the prior release. The recorded protocol and effective configuration match the local releases, and the frozen training manifest and split configuration match their pinned hashes. The prior archive and its 167 checksummed extracted files were reverified for the cross-diagnostic comparison.

The run used PyTorch {a['versions']['torch']}, TorchVision {a['versions']['torchvision']}, CUDA 12.1 and an {a['device']} on {a['host']}. Strict algorithms were enabled with `warn_only=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic cuDNN, benchmarking off and TF32 off. The source checks strict flags at every recorded phase. The installed RoIAlign source is preserved and hashed; its strict CUDA dispatch was observed on every update. There are {a['python_roi_align_update_calls']} recorded Python RoIAlign calls across the 30 updates and {a['python_roi_align_probe_calls']} during probing. No custom numerical kernel or compilation was introduced.

This behavior is consistent with the pinned [TorchVision 0.18 implementation](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/ops/roi_align.py), which dispatches strict CUDA RoIAlign to its differentiable tensor implementation. This observation does not isolate RoIAlign as the sole cause of the earlier divergence.

## What matched

The diagnostic uses 26 frozen training images: {a['positive_images']} with target objects and {a['negative_images']} without them. Batch sizes are 4/4/4/4/4/4/2, including a final entirely negative batch. Appearance is identity in every condition; the common geometry and labels are unchanged. Geometry/decision hashes were reconstructed, and the complete materialization record is identical to the earlier diagnostic. No raw training images were decoded again locally.

| Comparison | Updates compared | Recorded phases matched | Result |
|---|---:|---:|---|
| Independent first-update repeat | 1 | 4 | Exact |
| Plain seven-update repeat | 7 | 28 | Exact |
| Probe then seven updates | 7 | 28 | Exact |
| Save/resume during seven updates | 7 | 28 | Exact |

The audit reconciles all {a['phase_records_verified']} phase records with the update ledger and recomputes {a['raw_rng_fingerprints_recomputed']} raw RNG component fingerprints, covering Python, NumPy, CPU Torch and CUDA state. All {a['phase_comparisons_exact']} comparison-phase records match their references, including inputs, pre-forward model/optimizer state, losses/buffers, gradients, optimizer updates and RNG. Thirty pre-forward state hashes and both probe/resume boundary hashes were independently reconstructed from the saved values. The gate and suite first-update records also match in eight cross-stage checks, confirming the restart from the original state. Distinct pre-forward model hashes across successive steps rule out a trivial no-update trajectory in the saved evidence.

The checkpoint restore occurs after update 1; all six subsequent updates match uninterrupted training. The probe's model, optimizer, RNG and module modes match before/after. The original observer also records unchanged gradients. Replaying the controller from saved probe observations reproduces its policy exactly. The two-image probe has detected-object/cluster support 0/0, 1/1, 1/1, 0/0 across the four classes and correctly returns identity. This deliberately tiny probe tests observation side effects; it does not test an active class-adaptive policy's accuracy.

Full model/gradient/optimizer/reference tensors remain on CCAST. Therefore the local audit verifies their returned fingerprints and the recorded GPU-side byte comparisons; it does not independently recompute those full tensor comparisons. The checkpoint hash is pinned by the executed input check, rather than by reloading the checkpoint locally.

## Resource measurements

| Measurement | Observed value |
|---|---:|
| Worker elapsed time | {a['worker_elapsed_seconds']:.2f} seconds |
| Parent-supervised process elapsed time | {a['supervisor_elapsed_seconds']:.2f} seconds |
| Optimizer updates observed | {a['updates']} |
| Median instrumented update | {a['update_seconds_median']:.3f} seconds |
| Update time range | {a['update_seconds_min']:.3f}–{a['update_seconds_max']:.3f} seconds |
| Peak update allocated GPU memory | {a['peak_update_allocated_GiB']:.3f} GiB |
| Peak update reserved GPU memory | {a['peak_update_reserved_GiB']:.3f} GiB |
| Device memory reported by PyTorch | {a['gpu_total_GiB']:.3f} GiB |
| Two-image probe elapsed time | {a['probe_seconds']:.3f} seconds |
| Reference snapshots retained on CCAST | {a['reference_snapshot_bytes_on_ccast']/2**30:.3f} GiB |

All recorded measurements are below the declared limits. Allocated/reserved memory are PyTorch allocator measurements and exclude some external CUDA allocations. Timings include synchronization, CPU tensor copying, hashing and record writes; the overall figures exclude queue time and preceding unit tests. Neither these timings nor the earlier diagnostic's different duration establish a training speedup. The full 256-image controller panel has not yet been costed under strict execution.

## Interpretation and limits

Fact: the previous plain repeat first differed in backward at update 1; these strict repeats do not differ over the tested trajectory. The scientific inputs, training configuration and initial model/optimizer/RNG fingerprints agree between diagnostics. The runs used the same GPU model and software versions but different physical GPU UUIDs, and the diagnostic wrapper changed to enforce strictness and measure resources.

Inference: numerical execution was a credible source of the prior discrepancy, and strict execution provides a working remedy for this short comparison. This is not an operator-isolation experiment and does not prove all training code is correct. There is no evidence of a probe/resume state leak in the paths tested here.

Unknown: full-run repeatability, the cost of strict execution at all later training states, and whether class adaptation improves detection. There was no validation or test evaluation. The previous pilot's 0.2003 AP-point class-adaptive deficit against geometry-only remains an observed result; it cannot be retrospectively attributed to nondeterminism. [PyTorch's pinned reproducibility guidance](https://github.com/pytorch/pytorch/blob/v2.3.0/docs/source/notes/randomness.rst) also limits reproducibility guarantees across releases and platforms.

## Recommended next stage

1. Freeze a new comparison of geometry-only and class-adaptive local augmentation to the same epoch-8 endpoint. Restart both from the pinned epoch-1 identity checkpoint, so epochs 2–8 use the same strict settings. Preserve the earlier pilot as a separate experiment rather than mixing it with this execution regime.
2. Retain the detector, geometry, local appearance bank, sampling rule, support gates and retention threshold. Preselect any additional training probe panels and update epochs before launching. Keep data order, training steps and evaluation fixed across arms.
3. Put a short equality check through the actual training entry point at the beginning of that job, then measure the first full controller panel and early normal-training cost. This verifies integration of the strict settings and sets a practical runtime bound; it is not another open-ended diagnostic campaign. Stop on a strictness, integrity or resource failure.
4. Use clean validation AP50:95 at epoch 8 as the primary development comparison. Keep per-class AP, recall, false positives, controller support, augmentation exposure and extra compute as declared secondary diagnostics. Keep the test set closed and avoid choosing a favorable epoch after the run.
5. A favorable result would motivate paired independent seeds and the online-shared comparator before claiming a class-conditioning benefit. If the method remains worse without a declared compensating benefit, stop scaling this recipe and reassess the augmentation intervention. This short study is not a convergence test or proof of equivalence.

## Reproduction and evidence

Run `python reports/rdd_v2_strict_v1/audit_records.py` from the project directory. The program reads the verified archive, frozen source packages, training manifest and earlier diagnostic records. It performs no training or detector inference.

- `audit.json`: machine-readable findings and limitations.
- `archive_audit.json`: per-member hashes and source provenance.
- `exact_comparisons.csv`: all 22 update comparisons.
- `update_resources.csv`: all 30 update resource records.
- `returned_records/`: preserved downloaded evidence.
- `analysis_checksums.json`: hashes of the analysis code and outputs.
'''
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
