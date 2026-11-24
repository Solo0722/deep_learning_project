"""Audit saved CCAST diagnostic evidence; no training or detector inference."""
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'rdd_v2_repro_v2'
PREFIX = 'rdd_v2_repro_check_v2'
ARCHIVE = ROOT / (PREFIX + '_records.tar.gz')
EXPECTED = '3ec5a81f35d667f71994f80956fa9ec691bbd2c9fe0b704173f6d2a97f9eedd9'
RECORDS = HERE / 'returned_records' / PREFIX
CONDITIONS = ['control_a', 'control_b', 'probe_only', 'resume_only']
PHASES = ['before', 'forward', 'backward', 'update']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n',
                             encoding='utf-8', newline='\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_verified():
    assert sha(ARCHIVE) == EXPECTED
    inventory = {}
    with tarfile.open(ARCHIVE, 'r:gz') as source:
        for member in source:
            p = PurePosixPath(member.name)
            assert member.isfile() and p.parts[0] == PREFIX and not p.is_absolute()
            assert '..' not in p.parts and '\\' not in member.name and ':' not in member.name
            relative = '/'.join(p.parts[1:])
            assert relative and relative not in inventory and p.suffix not in ('.pt', '.tmp')
            assert member.size < 10_000_000
            target = RECORDS.joinpath(*p.parts[1:])
            assert RECORDS.resolve() in target.resolve().parents
            data = source.extractfile(member).read()
            assert len(data) == member.size
            digest = hashlib.sha256(data).hexdigest()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == digest, 'Existing extraction changed: ' + relative
            else:
                with target.open('xb') as f:
                    f.write(data)
            inventory[relative] = {'sha256': digest, 'bytes': member.size}
    sums = read(RECORDS / 'checksums.json')
    assert set(sums) == set(inventory) - {'checksums.json'}
    assert all(inventory[n]['sha256'] == h for n, h in sums.items())
    assert len(inventory) == 168
    return {'archive': ARCHIVE.name, 'sha256': EXPECTED, 'bytes': ARCHIVE.stat().st_size,
            'files': len(inventory), 'manifest_entries_verified': len(sums),
            'uncompressed_bytes': sum(v['bytes'] for v in inventory.values()), 'inventory': inventory}


def reconstruct_rng(r):
    return {'python': (r['python'][0], tuple(r['python'][1]), r['python'][2]),
            'numpy': r['numpy'], 'torch': torch.tensor(r['torch'], dtype=torch.uint8),
            'cuda': [torch.tensor(x, dtype=torch.uint8) for x in r['cuda']]}


def reconstruct_boundary(r):
    return {**r, 'rng': reconstruct_rng(r['rng'])}


def check_pair(record, comparison, condition, phase):
    info = comparison['first_tensor_pair']
    assert info['saved']
    path = RECORDS / 'suite' / condition / 'tensor_evidence' / info['file']
    assert sha(path) == info['sha256']
    change = next(c for c in comparison['changes'] if c['path'] == info['path'])
    with np.load(path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {'expected', 'actual'}
        a, b = arrays['expected'], arrays['actual']
    assert list(a.shape) == change['expected_shape'] and list(b.shape) == change['actual_shape']
    assert 'torch.' + a.dtype.name == change['expected_dtype'] == change['actual_dtype']
    assert a.nbytes + b.nbytes == info['bytes_uncompressed']
    assert np.isfinite(a).all() and np.isfinite(b).all()
    left, right = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    delta = right - left
    values = {'elements': delta.size, 'numerically_changed_elements': int(np.count_nonzero(delta)),
              'max_abs': float(np.max(np.abs(delta), initial=0)),
              'l2': float(np.linalg.norm(delta)), 'reference_l2': float(np.linalg.norm(left))}
    for k, v in values.items():
        assert math.isclose(v, change[k], rel_tol=1e-12, abs_tol=1e-20), (k, v, change[k])
    for sample in change['values']:
        i = sample['flat_index']
        assert float(left[i]) == sample['expected'] and float(right[i]) == sample['actual']
    if change['reference_l2']:
        assert math.isclose(values['l2']/values['reference_l2'], change['relative_l2'], rel_tol=1e-12)
    record.append({'condition': condition, 'phase': phase, 'file': info['file'],
                   'path': info['path'], 'independent_numeric_verification': True, **values})


def magnitudes(changes, group):
    selected = [c for c in changes if c['kind'] == 'tensor_values'
                and c['path'].startswith("$/str:'" + group + "'/")]
    return {'changed_tensors': len(selected),
            'changed_elements': sum(c['numerically_changed_elements'] for c in selected),
            'max_abs': max((c['max_abs'] for c in selected), default=0),
            'max_tensor_relative_l2': max((c['relative_l2'] or 0 for c in selected), default=0),
            'difference_l2': math.sqrt(sum(c['l2'] ** 2 for c in selected))}


def write_report(output, provenance):
    rows = output['numeric_comparisons']
    table = []
    for condition in CONDITIONS[1:]:
        first = next(r for r in rows if r['condition'] == condition and r['step'] == 1)
        last = next(r for r in rows if r['condition'] == condition and r['step'] == 7)
        table.append(f"| {condition} | {first['gradients_max_abs']:.3e} | "
                     f"{last['model_max_abs']:.3e} | {last['total_loss_difference_percent']:+.3f}% | "
                     f"{100*last['gradient_difference_relative_to_full_reference_norm']:.3f}% |")
    text = '''# Audit of the V2 reproducibility diagnostic

Audit date: 2026-10-29. CCAST PBS job: 797450.bright04. This report analyzes saved evidence and launches no GPU job.

The diagnostic reproduces divergence in a plain repeated training run, starting with small gradient differences during the first backward pass. Probing and checkpoint resumption are unnecessary for that divergence. No mismatch was found in the observed state around the probe or checkpoint restore. Numerical nondeterminism in backward computation is the leading explanation; a specific CUDA operation has not been isolated experimentally.

Recommendation: keep the augmentation method unchanged, preserve the existing pilot, and perform one bounded strict-determinism comparison before a longer experiment. Do not claim the earlier class-adaptive accuracy deficit was caused by numerical variation. This diagnostic did not evaluate accuracy.

## What was verified

The downloaded archive is `rdd_v2_repro_check_v2_records.tar.gz`, 4,152,302 bytes, with SHA-256:

`3ec5a81f35d667f71994f80956fa9ec691bbd2c9fe0b704173f6d2a97f9eedd9`

Its 168 regular files total 22,073,219 uncompressed bytes. All 167 checksummed content files match the returned manifest. The archive hash matches the CCAST value supplied by the user. Safe extraction rejects links, traversal, duplicate entries and unexpected checkpoint binaries. Existing extracted files are verified rather than replaced.

All 40 released package files match the source hashes recorded by the job. The 29 vendored pilot source files match both their original manifest and the original local pilot release. Protocol, source configuration, effective configuration, training manifest and split-configuration hashes agree. The effective diagnostic changes only the declared loader-worker and warmup settings. The pinned source checkpoint identity agrees across configuration records; its full tensors remain on CCAST and were not reloaded locally.

The job completed all 28 updates: four serial conditions with seven updates each. It used an NVIDIA A10, PyTorch 2.3.0, TorchVision 0.18.0 and CUDA 12.1. The recorded diagnostic elapsed time is 246.9 seconds, including preparation and instrumentation but excluding the preceding unit tests. The prior hashing failure prevented all real diagnostic updates in release v1; it is separate from this completed release-v2 result.

## Controlled inputs and state

All conditions use the same 26 training images, consisting of 17 positive images and nine images without target-class objects. The seven batch sizes are 4, 4, 4, 4, 4, 4 and 2. The final batch contains only negative images. Appearance is identity throughout; common geometry is applied according to the original smoke treatment. Nineteen geometry candidates were accepted and seven fell back because an annotation would be cropped.

Recomputing the image-order/geometry trace and complete decision hash reproduces the declared smoke pins. Recorded image IDs use the original zero-based training-manifest indices. Every per-step input hash agrees across all four conditions. Raw pixels were not included in the archive; this is a verification of the saved traces and executed input checks, not a new image decode or render.

Initial model, optimizer and RNG fork fingerprints agree across all four conditions. For update 1, the entire recorded pre-forward state agrees, followed by exactly equal loss components, buffers and RNG state. Internal activations and proposals were not separately recorded, so equality of every intermediate forward tensor is not asserted.

All 112 saved phase RNG fingerprints were reconstructed from their raw Python, NumPy, CPU Torch and CUDA state records. Every one of the 84 condition-versus-reference phase comparisons has identical RNG state. Thus a differing saved RNG stream is not the first observed discrepancy in this diagnostic.

## First difference and its magnitude

For `control_b`, `probe_only` and `resume_only`, the first observed difference is update 1, backward. There are respectively 69, 69 and 68 differing gradient tensors, all within the backbone/FPN. No first-update gradient difference was reported in the RPN or RoI box-head parameters. These counts are tensor entries in a comparison, not counts of bugs. The first-step losses are identical.

All quantities below compare with `control_a`. Loss difference is a percentage of the reference loss on that same batch, not an AP-point difference. The final column is the L2 norm of the complete gradient difference divided by the recorded reference gradient norm; it is not the maximum per-layer relative error.

| Condition | Update-1 largest gradient difference | Update-7 largest model-weight difference | Update-7 total-loss difference | Update-7 relative gradient L2 difference |
|---|---:|---:|---:|---:|
''' + '\n'.join(table) + '''

The initial maximum gradient differences are approximately 2.3e-8 to 2.8e-8. Their full-gradient relative L2 sizes are approximately 6.7e-7 to 6.9e-7. After the first optimizer update, the maximum parameter difference is 1.49e-8 in each comparison. These observations are consistent with small floating-point accumulation differences at the beginning.

Differences amplify over subsequent updates: model parameters, momentum, gradients and losses become different. At update 7, the maximum model-weight difference reaches approximately 1.16e-5. Some individual gradient tensors have larger relative differences than the full-gradient figures in the table; `resume_only` reaches a per-tensor relative L2 difference of about 8.29%. This does not establish a resume effect, because its trajectory already differed before the reload and there is only one execution of each condition.

The nine exported tensor pairs were independently recomputed with NumPy, including shape, dtype, changed-element counts, maximum absolute difference, L2 norms and recorded sample values. All agree with their comparison records. The remaining full-tensor magnitudes are GPU-side comparison results preserved in verified JSON; they were not independently recomputed locally because the full snapshots were intentionally excluded. Approximately 4.64 GB of reference snapshots remain on CCAST.

## Probe and checkpoint boundaries

The decoded before/after records match exactly around probing: model hash, optimizer hash, complete RNG state and every saved module mode. The probe's gradient-preservation check also reports success. Replaying the two-image probe's policy from its saved observations reproduces the saved policy exactly. Independent support recounts give 0, 1, 1 and 0 reference-detected objects for D00, D10, D20 and D40, respectively, with the same cluster counts. All routes correctly fall back to identity under the unchanged 16-object/eight-cluster support gate. No probe decision was applied to training in this diagnostic.

Checkpoint restore occurs after update 1. Its before/after model hash, optimizer hash, RNG and module-mode records are identical, and its saved metadata agree. Gradients are intentionally not restored: all conditions clear them before the next forward pass. The first divergence in `resume_only` precedes this checkpoint operation. These are positive checks of the observed boundaries; they do not prove that probing or rebuilding a model has no additional effect on CUDA scheduling, caches or later numerical behavior.

## Interpretation and limits

The configured cuDNN convolution setting was deterministic, with benchmarking and TF32 disabled. General PyTorch deterministic-algorithm mode was disabled, and `CUBLAS_WORKSPACE_CONFIG` was unset. The convolution flag alone does not make every GPU operation deterministic, as explained in the [PyTorch 2.3.0 reproducibility documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.3.0/docs/source/notes/randomness.rst).

TorchVision 0.18.0's [CUDA RoIAlign backward source](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/csrc/ops/cuda/roi_align_kernel.cu#L412) marks the operation nondeterministic and uses atomic gradient accumulation. The observed first-step pattern, with differing backbone/FPN gradients and equal head gradients, is consistent with this path. That makes RoIAlign a concrete candidate, not a proven unique cause. There are no operator-level traces or fixed-intermediate-tensor repetitions here. The cuDNN plan warnings alone do not identify the cause either.

The evidence supports the following bounded conclusions:

- The setup is not bitwise repeatable under its current GPU settings, even without probing or resumption.
- No recorded input, RNG or boundary-restoration mismatch was found in the tested path.
- The original smoke's trajectory discrepancy is plausible without an augmentation-controller state leak. This is not a blanket claim that all code is correct.
- Seven updates on one checkpoint, one machine and one serial execution per condition cannot estimate full-training variation or the additional effect of probe/resume.
- No validation/test images or metrics were evaluated in this diagnostic. No AP correction, equivalence margin or significance claim can be derived from it.

The existing four-arm pilot therefore remains a descriptive single-seed result. Its class-adaptive endpoint was about 0.20 AP points below geometry-only. This audit neither erases that deficit nor establishes that it is repeatable. The method has not yet demonstrated an advantage over the strongest simple control.

## Recommended next step

Run one narrowly bounded deterministic-backward check before extending training. Keep the same checkpoint, materialized inputs, optimizer and loss; enable strict deterministic algorithms, with the documented cuBLAS workspace configuration set before CUDA initialization. Do not use warning-only mode or silently disable determinism after an error.

Start by repeating the same first update. Record exact equality, runtime and peak GPU memory. If the path works within the declared resource cap, extend the repeats to the existing seven-update horizon; then verify the probe and resume paths under that same setting if they remain part of the intended training workflow. If an operation fails, report its identity and preserve the partial evidence. This is a new execution-setting intervention, not a change to the augmentation controller.

TorchVision 0.18.0's [RoIAlign dispatcher](https://github.com/pytorch/vision/blob/v0.18.0/torchvision/ops/roi_align.py) selects an alternative Python tensor implementation for CUDA when deterministic algorithms are enabled. Its resource cost must be measured on CCAST; it must not be assumed to fit or run at the native-kernel speed. Agreement between repeated executions of the same new path is the relevant check. Bitwise agreement between native and alternative implementations is not a reasonable automatic requirement; any cross-path numerical differences must be recorded separately.

If the deterministic path is stable and affordable, freeze it for both sides of the next controlled comparison and return to the scientific question: class-adaptive versus geometry-only under matched training budgets, with independent paired seeds before claiming a repeatable advantage. A common checkpoint and horizon are required; do not compare a changed backend in one arm against an old-backend arm as if only augmentation changed.

If strict execution is unsupported or too expensive, do not start an open-ended kernel rewrite. Document the limitation, retain the matched execution configuration, and assess run-to-run variation with a bounded replication plan. Bitwise equality is useful for debugging, but is not itself a requirement for a valid empirical study. Numerical determinism also cannot replace independent seeds or establish augmentation efficacy.

## Audit artifacts

- `audit_records.py`: reproducible archive, source, state, tensor-pair and policy audit.
- `archive_audit.json`: complete archive inventory and provenance checks.
- `audit.json`: full numerical findings and explicit verification boundaries.
- `step_comparison.csv`: all 21 non-reference update comparisons.
- `analysis_checksums.json`: hashes of the audit code and derived outputs.
- `returned_records/`: preserved downloaded records, including the nine tensor-pair files.

Run locally with `python reports/rdd_v2_repro_v2/audit_records.py`. It reads only the existing training manifest and saved diagnostic evidence; it performs no detector inference and contacts no remote host. The source links above were inspected on 2026-10-29.
'''
    (HERE / 'RESULTS.md').write_text(text, encoding='utf-8', newline='\n')


def main():
    provenance = extract_verified()
    context = read(RECORDS / 'context.json')
    protocol = read(PACKAGE / 'protocol.json')
    assert context['protocol'] == protocol
    source_sums = dict(line.split('  ', 1)[::-1] for line in (PACKAGE / 'SHA256SUMS').read_text().splitlines())
    assert source_sums == context['sources']
    for name, expected in source_sums.items():
        assert sha(PACKAGE / name) == expected, name
    pilot = PACKAGE / 'source_pilot'
    assert sha(pilot / 'SHA256SUMS') == protocol['source_manifest_sha256']
    originals = {}
    for line in (pilot / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert sha(pilot / name) == expected == sha(ROOT / 'rdd_v2_pilot_v1' / name)
        originals[name] = expected
    cfg = context['source_configuration']
    assert cfg == read(pilot / 'protocol.json')
    assert context['effective_configuration'] == {**cfg, 'workers': 0, 'warmup_steps': 0}
    assert cfg['checkpoint'] == protocol['source_checkpoint']
    train_path = ROOT / 'detection_splits_v1' / 'train.jsonl'
    assert sha(train_path) == cfg['manifests']['train']
    assert sha(train_path.with_name('config.json')) == cfg['split_config_sha256']
    rows = [json.loads(s) for s in train_path.read_text().splitlines()]
    assert len(rows) == cfg['expected_rows']['train'] and all(r['split'] == 'train' for r in rows)
    lookup = {r['image']: r for r in rows}
    row_ids = {r['image']: i for i, r in enumerate(rows)}
    provenance.update(package_sources_verified=len(source_sums), original_pilot_sources_verified=len(originals),
                      training_manifest_sha256=sha(train_path), train_rows=len(rows))
    ts = load_module('audit_trace_state', PACKAGE / 'trace_state.py')
    controller = load_module('audit_controller', pilot / 'controller.py')
    inputs = read(RECORDS / 'materialized_inputs.json')
    assert inputs['materialization_rng_unchanged']
    batches = inputs['batches']
    assert len(batches) == 7
    names = [n for batch in batches for n in batch['images']]
    plan = read(pilot / 'panel_plan.json')['2'][:2]
    assert sorted(names) == sorted(set(read(pilot / 'smoke_selection.json')['train']) | {p['image'] for p in plan})
    assert len(names) == len(set(names)) == 26
    pairing = decision = '0'*64
    for batch in batches:
        assert len(batch['images']) == len(batch['image_ids']) == len(batch['target_counts']) == len(batch['traces'])
        for name, image_id, count, event in zip(batch['images'], batch['image_ids'], batch['target_counts'], batch['traces']):
            assert name in lookup and name == event['image'] and image_id == row_ids[name]
            assert count == len(lookup[name]['targets']) == event['targets']
            assert event['operation'] == 'identity' and event['input_sha256'] == event['reference_sha256']
            assert event['pixel_changes_before_geometry'] == event['pixel_changes_after_geometry'] == 0
            paired = {k: event[k] for k in ('image', 'opportunity', 'group_index', 'group_ids', 'candidate',
                      'geometry_accepted', 'geometry_reason', 'cropped_ids', 'boxes', 'reference_sha256')}
            encode = lambda obj: json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            pairing = hashlib.sha256(bytes.fromhex(pairing) + encode(paired)).hexdigest()
            decision = hashlib.sha256(bytes.fromhex(decision) + encode(event)).hexdigest()
    assert pairing == protocol['expected_pairing_sha256'] == inputs['augmentation']['pairing_sha256']
    assert decision == protocol['expected_decision_sha256'] == inputs['augmentation']['decision_sha256']
    summary = read(RECORDS / 'suite' / 'comparison.json')
    status, complete = read(RECORDS / 'status.json'), read(RECORDS / 'COMPLETE.json')
    assert summary['status'] == status['status'] == complete['status'] == 'complete_with_divergence'
    assert status['exit_code'] == 0 and complete['updates'] == 28
    assert 0 < status['elapsed_seconds'] < protocol['diagnostic_seconds']
    assert set(summary['conditions']) == set(CONDITIONS)
    fork = read(RECORDS / 'suite/control_a/fork.json')
    phases, pairs, numeric, first = {}, [], [], {}
    for condition in CONDITIONS:
        assert read(RECORDS / 'suite' / condition / 'fork.json') == fork == summary['conditions'][condition]['fork']
        assert summary['conditions'][condition]['updates_complete'] == 7
        for step in range(1, 8):
            for phase in PHASES:
                r = read(RECORDS / 'suite' / condition / 'steps' / f'{step:02d}_{phase}.json')
                phases[condition, step, phase] = r
                rng = reconstruct_rng(r['rng_state'])
                assert {k: ts.digest(v) for k, v in rng.items()} == r['rng_sha256']
                if phase == 'before':
                    state = {k: r[k] for k in ('inputs_sha256','model_sha256','optimizer_sha256','modes',
                                               'gradients_cleared','learning_rates')}
                    assert ts.digest({**state, 'rng': rng}) == r['state']['sha256']
                    assert r['inputs_sha256'] == batches[step-1]['sha256']
                    assert r['gradients_cleared'] and r['learning_rates'] == [cfg['learning_rate']]
                if condition == 'control_a':
                    continue
                ref = phases['control_a', step, phase]
                assert r['rng_state'] == ref['rng_state'], (condition, step, phase, 'RNG')
                if phase == 'before':
                    assert r['modes'] == ref['modes']
            if condition == 'control_a':
                continue
            comparisons = read(RECORDS / 'suite' / condition / 'steps' / f'{step:02d}_comparison.json')
            ref_loss = phases['control_a', step, 'forward']['losses']
            actual_loss = phases[condition, step, 'forward']['losses']
            result = {'condition': condition, 'step': step, 'reference_total_loss': sum(ref_loss.values()),
                      'total_loss': sum(actual_loss.values())}
            result['total_loss_difference'] = result['total_loss'] - result['reference_total_loss']
            result['total_loss_difference_percent'] = 100 * result['total_loss_difference']/result['reference_total_loss']
            eq = {}
            for phase in PHASES:
                c = comparisons[phase]
                assert not c['missing_paths'] and not c['unexpected_paths']
                assert c['expected_sha256'] == phases['control_a', step, phase]['state']['sha256']
                assert c['actual_sha256'] == phases[condition, step, phase]['state']['sha256']
                assert c['equal'] == (c['expected_sha256'] == c['actual_sha256']) == (not c['changes'])
                eq[phase] = c['equal']
                if not c['equal']:
                    assert c['changed_paths'] == len(c['changes'])
                    first.setdefault(condition, {'step': step, 'phase': phase, 'changed_paths': c['changed_paths']})
                for change in c['changes']:
                    assert change['kind'] in ('tensor_values', 'value_or_type')
                    if change['kind'] == 'tensor_values':
                        assert change['expected_shape'] == change['actual_shape']
                        assert change['expected_dtype'] == change['actual_dtype']
                        assert all(math.isfinite(change[k]) for k in ('max_abs','l2','reference_l2'))
                if c.get('first_tensor_pair') is not None:
                    check_pair(pairs, c, condition, phase)
            assert summary['conditions'][condition]['comparisons'][step-1]['phase_equal'] == eq
            for phase, group in [('backward','gradients'), ('update','model'),
                                 ('update','parameter_updates'), ('update','optimizer')]:
                for k, v in magnitudes(comparisons[phase]['changes'], group).items():
                    result[group + '_' + k] = v
            result['gradient_difference_relative_to_full_reference_norm'] = (
                result['gradients_difference_l2']/phases['control_a', step, 'backward']['grad_norm'])
            numeric.append(result)
            if step == 1:
                assert eq['before'] and eq['forward'] and not eq['backward']
                changes = comparisons['backward']['changes']
                assert all(c['path'].startswith("$/str:'gradients'/str:'backbone.") for c in changes)
        if condition != 'control_a':
            assert summary['conditions'][condition]['first_difference'] == first[condition]
    assert len(pairs) == 9 and len(phases) == 112 and len(numeric) == 21
    before = read(RECORDS / 'suite/probe_only/probe_before.json')
    after = read(RECORDS / 'suite/probe_only/probe_after.json')
    assert before == after['state'] and after['comparison']['equal']
    assert ts.digest(reconstruct_boundary(before)) == after['comparison']['expected_sha256']
    assert after['comparison']['expected_sha256'] == after['comparison']['actual_sha256']
    resume = read(RECORDS / 'suite/resume_only/resume_boundary.json')
    assert resume['before'] == resume['after'] and resume['comparison']['equal']
    assert ts.digest(reconstruct_boundary(resume['before'])) == resume['comparison']['expected_sha256']
    assert resume['comparison']['expected_sha256'] == resume['comparison']['actual_sha256']
    for n in ('resume_checkpoint/last.json', 'resume_checkpoint/checkpoints/'+resume['checkpoint']['file'].replace('.pt','.json')):
        assert read(RECORDS / 'suite/resume_only' / n) == resume['checkpoint']
    assert resume['checkpoint']['step'] == protocol['initial_smoke_step'] + 1
    entry = read(RECORDS / 'suite/probe_only/probe_entry.json')
    probe_file = RECORDS / 'suite/probe_only' / entry['file']
    assert sha(probe_file) == entry['sha256'] and entry['plan_sha256'] == controller.digest(plan)
    with gzip.open(probe_file, 'rt', encoding='utf-8') as handle:
        panel = [json.loads(line) for line in handle]
    assert [p['selection'] for p in panel] == plan
    rebuilt = controller.build_policy(controller.units_from_panel(panel, lookup), cfg['controller'])
    assert rebuilt == entry['policy']
    assert not any(rebuilt['shared_allowed']) and all(not any(x['allowed']) for x in rebuilt['routed'].values())
    independent_support = {}
    for cl in range(4):
        hits = [(p['image'],p['object_indices'][i],p['source_cluster_id']) for p in panel for i in p['target_positions']
                if p['classes'][i] == cl and p['reference']['detected'][i]]
        independent_support[str(cl)] = {'detected_objects': len(hits), 'detected_clusters': len({h[2] for h in hits})}
        assert all(rebuilt['pooled'][str(cl)][k] == v for k,v in independent_support[str(cl)].items())
    reference_bytes = sum(read(RECORDS / 'suite/control_a/reference_steps' / f'step_{s:02d}.json')['bytes'] for s in range(1,8))
    assert reference_bytes == summary['reference_snapshot_bytes_retained_on_ccast']
    save('archive_audit.json', provenance)
    with (HERE / 'step_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(numeric[0]))
        writer.writeheader(); writer.writerows(numeric)
    output = {'status': 'audit_complete', 'job': context['job'], 'versions': context['versions'],
              'device': context['device'], 'backends': context['backends'], 'elapsed_seconds': status['elapsed_seconds'],
              'updates': 28, 'images': 26, 'positive_images': sum(bool(lookup[n]['targets']) for n in names),
              'negative_images': sum(not lookup[n]['targets'] for n in names), 'appearance_edits': 0,
              'batch_sizes': [len(b['images']) for b in batches], 'first_differences': first,
              'initial_forks_identical': True, 'input_hashes_match_every_step': True,
              'first_step_loss_components_buffers_rng_exact': True,
              'all_112_phase_rng_hashes_recomputed': True,
              'all_84_comparison_phase_rng_states_exact': True,
              'probe_boundary_exact': True, 'resume_boundary_exact': True,
              'probe_policy_replay_exact': True, 'probe_support_independent_recount': independent_support,
              'tensor_pairs_independently_recomputed': pairs,
              'numeric_comparisons': numeric, 'reference_snapshot_bytes_retained_on_ccast': reference_bytes,
              'limits': ['No detector inference or new training in this audit.',
                        'Raw training pixels and full checkpoint/reference tensors were not downloaded; hashes and executed checks bound their verification.',
                        'Nine bounded tensor pairs were independently recomputed. Other tensor magnitudes are verified records of GPU-side comparisons, not locally recomputed full tensors.',
                        'No hidden activation/proposal/operator trace; first observed differing phase is not a uniquely localized CUDA operation.',
                        'One serial run per condition cannot estimate run-to-run variance or additional probe/resume effects.',
                        'No accuracy measurement; no attribution of the full-pilot AP gap to numerical variation.',
                        'Local analysis uses CPU torch '+torch.__version__+' and numpy '+np.__version__+'.']}
    save('audit.json', output)
    write_report(output, provenance)
    save('analysis_checksums.json', {n:sha(HERE/n) for n in ['audit_records.py','archive_audit.json','step_comparison.csv','audit.json','RESULTS.md']})
    print(json.dumps({k:v for k,v in output.items() if k not in ['tensor_pairs_independently_recomputed','numeric_comparisons','limits']},indent=2))
    print('FIRST AND FINAL NUMERICS')
    print(json.dumps([r for r in numeric if r['step'] in (1,7)],indent=2))


if __name__ == '__main__':
    main()
