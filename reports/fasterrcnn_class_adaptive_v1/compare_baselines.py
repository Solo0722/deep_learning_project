"""Five frozen conditions: verified records, descriptive comparison, diagnostics.

Run after audit_ccast_results.py. Reuses prior audited baseline COCO evaluations
after checking original archive hashes and records; no new training or inference.
"""
from collections import Counter
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
spec = importlib.util.spec_from_file_location('prior_comparison', ROOT/'reports/fasterrcnn_local_v1/compare_baselines.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
prior.OUT = OUT
read, save, save_csv, digest = prior.read, prior.save, prior.save_csv, prior.digest
CONDITIONS = {
    'identity': ('fasterrcnn_baseline_v1', 'Identity', '#176B86'),
    'image_level': ('fasterrcnn_shared_v1', 'Fixed image-level', '#BA5B24'),
    'local': ('fasterrcnn_local_v1', 'Fixed local', '#6D559C'),
    'online_shared': ('fasterrcnn_online_shared_v1', 'Online shared local', '#147D50'),
    'class_adaptive': ('fasterrcnn_class_adaptive_v1', 'Online class-adaptive local', '#B33951'),
}
METRICS = ('AP50_95', 'AP50', 'AP75', 'AR100')
CLASSES = prior.CLASSES


def main():
    reports, data, evidence = {}, {}, {}
    for key, (package, _, _) in CONDITIONS.items():
        reports[key], data[key], evidence[key] = prior.load_evidence(key, package)
        print(f'Original archive and audited records rechecked: {key}', flush=True)
    adaptive = reports['class_adaptive']
    controller = read(OUT/'controller_verification.json')
    shared_controller = read(ROOT/'reports/fasterrcnn_online_shared_v1/controller_verification.json')
    assert controller['status'] == shared_controller['status'] == 'passed_saved_observation_and_metadata_audit'
    assert digest(OUT/'audit_class_controller.py') == adaptive['controller_audit_source_sha256']
    assert digest(OUT/'independent_feedback.py') == controller['independent_feedback_source_sha256']
    assert digest(ROOT/'reports/fasterrcnn_online_shared_v1/audit_online_controller.py') == reports['online_shared']['controller_audit_source_sha256']
    allowed = {'augmentation', 'augmentation_policy', 'local_policy', 'controller', 'version',
               'purpose', 'recipe_status', 'comparator', 'sources'}
    strip = lambda c: {k: v for k, v in c.items() if k not in allowed}
    reference = data['identity']['context']
    for records in data.values():
        context = records['context']
        assert strip(context['configuration']) == strip(reference['configuration'])
        assert context['versions'] == reference['versions']
        assert all(context[k] == reference[k] for k in ('raw_root', 'split_root', 'profile'))
        assert records['validation_ground_truth'] == data['identity']['validation_ground_truth']
        for a, b in zip(records['history'], data['identity']['history']):
            assert all(a[k] == b[k] for k in ('epoch', 'step', 'images_seen', 'train_images', 'val_images'))
    for a, b in zip(data['online_shared']['history'], data['class_adaptive']['history']):
        assert a['augmentation']['counts'].get('augmentation_opportunities', 0) == b['augmentation']['counts'].get('augmentation_opportunities', 0)
        for key in ('images', 'target_images', 'reason/warmup', 'reason/no_target', 'reason/probability'):
            assert a['augmentation']['counts'].get(key, 0) == b['augmentation']['counts'].get(key, 0)
    # Full decision hashes include chosen actions, policy IDs and routing fields;
    # they must not be equated across distinct policies. Identical keyed source,
    # configuration and replayed epoch order establish opportunity/group pairing.
    for filename in ('shared_augmentation.py', 'local_augmentation.py'):
        assert digest(ROOT/'fasterrcnn_class_adaptive_v1'/filename) == digest(ROOT/'fasterrcnn_online_shared_v1'/filename)
    assert digest(ROOT/'fasterrcnn_class_adaptive_v1/shared_controller.py') == digest(ROOT/'fasterrcnn_online_shared_v1/controller.py')
    assert controller['compute']['image_forwards'] == shared_controller['compute']['image_forwards'] == 71680

    aggregate, classes, curves, negatives, runtimes = [], [], [], [], []
    for selection in ('best', 'final'):
        for key, (_, label, _) in CONDITIONS.items():
            r = reports[key]
            aggregate.append({'selection': selection, 'condition': key, 'label': label,
                              'epoch': r['best_epoch'] if selection == 'best' else 20,
                              **{m: 100*r[f'{selection}_metrics'][m] for m in METRICS}})
    for name, label in CLASSES.items():
        values = {k: r['best_metrics']['per_class'][name] for k, r in reports.items()}
        assert len({v['target_count'] for v in values.values()}) == 1
        row = {'class': name, 'description': label, 'validation_targets': values['identity']['target_count']}
        for metric in ('AP50_95', 'AP50'):
            row.update({f'{k}_{metric}': 100*v[metric] for k, v in values.items()})
            for other in (k for k in CONDITIONS if k != 'class_adaptive'):
                row[f'class_adaptive_minus_{other}_{metric}_pp'] = 100*(values['class_adaptive'][metric]-values[other][metric])
        classes.append(row)
    for epoch in range(20):
        row = {'epoch': epoch+1}
        for key, records in data.items():
            h = records['history'][epoch]
            row.update({f'{key}_{m}': 100*h['validation'][m] for m in ('AP50_95', 'AP50')})
            row[f'{key}_augmentation_applied'] = h.get('augmentation', {}).get('counts', {}).get('applied_images', 0)
        curves.append(row)
    for key, r in reports.items():
        m = data[key]['history'][r['best_epoch']-1]['validation']
        counts = m['no_target_predicted_boxes']
        assert len(counts) == 2193
        negatives.append({'condition': key, 'epoch': r['best_epoch'], 'negative_images': len(counts),
                          'images_with_predictions': sum(v > 0 for v in counts.values()),
                          'prediction_boxes': sum(counts.values()),
                          'fraction_percent': 100*sum(v > 0 for v in counts.values())/len(counts),
                          'confidence_threshold': .25})
        runtimes.append({'condition': key, 'hours': r['run']['elapsed_seconds']/3600,
                         'host': r['run']['host'], 'gpu': r['run']['gpu'], 'pbs_job': r['run']['pbs_job']})
    changes = {s: {other: {m: 100*(adaptive[f'{s}_metrics'][m]-reports[other][f'{s}_metrics'][m]) for m in METRICS}
                   for other in CONDITIONS if other != 'class_adaptive'} for s in ('best', 'final')}
    assert all(curves[0][f'{key}_augmentation_applied'] == 0 for key in CONDITIONS)
    reasons = {reason: sum(r[reason] for r in controller['class_set_exposure'])
               for reason in ('unobserved_class_set', 'insufficient_support', 'no_admissible_family')}
    strength = {level: sum(r['images'] for r in controller['operation_strength_training_exposure'] if r['level'] == level)
                for level in range(1, 4)}
    per_set = []
    for s in controller['class_set_exposure']:
        cells = [c for c in controller['class_set_rounds'] if c['class_set'] == s['class_set']]
        per_set.append({**s, 'active_updates': sum(c['reason'] == 'active' for c in cells),
                        'observed_updates': sum(c['panel_images'] > 0 for c in cells)})
    p_photo = [f['probability'] for r in controller['rounds'] for cell in r['policy']['class_sets'].values()
               if len(cell['families']) == 2 for f in cell['families'] if f['family'] == 'photometric']
    diagnostic = {
        'opportunities': controller['total']['augmentation_opportunities'],
        'class_adaptive_applied': controller['total']['applied_images'],
        'class_adaptive_changed': controller['reported_changed_images'],
        'online_shared_applied': shared_controller['total']['applied_images'],
        'online_shared_changed': shared_controller['reported_changed_images'],
        'identity_fallbacks': controller['total']['controller_identity_images'],
        'identity_fraction_percent': 100*controller['identity_opportunity_fraction'],
        'fallback_reasons': reasons, 'applied_by_level': strength,
        'applied_percent_by_level': {k: 100*v/controller['total']['applied_images'] for k, v in strength.items()},
        'class_set_summary': per_set,
        'photometric_probability_when_two_families_eligible': {
            'min': min(p_photo), 'max': max(p_photo),
            'mean_absolute_deviation_from_half': float(np.mean(np.abs(np.asarray(p_photo)-.5)))},
        'probe_minutes_class_adaptive': controller['compute']['seconds']/60,
        'probe_minutes_online_shared': shared_controller['compute']['seconds']/60,
        'additional_image_forwards_each': 71680,
    }
    report = {
        'status': 'verified_descriptive_comparison_one_run_per_condition',
        'primary_comparison': 'Class-adaptive versus online shared: full-validation AP50:95 at independently selected checkpoints; earliest maximum wins',
        'scale': 'Metrics on 0-100 scale; differences are percentage points',
        'selected_epochs': {k: r['best_epoch'] for k, r in reports.items()},
        'seed': 42, 'training_images': 26863, 'validation_images': 5760, 'validation_targets': 8225,
        'epochs_each': 20, 'optimizer_steps_each': 134320, 'image_presentations_each': 537260,
        'matched_nonaugmentation_configuration': True, 'identical_validation_ground_truth': True,
        'matched_online_probe_budget_and_prospective_panels': True,
        'matched_application_opportunities': True, 'matched_actual_edit_exposure': False,
        'aggregate_comparison': aggregate, 'class_adaptive_minus_control_changes_pp': changes,
        'per_class_selected_comparison': classes, 'negative_images_at_selected_epoch': negatives,
        'negative_prediction_scope': 'False positives relative to retained target annotations; no fresh visual audit of annotation completeness',
        'runtime': runtimes,
        'runtime_scope': 'Full-run invocation including validation and probes, excluding smoke/tests; between-run times do not isolate controller overhead',
        'controller_diagnostics': diagnostic,
        'unaugmented_epoch1': {'metrics': curves[0],
            'interpretation': 'Trajectories differ before augmentation; cause unknown. Not a standard error and not a valid correction to later scores.'},
        'verdict': 'The implemented class-adaptive controller executed its declared policy, but no observed AP50:95 gain over online shared or identity was obtained in this development run.',
        'limits': [
            'One seed/run per condition; no estimate of variation across training repetitions, no significance or equivalence claim.',
            'Validation selects checkpoints and supplies comparisons; no untouched test estimate.',
            'Pre-augmentation divergence prevents assigning small later score differences solely to augmentation.',
            'Exact-set conditioning changes support, fallback frequency, strengths and weighting together; their separate downstream effects are not identified.',
            'Fixed local uses level 2 while online methods select from a three-level grid; no fixed mild-strength or uniform-weight ablation has been run.',
            'Policy arithmetic reconstructed from saved observations; raw detector matching, per-ROI losses, pixels and checkpoint tensor bytes were not independently rerun.',
            'The run records do not independently supply the batch job test/smoke logs; full-run completion is separately established.'
        ],
        'next_action': 'Preserve the negative primary comparison. Inspect retention versus useful augmentation strength and exact-set support before further GPU work; any revised estimator is a new declared condition. Use matched repetitions to assess reliability, not outcome-driven threshold changes.',
        'evidence': evidence, 'analysis_source_sha256': digest(Path(__file__)),
        'reused_evidence_loader_sha256': digest(Path(prior.__file__)),
        'controller_record_sha256': digest(OUT/'controller_verification.json'),
        'test_data_read': False, 'new_training_or_detector_inference_executed': False,
    }
    save('comparison.json', report)
    for name, rows in [('aggregate_comparison.csv', aggregate), ('per_class_comparison.csv', classes),
                       ('comparison_curves.csv', curves), ('negative_image_comparison.csv', negatives),
                       ('runtime_comparison.csv', runtimes), ('class_set_summary.csv', per_set)]:
        save_csv(name, rows)
    make_plots(reports, data, controller)
    write_report(report, controller)
    print(json.dumps({'selected_metrics': aggregate[:5], 'changes_pp': changes['best'],
                      'controller': {k: v for k, v in diagnostic.items() if k != 'class_set_summary'},
                      'class_sets': per_set, 'negatives': negatives, 'runtime': runtimes}, indent=2), flush=True)


def make_plots(reports, data, controller):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for filename in ('comparison.png', 'controller_strengths.png'):
        assert not (OUT/filename).exists(), 'Preserve prior figures'
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for key, (_, label, color) in CONDITIONS.items():
        axes[0].plot(range(1, 21), [100*h['validation']['AP50_95'] for h in data[key]['history']],
                     label=label, color=color, lw=1.6)
    axes[0].set(xlabel='Epoch', ylabel='Validation AP50:95 (0-100)', title='One training run per condition', xticks=[1, 4, 8, 12, 16, 20])
    axes[0].legend(fontsize=8, loc='lower right')
    keys = list(CONDITIONS)
    values = [100*reports[k]['best_metrics']['AP50_95'] for k in keys]
    axes[1].barh(range(5), values, color=[CONDITIONS[k][2] for k in keys])
    axes[1].set(yticks=range(5), yticklabels=[CONDITIONS[k][1] for k in keys], xlim=(0, 27),
                xlabel='Selected validation AP50:95 (0-100)', title='Selected checkpoint; no test evaluation')
    axes[1].invert_yaxis()
    for i, value in enumerate(values):
        axes[1].text(value+.2, i, f'{value:.3f}', va='center', fontsize=9)
    for axis in axes:
        axis.grid(axis='x', alpha=.18)
    fig.savefig(OUT/'comparison.png', dpi=180)
    plt.close(fig)

    sets = [r for r in controller['class_set_exposure'] if r['applied']]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, family in zip(axes, ('photometric', 'blur_noise')):
        values = [[next(c for c in controller['class_set_rounds'] if c['epoch'] == e and c['class_set'] == s['class_set'])[f'{family}_level']
                   for e in range(2, 21, 2)] for s in sets]
        axis.imshow(values, cmap=plt.get_cmap('Blues', 4), vmin=0, vmax=3, aspect='auto')
        axis.set(xticks=range(10), xticklabels=range(2, 21, 2), yticks=range(len(sets)),
                 yticklabels=[s['labels'] for s in sets], xlabel='Update before epoch',
                 title=('Photometric' if family == 'photometric' else 'Blur / noise'))
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                axis.text(j, i, str(value), ha='center', va='center', color='white' if value >= 2 else 'black', fontsize=9)
    fig.suptitle('Selected strengths by exact class set: 0 = family unavailable, 1-3 = grid level')
    fig.savefig(OUT/'controller_strengths.png', dpi=180)
    plt.close(fig)


def write_report(report, controller):
    d = report['controller_diagnostics']
    delta = report['class_adaptive_minus_control_changes_pp']['best']
    lines = [
        '# Online class-adaptive local augmentation: completed-run audit', '',
        '## Finding', '',
        'The controller executed its declared exact-class-set policy, but the primary comparison did not show an accuracy gain. '
        f"Selected validation AP50:95 is 22.563: {delta['online_shared']['AP50_95']:+.3f} percentage points relative to online shared, "
        f"{delta['identity']['AP50_95']:+.3f} relative to identity, and {delta['local']['AP50_95']:+.3f} relative to fixed local. "
        'These are descriptive results from one training run per condition, not an equivalence or statistical significance result.', '',
        '## Controlled comparison', '',
        'Each condition started fresh with the same COCO initialization recipe, seed 42, frozen manifests, detector, optimizer schedule and evaluation. '
        'All completed 20 epochs, 134,320 optimizer steps and 537,260 image presentations on 26,863 training images. '
        'The 5,760-image validation set contains 8,225 retained target boxes. Maximum validation AP50:95 selects a checkpoint; earliest ties win. '
        'The test partition was not evaluated.', '',
        '| Condition | Selected epoch | AP50:95 | AP50 | Final AP50:95 |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in report['aggregate_comparison'][:5]:
        final = next(v for v in report['aggregate_comparison'] if v['condition'] == r['condition'] and v['selection'] == 'final')
        lines.append(f"| {r['label']} | {r['epoch']} | {r['AP50_95']:.3f} | {r['AP50']:.3f} | {final['AP50_95']:.3f} |")
    lines += ['', 'All scores use a 0-100 scale; differences are percentage points.', '', '## Class-level results', '',
              '| Class | Targets | Identity AP50:95 | Online shared AP50:95 | Class-adaptive AP50:95 | Adaptive minus shared |',
              '|---|---:|---:|---:|---:|---:|']
    for r in report['per_class_selected_comparison']:
        lines.append(f"| {r['class']} ({r['description']}) | {r['validation_targets']} | {r['identity_AP50_95']:.3f} | "
                     f"{r['online_shared_AP50_95']:.3f} | {r['class_adaptive_AP50_95']:.3f} | {r['class_adaptive_minus_online_shared_AP50_95_pp']:+.3f} |")
    lines += ['', '## Controller behavior and exposure', '',
              'All ten updates used the same prospective 256-image/group panels as online shared. The budget was 71,680 additional image forwards, plus auxiliary ROI computation. '
              'All 15 exact-set states per update, including unsupported states, were independently reconstructed from saved per-object observations.', '',
              f"Of {d['opportunities']:,} matched application opportunities, {d['class_adaptive_applied']:,} applied an edit and "
              f"{d['identity_fallbacks']:,} ({d['identity_fraction_percent']:.2f}%) used identity. "
              f"The recorded changed-image count is {d['class_adaptive_changed']:,}. Online shared applied {d['online_shared_applied']:,} edits, "
              f"of which {d['online_shared_changed']:,} changed pixels. Thus opportunity counts match, while actual edit exposure does not.", '',
              'Fallback reasons: '+', '.join(f"{k.replace('_', ' ')} = {v:,}" for k, v in d['fallback_reasons'].items())+'.', '',
              'Applied strength distribution: '+', '.join(f"level {k} = {v:,} ({d['applied_percent_by_level'][k]:.2f}%)" for k, v in d['applied_by_level'].items())+'.', '',
              '| Exact class set | Active updates / 10 | Opportunities | Applied edits | Identity |',
              '|---|---:|---:|---:|---:|']
    for r in d['class_set_summary']:
        lines.append(f"| {r['labels']} | {r['active_updates']} | {r['opportunities']:,} | {r['applied']:,} | {r['identity']:,} |")
    lines += ['', 'Exact-set support constraints and class-specific strengths are observable implementation behavior. '
              'Neither passing the 80% retention gate nor increasing the auxiliary training loss establishes useful regularization or label preservation. '
              'The saved results do not identify whether stronger edits, missing support, family weighting, or training variability caused the downstream difference.', '',
              'The same-snapshot pooled-policy diagnostic in `controller_verification.json` is exploratory arithmetic on the class-adaptive detector snapshots. '
              'It is not an alternative trained detector and supplies no counterfactual AP estimate.', '',
              '## False alarms and runtime', '',
              '| Condition | No-target images with detections / 2,193 | Fraction | Full-run hours |',
              '|---|---:|---:|---:|']
    for n, runtime in zip(report['negative_images_at_selected_epoch'], report['runtime']):
        lines.append(f"| {CONDITIONS[n['condition']][1]} | {n['images_with_predictions']} | {n['fraction_percent']:.2f}% | {runtime['hours']:.2f} |")
    lines += ['', 'Detections here use confidence 0.25 and are false positives relative to retained annotations; negative images were not visually re-audited. '
              'Full-run time includes validation and probes but excludes tests and smoke; between-run timing does not isolate controller overhead. '
              f"Recorded probe time was {d['probe_minutes_class_adaptive']:.2f} minutes for class-adaptive and {d['probe_minutes_online_shared']:.2f} minutes for online shared.", '',
              '## Verification and limits', '',
              'The returned archive SHA-256 is `c4817b3cb3bc69813776d06662a9070a7415e9552cab7c6b331c5237414d9b74`. '
              'All 123 archive members and checksum coverage were verified. Archived source bytes match the released package. '
              'The frozen training/validation manifests, original-coordinate validation ground truth, version/configuration records, training budgets, '
              'all 20 prediction files, epoch histories and best/final checkpoint metadata were checked. '
              'Direct COCO evaluation independently recomputed epochs 18 and 20, including per-class metrics; maximum discrepancy was zero. '
              'All 20 epochs of keyed training decisions, order, group geometry, fallbacks and exposure counters were replayed. '
              'Pixel counters and state-preservation flags were checked for internal consistency; raw pixels, probe logits/boxes, live detector state and full checkpoints were not re-executed.', '',
              'Before augmentation began, epoch-1 AP50:95 already differed across conditions: '+
              ' / '.join(f"{report['unaugmented_epoch1']['metrics'][f'{k}_AP50_95']:.3f}" for k in CONDITIONS)+
              ' (table order). The cause is unestablished. These differences are not an estimate of final-run variance and must not be subtracted from later scores. '
              'One run per condition cannot establish repeatability or equivalence; validation-based selection also means these are development results. '
              'The first four conditions retain their earlier independent metric audits; this analysis rechecks their original archive hashes and comparison records.', '',
              '## Decision', '',
              'The planned five-condition development sequence is complete. Report a working, audited class-adaptive prototype with no observed gain over its primary online-shared comparator under this protocol. '
              'Keep this result rather than changing thresholds after observing it. Before further GPU work, inspect whether the retention/loss criterion selects useful augmentation strengths and whether exact-set support is adequate for the intended cases. '
              'A matched fixed-strength or uniform-family ablation can address attribution; matched seed repetitions are needed to assess reliability. '
              'Any revised controller is a new experimental condition. No additional training, detector inference, test evaluation, manuscript change or publication was performed by this audit.', '',
              'Reproduce with `python reports/fasterrcnn_class_adaptive_v1/audit_ccast_results.py` then '
              '`python reports/fasterrcnn_class_adaptive_v1/compare_baselines.py`. The audit records Python/library versions and source hashes. '
              'Plot creation refuses existing figures; preserve or choose a fresh output directory when regenerating them. '
              'Primary outputs: `ccast_verification.json`, `controller_verification.json`, `comparison.json`, CSV tables, `comparison.png` and `controller_strengths.png`.', '']
    save('RESULTS.md', '\n'.join(lines).encode())


if __name__ == '__main__':
    main()
