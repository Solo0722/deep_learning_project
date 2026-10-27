"""Three frozen development runs: source verification, matched comparison, plots."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONDITIONS = {
    'identity': ('fasterrcnn_baseline_v1', 'Identity', '#176B86'),
    'image_level': ('fasterrcnn_shared_v1', 'Fixed image-level', '#BA5B24'),
    'local': ('fasterrcnn_local_v1', 'Fixed local', '#5E4AA5'),
}
METRICS = ('AP50_95', 'AP50', 'AP75', 'AR100')
CLASSES = {'D00': 'Longitudinal cracks', 'D10': 'Transverse cracks',
           'D20': 'Alligator cracks', 'D40': 'Potholes'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(name, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    path = OUT/name
    if path.exists():
        assert path.read_bytes() == data, f'Preserve different existing output: {path}'
    else:
        with path.open('xb') as handle:
            handle.write(data)


def save_csv(name, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    save(name, stream.getvalue().encode())


def load_evidence(key, package):
    folder = ROOT/'reports'/package
    report = read(folder/'ccast_verification.json')
    assert report['status'] == 'verified_with_selected_and_final_metrics_recomputed'
    assert report['maximum_recomputed_metric_error'] < 1e-12
    assert digest(folder/'audit_ccast_results.py') == report['audit_source_sha256']
    assert digest(ROOT/report['archive']) == report['archive_sha256']
    recomputed = read(folder/'recomputed_metrics.json')
    assert recomputed[str(report['best_epoch'])] == report['best_metrics']
    assert recomputed[str(report['completed_epochs'])] == report['final_metrics']
    prefix = report['archive'].split('_records_')[0]
    wanted = {'history.json', 'context.json', 'run.json', 'best.json', 'validation_ground_truth.json', 'checksums.json'}
    captured = {}
    # Read original bytes, without extraction or rerunning inference on previous controls.
    with tarfile.open(ROOT/report['archive'], 'r|gz') as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if path.parent.as_posix() == prefix and path.name in wanted:
                assert member.isfile() and path.name not in captured
                captured[path.name] = bundle.extractfile(member).read()
    assert captured.keys() == wanted
    sums = json.loads(captured['checksums.json'])
    for name in wanted-{'checksums.json'}:
        assert hashlib.sha256(captured[name]).hexdigest() == sums[name]
    data = {name[:-5]: json.loads(content) for name, content in captured.items()}
    history, context = data['history'], data['context']
    assert len(history) == report['completed_epochs'] == 20
    chosen = max(history, key=lambda row: row['validation']['AP50_95'])
    assert chosen['epoch'] == report['best_epoch'] == data['best']['epoch']
    for selection, row in [('best', chosen), ('final', history[-1])]:
        for metric in METRICS:
            assert row['validation'][metric] == report[f'{selection}_metrics'][metric]
        assert row['validation']['per_class'] == report[f'{selection}_metrics']['per_class']
    assert context == read(folder/'verified_context.json')
    assert data['run'] == read(folder/'verified_run.json')
    assert context['configuration'] == read(ROOT/package/'protocol.json')
    if key != 'local':
        save(f'{key}_history.json', captured['history.json'])
    return report, data, {'archive_sha256': report['archive_sha256'],
                          'verification_sha256': digest(folder/'ccast_verification.json'),
                          'audit_source_sha256': report['audit_source_sha256'],
                          'recomputed_epochs': report['recomputed_epochs'],
                          'maximum_recomputed_metric_error': report['maximum_recomputed_metric_error'],
                          'scope': 'Archive and source hashes and selected records rechecked; existing direct COCO recomputation retained'}


def main():
    reports, data, evidence = {}, {}, {}
    for key, (package, _, _) in CONDITIONS.items():
        reports[key], data[key], evidence[key] = load_evidence(key, package)
    allowed = {'augmentation', 'augmentation_policy', 'local_policy', 'version', 'purpose',
               'recipe_status', 'comparator', 'sources'}
    reference = data['identity']['context']
    strip = lambda cfg: {k: v for k, v in cfg.items() if k not in allowed}
    for key, records in data.items():
        context = records['context']
        assert strip(context['configuration']) == strip(reference['configuration'])
        assert context['versions'] == reference['versions']
        assert all(context[n] == reference[n] for n in ('raw_root', 'split_root', 'profile'))
        assert records['validation_ground_truth'] == data['identity']['validation_ground_truth']
        for a, b in zip(records['history'], data['identity']['history']):
            assert all(a[n] == b[n] for n in ('epoch', 'step', 'images_seen', 'train_images', 'val_images'))
    appearance = lambda cfg: {k: v for k, v in cfg['augmentation_policy'].items() if k != 'scope'}
    assert appearance(data['image_level']['context']['configuration']) == appearance(data['local']['context']['configuration'])
    for a, b in zip(data['image_level']['history'], data['local']['history']):
        assert a['augmentation']['decision_sha256'] == b['augmentation']['shared_decision_sha256']
        assert all(b['augmentation']['counts'].get(k, 0) == v for k, v in a['augmentation']['counts'].items())
    rows, classes, curves, negatives, runtimes = [], [], [], [], []
    for selection in ('best', 'final'):
        for key, (_, label, _) in CONDITIONS.items():
            r = reports[key]
            rows.append({'selection': selection, 'condition': key, 'label': label,
                         'epoch': r['best_epoch'] if selection == 'best' else 20,
                         **{m: 100*r[f'{selection}_metrics'][m] for m in METRICS}})
    for name, label in CLASSES.items():
        values = {k: r['best_metrics']['per_class'][name] for k, r in reports.items()}
        assert len({v['target_count'] for v in values.values()}) == 1
        row = {'class': name, 'description': label, 'validation_targets': values['local']['target_count']}
        for metric in ('AP50_95', 'AP50'):
            row.update({f'{k}_{metric}': 100*v[metric] for k, v in values.items()})
            for comparison in ('identity', 'image_level'):
                row[f'local_minus_{comparison}_{metric}_pp'] = 100*(values['local'][metric]-values[comparison][metric])
        classes.append(row)
    for epoch in range(20):
        row = {'epoch': epoch+1}
        for key, records in data.items():
            h = records['history'][epoch]
            row.update({f'{key}_{m}': 100*h['validation'][m] for m in ('AP50_95', 'AP50')})
            row[f'{key}_augmentation_applied'] = h.get('augmentation', {}).get('counts', {}).get('applied_images', 0)
        curves.append(row)
    negative_counts = {}
    for key, r in reports.items():
        metrics = data[key]['history'][r['best_epoch']-1]['validation']
        counts = metrics['no_target_predicted_boxes']
        negative_counts[key] = counts
        assert len(counts) == 2193
        negatives.append({'condition': key, 'epoch': r['best_epoch'], 'negative_images': len(counts),
                          'images_with_predictions': sum(v > 0 for v in counts.values()),
                          'prediction_boxes': sum(counts.values()),
                          'fraction_percent': 100*sum(v > 0 for v in counts.values())/len(counts),
                          'confidence_threshold': .25})
        runtimes.append({'condition': key, 'hours': r['run']['elapsed_seconds']/3600,
                         'host': r['run']['host'], 'gpu': r['run']['gpu'], 'pbs_job': r['run']['pbs_job']})
    assert all(counts.keys() == negative_counts['identity'].keys() for counts in negative_counts.values())
    changes = {selection: {other: {m: 100*(reports['local'][f'{selection}_metrics'][m]-reports[other][f'{selection}_metrics'][m])
                                    for m in METRICS} for other in ('identity', 'image_level')} for selection in ('best', 'final')}
    warmup = {'epoch': 1, 'metrics': curves[0],
              'local_minus_identity_AP50_95_pp': curves[0]['local_AP50_95']-curves[0]['identity_AP50_95'],
              'local_minus_image_level_AP50_95_pp': curves[0]['local_AP50_95']-curves[0]['image_level_AP50_95'],
              'interpretation': 'Training trajectories differ before augmentation starts; the cause is not established.'}
    assert all(curves[0][f'{k}_augmentation_applied'] == 0 for k in CONDITIONS)
    local_best = reports['local']['best_metrics']['AP50_95']
    if all(local_best <= reports[k]['best_metrics']['AP50_95'] for k in ('identity', 'image_level')):
        verdict = 'The fixed local recipe has no observed selected-checkpoint AP50:95 gain over either completed control in this seed.'
    else:
        verdict = 'The fixed local recipe has an observed selected-checkpoint AP50:95 gain over at least one control in this seed; reliability remains unmeasured.'
    report = {
        'status': 'verified_descriptive_comparison_one_run_per_condition',
        'primary_comparison': 'Full-validation AP50:95 at each independently selected checkpoint; earliest maximum wins',
        'scale': 'Metric points on 0-100 scale; differences are percentage points',
        'selected_epochs': {k: v['best_epoch'] for k, v in reports.items()},
        'seed': 42, 'training_images': 26863, 'validation_images': 5760, 'validation_targets': 8225,
        'epochs_each': 20, 'updates_each': 134320, 'image_presentations_each': 537260,
        'matched_nonaugmentation_configuration': True, 'identical_validation_ground_truth': True,
        'appearance_choices_and_counts_match_image_level_exactly_all_epochs': True,
        'aggregate_comparison': rows, 'local_minus_control_changes_pp': changes,
        'per_class_selected_comparison': classes, 'negative_images_at_selected_epoch': negatives,
        'negative_prediction_scope': 'Relative to retained annotations; no visual re-audit of annotation completeness',
        'runtime': runtimes, 'runtime_scope': 'Whole full-run invocation; excludes smoke and tests; not isolated augmentation overhead',
        'augmentation': reports['local']['augmentation_verification'], 'unaugmented_epoch1': warmup,
        'verdict': verdict,
        'limits': ['One run per condition; seed variability and statistical equivalence are not estimated.',
                   'Validation selected checkpoints and supplied these metrics; no untouched test estimate.',
                   'Pre-augmentation divergence prevents attributing small later differences exclusively to augmentation.',
                   'Local versus image-level also changes edit area and boundary blending; location alone is not isolated.',
                   'Fixed recipes do not test online adaptation or class conditioning.',
                   'No new detector inference or raw-image/checkpoint-byte verification was performed locally.'],
        'next_action': 'Keep all three controls. Specify the planned online shared-local controller and its matched feedback protocol before implementing stage 4; reserve class conditioning for stage 5. Replication is needed before claiming reliable gains.',
        'evidence': evidence, 'analysis_source_sha256': digest(Path(__file__)),
        'test_data_read': False, 'new_training_or_inference_executed': False}
    save('comparison.json', report)
    for name, records in [('aggregate_comparison.csv', rows), ('per_class_comparison.csv', classes),
                          ('comparison_curves.csv', curves), ('negative_image_comparison.csv', negatives)]:
        save_csv(name, records)
    plot_paths = [OUT/f'comparison.{ext}' for ext in ('png', 'pdf')]
    assert not any(path.exists() for path in plot_paths), 'Preserve existing plots'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [1.3, 1]})
    for key, (_, label, color) in CONDITIONS.items():
        axes[0].plot(range(1, 21), [r[f'{key}_AP50_95'] for r in curves], color=color, label=label, linewidth=1.7)
        epoch = reports[key]['best_epoch']
        axes[0].scatter([epoch], [100*reports[key]['best_metrics']['AP50_95']], color=color, marker='D', s=36, zorder=5)
    axes[0].set(title='Full-validation learning curves', xlabel='Completed epoch', ylabel='AP50:95 (0-100)',
                xlim=(.5, 20.5), ylim=(0, 27), xticks=[1, 5, 10, 15, 20])
    axes[0].legend(frameon=False, fontsize=9, loc='lower right')
    x = np.arange(4)
    magnitude = .2
    for offset, reference_key, color, label in [(-.19, 'identity', '#176B86', 'Local minus identity'),
                                              (.19, 'image_level', '#BA5B24', 'Local minus image-level')]:
        values = [r[f'local_minus_{reference_key}_AP50_95_pp'] for r in classes]
        magnitude = max(magnitude, max(abs(v) for v in values))
        bars = axes[1].bar(x+offset, values, width=.35, color=color, label=label)
        axes[1].bar_label(bars, labels=[f'{v:+.2f}' for v in values], padding=3, fontsize=8)
    axes[1].axhline(0, color='#444444', linewidth=.8)
    axes[1].set(title='Per-class change at selected checkpoints', ylabel='AP50:95 change (percentage points)',
                xticks=x, xticklabels=list(CLASSES), ylim=(-magnitude*1.35, magnitude*1.35))
    axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#DDDDDD', linewidth=.6)
    fig.suptitle('Fixed augmentation baselines | One run per condition', fontsize=14, y=.98)
    selected_text = ', '.join(f'{CONDITIONS[k][1]}: {v}' for k, v in report['selected_epochs'].items())
    fig.text(.5, .035, 'Selected epochs — '+selected_text+'. Diamonds mark selected checkpoints.', ha='center', fontsize=8.5)
    fig.text(.5, .008, 'Validation-selected results. Test data unopened. No estimate of training-seed variability.', ha='center', fontsize=8.5, color='#444444')
    fig.tight_layout(rect=(0, .075, 1, .93))
    fig.savefig(plot_paths[0], dpi=180, facecolor='white')
    fig.savefig(plot_paths[1], facecolor='white')
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ('verdict', 'selected_epochs', 'aggregate_comparison',
        'local_minus_control_changes_pp', 'per_class_selected_comparison', 'negative_images_at_selected_epoch',
        'runtime', 'unaugmented_epoch1')}, indent=2))


if __name__ == '__main__':
    main()
