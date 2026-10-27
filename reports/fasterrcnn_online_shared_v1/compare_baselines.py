"""Four frozen runs: matched records, development metrics, controller diagnostics."""
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
prior.OUT = OUT  # Reuse read-only source checks; write only this audit's new outputs.
read, save, save_csv, digest = prior.read, prior.save, prior.save_csv, prior.digest
CONDITIONS = {
    'identity': ('fasterrcnn_baseline_v1', 'Identity', '#176B86'),
    'image_level': ('fasterrcnn_shared_v1', 'Fixed image-level', '#BA5B24'),
    'local': ('fasterrcnn_local_v1', 'Fixed local', '#6D559C'),
    'online_shared': ('fasterrcnn_online_shared_v1', 'Online shared local', '#147D50'),
}
METRICS = ('AP50_95', 'AP50', 'AP75', 'AR100')
CLASSES = prior.CLASSES


def main():
    reports, data, evidence = {}, {}, {}
    for key, (package, _, _) in CONDITIONS.items():
        reports[key], data[key], evidence[key] = prior.load_evidence(key, package)
        print(f'Original archive and audited records rechecked: {key}', flush=True)
    online = reports['online_shared']
    assert digest(OUT/'audit_online_controller.py') == online['controller_audit_source_sha256']
    controller = read(OUT/'controller_verification.json')
    assert controller['status'] == 'passed_saved_observation_and_metadata_audit'
    allowed = {'augmentation', 'augmentation_policy', 'local_policy', 'controller', 'version',
               'purpose', 'recipe_status', 'comparator', 'sources'}
    stripped = lambda c: {k: v for k, v in c.items() if k not in allowed}
    reference = data['identity']['context']
    for records in data.values():
        context = records['context']
        assert stripped(context['configuration']) == stripped(reference['configuration'])
        assert context['versions'] == reference['versions']
        assert all(context[k] == reference[k] for k in ('raw_root', 'split_root', 'profile'))
        assert records['validation_ground_truth'] == data['identity']['validation_ground_truth']
        for a, b in zip(records['history'], data['identity']['history']):
            assert all(a[k] == b[k] for k in ('epoch', 'step', 'images_seen', 'train_images', 'val_images'))
    assert data['local']['context']['configuration']['local_policy'] == data['online_shared']['context']['configuration']['local_policy']
    # Identical application opportunities and union masks; online families/strengths can differ.
    for a, b in zip(data['local']['history'], data['online_shared']['history']):
        for field in ('applied_images', 'selected_groups', 'selected_boxes', 'selected_multi_box_groups',
                      'local_reference_pixels', 'local_union_pixels', 'local_alpha_units'):
            assert a['augmentation']['counts'].get(field, 0) == b['augmentation']['counts'].get(field, 0)

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
            for other in ('identity', 'image_level', 'local'):
                row[f'online_minus_{other}_{metric}_pp'] = 100*(values['online_shared'][metric]-values[other][metric])
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
        m = data[key]['history'][r['best_epoch']-1]['validation']
        counts = m['no_target_predicted_boxes']
        assert len(counts) == 2193
        negative_counts[key] = counts
        negatives.append({'condition': key, 'epoch': r['best_epoch'], 'negative_images': len(counts),
                          'images_with_predictions': sum(v > 0 for v in counts.values()),
                          'prediction_boxes': sum(counts.values()),
                          'fraction_percent': 100*sum(v > 0 for v in counts.values())/len(counts),
                          'confidence_threshold': .25})
        runtimes.append({'condition': key, 'hours': r['run']['elapsed_seconds']/3600,
                         'host': r['run']['host'], 'gpu': r['run']['gpu'], 'pbs_job': r['run']['pbs_job']})
    assert all(v.keys() == negative_counts['identity'].keys() for v in negative_counts.values())
    changes = {s: {other: {m: 100*(online[f'{s}_metrics'][m]-reports[other][f'{s}_metrics'][m]) for m in METRICS}
                   for other in ('identity', 'image_level', 'local')} for s in ('best', 'final')}
    assert all(curves[0][f'{key}_augmentation_applied'] == 0 for key in CONDITIONS)
    warmup = {'epoch': 1, 'metrics': curves[0],
              'online_minus_control_AP50_95_pp': {key: curves[0]['online_shared_AP50_95']-curves[0][f'{key}_AP50_95'] for key in ('identity', 'image_level', 'local')},
              'interpretation': 'All four trajectories differ before augmentation starts; the cause is not established. This is not a standard error or an estimate of the final treatment effect.'}
    report = {
        'status': 'verified_descriptive_comparison_one_run_per_condition',
        'primary_comparison': 'Full-validation AP50:95 at each independently selected checkpoint; earliest maximum wins',
        'scale': 'Metric points on 0-100 scale; differences are percentage points',
        'selected_epochs': {k: r['best_epoch'] for k, r in reports.items()},
        'seed': 42, 'training_images': 26863, 'validation_images': 5760, 'validation_targets': 8225,
        'epochs_each': 20, 'updates_each': 134320, 'image_presentations_each': 537260,
        'matched_nonaugmentation_configuration': True, 'identical_validation_ground_truth': True,
        'identical_local_application_opportunities_groups_and_mask_exposure': True,
        'aggregate_comparison': aggregate, 'online_minus_control_changes_pp': changes,
        'per_class_selected_comparison': classes, 'negative_images_at_selected_epoch': negatives,
        'negative_prediction_scope': 'Relative to retained annotations; no fresh visual audit of annotation completeness',
        'runtime': runtimes, 'runtime_scope': 'Whole full-run invocation, including validation and online probes; excludes smoke and tests. Between-run differences do not isolate controller overhead.',
        'controller_record': 'controller_verification.json', 'controller_record_sha256': digest(OUT/'controller_verification.json'),
        'unaugmented_epoch1': warmup,
        'verdict': 'Online shared local has the largest observed selected-checkpoint AP50:95 of these four runs. Its small gain is not established as reliable or attributable specifically to online adaptation.',
        'limits': ['One seed and one run per condition; run-to-run variability and statistical reliability are unmeasured.',
                   'Validation both selects checkpoints and supplies reported metrics; the test set remains untouched.',
                   'Pre-augmentation divergence prevents causal attribution of small later differences from this comparison alone.',
                   'Online uses a strength grid and selects mild blur/noise; fixed local uses only level 2. No matched fixed mild or uniform-family ablation isolates the online mechanism.',
                   'Saved observation reconstruction verifies controller arithmetic, not fresh detection/matching or raw-pixel inference.',
                   'No class-adaptive run has been performed. Pooled per-class feedback cannot establish benefit from class-set conditioning.'],
        'next_action': 'Preserve these controls and the frozen online-shared protocol. Stage 5 may test class conditioning with the same grid, budget, gates and training settings. Treat it as an untested hypothesis; use a matched fixed mild control and repeated seeds before claiming an online-adaptation advantage.',
        'evidence': evidence,
        'analysis_source_sha256': digest(Path(__file__)),
        'reused_evidence_loader_sha256': digest(Path(prior.__file__)),
        'test_data_read': False, 'new_training_or_inference_executed': False,
    }
    save('comparison.json', report)
    for name, rows in [('aggregate_comparison.csv', aggregate), ('per_class_comparison.csv', classes),
                       ('comparison_curves.csv', curves), ('negative_image_comparison.csv', negatives), ('runtime_comparison.csv', runtimes)]:
        save_csv(name, rows)
    make_plots(reports, data, controller)
    print(json.dumps({'selected_metrics': aggregate[:4], 'changes_pp': changes, 'warmup': warmup,
                      'classes': classes, 'negative_images': negatives, 'runtime': runtimes}, indent=2), flush=True)


def make_plots(reports, data, controller):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    for stem in ('comparison', 'controller_policy'):
        assert not any((OUT/f'{stem}.{ext}').exists() for ext in ('png', 'pdf')), 'Preserve existing figures'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for key, (_, label, color) in CONDITIONS.items():
        history = data[key]['history']
        for ax, metric in zip(axes[0], ('AP50_95', 'AP50')):
            ax.plot(range(1, 21), [100*h['validation'][metric] for h in history], label=label, color=color, lw=1.8)
            best = reports[key]['best_epoch']
            ax.scatter([best], [100*history[best-1]['validation'][metric]], color=color, s=36, zorder=4)
        axes[0, 0].set_ylabel('Validation AP50:95')
        axes[0, 1].set_ylabel('Validation AP50')
    for ax in axes[0]:
        ax.set_xlabel('Epoch'); ax.set_xticks([1, 4, 8, 12, 16, 20]); ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8, loc='lower right')
    axes[0, 0].set_title('Full validation trajectories')
    axes[0, 1].set_title('Dots mark AP50:95-selected checkpoints')
    x = np.arange(4)
    for index, (key, (_, label, color)) in enumerate(CONDITIONS.items()):
        vals = [100*reports[key]['best_metrics']['per_class'][c]['AP50_95'] for c in CLASSES]
        axes[1, 0].bar(x+(index-1.5)*.19, vals, width=.18, label=label, color=color)
    axes[1, 0].set_xticks(x, ['D00\nLongitudinal', 'D10\nTransverse', 'D20\nAlligator', 'D40\nPotholes'])
    axes[1, 0].set_ylabel('Selected per-class AP50:95'); axes[1, 0].set_ylim(0, 35)
    axes[1, 0].set_title('Observed class results'); axes[1, 0].grid(axis='y', alpha=.18)
    labels = ['Identity', 'Fixed\nimage-level', 'Fixed local', 'Online shared\nlocal']
    ap = [100*r['best_metrics']['AP50_95'] for r in reports.values()]
    bars = axes[1, 1].bar(x, ap, color=[v[2] for v in CONDITIONS.values()], width=.65)
    axes[1, 1].bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    axes[1, 1].set_xticks(x, labels); axes[1, 1].set_ylim(0, 26)
    axes[1, 1].set_ylabel('Selected overall AP50:95'); axes[1, 1].set_title('Single-run descriptive comparison')
    axes[1, 1].grid(axis='y', alpha=.18)
    fig.suptitle('RDD2022 / Faster R-CNN / seed 42\nFour development conditions; no test evaluation or replication', fontsize=14)
    for ext in ('png', 'pdf'): fig.savefig(OUT/f'comparison.{ext}', dpi=180, metadata={'Title': 'Four-condition RDD2022 development comparison'})
    plt.close(fig)
    rounds = controller['rounds']
    epochs = [r['epoch'] for r in rounds]
    family = lambda r, f: next((v for v in r['policy']['families'] if v['family'] == f), {'level': 0, 'probability': 0.})
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.7), sharex=True, layout='constrained')
    for f, label, color in [('photometric', 'Photometric', '#BA5B24'), ('blur_noise', 'Blur / noise', '#176B86')]:
        axes[0].step(epochs+[21], [family(r, f)['level'] for r in rounds]+[family(rounds[-1], f)['level']], where='post', label=label, color=color, lw=2)
        axes[0].scatter(epochs, [family(r, f)['level'] for r in rounds], color=color, s=28)
        axes[1].step(epochs+[21], [family(r, f)['probability'] for r in rounds]+[family(rounds[-1], f)['probability']], where='post', label=label, color=color, lw=2)
    axes[0].axhline(2, ls=':', color='#777777', label='Fixed-local strength', lw=1)
    axes[0].set_yticks([0, 1, 2, 3], ['Unavailable', '1: mild', '2: medium', '3: strong'])
    axes[0].set_ylim(-.2, 3.4); axes[0].set_ylabel('Selected strength'); axes[0].legend(loc='lower right', fontsize=9)
    axes[1].axhline(.5, ls=':', color='#777777', lw=1)
    axes[1].set_ylim(-.03, 1.03); axes[1].set_ylabel('Family probability'); axes[1].set_xlabel('Epoch receiving the updated policy')
    axes[1].set_xticks(epochs); axes[1].set_xlim(1.6, 20.9)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle('What the online shared controller actually changed\nFull run: 10 updates, sufficient support throughout, no identity fallback', fontsize=13)
    for ext in ('png', 'pdf'): fig.savefig(OUT/f'controller_policy.{ext}', dpi=180, metadata={'Title': 'Online shared-local controller policy'})
    plt.close(fig)


if __name__ == '__main__':
    main()
