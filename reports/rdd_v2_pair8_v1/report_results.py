"""Generate the bounded conclusion, tables and monochrome figure from audited records."""
from collections import Counter
import csv
import json
from pathlib import Path

from audit_returned import ROOT, REPORT, RETURNED, PACKAGE, load, save, sha


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def csvfile(name, rows):
    with (REPORT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    integrity=load(REPORT/'archive_audit.json')
    execution=load(REPORT/'execution_audit.json')
    replay=load(REPORT/'exposure_replay.json')
    controller=load(REPORT/'controller_audit.json')
    metric=load(REPORT/'metric_recomputation.json')
    assert execution['status']==replay['status']=='pass' and len(controller['updates'])==4
    assert all(u['status']=='pass' for u in controller['updates'].values())
    results=metric['results'];g=results['geometry_only'];c=results['class_adaptive']
    assert all(x['maximum_metric_error']==0 for x in results.values())
    histories={a:load(RETURNED/a/'history.json') for a in ['geometry_only','class_adaptive']}
    epochs=[]
    for gh,ch in zip(histories['geometry_only'],histories['class_adaptive']):
        epochs.append({'epoch':gh['epoch'],'geometry_AP50_95':gh['validation']['AP50_95']*100,
            'class_AP50_95':ch['validation']['AP50_95']*100,
            'class_minus_geometry_AP50_95_points':(ch['validation']['AP50_95']-gh['validation']['AP50_95'])*100,
            'geometry_AP50':gh['validation']['AP50']*100,'class_AP50':ch['validation']['AP50']*100,
            'class_applied':ch['augmentation']['counts']['all/applied'],
            'class_changed':ch['augmentation']['counts']['all/changed']})
    assert all(r['class_minus_geometry_AP50_95_points']<0 for r in epochs)
    csvfile('epoch_comparison.csv',epochs)
    endpoint=[]
    for k in ['AP50_95','AP50','AP75','AR100','AP_small','AP_medium','AP_large']:
        endpoint.append({'metric':k,'geometry_only_percent':100*g['metrics'][k],
            'class_adaptive_percent':100*c['metrics'][k],
            'class_minus_geometry_points':100*(c['metrics'][k]-g['metrics'][k])})
    csvfile('endpoint_comparison.csv',endpoint)
    per_class=[{'class':cl,'geometry_AP50_95':100*g['per_class'][cl]['AP50_95'],
        'class_AP50_95':100*c['per_class'][cl]['AP50_95'],
        'class_minus_geometry_points':100*(c['per_class'][cl]['AP50_95']-g['per_class'][cl]['AP50_95'])}
        for cl in ['D00','D10','D20','D40']]
    assert all(r['class_minus_geometry_points']<0 for r in per_class)
    csvfile('per_class_comparison.csv',per_class)
    actions=Counter();borrowed=0
    for e in replay['epochs'].values():
        actions.update(e['sampled_action_counts']['class_adaptive'])
        borrowed+=e['borrowed_opportunities']['class_adaptive']
    applied=sum(actions.values());changed=sum(e['class_changed'] for e in epochs)
    quality=sum(v for k,v in actions.items() if k.startswith('gaussian_'))
    csvfile('action_counts.csv',[{'action':k,'count':v,'percent_of_applied':100*v/applied} for k,v in sorted(actions.items())])
    # These distributions replay alternative rules on this arm's observations;
    # they do not constitute trained shared/fixed comparator results.
    last=histories['class_adaptive'][-1]['augmentation']['counts']
    run=load(RETURNED/'class_adaptive/run.json')
    singleton=run['controller_updates'][-1]['policy']['direct']['3']['3']
    probe_seconds=execution['resources']['class_adaptive']['probe_seconds']
    recorded_hours=(execution['main_invocation_seconds']+execution['gpu_gate_seconds'])/3600
    delta=endpoint[0]['class_minus_geometry_points']
    summary={'status':'audited','epoch':8,'seed':42,'primary_delta_AP50_95_points':delta,
        'primary_relative_change_percent':100*(c['metrics']['AP50_95']/g['metrics']['AP50_95']-1),
        'all_seven_recorded_epochs_have_negative_AP_difference':True,'all_four_classes_have_negative_final_AP_difference':True,
        'applied_appearance_opportunities':applied,'changed_appearance_images':changed,
        'blur_or_noise_actions':quality,'blur_or_noise_percent':100*quality/applied,
        'borrowed_opportunities':borrowed,'borrowed_percent':100*borrowed/applied,
        'controller_probe_minutes':probe_seconds/60,'main_and_gate_recorded_hours':recorded_hours,
        'verdict':'Current v2 class-adaptive recipe did not beat geometry-only at the declared fixed endpoint; do not scale the unchanged recipe.',
        'limits':'One development seed and horizon; no seed-variance or test-generalization claim. No full tensor re-inference.'}
    save('analysis_summary.json',summary)
    endpoints=table(['Metric (%)','Geometry-only','Class-adaptive','Difference (points)'],
        [[r['metric'],f"{r['geometry_only_percent']:.4f}",f"{r['class_adaptive_percent']:.4f}",f"{r['class_minus_geometry_points']:+.4f}"] for r in endpoint])
    classes=table(['Class','Geometry AP','Class-adaptive AP','Difference'],
        [[r['class'],f"{r['geometry_AP50_95']:.4f}",f"{r['class_AP50_95']:.4f}",f"{r['class_minus_geometry_points']:+.4f}"] for r in per_class])
    curve=table(['Epoch','Geometry AP','Class-adaptive AP','Difference'],
        [[r['epoch'],f"{r['geometry_AP50_95']:.4f}",f"{r['class_AP50_95']:.4f}",f"{r['class_minus_geometry_AP50_95_points']:+.4f}"] for r in epochs])
    support_rows=[]
    for epoch in [2,4,6,8]:
        u=controller['updates'][f'class_adaptive/epoch_{epoch}']
        support_rows.append([epoch,*[f"{u['pooled'][str(cl)]['detected_objects']}/{u['pooled'][str(cl)]['detected_clusters']}/{u['pooled'][str(cl)]['allowed_actions']}" for cl in range(4)]])
    support=table(['Before epoch','D00 objects/clusters/actions','D10 objects/clusters/actions','D20 objects/clusters/actions','D40 objects/clusters/actions'],support_rows)
    attempts=table(['PBS job','Outcome','Main invocation hours'],
        [[a['job'],a['status'],f"{a['elapsed_seconds']/3600:.3f}"] for a in execution['attempts']])
    thresholds=table(['Diagnostic at confidence 0.25, IoU 0.5','Geometry-only','Class-adaptive'],[
        ['Correct matched targets',g['threshold_diagnostic']['true_positives'],c['threshold_diagnostic']['true_positives']],
        ['Recall (%)',f"{100*g['threshold_diagnostic']['recall']:.3f}",f"{100*c['threshold_diagnostic']['recall']:.3f}"],
        ['Precision (%)',f"{100*g['threshold_diagnostic']['precision']:.3f}",f"{100*c['threshold_diagnostic']['precision']:.3f}"],
        ['False-positive boxes on no-target images',g['no_target_boxes'],c['no_target_boxes']],
        ['No-target images with predictions, out of 2,193',g['no_target_images_with_predictions'],c['no_target_images_with_predictions']]])
    text=f'''# RDD v2 paired epoch-8 experiment: audited result

Audit date: 2026-11-03 (America/Chicago). Run: `rdd_v2_pair8_seed42_v1`. Original source package and returned records are preserved.

## Verdict

The class-adaptive local-augmentation recipe did not beat geometry-only at the declared endpoint. Epoch-8 AP50:95 is {100*c['metrics']['AP50_95']:.4f}% versus {100*g['metrics']['AP50_95']:.4f}%, a {delta:.4f}-percentage-point difference ({summary['primary_relative_change_percent']:.2f}% relative). All four classes have lower final AP, and class-adaptive trails on AP at every recorded epoch from 2 through 8. This is an unfavorable one-seed development result for this recipe, not proof that every class-adaptive method is ineffective.

Recommendation: stop scaling the unchanged recipe. Keep geometry-only as the reference for this v2 comparison. Another longer run is not justified by the declared objective or the observed curves. The checked execution records substantiate implementation of the intended protocol; the main unresolved issue is augmentation utility.

## Planned and executed methodology

Both arms independently restored the fixed identity epoch-1 checkpoint (`dbd619334a57ad4422e929b90b327f80d1ca3bfc5b86f6f37594714ec138e858`), including model, optimizer and RNG. They used the same pretrained Faster R-CNN ResNet-50-FPN, 640-pixel letterbox, batch four, seed 42 and SGD recipe. All 26,863 training images were processed once per epoch, including 10,233 without target-class boxes. Each arm performed 47,012 additional optimizer updates and 188,041 additional image presentations through epoch 8. Full clean validation used the same 5,760 images and 8,225 boxes. No test evaluation was performed by the audited entry path.

Both arms used the same keyed 54-candidate reflection/scale/translation geometry, with annotation-preserving whole-candidate rejection. Only the class-adaptive arm applied local appearance edits in selected overlapping-box unions. A 0.5 opportunity on positive images was shared by the protocol; eligible actions were sampled uniformly by family, operation and tested level. The eight operations and three levels, support rules, sampling and detector-retention threshold were unchanged from the earlier v2 pilot.

The controller measured the current detector before epochs 2, 4, 6 and 8 on four frozen 256-image training panels. Correct-class detections at confidence 0.25 and IoU 0.5 defined reference support. An action needed at least 0.8 retention with 16 detected objects from eight clusters. Pooled per-class evidence, direct class-set vetoes and explicitly marked borrowed evidence determined routing. These measurements estimate tolerance to a transformation, not the validation benefit of training on it.

Strict execution used deterministic algorithms with warn-only disabled, deterministic cuDNN, benchmark and TF32 disabled, the declared CUBLAS workspace and pinned RoIAlign source. Class-adaptive ran first. Two administrative time-limit stops were followed by explicit resumes. Both arms completed the fixed endpoint, with no accuracy-dependent stopping or epoch selection. The original epoch-4 pilot used a different execution regime and is not pooled with this run as another seed.

## Primary and secondary results

AP, AR and differences below are reported on the 0-100 scale. The endpoint is epoch 8, not a selected best checkpoint.

{endpoints}

{classes}

The largest class AP deficit is D20. D00, D10 and D40 also decline. The area metrics use original-image pixel area and are resolution-dependent; they are diagnostic rather than an invariant crack-size definition. A small D20 area-specific gain does not overturn the class's overall AP loss.

{curve}

The common warmup AP50:95 was {100*results['common_warmup']['metrics']['AP50_95']:.4f}%. Both arms learned beyond that starting point. The seven epoch differences are correlated observations from one paired run, not seven independent replications. They describe the observed trajectory and do not provide a seed-variance estimate or a significance test.

![Validation trajectories and paired differences](epoch8_comparison.png)

## Counterevidence and threshold trade-off

{thresholds}

Class-adaptive produces fewer high-confidence false-positive detections on the 2,193 images without target-class annotations and slightly higher overall precision at the declared threshold. It also finds fewer target objects, and AP50:95, AP50 and AR100 are lower. Thus, the false-positive reduction is a trade-off, not evidence that the primary AP objective succeeded. Confidence was not tuned after inspection. D10 threshold recall improves slightly despite its AP loss; do not describe every diagnostic as uniformly worse.

## Controller operation, support and exposure

All 16 pooled class-by-update support cells passed the declared 16-object/eight-cluster requirement. Counts below are detected reference objects / distinct supporting clusters / admissible actions out of 24.

{support}

The audit independently replayed matching, all-annotation affine geometry, retention and routing for 29,696 saved controller views (four panels x 256 images x 29 views). Including repeat-parity inputs, the probes processed 29,712 detector images. The model/optimizer/RNG/mode audit records match the preceding fork or validation state. No probe exceeded its 30-minute limit and no incomplete policy was adopted.

Across epochs 2-8 there were {applied:,} appearance opportunities. Every opportunity selected an action; {changed:,} changed pixels after geometry. Consequently this run did not collapse to identity. There were {borrowed:,} borrowed-evidence opportunities ({100*borrowed/applied:.2f}%). Gaussian blur or noise accounted for {quality:,} selected actions ({100*quality/applied:.2f}%), consistent with the equal-family sampler. This frequency does not establish that those operations caused the loss; their effect was not isolated in this comparison.

Pooled support does not eliminate sparse direct class-set support. At epoch 8, the singleton D40 set had only {singleton['detected_objects']} detected objects in {singleton['detected_clusters']} clusters, below the 16-object rule; its route borrowed pooled D40 evidence. {last['group_class/3/borrowed_opportunities']:,} of {last['group_class/3/opportunities']:,} opportunities involving D40 borrowed evidence ({100*last['group_class/3/borrowed_opportunities']/last['group_class/3/opportunities']:.2f}%). This remains a methodological weakness, but the data do not establish it as the cause of the loss, especially because all four classes lose AP.

The controller admits any action satisfying tolerance and continues to apply edits whenever an action is eligible. It does not compare the training utility of that action against leaving the image unchanged. This is a plausible mismatch between the measured quantity and the desired outcome. The current result supports investigating that mismatch, not claiming it has been causally proven.

`exposure_replay.json` also compares this arm's routed distribution with fixed and global rules on the same saved observations. Those are descriptive counterfactual sampling rules; they are not newly trained fixed-local or online-shared baselines.

## Execution and reproducibility audit

- Downloaded archive SHA-256: `{integrity['archive_sha256']}`; {integrity['archive_bytes']:,} bytes.
- All {integrity['internal_checksums']} payload checksums, {integrity['source_checksums']} package sources, protocol, package versions, train/validation manifests and split configuration matched.
- Three GPU gates: 26 training and 24 validation images each, seven updates per branch. All 42 paired before/after phase comparisons matched exactly. Another 28 cross-job reference phase comparisons matched across two physical A10 devices. All three saved restore boundaries and paired prediction lists matched.
- Both arms had identical warmup fork metadata. All seven epoch pairing hashes agreed. Independent metadata replay matched all non-pixel exposure counters, geometry rejection, group selection and sampled actions across 376,082 training-image presentations.
- All 14 epoch validation records, clipping coverage and available completed-checkpoint metadata were consistent. Both endpoints recorded step 53,728 including warmup, with 214,904 total image presentations each.
- Independently assembled ground truth plus official COCOeval reproduced the common warmup and both endpoint metrics with maximum absolute difference zero. Evaluation ran locally with NumPy {metric['versions']['numpy']} / pycocotools {metric['versions']['pycocotools']}; the recorded CCAST pycocotools version is 2.0.10.
- The final archive intentionally omits full checkpoint tensors. Its writer also omits nested `checksums.json` files by basename. Their three redundant gate manifests were reconstructed from the preserved copied hashes and verified exactly; the gate payloads themselves were present and checked.

These checks found no failure in the audited paths that explains away the AP deficit. They are not a proof that all code is bug-free. Full-run checkpoints and source RGB images remain on CCAST, so the audit did not repeat detector inference or reproduce a second uninterrupted full-training trajectory. Main-run resumes are supported by immutable context, saved counters, paired hashes and available checkpoint metadata; exact short restore behavior is directly evidenced by the gates. Logged pixel hashes and changed-pixel counts were not independently rerendered.

## Resource use

{attempts}

Main invocations plus the three GPU gates recorded {recorded_hours:.2f} hours. This excludes queueing, some startup/unit-test work and archive creation, so it is not a complete billing record. The four full controller probes added {probe_seconds/60:.2f} minutes of measured work. Training-loop wall times were {execution['resources']['geometry_only']['training_seconds']/3600:.3f} hours for geometry-only and {execution['resources']['class_adaptive']['training_seconds']/3600:.3f} hours for class-adaptive; probe costs are separate. Sequential timings are descriptive, not a controlled throughput benchmark.

The four 128-update timing windows had median update times between 0.906 and 0.915 seconds. Recorded full-run peak reserved memory remained below the 90% allocator cap. Resource overhead is real, but the decision here is driven by worse primary accuracy rather than cost alone.

## Next decision

Do not launch a longer unchanged v2 run or another general determinism diagnostic. The existing evidence is sufficient to make this recipe a lower priority for additional compute; it is not sufficient to claim universal inferiority or to end all research on class-adaptive augmentation.

If development continues, the next bounded experiment should first isolate whether useful appearance augmentation exists under the same geometry and detector recipe: compare a small, prespecified intensity-only treatment and a blur/noise-only treatment with geometry-only, using the same predetermined images/groups and preserving each family's original marginal application rate (identity in place of the removed family), with no threshold tuning. Renormalizing the surviving family to twice its exposure would change two factors at once. This would test a specific mechanism suggested by the observed 50% blur/noise exposure; it has not been executed or authorized by this audit. Only a treatment with a defensible benefit would justify redesigning the adaptive selection criterion around training utility. Sparse direct class-set evidence must also remain explicit in any revision. Adding seeds or epochs to rescue the unchanged recipe is not the immediate recommendation.

The current report can support a negative empirical finding and documented working prototype. It cannot support a superiority claim for the proposed augmentation controller. No new GPU job, manuscript update, repository commit or push was performed during this audit.

## Reproduction and evidence files

From the project root, run the following local analysis scripts in order: `audit_returned.py`, `audit_execution.py`, `audit_controller.py`, `replay_exposure.py`, `recompute_metrics.py`, `report_results.py`, all under `reports/rdd_v2_pair8_v1/`. The scripts preserve the downloaded source archive and released training package.

Central outputs: `archive_audit.json`, `execution_audit.json`, `controller_audit.json`, `exposure_replay.json`, `metric_recomputation.json`, `analysis_summary.json`, `epoch_comparison.csv`, `endpoint_comparison.csv`, `per_class_comparison.csv` and `action_counts.csv`. The returned records are under `returned_records/`. Analysis file hashes and reused helper hashes are in `analysis_checksums.json`.
'''
    # Scientific figure: monochrome, explicit point-scale differences, no seed CIs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[e['epoch'] for e in epochs]
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axes[0].plot(x,[e['geometry_AP50_95'] for e in epochs],color='black',marker='o',label='Geometry only')
    axes[0].plot(x,[e['class_AP50_95'] for e in epochs],color='black',marker='s',linestyle='--',label='Class adaptive')
    axes[0].set(xlabel='Training epoch',ylabel='Validation AP50:95 (%)',ylim=(0,25),xticks=x)
    axes[0].legend(frameon=False)
    axes[1].axhline(0,color='black',linewidth=0.8)
    axes[1].plot(x,[e['class_minus_geometry_AP50_95_points'] for e in epochs],color='black',marker='o')
    axes[1].set(xlabel='Training epoch',ylabel='Class adaptive minus geometry (AP points)',ylim=(-2,0.15),xticks=x)
    fig.suptitle('Fixed epoch-8 comparison, seed 42; one paired development run')
    for ax in axes:
        ax.grid(axis='y',color='0.85',linewidth=.6);ax.spines[['right','top']].set_visible(False)
    fig.savefig(REPORT/'epoch8_comparison.png',dpi=180)
    fig.savefig(REPORT/'epoch8_comparison.pdf')
    plt.close(fig)
    (REPORT/'RESULTS.md').write_text(text,encoding='utf-8',newline='\n')
    names=['audit_returned.py','audit_execution.py','audit_controller.py','replay_exposure.py','recompute_metrics.py','report_results.py',
        'archive_audit.json','execution_audit.json','controller_audit.json','exposure_replay.json','metric_recomputation.json',
        'analysis_summary.json','epoch_comparison.csv','endpoint_comparison.csv','per_class_comparison.csv','action_counts.csv',
        'epoch8_comparison.png','epoch8_comparison.pdf','RESULTS.md']
    checks={str((REPORT/n).relative_to(ROOT)).replace('\\','/'):sha(REPORT/n) for n in names}
    for n in ['reports/rdd_v2_controller_check_v1/audit_returned.py','reports/rdd_v2_geometry_check_v1/audit_returned.py']:
        checks[n]=sha(ROOT/n)
    save('analysis_checksums.json',checks)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
