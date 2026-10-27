"""Validate the stage-5 implementation and its fixed-budget comparison."""
import ast
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PACKAGE=ROOT/'fasterrcnn_class_adaptive_v1'
SHARED=ROOT/'fasterrcnn_online_shared_v1'
sys.path.insert(0,str(PACKAGE))
import baseline as b
import controller as c
import local_augmentation as local
import probe_runner as runner


def save(name,value): (OUT/name).write_bytes(b.json_bytes(value))


def main():
    started=time.monotonic();log=io.StringIO()
    with contextlib.redirect_stdout(log):
        suite=unittest.defaultTestLoader.discover(str(PACKAGE/'tests'),pattern='test_*.py')
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (OUT/'unit_tests.txt').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not result.wasSuccessful():print(log.getvalue());raise SystemExit('Tests failed')
    tests={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'seconds':time.monotonic()-started}
    save('unit_tests.json',tests);print(tests,flush=True)
    cfg=json.loads((PACKAGE/'protocol.json').read_text())
    refs=[json.loads((PACKAGE/f'{s}_protocol.json').read_text()) for s in ('identity','shared','local','online_shared')]
    c.verify_pairing(cfg,*refs)
    for name,key in [('online_shared_protocol.json','online_shared_protocol_sha256'),('shared_controller.py','online_shared_controller_sha256')]:
        assert b.checkpoints.file_hash(PACKAGE/name)==cfg['comparator'][key]
    unchanged=['rdd_data.py','models.py','metrics.py','checkpoints.py','integration.py','select_training_cases.py',
               'smoke_selection.json','local_augmentation.py','shared_augmentation.py','smoke_augmentation.py','probe_detector.py']
    for name in unchanged:assert (PACKAGE/name).read_bytes()==(SHARED/name).read_bytes(),name
    parse=lambda path:{n.name:ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old,new=parse(SHARED/'baseline.py'),parse(PACKAGE/'baseline.py')
    functions=['Dataset','epoch_batches','learning_rate','update','validate','commit','restore','write_history','verify_partitions','archive_records','loader','fit','initial_progress']
    for name in functions:assert old[name]==new[name],name
    assert parse(SHARED/'probe_runner.py')['validate_saved_updates']==parse(PACKAGE/'probe_runner.py')['validate_saved_updates']
    assert (PACKAGE/'shared_controller.py').read_bytes()==(SHARED/'controller.py').read_bytes()
    # Enforce the precisely scoped change to the retained numerical aggregation core.
    source=(SHARED/'controller.py').read_text()
    expected=source[source.index('def aggregate('):source.index('\n\ndef decision(')]
    expected=expected.replace('def aggregate(panel, settings, epoch):','def _aggregate_set(panel, settings, epoch, represented):')
    expected=expected.replace("if not panel or len({p['image'] for p in panel}) != len(panel):","if len({p['image'] for p in panel}) != len(panel):")
    expected=expected.replace("class_objects = {c: [] for c in settings['classes']}","class_objects = {c: [] for c in represented}")
    expected=expected.replace("if any(c not in class_objects for c in p['classes']) or set(p['views']) != keys:","if any(c not in settings['classes'] for c in p['classes']) or set(p['views']) != keys:")
    expected=expected.replace("for index in positions:\n            class_objects","if tuple(sorted({p['classes'][i] for i in positions})) != tuple(represented):\n            raise ValueError('Feedback belongs to a different exact class set')\n        for index in positions:\n            class_objects")
    actual=(PACKAGE/'class_set_feedback.py').read_text()
    assert actual[actual.index('def _aggregate_set('):].strip()==expected.strip()
    parity={'status':'passed','unchanged_dependency_files':unchanged,'unchanged_baseline_functions':functions,
            'observer_panel_grid_noise_training_draws_identical':True,'shared_controller_bytes_preserved':True,
            'only_measurement_core_changes':'Restrict target-group pooling to the represented exact class set; validate all full-image observations; permit empty sets as unsupported.',
            'online_shared_protocol_sha256':cfg['comparator']['online_shared_protocol_sha256'],
            'protocol_sha256':b.checkpoints.file_hash(PACKAGE/'protocol.json')}
    save('comparator_checks.json',parity)
    train=b.read_manifest(ROOT/'detection_splits_v1/train.jsonl',cfg['manifests']['train'],'train')
    assert len(train)==26863
    prior=json.loads((ROOT/'reports/fasterrcnn_online_shared_v1/panel_plan.json').read_text())
    prior_by_epoch={p['epoch']:p for p in prior['epochs']}
    sets=c.all_class_sets(cfg['controller']);planned=[];support_by_update={}
    for epoch in range(20):
        if not runner.due(cfg,epoch):continue
        panel=c.select_panel(list(enumerate(train)),640,42,epoch,256)
        assert panel==prior_by_epoch[epoch+1]['panel']
        cells={k:{'panel_images':0,'objects':Counter(),'clusters':{str(v):set() for v in labels}} for k,labels in sets.items()}
        for p in panel:
            cell=cells[c.panel_class_set(p)];cell['panel_images']+=1
            for i in p['target_positions']:
                label=str(p['classes'][i]);cell['objects'][label]+=1;cell['clusters'][label].add(p['source_cluster_id'])
        summaries={}
        for key,v in cells.items():
            summaries[key]={'panel_images':v['panel_images'],'annotated_objects':{str(i):v['objects'][str(i)] for i in sets[key]},
                            'source_clusters':{i:len(s) for i,s in v['clusters'].items()}}
            summaries[key]['could_meet_support_if_all_detected']=all(v['objects'][str(i)]>=8 and len(v['clusters'][str(i)])>=4 for i in sets[key])
        support_by_update[epoch]=summaries
        planned.append({'epoch':epoch+1,'panel_sha256':c.digest(panel),'panel':panel,'class_set_annotation_support':summaries,
                        'images':256,'distinct_source_clusters':len({p['source_cluster_id'] for p in panel})})
        print(f'Panel {epoch+1}: exact match to online shared; '+str(sum(v['could_meet_support_if_all_detected'] for v in summaries.values()))+' potentially supported sets',flush=True)
    exposures=json.loads((PACKAGE/'shared_exposure.json').read_text())
    counts=[];total=0;upper=0
    for epoch in range(20):
        opportunities=Counter()
        for row in train:
            choice=local.decision(row,cfg['augmentation_policy'],42,epoch,640)
            if choice['local']:
                key=c.class_set_key(c.selected_classes(row,640,choice['local']['group_index']))
                opportunities[key]+=1
        n=sum(opportunities.values());assert n==exposures['epochs'][epoch]['counts'].get('applied_images',0)
        supported=0
        if epoch:
            evidence=support_by_update[max(e for e in support_by_update if e<=epoch)]
            supported=sum(v for k,v in opportunities.items() if evidence[k]['could_meet_support_if_all_detected'])
        counts.append({'epoch':epoch+1,'opportunities':n,'class_sets':dict(opportunities),'maximum_supported_opportunities_from_annotations':supported})
        total+=n;upper+=supported
    assert total==158151
    plan={'scope':'Frozen training metadata only. Annotation counts give support upper bounds, not detector retention or future active-policy predictions.',
          'scheduled_updates':10,'probe_image_forwards':71680,'epochs':planned,'application_opportunities':counts,
          'total_application_opportunities':total,'same_panels_and_group_choices_as_online_shared':True,
          'all_opportunities_match_fixed_controls':True,'maximum_supported_opportunities_from_annotations':upper,
          'minimum_identity_opportunities_due_to_annotation_sparsity':total-upper,
          'probe_settings_sha256':c.digest(cfg['controller']),'test_data_read':False}
    save('panel_plan.json',plan)
    sources={p.relative_to(PACKAGE).as_posix():b.checkpoints.file_hash(p) for p in PACKAGE.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json')}
    save('tested_sources.json',sources)
    print(json.dumps({'parity':'passed','probe_views':71680,'opportunities':total,'minimum_sparse_identity':total-upper}),flush=True)


if __name__=='__main__':main()
