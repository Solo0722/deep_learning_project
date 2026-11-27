import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import baseline as b
import checkpoints
import gate
import pilot
import rdd_data
import strict_runtime as strict
import test_pilot as fixtures


class Pair8Tests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.PilotTests();self.fixture.setUp()
        self.addCleanup(self.fixture.fixture.temp.cleanup)
        self.cfg=copy.deepcopy(self.fixture.cfg)
        self.cfg.update(epochs=2,update_epochs=[2],arms=['geometry_only','class_adaptive'])
        self.cfg['controller'].update(minimum_detected_objects=16,minimum_detected_clusters=8)
        self.plans={'2':self.fixture.plan[:2]}
        self.rows=self.fixture.rows
        self.root=self.fixture.fixture.root

    def payload(self):
        model,opt=self.fixture.fresh()
        return {'format':1,'model':checkpoints.cpu_tree(model.state_dict()),
            'optimizer':checkpoints.cpu_tree(opt.state_dict()),'rng':checkpoints.rng_state()}

    def compare(self,name='gate'):
        return gate.compare_run(self.root,self.rows,self.fixture.fixture.val,self.payload(),self.plans,
            sorted(r['image'] for r in self.rows),self.cfg,self.root/name,
            self.fixture.fresh,'cpu',lambda:False)

    def test_actual_fit_gate_matches_probe_resume_and_validation(self):
        result=self.compare()
        self.assertTrue(result['production_loop_exact'] and result['resume_exact'])
        self.assertTrue(result['validation_predictions_exact'])
        self.assertEqual(result['optimizer_updates_each'],3)
        self.assertEqual(result['trace_records_per_arm'],{'geometry_only':6,'class_adaptive':6})
        row=json.loads((self.root/'gate/class_adaptive/history.json').read_text())[0]
        self.assertEqual(row['augmentation']['counts'].get('all/applied',0),0)

    def test_gate_detects_extra_rng_consumption_in_real_update_path(self):
        update=b.update;calls=[0]
        def changed(*args,**kwargs):
            result=update(*args,**kwargs);calls[0]+=1
            if calls[0]==4:torch.rand(1)
            return result
        with patch.object(b,'update',changed),self.assertRaisesRegex(ValueError,'exactness failed'):
            self.compare('leak')

    def test_gate_refuses_existing_output(self):
        (self.root/'existing').mkdir()
        with self.assertRaises(FileExistsError):self.compare('existing')

    def test_strict_flags_detect_disabled_and_warn_only_modes(self):
        try:
            torch.use_deterministic_algorithms(False)
            with self.assertRaisesRegex(ValueError,'Strict execution'):strict.check(self.cfg)
            torch.use_deterministic_algorithms(True,warn_only=True)
            with self.assertRaisesRegex(ValueError,'Strict execution'):strict.check(self.cfg)
        finally:strict.configure(self.cfg,fresh_cuda=False)

    def test_strict_configuration_rejects_late_cuda_or_wrong_workspace(self):
        with patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':16:8'}):
            with self.assertRaisesRegex(ValueError,'before starting'):strict.configure(self.cfg)
        with patch.object(torch.cuda,'is_initialized',return_value=True):
            with self.assertRaisesRegex(ValueError,'CUDA initialized'):strict.configure(self.cfg)

    def histories(self):
        cfg=copy.deepcopy(self.cfg);cfg['epochs']=8
        h={}
        for arm,score in [('geometry_only',0.20),('class_adaptive',0.19)]:
            h[arm]=[{'epoch':e,'augmentation':{'pairing_sha256':str(e)},
                'validation':{'AP50_95':score if e==8 else 0.9,
                    'per_class':{c:{'AP50_95':score} for c in rdd_data.CLASSES}}} for e in range(2,9)]
        return cfg,h

    def test_pairing_summary_uses_fixed_epoch8_instead_of_best_epoch(self):
        cfg,h=self.histories();result=pilot.paired_summary(h,cfg)
        self.assertEqual(result['epoch'],8)
        self.assertAlmostEqual(result['comparisons']['class_adaptive minus geometry_only']['AP50_95_points'],-1.)

    def test_pairing_summary_rejects_incomplete_or_unpaired_histories(self):
        cfg,h=self.histories();h['class_adaptive'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete history'):pilot.paired_summary(h,cfg)
        cfg,h=self.histories();h['class_adaptive'][3]['augmentation']['pairing_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'Unpaired'):pilot.paired_summary(h,cfg)

    def test_probe_time_limit_leaves_preprobe_checkpoint_authoritative(self):
        cfg=copy.deepcopy(self.cfg);cfg['resource_checks']['maximum_probe_seconds']=0
        out=self.fixture.fixture.output('probe_limit')
        model,opt=self.fixture.fresh();p=pilot.new_progress(cfg,len(self.rows))
        train=pilot.Dataset(self.root,self.rows,cfg,'class_adaptive')
        context={'cfg':cfg}
        with self.assertRaisesRegex(RuntimeError,'probe reached'):
            pilot.fit(model,opt,train,self.fixture.fixture.val,cfg,'cpu',out,context,p,self.plans)
        restored=b.restore(out,model,opt,context)
        self.assertEqual(restored['step'],3);self.assertFalse(restored['controller_updates'])
        record=json.loads(next((out/'resources').glob('probe_*.json')).read_text())
        self.assertEqual(record['status'],'resource_limit')

    def test_epoch8_resume_keeps_all_four_controller_updates_and_budget(self):
        cfg=copy.deepcopy(self.fixture.cfg)
        cfg.update(epochs=8,update_epochs=[2,4,6,8])
        plans={str(e):self.fixture.plan for e in cfg['update_epochs']}
        context={'cfg':cfg,'fixture':'eight_epochs'}
        def fit(out,model,opt,p,stop=lambda:False):
            train=pilot.Dataset(self.root,self.rows,cfg,'class_adaptive')
            return pilot.fit(model,opt,train,self.fixture.fixture.val,cfg,'cpu',out,context,p,plans,stop)
        model,opt=self.fixture.fresh();p=pilot.new_progress(cfg,5)
        self.assertEqual(fit(self.fixture.fixture.output('full8'),model,opt,p),'complete')
        expected=(checkpoints.cpu_tree(model.state_dict()),checkpoints.cpu_tree(opt.state_dict()),checkpoints.rng_state())
        model,opt=self.fixture.fresh();p=pilot.new_progress(cfg,5);out=self.fixture.fixture.output('resume8')
        self.assertEqual(fit(out,model,opt,p,lambda:p['step']>=16),'budget_stopped')
        model,opt=self.fixture.fresh();p=b.restore(out,model,opt,context)
        self.assertEqual(fit(out,model,opt,p),'complete')
        checkpoints.assert_same(model.state_dict(),expected[0]);checkpoints.assert_same(opt.state_dict(),expected[1])
        checkpoints.assert_same(checkpoints.rng_state(),expected[2])
        self.assertEqual(p['step'],24);self.assertEqual(p['images_seen'],40)
        self.assertEqual([x['epoch'] for x in p['controller_updates']],[2,4,6,8])
        self.assertEqual([x['epoch'] for x in p['history']],list(range(2,9)))


if __name__=='__main__':unittest.main()
