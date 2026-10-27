import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diagnose


class DiagnosticTests(unittest.TestCase):
    def test_batch_statistics_match_functional_without_changing_buffers_or_flags(self):
        model = nn.Sequential(nn.BatchNorm2d(1), nn.Dropout2d(.5))
        model.train()
        with torch.no_grad():
            model[0].running_mean.fill_(10)
            model[0].running_var.fill_(4)
            model[0].weight.fill_(2)
            model[0].bias.fill_(3)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        x = torch.arange(8,dtype=torch.float32).reshape(2,1,2,2)
        expected = F.batch_norm(x,None,None,model[0].weight,model[0].bias,training=True,eps=model[0].eps)
        with diagnose.normalization_mode(model,"batch_stats") as count:
            self.assertEqual(count,1)
            self.assertTrue(model[0].training)
            self.assertFalse(model[1].training)
            self.assertFalse(model[0].track_running_stats)
            actual = model(x)
        torch.testing.assert_close(actual,expected)
        self.assertTrue(model.training and model[0].training and model[1].training)
        self.assertTrue(model[0].track_running_stats)
        for key,value in before.items():
            self.assertTrue(torch.equal(value,model.state_dict()[key]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_running_statistics_reproduce_after_batch_stat_intervention(self):
        model = nn.Sequential(nn.BatchNorm2d(1))
        x = torch.arange(8,dtype=torch.float32).reshape(2,1,2,2)
        with diagnose.normalization_mode(model,"running_stats"):
            before = model(x)
        with diagnose.normalization_mode(model,"batch_stats"):
            changed = model(x)
        with diagnose.normalization_mode(model,"running_stats_after"):
            after = model(x)
        self.assertFalse(torch.allclose(before,changed))
        self.assertTrue(torch.equal(before,after))

    def test_exception_restores_training_and_tracking_flags(self):
        model = nn.Sequential(nn.BatchNorm2d(1),nn.Dropout2d())
        model.train();model[1].eval()
        with self.assertRaisesRegex(RuntimeError,"test failure"):
            with diagnose.normalization_mode(model,"batch_stats"):
                raise RuntimeError("test failure")
        self.assertTrue(model.training and model[0].training and model[0].track_running_stats)
        self.assertFalse(model[1].training)

    def test_running_buffer_mutation_is_detected(self):
        model = nn.Sequential(nn.BatchNorm2d(1))
        with self.assertRaisesRegex(ValueError,"Changed diagnostic model state"):
            with diagnose.normalization_mode(model,"batch_stats"):
                model[0].running_mean.add_(1)

    def test_raw_geometry_distinguishes_localization_and_score_filter(self):
        example = SimpleNamespace(boxes=np.array([[10.,10.,20.,20.]]),labels=np.array([0]),image_id=7)
        # Two raw candidates: exact target with low class-0 score, and a disjoint high-score box.
        decoded = torch.tensor([[[15.,50.],[15.,50.],[10.,10.],[10.,10.],
                                 [.0001,.9],[.8,.0001],[.0001,.0001],[.0001,.0001]]])
        target = diagnose.raw_geometry(decoded,[example])[0]["targets"][0]
        self.assertEqual(target["best_raw_iou"],1.)
        self.assertEqual(target["best_any_class_eligible_iou"],1.)
        self.assertEqual(target["best_same_class_eligible_iou"],0.)

    def test_changed_record_and_checkpoint_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory/"checkpoint.pt"
            path.write_bytes(b"original")
            expected = {"bytes":8,"sha256":hashlib.sha256(b"original").hexdigest()}
            diagnose.verify_files(directory,{"checkpoint.pt":expected["sha256"]})
            path.write_bytes(b"changed!")
            with self.assertRaisesRegex(ValueError,"Changed input"):
                diagnose.verify_files(directory,{"checkpoint.pt":expected["sha256"]})
            with patch("diagnose.torch.load") as load:
                with self.assertRaisesRegex(ValueError,"Checkpoint bytes/hash differ"):
                    diagnose.load_existing(path,expected,{},nn.Linear(1,1))
                load.assert_not_called()

    def test_assignment_capture_keeps_empty_images_and_soft_scores(self):
        scores=torch.tensor([[[.2,0.],[0.,.3]],[[0.,0.],[0.,0.]]])
        mask=torch.tensor([[True,True],[False,False]])
        indices=torch.tensor([[0,1],[0,0]])
        result=diagnose.capture_assignments((None,None,scores,mask,indices))
        self.assertEqual(result["foreground_count_per_image"],[2,0])
        self.assertEqual(result["assigned_gt_indices_per_image"],[[0,1],[]])
        self.assertAlmostEqual(result["target_score_sum_per_image"][0],.5)
        self.assertEqual(result["target_score_sum_per_image"][1],0.)

    def test_reference_prediction_mismatch_stops_before_intervention(self):
        with patch("diagnose.measure",return_value=({"metrics":{"AP50":0.,"AP50_95":0.}},[])) as measure:
            with self.assertRaisesRegex(ValueError,"prediction count"):
                diagnose.diagnose_checkpoint(None,[],{},"cpu",[{"image_id":1}],
                                             ["running_stats","batch_stats","running_stats_after"])
            self.assertEqual(measure.call_count,1)


if __name__ == "__main__":
    unittest.main()
