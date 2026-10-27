import copy
import json
from pathlib import Path
import random
import sys
import tarfile
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checkpoints
import learning
import metrics
from integration import checked_update, paired_batch
from rdd_data import Example


def example(index=0, positive=True):
    boxes = np.array([[1,1,7,7], [9,1,15,7], [1,10,7,16], [9,10,15,16]], dtype=float)
    if not positive:
        boxes = np.empty((0,4))
    labels = np.arange(len(boxes), dtype=np.int64)
    row = {"image": f"synthetic/{index}.jpg", "width":32, "height":32,
           "image_sha256": str(index)*64,
           "targets": [{"class_index": int(c), "box_xyxy": b.tolist()} for b,c in zip(boxes, labels)]}
    return Example(row, index, np.full((32,32,3), 32+index, dtype=np.uint8), boxes, labels,
                   {"original_width":32,"original_height":32,"resized_width":32,"resized_height":32,
                    "pad_left":0,"pad_top":0,"size":32})


def perfect(gt):
    return [{"image_id": a["image_id"], "category_id":a["category_id"],
             "bbox":a["bbox"].copy(), "score":0.9} for a in gt["annotations"]]


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = metrics.ground_truth([example(), example(1, False)])

    def test_perfect_predictions(self):
        result = metrics.evaluate(self.gt, perfect(self.gt))
        for key in ("AP50_95", "AP50", "AP75", "AR100"):
            self.assertAlmostEqual(result[key], 1)
        self.assertEqual(result["evaluation_images"], 2)
        self.assertEqual(result["no_target_images"], 1)
        self.assertEqual(result["no_target_predicted_boxes"], {"1":0})

    def test_empty_predictions_are_zero_not_error(self):
        result = metrics.evaluate(self.gt, [])
        self.assertEqual(result["AP50"], 0)
        self.assertEqual(result["AR100"], 0)

    def test_wrong_classes_and_disjoint_boxes_have_zero_ap(self):
        predictions = perfect(self.gt)
        for p in predictions:
            p["category_id"] = p["category_id"] % 4 + 1
        self.assertEqual(metrics.evaluate(self.gt, predictions)["AP50"], 0)
        predictions = perfect(self.gt)
        for p in predictions:
            p["bbox"] = [25,25,3,3]
        self.assertEqual(metrics.evaluate(self.gt, predictions)["AP50"], 0)

    def test_missed_class_counts_against_macro_ap(self):
        result = metrics.evaluate(self.gt, perfect(self.gt)[:-1])
        self.assertAlmostEqual(result["AP50"], 0.75)
        self.assertEqual(result["per_class"]["D40"]["AP50"], 0)

    def test_iou_thresholds_separate_localization_quality(self):
        predictions = perfect(self.gt)
        predictions[0]['bbox'][2] = 9  # IoU=36/54: matches at 0.50 through 0.65 only.
        result = metrics.evaluate(self.gt, predictions)
        self.assertAlmostEqual(result['AP50'], 1)
        self.assertAlmostEqual(result['AP75'], 0.75)
        self.assertAlmostEqual(result['AP50_95'], 0.85)

    def test_false_positive_on_empty_image_reduces_ap(self):
        predictions = perfect(self.gt)
        predictions.append({"image_id":1,"category_id":1,"bbox":[1,1,6,6],"score":0.99})
        result = metrics.evaluate(self.gt, predictions)
        self.assertAlmostEqual(result["per_class"]["D00"]["AP50"], 0.5)
        self.assertAlmostEqual(result["AP50"], 0.875)
        self.assertEqual(result["no_target_predicted_boxes"], {"1":1})

    def test_all_negative_ground_truth_is_undefined(self):
        result = metrics.evaluate(metrics.ground_truth([example(1, False)]), [])
        self.assertIsNone(result["AP50"])
        self.assertIsNone(result["per_class"]["D00"]["AP50"])

    def test_unknown_image_class_or_invalid_prediction_rejected(self):
        for key, value in (("image_id",99),("category_id",0),("score",float("nan")),("bbox",[1,1,0,2])):
            with self.subTest(key=key):
                predictions = perfect(self.gt)
                predictions[0][key] = value
                with self.assertRaises(ValueError):
                    metrics.evaluate(self.gt, predictions)

    def test_original_coordinate_conversion_fractional_and_padding(self):
        e = example()
        e.record.update(width=101, height=53)
        e.geometry.update(original_width=101, original_height=53, resized_width=64,
                          resized_height=34, pad_top=15, size=64)
        prediction = {"boxes":torch.tensor([[6.4,21.8,32,35.4],[1,0,3,10]], dtype=torch.float32),
                      "scores":torch.tensor([0.8,0.9]), "labels":torch.tensor([3,0])}
        before = copy.deepcopy(prediction)
        records, clipping = metrics.convert_prediction(e, prediction)
        self.assertEqual(len(records),1)
        np.testing.assert_allclose(records[0]["bbox"], [10.1,10.6,40.4,21.2], atol=1e-5)
        self.assertEqual(records[0]["category_id"],4)
        self.assertEqual(clipping["collapsed_after_clipping"],1)
        for key in prediction:
            self.assertTrue(torch.equal(before[key],prediction[key]))

    def test_nonfinite_or_invalid_class_before_conversion_rejected(self):
        for value in (4, float("nan")):
            prediction = {"boxes":torch.tensor([[1.,1.,2.,2.]]),"scores":torch.tensor([0.5]),
                          "labels":torch.tensor([value])}
            with self.assertRaises(ValueError):
                metrics.convert_prediction(example(), prediction)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def make_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(0.25),torch.nn.Linear(4,1))
        return model, torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def step(self, model, optimizer):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.rand(2,3) + float(np.random.rand()) + random.random()
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_resumed_training_matches_uninterrupted_cpu_trajectory(self):
        model, optimizer = self.make_model()
        self.step(model,optimizer)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pt'
            metadata = checkpoints.save(path,model,optimizer,'test',1,{'protocol':'fixed'})
            expected_losses = [self.step(model,optimizer) for _ in range(3)]
            expected_model = checkpoints.cpu_tree(model.state_dict())
            expected_optimizer = checkpoints.cpu_tree(optimizer.state_dict())
            restored, new_optimizer = self.make_model()
            result = checkpoints.restore(path,metadata,restored,new_optimizer,'test',{'protocol':'fixed'})
            actual_losses = [self.step(restored,new_optimizer) for _ in range(3)]
            self.assertEqual(expected_losses, actual_losses)
            self.assertEqual(result['next_step'],2)
            checkpoints.assert_same(restored.state_dict(), expected_model)
            checkpoints.assert_same(new_optimizer.state_dict(), expected_optimizer)

    def test_changed_checkpoint_context_or_step_rejected(self):
        model, optimizer = self.make_model()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pt'
            metadata = checkpoints.save(path,model,optimizer,'test',1,{'protocol':'fixed'})
            for context, changed in (({'protocol':'other'},metadata),({'protocol':'fixed'},{**metadata,'completed_steps':2})):
                with self.assertRaises(ValueError):
                    checkpoints.restore(path,changed,model,optimizer,'test',context)
            with path.open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                checkpoints.restore(path,metadata,model,optimizer,'test',{'protocol':'fixed'})

    def test_existing_checkpoint_is_preserved(self):
        model, optimizer = self.make_model()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pt'
            checkpoints.save(path,model,optimizer,'test',1,{})
            old = path.read_bytes()
            with self.assertRaises(FileExistsError):
                checkpoints.save(path,model,optimizer,'test',1,{})
            self.assertEqual(old,path.read_bytes())


class OrchestrationTests(unittest.TestCase):
    def test_weight_hash_and_saved_checksum_manifest_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root/'test.pt'
            path.write_bytes(b'abc')
            digest = checkpoints.file_hash(path)
            protocol = {'pretrained_weights':{'test':{'filename':'test.pt','bytes':3,'sha256':digest}}}
            (root/'SHA256SUMS').write_text(f'{digest}  test.pt\n')
            self.assertEqual(learning.verify_weights(root,protocol)['test']['sha256'],digest)
            path.write_bytes(b'xyz')
            with self.assertRaisesRegex(ValueError,'checksum'):
                learning.verify_weights(root,protocol)

    def test_checkpoint_reload_prediction_change_detected(self):
        before = perfect(metrics.ground_truth([example()]))
        learning.compare_predictions(before, copy.deepcopy(before))
        after = copy.deepcopy(before)
        after[0]['bbox'][0] += 1
        with self.assertRaises(ValueError):
            learning.compare_predictions(before,after)

    def test_completed_without_learning_is_needs_review(self):
        rule = {'minimum_final_training_AP50':0.5,'minimum_training_AP50_gain':0.1}
        self.assertEqual(learning.learning_review({'AP50':0.1},{'AP50':0.15},rule)['decision'],'needs_review')
        self.assertEqual(learning.learning_review({'AP50':0.1},{'AP50':0.7},rule)['decision'],'learning_signal_observed')

    def test_archive_excludes_full_checkpoints_and_preserves_previous_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root/'result'
            (out/'checkpoints').mkdir(parents=True)
            (out/'checkpoints/large.pt').write_bytes(b'checkpoint')
            (out/'run.json').write_text('{}')
            archive = root/'result.tar.gz'
            learning.archive_records(out,archive)
            with tarfile.open(archive) as t:
                self.assertEqual(set(t.getnames()),{'result/run.json','result/checksums.json'})
            old = archive.read_bytes()
            old_checksums = (out/'checksums.json').read_bytes()
            with self.assertRaises(FileExistsError):
                learning.archive_records(out,archive)
            self.assertEqual(archive.read_bytes(),old)
            self.assertEqual((out/'checksums.json').read_bytes(),old_checksums)

    def test_two_model_loop_reloads_and_keeps_data_paired(self):
        protocol = json.loads((Path(__file__).resolve().parents[1]/'protocol.json').read_text())
        protocol.update(image_size=32,total_steps=4,evaluate_steps=[0,2,4],checkpoint_steps=[2,4])
        builds = []
        def builder(name,size,device,evaluation,weights):
            builds.append((name,weights))
            model = torch.nn.Linear(1,1,bias=False)
            return model,model.weight,{}
        def trainer(model,head,name,examples,optimizer,device):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(torch.ones(1,1))-1).square().sum()
            return {'inputs':paired_batch(examples)[3],
                    'update':checked_update(model,optimizer,head,loss,{'loss':loss})}
        def predictor(model,name,images,evaluation,device):
            return [{'boxes':torch.empty(0,4),'scores':torch.empty(0),'labels':torch.empty(0)} for _ in images]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            report = {'models':{}}
            weights = {name:{'path':'unused'} for name in protocol['models']}
            learning.run_models([example(),example(1,False)],protocol,{'protocol':'test'},out,
                                report,weights,'cpu',builder,trainer,predictor)
            self.assertEqual(len(builds),6)
            self.assertTrue(report['paired_inputs_verified'])
            for state in report['models'].values():
                self.assertEqual(state['steps_completed'],4)
                self.assertEqual(state['status'],'complete')
                self.assertEqual(state['learning_review']['decision'],'needs_review')
                self.assertEqual(len(state['checkpoint_reloads']),2)
            records = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['input_signature'] for r in records[:4]],
                             [r['input_signature'] for r in records[4:]])


if __name__ == '__main__':
    unittest.main()
