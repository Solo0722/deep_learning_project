import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import integration as subject
from rdd_data import Example


def example(index, positive=True):
    labels = np.arange(4, dtype=np.int64) if positive else np.empty(0, dtype=np.int64)
    boxes = np.array([[1.5, 2.5, 12.5, 18.5]] * len(labels), dtype=np.float64).reshape(-1, 4)
    return Example({"image": f"synthetic/image_{index}.jpg", "image_sha256": str(index) * 64}, index,
                   np.full((32, 32, 3), index + 20, dtype=np.uint8), boxes, labels,
                   {"size": 32, "original_width": 32, "original_height": 32,
                    "resized_width": 32, "resized_height": 32, "pad_left": 0, "pad_top": 0})


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_paired_inputs_all_labels_empty_image_and_no_mutation(self):
        examples = [example(0), example(1, False)]
        before = copy.deepcopy(examples)
        images, targets, yolo, record = subject.paired_batch(examples)
        self.assertTrue(torch.equal(torch.stack(images), yolo["img"]))
        self.assertEqual(targets[0]["labels"].tolist(), [1, 2, 3, 4])
        self.assertEqual(yolo["cls"].flatten().tolist(), [0, 1, 2, 3])
        self.assertEqual(targets[1]["boxes"].shape, (0, 4))
        self.assertEqual(yolo["batch_idx"].tolist(), [0, 0, 0, 0])
        self.assertEqual([item["target_count"] for item in record["images"]], [4, 0])
        for original, after in zip(before, examples):
            np.testing.assert_array_equal(original.rgb, after.rgb)
            np.testing.assert_array_equal(original.boxes, after.boxes)

    def test_entirely_empty_batch_retained(self):
        images, targets, yolo, record = subject.paired_batch([example(0, False), example(1, False)])
        self.assertEqual(len(images), 2)
        self.assertEqual(yolo["bboxes"].shape, (0, 4))
        self.assertEqual(len(targets), 2)
        self.assertEqual([item["target_count"] for item in record["images"]], [0, 0])

    def test_changed_pixels_change_input_signature(self):
        images = [example(0)]
        old = subject.paired_batch(images)[3]["signature"]
        images[0].rgb[0, 0, 0] += 1
        self.assertNotEqual(old, subject.paired_batch(images)[3]["signature"])

    def test_bad_adapter_pixels_labels_and_boxes_are_rejected(self):
        real = subject.yolo_collate
        for change in ("pixels", "class", "fractional_class", "box"):
            with self.subTest(change=change):
                def broken(examples):
                    batch, metadata = real(examples)
                    if change == "pixels":
                        batch["img"][0, 0, 0, 0] += 0.1
                    elif change == "class":
                        batch["cls"][0, 0] = 3
                    elif change == "fractional_class":
                        batch["cls"][0, 0] = 0.5
                    else:
                        batch["bboxes"][0, 0] += 0.1
                    return batch, metadata
                with patch.object(subject, "yolo_collate", broken):
                    with self.assertRaises(ValueError):
                        subject.paired_batch([example(0)])

    def test_real_optimizer_update_is_verified(self):
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(torch.ones(1, 2)) - 3).square().sum()
        result = subject.checked_update(model, optimizer, model.weight, loss, {"loss": loss})
        self.assertGreater(result["classifier_max_absolute_update"], 0)

    def test_nonfinite_loss_rejected_before_update(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.weight.detach().clone()
        loss = model.weight.sum() * float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            subject.checked_update(model, optimizer, model.weight, loss, {"loss": loss})
        self.assertTrue(torch.equal(before, model.weight))

    def test_nonfinite_gradient_rejected_before_update(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.weight.detach().clone()
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        loss = model.weight.square().sum()
        with self.assertRaisesRegex(ValueError, "Nonfinite gradient"):
            subject.checked_update(model, optimizer, model.weight, loss, {"loss": loss})
        self.assertTrue(torch.equal(before, model.weight))

    def test_absent_classifier_gradient_and_no_update_rejected(self):
        for mode in ("no_gradient", "zero_learning_rate"):
            with self.subTest(mode=mode):
                model = torch.nn.Linear(2, 1)
                optimizer = torch.optim.SGD(model.parameters(), lr=0 if mode == "zero_learning_rate" else 0.01)
                loss = model.bias.square().sum() if mode == "no_gradient" else model.weight.square().sum()
                with self.assertRaisesRegex(ValueError, "Classifier"):
                    subject.checked_update(model, optimizer, model.weight, loss, {"loss": loss})

    def test_valid_predictions_and_empty_output(self):
        result = subject.prediction_summary(torch.tensor([[1., 2., 3., 4.]]), torch.tensor([0.5]), torch.tensor([1]), 1, 32)
        self.assertEqual(result["detections"], 1)
        empty = subject.prediction_summary(torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64), 1, 32)
        self.assertEqual(empty["detections"], 0)

    def test_invalid_prediction_geometry_scores_and_labels(self):
        cases = [([[1., 2., 1., 4.]], [0.5], [1]), ([[1., 2., 3., 4.]], [float("nan")], [1]),
                 ([[1., 2., 3., 4.]], [1.5], [1]), ([[1., 2., 3., 4.]], [0.5], [0]),
                 ([[1., 2., 3., 4.]], [0.5], [1.5]), ([[-1., 2., 3., 4.]], [0.5], [1])]
        for boxes, scores, labels in cases:
            with self.subTest(boxes=boxes, scores=scores, labels=labels), self.assertRaises(ValueError):
                subject.prediction_summary(torch.tensor(boxes), torch.tensor(scores), torch.tensor(labels), 1, 32)

    def test_output_cannot_overlap_inputs_or_replace_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            data = root / "data"
            data.mkdir()
            for out in (data, data / "nested", root):
                with self.assertRaises(ValueError):
                    subject.check_output(out, (data,))
            old = root / "old"
            old.mkdir()
            with self.assertRaises(FileExistsError):
                subject.check_output(old, (data,))
            out = root / "new"
            out.with_name("new.tar.gz").write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                subject.check_output(out, (data,))

    def make_cli_fixture(self, root):
        package, splits, raw = root / "package", root / "splits", root / "raw"
        for directory in (package, splits, raw):
            directory.mkdir()
        samples = [example(0), example(1, False), example(2, False), example(3, False)]
        rows = [{"targets": [0] * len(sample.boxes)} for sample in samples]
        (splits / "config.json").write_bytes(b"{}")
        reference = {"expected_environment": sys.prefix, "config_sha256": subject.sha256(b"{}"),
                     "manifest_sha256": "0" * 64, "expected_training_images": 4,
                     "expected_training_targets": 4, "image_size": 32, "batch_size": 2}
        selection = {"selected": [{"image": sample.record["image"]} for sample in samples]}
        protocol = {"purpose": "synthetic CLI fixture only", "versions": {}, "loader_source_sha256": {},
                    "models": ["fasterrcnn_resnet50_fpn", "yolov8n"], "seed": 42, "cpu_threads": 1,
                    "image_size": 32, "batch_size": 2, "expected_images": 4, "expected_targets": 4,
                    "learning_rate": 0.01}
        for name, value in (("protocol.json", protocol), ("reference.json", reference), ("selection.json", selection)):
            subject.write_json(package / name, value)
        for name in ("integration.py", "rdd_data.py", "select_training_cases.py"):
            (package / name).write_text("# synthetic fixture\n", encoding="utf-8")
        checksum_lines = [f"{subject.sha256(path.read_bytes())}  {path.name}\n" for path in sorted(package.iterdir())]
        (package / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
        return package, splits, raw, samples, rows, selection

    def call_cli(self, root, *, mismatch=False):
        package, splits, raw, samples, rows, selection = self.make_cli_fixture(root)
        out = root / "results"

        def builder(name, size, device):
            model = torch.nn.Linear(2, 1)
            return model, model.weight, {"fixture": True}

        def stepper(model, head, name, examples, optimizer, device, size):
            signature = subject.paired_batch(examples)[3]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(torch.ones(1, 2)) - 3).square().sum()
            update = subject.checked_update(model, optimizer, head, loss, {"loss": loss})
            if mismatch and name == "yolov8n":
                signature["signature"] = "different"
            return {"inputs": signature, "update": update, "inference": [],
                    "target_counts": [len(example.boxes) for example in examples]}

        actual_run = subject.run_models
        def fixture_run(batches, protocol, destination, report):
            return actual_run(batches, protocol, destination, report, "cpu", builder, stepper)

        argv = ["integration.py", "--root", str(raw), "--splits", str(splits), "--out", str(out)]
        with patch.object(subject, "__file__", str(package / "integration.py")), \
             patch.object(subject.sys, "argv", argv), \
             patch.dict(subject.os.environ, {"PBS_JOBID": "synthetic-test"}), \
             patch.object(subject.socket, "gethostname", return_value="test-compute"), \
             patch.object(subject, "read_manifest", return_value=rows), \
             patch.object(subject, "make_selection", return_value=selection), \
             patch.object(subject, "TrainingDataset", return_value=samples), \
             patch.object(subject, "run_models", side_effect=fixture_run), \
             patch.object(subject.torch.cuda, "is_available", return_value=True), \
             patch.object(subject.torch.cuda, "device_count", return_value=1), \
             patch.object(subject.torch.cuda, "get_device_name", return_value="synthetic GPU"), \
             patch.object(subject.torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=0)):
            if mismatch:
                with self.assertRaisesRegex(ValueError, "different input batches"):
                    subject.main()
            else:
                subject.main()
                with self.assertRaises(FileExistsError):
                    subject.main()
        return out

    def test_cli_success_records_both_models_archives_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.call_cli(Path(folder).resolve())
            run = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["status"], "complete")
            self.assertTrue(run["paired_input_signatures_match"])
            self.assertEqual(run["all_empty_batches"], [2])
            for result in run["models"].values():
                self.assertEqual(result["steps_completed"], 2)
                self.assertEqual(result["images_processed"], 4)
            self.assertTrue(out.with_name("results.tar.gz").is_file())

    def test_cli_mismatch_records_failure_and_no_success_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.call_cli(Path(folder).resolve(), mismatch=True)
            run = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["models"]["fasterrcnn_resnet50_fpn"]["status"], "complete")
            self.assertEqual(run["models"]["yolov8n"]["status"], "failed")
            self.assertFalse(out.with_name("results.tar.gz").exists())

    def test_changed_package_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            package, *_ = self.make_cli_fixture(Path(folder).resolve())
            subject.verified_package(package)
            (package / "rdd_data.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Changed package file"):
                subject.verified_package(package)


if __name__ == "__main__":
    unittest.main()
