import copy
import json
from pathlib import Path
import random
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as b
import checkpoints
from rdd_data import CLASSES, sha256


class ToyDetector(torch.nn.Module):
    """Stochastic training path tests checkpoint RNG; known boxes test validation."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, images, targets=None):
        if self.training:
            jitter = torch.rand(()) + random.random() + float(np.random.rand())
            return {"loss_classifier": (self.weight-jitter).square() + 0.01}
        self.last_eval_grad_enabled = torch.is_grad_enabled()
        result = []
        for _ in images:
            result.append({"boxes": torch.tensor([[4., 4., 20., 20.]] * 4),
                           "labels": torch.tensor([1, 2, 3, 4]), "scores": torch.full((4,), 0.9)})
        return result


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix="rdd_baseline_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = json.loads((b.PACKAGE / "local_protocol.json").read_text())
        self.cfg.update(epochs=2, batch_size=2, validation_batch_size=2, image_size=32, workers=0,
                        checkpoint_every_steps=2, log_every_steps=1000, warmup_steps=2, lr_decay_epochs=[1])
        self.train_rows = [self.row(f"t{i}", "train", i if i < 4 else None) for i in range(5)]
        self.val_rows = [self.row(f"v{i}", "val", i if i < 4 else None) for i in range(5)]
        self.train = b.Dataset(self.root, self.train_rows, "train", 32)
        self.val = b.Dataset(self.root, self.val_rows, "val", 32)
        self.context = {"protocol": self.cfg, "fixture": True}

    def row(self, name, split, label):
        img_name, xml_name = f"Fixture/train/images/{name}.png", f"Fixture/train/annotations/{name}.xml"
        image_path, xml_path = self.root / img_name, self.root / xml_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        # Distinct pixels for each row make decoded-duplicate checks meaningful.
        generator = np.random.default_rng(int(sha256(name.encode())[:8], 16))
        image = Image.fromarray(generator.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        image.save(image_path)
        xml_path.write_bytes(b"<annotation/>")
        targets = [] if label is None else [{"box_xyxy": [4., 4., 20., 20.], "class_index": label,
                    "label": CLASSES[label], "object_index": 0, "difficult": "1", "truncated": "0"}]
        return {"image": img_name, "xml": xml_name, "source": "Fixture", "source_cluster_id": name,
                "width": 32, "height": 32, "split": split, "exif_orientation": 1,
                "image_sha256": sha256(image_path.read_bytes()), "xml_sha256": sha256(xml_path.read_bytes()),
                "rgb_sha256": sha256(b"RGB:32:32:" + image.tobytes()), "targets": targets,
                "other_annotations": [], "annotation_category": "target_bearing" if targets else "empty_xml"}

    def output(self, name):
        out = self.root / name
        out.mkdir()
        (out / "checkpoints").mkdir()
        return out

    def fresh(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = ToyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        return model, optimizer

    def test_val_loader_uses_val_without_relabelling_and_rejects_test(self):
        example = self.val[0]
        self.assertEqual(example.record["split"], "val")
        self.assertEqual(example.record, self.val_rows[0])
        with self.assertRaises(ValueError):
            b.Dataset(self.root, self.val_rows, "train", 32)
        with self.assertRaises(ValueError):
            b.Dataset(self.root, self.val_rows, "test", 32)

    def test_cross_partition_image_cluster_and_rgb_overlap_rejected(self):
        b.verify_partitions(self.train_rows, self.val_rows)
        for key in ("image", "source_cluster_id", "rgb_sha256"):
            bad = copy.deepcopy(self.val_rows)
            bad[0][key] = self.train_rows[0][key]
            with self.assertRaises(ValueError):
                b.verify_partitions(self.train_rows, bad)

    def test_image_modification_rejected(self):
        (self.root / self.val_rows[0]["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image checksum"):
            self.val[0]

    def test_sampler_preserves_last_batch_and_model_rng(self):
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        batches = b.epoch_batches(5, 2, 42, 0)
        self.assertEqual(sorted(x for batch in batches for x in batch), list(range(5)))
        self.assertEqual([len(x) for x in batches], [2, 2, 1])
        self.assertEqual(b.epoch_batches(5, 2, 42, 0, 1), batches[1:])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertNotEqual(batches, b.epoch_batches(5, 2, 42, 1))

    def test_validation_keeps_all_images_no_grad_no_parameter_update(self):
        model, _ = self.fresh()
        before = checkpoints.cpu_tree(model.state_dict())
        result = b.validate(model, self.val, self.cfg, "cpu")
        checkpoints.assert_same(model.state_dict(), before)
        self.assertFalse(model.last_eval_grad_enabled)
        self.assertEqual(result["metrics"]["evaluation_images"], 5)
        self.assertEqual(result["metrics"]["target_boxes"], 4)
        self.assertEqual(result["metrics"]["no_target_images"], 1)
        self.assertEqual(result["metrics"]["no_target_images_with_predictions"], 1)
        self.assertEqual(len(result["clipping"]), 5)
        self.assertEqual(len(result["predictions"]), 20)

    def test_all_negative_training_batch_kept(self):
        model, optimizer = self.fresh()
        images, targets, _ = b.collate([self.train[4], self.train[4]])
        self.assertEqual(targets[0]["boxes"].shape, (0, 4))
        before = model.weight.item()
        values, norm = b.update(model, optimizer, images, targets, "cpu")
        self.assertNotEqual(before, model.weight.item())
        self.assertGreater(norm, 0)
        self.assertTrue(values)

    def test_nonfinite_loss_stops_before_optimizer_update(self):
        model, optimizer = self.fresh()
        images, targets, _ = b.collate([self.train[0]])
        before = model.weight.item()
        with patch.object(model, "forward", return_value={"loss": model.weight * float("nan")}):
            with self.assertRaises(ValueError):
                b.update(model, optimizer, images, targets, "cpu")
        self.assertEqual(before, model.weight.item())

    def test_warmup_and_decay_use_original_global_step_and_epoch(self):
        self.assertAlmostEqual(b.learning_rate(self.cfg, 0, 0), .0025*.001)
        self.assertAlmostEqual(b.learning_rate(self.cfg, 2, 0), .0025)
        self.assertAlmostEqual(b.learning_rate(self.cfg, 3, 1), .00025)

    def test_mid_epoch_resume_matches_uninterrupted_stochastic_training(self):
        model, optimizer = self.fresh()
        full = b.fit(model, optimizer, self.train, self.val, self.cfg, "cpu", self.output("full"),
                     self.context, b.initial_progress())
        expected_model = checkpoints.cpu_tree(model.state_dict())
        expected_opt = checkpoints.cpu_tree(optimizer.state_dict())
        expected_rng = checkpoints.rng_state()
        self.assertEqual(full, "complete")
        model, optimizer = self.fresh()
        progress = b.initial_progress()
        out = self.output("resume")
        status = b.fit(model, optimizer, self.train, self.val, self.cfg, "cpu", out, self.context,
                       progress, lambda: progress["step"] >= 1)
        self.assertEqual(status, "budget_stopped")
        self.assertEqual(progress["images_seen"], 2)
        model, optimizer = self.fresh()
        progress = b.restore(out, model, optimizer, self.context)
        status = b.fit(model, optimizer, self.train, self.val, self.cfg, "cpu", out, self.context, progress)
        self.assertEqual(status, "complete")
        self.assertEqual(progress["images_seen"], 10)
        self.assertEqual(progress["step"], 6)
        self.assertEqual(progress["best"]["epoch"], 1)  # ties keep earlier epoch
        checkpoints.assert_same(model.state_dict(), expected_model)
        checkpoints.assert_same(optimizer.state_dict(), expected_opt)
        checkpoints.assert_same(checkpoints.rng_state(), expected_rng)

    def test_validation_interruption_resumes_without_repeating_training(self):
        model, optimizer = self.fresh()
        cfg = {**self.cfg, "epochs": 1}
        out, progress = self.output("val_resume"), b.initial_progress()
        status = b.fit(model, optimizer, self.train, self.val, cfg, "cpu", out, self.context,
                       progress, lambda: progress["step"] >= 3)
        self.assertEqual(status, "budget_stopped")
        self.assertEqual(progress["images_seen"], 5)
        model, optimizer = self.fresh()
        progress = b.restore(out, model, optimizer, self.context)
        self.assertEqual(b.fit(model, optimizer, self.train, self.val, cfg, "cpu", out, self.context, progress), "complete")
        self.assertEqual(progress["images_seen"], 5)
        self.assertEqual(progress["step"], 3)

    def test_changed_checkpoint_or_context_rejected_and_epochs_retained(self):
        model, optimizer = self.fresh()
        out = self.output("integrity")
        b.fit(model, optimizer, self.train, self.val, self.cfg, "cpu", out, self.context, b.initial_progress())
        self.assertEqual(len(list((out / "checkpoints").glob("epoch_*.pt"))), 2)
        with self.assertRaisesRegex(ValueError, "context|configuration"):
            b.restore(out, model, optimizer, {"different": True})
        record = json.loads((out / "last.json").read_text())
        (out / "checkpoints" / record["file"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            b.restore(out, model, optimizer, self.context)

    def test_atomic_pointer_failure_preserves_previous_checkpoint(self):
        model, optimizer = self.fresh()
        out = self.output("atomic")
        progress = b.initial_progress()
        original = b.commit(out, model, optimizer, progress, self.context, "initial")
        real = b.write_json
        def fail_pointer(path, value):
            if Path(path).name == "last.json":
                raise OSError("simulated interruption before pointer replacement")
            return real(path, value)
        with patch.object(b, "write_json", side_effect=fail_pointer):
            with self.assertRaises(OSError):
                b.commit(out, model, optimizer, progress, self.context, "intermediate")
        self.assertEqual(json.loads((out / "last.json").read_text()), original)
        self.assertEqual(b.restore(out, model, optimizer, self.context), progress)

    def test_archive_contains_verified_records_without_checkpoint_binaries(self):
        model, optimizer = self.fresh()
        out = self.output("archive")
        b.fit(model, optimizer, self.train, self.val, self.cfg, "cpu", out, self.context, b.initial_progress())
        b.archive_records(out, "fixture")
        path = out.parent / "archive_records_fixture.tar.gz"
        with tarfile.open(path) as bundle:
            files = {m.name.removeprefix("archive/"): bundle.extractfile(m).read() for m in bundle.getmembers()}
        self.assertFalse(any(name.endswith(".pt") for name in files))
        self.assertIn("history.csv", files)
        sums = json.loads(files["checksums.json"])
        self.assertEqual(set(sums), set(files)-{"checksums.json"})
        for name, digest in sums.items():
            self.assertEqual(sha256(files[name]), digest)
        with self.assertRaises(FileExistsError):
            b.archive_records(out, "fixture")


if __name__ == "__main__":
    unittest.main()
