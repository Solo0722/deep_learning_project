"""Behavior checks on synthetic data; these are not research results."""
import contextlib
import csv
import io
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

import train_baseline as baseline


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / "tmp"
        self.scratch.mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="baseline_test_", dir=str(self.scratch)))
        self.data = self.work / "data"
        self.data.mkdir()
        self.rows, self.checksums = [], {}
        for split, number in (("train", 2), ("val", 1), ("test", 1)):
            for label_id, label in enumerate(baseline.LABELS):
                for i in range(number):
                    name = "%s_%s_%d" % (split, label, i)
                    relative = "%s/%s/%s.png" % (split, label, name)
                    row = {"sample_id": name, "split": split, "source": "SYNTHETIC",
                           "image": name + ".jpg", "xml": name + ".xml",
                           "object_index": "0", "label": label, "label_id": str(label_id),
                           "group_id": name, "left": "0", "top": "0", "right": "224",
                           "bottom": "112", "file_sha256": baseline.sha256(name.encode()),
                           "crop_path": relative}
                    self.rows.append(row)
                    # Deliberately create no test pixel files. Development must work without them.
                    if split == "test":
                        continue
                    path = self.data / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image = Image.new("RGB", (224, 224), (127, 127, 127))
                    colors = ((200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 56, 223, 167), fill=colors[label_id])
                    draw.rectangle((20 + 4*i, 70, 60 + 4*i, 100), fill="white")
                    image.save(path)
                    self.checksums[relative] = baseline.sha256(path.read_bytes())
        self.write_manifest()

    def write_manifest(self):
        with (self.data / "crops.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        digest = baseline.sha256((self.data / "crops.csv").read_bytes())
        (self.data / "config.json").write_text(json.dumps({
            "manifest_sha256": digest, "crop_size": 224,
            "labels": {str(i): c for i, c in enumerate(baseline.LABELS)}}), encoding="utf-8")
        (self.data / "run.json").write_text(json.dumps({"status": "complete", "manifest_sha256": digest}), encoding="utf-8")
        with (self.data / "crop_checksums.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["crop_path", "sha256"])
            writer.writerows(self.checksums.items())

    def tearDown(self):
        target, scratch = self.work.resolve(), self.scratch.resolve()
        target.relative_to(scratch)
        if target == scratch or not target.name.startswith("baseline_test_"):
            raise RuntimeError("Unexpected cleanup path")
        shutil.rmtree(target)

    def test_metrics_with_missed_class_and_unequal_support(self):
        # Correct: 2 of 3 D00, 1 of 2 D10, 1 of 1 D20, 0 of 1 D40.
        truth = [0, 0, 0, 1, 1, 2, 3]
        prediction = [0, 0, 1, 1, 2, 2, 0]
        result = baseline.metrics(baseline.confusion_matrix(truth, prediction))
        self.assertAlmostEqual(result["accuracy"], 4/7)
        self.assertAlmostEqual(result["macro_f1"], (2/3 + 1/2 + 2/3 + 0)/4)
        self.assertEqual(result["per_class"]["D40"]["recall"], 0)
        self.assertEqual(result["per_class"]["D00"]["support"], 3)

    def test_augmentation_is_repeatable_and_preserves_padding(self):
        splits, _ = baseline.read_manifest(self.data)
        dataset = baseline.CropDataset(self.data, splits["train"], self.checksums, "shared")
        original = np.array(dataset.image(0, augmented=False))
        first = np.array(dataset.image(0))
        self.assertTrue(np.array_equal(first, np.array(dataset.image(0))))
        self.assertTrue(np.array_equal(first[:56], original[:56]))
        self.assertTrue(np.array_equal(first[168:], original[168:]))
        changed = False
        for epoch in range(1, 6):
            dataset.epoch = epoch
            changed |= not np.array_equal(first, np.array(dataset.image(0)))
        self.assertTrue(changed)
        val = baseline.CropDataset(self.data, splits["val"], self.checksums)
        before = val[0][0]
        val.epoch = 999
        self.assertTrue(torch.equal(before, val[0][0]))
        with self.assertRaisesRegex(ValueError, "restricted"):
            baseline.CropDataset(self.data, splits["val"], self.checksums, "shared")
        with self.assertRaisesRegex(ValueError, "Test images"):
            baseline.CropDataset(self.data, [self.rows[-1]], self.checksums)
        path = self.data / splits["train"][0]["crop_path"]
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "changed since generation"):
            dataset[0]

    def test_rejects_manifest_edit_and_group_leakage(self):
        with (self.data / "crops.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            baseline.read_manifest(self.data)
        self.rows[-1]["group_id"] = self.rows[0]["group_id"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Cross-split group_id"):
            baseline.read_manifest(self.data)

    def test_resume_matches_uninterrupted_training(self):
        splits, _ = baseline.read_manifest(self.data)
        checksums = baseline.read_checksums(self.data, splits)
        args = SimpleNamespace(data=self.data, augmentation="shared", seed=42, device="cpu",
                               lr=0.001, weight_decay=0.0001, epochs=2, resume=False,
                               batch_size=4, workers=0, stop_after_epoch=0, max_minutes=0)
        config = {"purpose": "SYNTHETIC SOFTWARE TEST, NOT A RESEARCH RESULT"}
        full, resumed = self.work / "full", self.work / "resumed"
        full.mkdir()
        resumed.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            baseline.seed_all(42, 2)
            baseline.train(args, splits, checksums, full, config)
            baseline.seed_all(42, 2)
            args.stop_after_epoch = 1
            baseline.train(args, splits, checksums, resumed, config)
            pause = json.loads((resumed / "run_status.json").read_text())
            self.assertEqual(pause["status"], "paused_after_epoch")
            # Disturb process RNG, as in a fresh process, before loading the checkpoint.
            baseline.seed_all(123, 2)
            random.random()
            torch.rand(9)
            args.resume, args.stop_after_epoch = True, 0
            baseline.train(args, splits, checksums, resumed, config)
        a = torch.load(full / "last.pt", map_location="cpu", weights_only=False)
        b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
        for name in a["model"]:
            self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)
        for ha, hb in zip(a["history"], b["history"]):
            for key in ("epoch", "lr", "train", "val"):
                self.assertEqual(ha[key], hb[key], key)
        self.assertEqual(a["scheduler"], b["scheduler"])
        self.assertEqual((full / "validation_predictions.csv").read_bytes(),
                         (resumed / "validation_predictions.csv").read_bytes())
        status = json.loads((resumed / "run_status.json").read_text())
        self.assertEqual(status["status"], "complete")
        self.assertFalse(status["test_evaluated"])
        self.assertEqual(a["history"][0]["train"]["per_class"]["D00"]["support"], 2)
        self.assertTrue(np.isfinite(a["history"][-1]["train"]["loss"]))


if __name__ == "__main__":
    unittest.main()
