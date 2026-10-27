"""Scientific invariants on synthetic pixels, not evidence of augmentation benefit."""
import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

import pilot_runner as pilot
import train_baseline as base


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / "tmp"
        self.scratch.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="pilot_test_", dir=str(self.scratch)))
        self.data = self.root / "data"
        self.data.mkdir()
        self.rows, checksums = [], []
        for split, count in (("train", 2), ("val", 4), ("test", 1)):
            for c, label in enumerate(base.LABELS):
                for i in range(count):
                    sid = "%s_%s_%d" % (split, label, i)
                    row = {"sample_id": sid, "split": split, "source": "SYNTHETIC",
                           "image": sid + ".jpg", "xml": sid + ".xml", "group_id": sid,
                           "label": label, "label_id": str(c), "object_index": "0",
                           "left": "0", "top": "0", "right": "224", "bottom": "112",
                           "file_sha256": base.sha256(sid.encode()),
                           "crop_path": "%s/%s/%s.png" % (split, label, sid)}
                    self.rows.append(row)
                    if split == "test":
                        continue  # Any accidental test access must fail.
                    image = Image.new("RGB", (224, 224), (127, 127, 127))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 56, 223, 167), fill=(50 + 30 * c, 70 + 15 * i, 100))
                    draw.line((10, 65 + i, 210, 145 - c), fill="white", width=4)
                    path = self.data / row["crop_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)
                    checksums.append((row["crop_path"], base.sha256(path.read_bytes())))
        with (self.data / "crops.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        sha = base.sha256((self.data / "crops.csv").read_bytes())
        base.json_write(self.data / "config.json", {"manifest_sha256": sha, "crop_size": 224,
                        "labels": {str(i): c for i, c in enumerate(base.LABELS)}})
        base.json_write(self.data / "run.json", {"manifest_sha256": sha, "status": "complete"})
        with (self.data / "crop_checksums.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["crop_path", "sha256"])
            writer.writerows(checksums)
        self.design = self.root / "design"
        self.call("prepare", "--manifest", self.data / "crops.csv", "--out", self.design,
                  "--seed", 19, "--eval-per-class", 1)

    def call(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return pilot.main(list(map(str, args)))

    def tearDown(self):
        target = self.root.resolve()
        target.relative_to(self.scratch.resolve())
        if target == self.scratch.resolve() or not target.name.startswith("pilot_test_"):
            raise RuntimeError("Unsafe fixture cleanup path")
        shutil.rmtree(target)

    def load(self):
        splits, config = base.read_manifest(self.data)
        checksums = base.read_checksums(self.data, splits)
        design, panels = pilot.load_design(self.design, splits, config["manifest_sha256"])
        return splits, config, checksums, design, panels

    def test_group_integrity_design_tampering_and_missing_test_pixels(self):
        splits, config, checksums, design, panels = self.load()
        for field in ("image", "group_id", "file_sha256"):
            a = {r[field] for r in panels["reference"]}
            b = {r[field] for r in panels["audit"]}
            self.assertFalse(a & b)
        rows = [dict(r) for r in splits["val"]]
        rows[0]["group_id"] = rows[4]["group_id"]
        rows[4]["image"] = rows[8]["image"]
        rows[8]["file_sha256"] = rows[12]["file_sha256"]
        roles = pilot.grouped_roles(rows, 19)
        self.assertEqual(len({roles[rows[i]["sample_id"]] for i in (0, 4, 8, 12)}), 1)
        self.assertEqual(roles, pilot.grouped_roles(rows, 19))
        design["bank"][1]["magnitude"] = 0.2
        base.json_write(self.design / "design.json", design)
        with self.assertRaisesRegex(ValueError, "integrity"):
            pilot.load_design(self.design, splits, config["manifest_sha256"])
        self.assertFalse((self.data / "test").exists())

    def test_operation_strength_padding_targeting_and_pairing(self):
        splits, _, checksums, _, _ = self.load()
        ds = base.CropDataset(self.data, splits["train"], checksums)
        original = ds.image(0)
        row = ds.rows[0]
        original_array = np.array(original)
        for a in pilot.DEFAULT_BANK:
            for sign in (-1, 1):
                transformed = np.array(pilot.transform(original, row, a, sign))
                self.assertTrue(np.array_equal(original_array[:56], transformed[:56]))
                self.assertTrue(np.array_equal(original_array[168:], transformed[168:]))
                zero = {**a, "magnitude": 0.0}
                self.assertTrue(np.array_equal(original_array, np.array(pilot.transform(original, row, zero, sign))))
        low = np.array(pilot.transform(original, row, pilot.DEFAULT_BANK[1], 1))
        high = np.array(pilot.transform(original, row, pilot.DEFAULT_BANK[2], 1))
        self.assertGreater(high[80, 10, 0], low[80, 10, 0])
        plan = pilot.balanced_plan(ds.rows, 2, 4, 29)
        self.assertEqual(plan, pilot.balanced_plan(ds.rows, 2, 4, 29))
        x0, y0, n0 = pilot.training_batch(ds, plan[0], "D00", pilot.DEFAULT_BANK[0], 29, 1)
        xa, ya, na = pilot.training_batch(ds, plan[0], "D00", pilot.DEFAULT_BANK[2], 29, 1)
        self.assertTrue(torch.equal(y0, ya))
        self.assertTrue(torch.equal(x0[y0 != 0], xa[y0 != 0]))
        self.assertFalse(torch.equal(x0[y0 == 0], xa[y0 == 0]))
        self.assertEqual((n0, na), (0, 1))
        path = self.data / ds.rows[0]["crop_path"]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed since generation"):
            ds.image(0)

    def test_preview_uses_verified_training_pngs(self):
        out = self.root / "preview"
        self.call("preview", "--data", self.data, "--design", self.design, "--out", out)
        manifest = pilot.read_json(out / "preview_manifest.json")
        self.assertEqual(manifest["source_kind"], "ORIGINAL CHECKSUM-VERIFIED TRAINING PNGS")
        self.assertEqual(len(manifest["images"]), 5)
        for entry in manifest["images"]:
            self.assertTrue(all(sid.startswith("train_") for sid in entry["sample_ids"]))
            with Image.open(out / entry["file"]) as sheet:
                self.assertEqual(sheet.height, 66 + 8 * 256)
                self.assertEqual(sheet.width, 928 if "blur" in entry["file"] else 1392)
        self.assertFalse((self.data / "test").exists())

    def train(self, out, *extra):
        return self.call("train", "--data", self.data, "--design", self.design, "--out", out,
                         "--device", "cpu", "--epochs", 2, "--snapshots", "1,2", "--batch-size", 4,
                         "--eval-batch-size", 4, "--workers", 0, "--max-minutes", 0, *extra)

    def test_resume_identity_and_nonidentity_end_to_end(self):
        full, resumed = self.root / "full", self.root / "resumed"
        self.train(full)
        self.train(resumed, "--stop-after-epoch", 1)
        self.train(resumed, "--resume")
        a = torch.load(full / "last.pt", map_location="cpu", weights_only=False)
        b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "scheduler", "rng"):
            self.assertTrue(pilot.equal_state(a[key], b[key]), key)
        identity_out = self.root / "identity"
        self.call("identity", "--data", self.data, "--design", self.design, "--out", identity_out,
                  "--checkpoint", full / "epoch_001.pt", "--device", "cpu", "--steps", 3,
                  "--batch-size", 4, "--eval-batch-size", 4)
        result = pilot.read_json(identity_out / "identity_control.json")
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_model_absolute_difference"], 0)
        self.assertTrue(all(result["checks"].values()))
        probe_out = self.root / "probe"
        command = ("probe", "--data", self.data, "--design", self.design, "--out", probe_out,
                   "--checkpoint", full / "epoch_001.pt", "--device", "cpu", "--batch-size", 4,
                   "--eval-batch-size", 4, "--horizons", "1,2", "--classes", "D00",
                   "--candidates", "identity,brightness_high", "--max-minutes", 0)
        self.call(*command)
        identity = pilot.read_json(probe_out / "trials/D00_e000_identity.json")
        for h in identity["outcomes"].values():
            for r in h.values():
                self.assertEqual(r["balanced_loss_gain"], 0)
                self.assertEqual(r["macro_f1_gain"], 0)
        self.assertEqual(identity["signals"]["adamw_linear_gain"], 0)
        changed = pilot.read_json(probe_out / "trials/D00_e000_brightness_high.json")
        self.assertEqual(changed["changed_target_examples"], 2)
        self.assertNotEqual(changed["signals"]["excess_loss"], 0)
        before = (probe_out / "results.csv").read_bytes()
        self.call(*command, "--resume")
        self.assertEqual(before, (probe_out / "results.csv").read_bytes())
        self.assertEqual(pilot.read_json(probe_out / "run_status.json")["status"], "complete")
        # Save an inspectable check report, explicitly marked as synthetic.
        evidence = Path(__file__).resolve().parents[1] / "reports/pilot_verification_v1"
        evidence.mkdir(parents=True, exist_ok=True)
        base.json_write(evidence / "identity_control_cpu.json", {
            **result, "input_kind": "SYNTHETIC FIXTURE; no road-defect effect estimate",
            "code_hashes": pilot.code_hashes(), "resume_exact_state_match": True,
            "nonidentity_changed_target_images": changed["changed_target_examples"],
            "environment": pilot.environment("cpu")})

    def test_class_balanced_evaluation_matches_manual_calculation(self):
        class ZeroModel(nn.Module):
            def forward(self, x):
                return torch.zeros((len(x), 4))
        splits, _, checksums, _, _ = self.load()
        ds = base.CropDataset(self.data, splits["val"], checksums)
        metric = pilot.evaluate(ZeroModel(), ds, "cpu", 5)
        self.assertAlmostEqual(metric["balanced_loss"], np.log(4), places=6)
        self.assertAlmostEqual(metric["accuracy"], 0.25)
        self.assertAlmostEqual(metric["macro_f1"], 0.1)

    def test_reference_gradient_with_unequal_class_support(self):
        class ScalarModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.3))

            def forward(self, x):
                v = x.mean((1, 2, 3)) * self.weight
                return torch.stack((v, -v, 2 * v, torch.zeros_like(v)), 1)
        splits, _, checksums, _, _ = self.load()
        rows = [r for r in splits["val"] if r["label"] == "D00"] + [
            next(r for r in splits["val"] if r["label"] == c) for c in base.LABELS[1:]]
        ds = base.CropDataset(self.data, rows, checksums)
        model = ScalarModel()
        actual = pilot.reference_gradient(model, ds, "cpu", 3).item()
        with torch.no_grad():
            model.weight.fill_(0.301)
        upper = pilot.evaluate(model, ds, "cpu", 3)["balanced_loss"]
        with torch.no_grad():
            model.weight.fill_(0.299)
        lower = pilot.evaluate(model, ds, "cpu", 3)["balanced_loss"]
        self.assertAlmostEqual(actual, (upper - lower) / 0.002, delta=0.0002)


if __name__ == "__main__":
    unittest.main()
