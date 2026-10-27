import copy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdd_data import (CLASSES, TrainingDataset, inside, load_example, prepare_example,
                      read_manifest, restore_boxes, sha256, validate_record,
                      torchvision_collate, yolo_collate)
from select_training_cases import make_selection
from verify_training_loader import check_batch
import verify_training_loader as runner


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rdd_loader_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def make_row(self, name="sample", boxes=None, labels=None, size=(4, 2), orientation=1,
                 category=None, source="Source", mode="RGB"):
        if boxes is None:
            boxes = [[0.5, 0.25, 3.5, 1.75]]
        if labels is None:
            labels = [0] * len(boxes)
        image_name = f"{source}/train/images/{name}.png"
        xml_name = f"{source}/train/annotations/xmls/{name}.xml"
        image_path, xml_path = self.root / image_name, self.root / xml_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_bytes(b"<annotation/>")
        values = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
        with Image.fromarray(values) as image:
            if mode != "RGB":
                image = image.convert(mode)
            exif = Image.Exif()
            exif[274] = orientation
            image.save(image_path, exif=exif)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            digest = sha256(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii") + rgb.tobytes())
            rgb.close()
        category = category or ("target_bearing" if boxes else "empty_xml")
        return {
            "image": image_name, "xml": xml_name, "source": source, "split": "train",
            "source_cluster_id": image_name, "width": size[0], "height": size[1],
            "image_sha256": sha256(image_path.read_bytes()), "xml_sha256": sha256(xml_path.read_bytes()),
            "rgb_sha256": digest, "exif_orientation": orientation, "annotation_category": category,
            "other_annotations": [{"label": "Repair", "box_xyxy": [0, 0, 1, 1]}] if category == "only_other_labels" else [],
            "targets": [{"box_xyxy": box, "class_index": index, "label": CLASSES[index],
                         "object_index": i, "difficult": "1" if i == 0 else None, "truncated": "0"}
                        for i, (box, index) in enumerate(zip(boxes, labels))],
        }

    def manifest(self, rows):
        path = self.root / "train.jsonl"
        content = ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")
        path.write_bytes(content)
        return path, sha256(content)

    def prepared(self, row):
        return prepare_example(load_example(self.root, row), 32)

    def test_known_letterbox_and_inverse_coordinates(self):
        row = self.make_row()
        example = self.prepared(row)
        np.testing.assert_array_equal(example.boxes, [[4, 10, 28, 22]])
        np.testing.assert_array_equal(restore_boxes(example.boxes, example.geometry), [[0.5, 0.25, 3.5, 1.75]])
        self.assertTrue(np.all(example.rgb[:8] == 114))
        self.assertEqual(example.geometry["pad_top"], 8)

    def test_rounded_resize_uses_actual_axis_scales(self):
        row = self.make_row(size=(7, 2), boxes=[[0, 0, 7, 2]])
        example = self.prepared(row)
        self.assertEqual(example.geometry["resized_height"], 9)
        np.testing.assert_array_equal(example.boxes, [[0, 11, 32, 20]])
        np.testing.assert_allclose(restore_boxes(example.boxes, example.geometry), [[0, 0, 7, 2]], rtol=0, atol=1e-12)

    def test_fractional_subpixel_box_is_not_filtered(self):
        row = self.make_row(boxes=[[1.25, 0.25, 1.5, 0.5]])
        example = self.prepared(row)
        np.testing.assert_array_equal(example.boxes, [[10, 10, 12, 12]])
        self.assertEqual(check_batch([example])[0]["targets"], 1)

    def test_all_class_mappings_and_same_pixels(self):
        row = self.make_row(boxes=[[0, 0, 2, 1]] * 4, labels=[0, 1, 2, 3])
        example = self.prepared(row)
        images, targets, _ = torchvision_collate([example])
        yolo, _ = yolo_collate([example])
        self.assertEqual(targets[0]["labels"].tolist(), [1, 2, 3, 4])
        self.assertEqual(yolo["cls"].view(-1).tolist(), [0, 1, 2, 3])
        self.assertTrue(torch.equal(images[0], yolo["img"][0]))
        self.assertEqual(check_batch([example])[0]["difficult_targets"], 1)

    def test_mixed_batch_keeps_empty_image_and_correct_batch_index(self):
        negative = self.prepared(self.make_row("negative", boxes=[]))
        positive = self.prepared(self.make_row("positive", labels=[3]))
        images, targets, _ = torchvision_collate([negative, positive])
        yolo, _ = yolo_collate([negative, positive])
        self.assertEqual(len(images), 2)
        self.assertEqual(tuple(targets[0]["boxes"].shape), (0, 4))
        self.assertEqual(yolo["batch_idx"].tolist(), [1])
        self.assertEqual([r["targets"] for r in check_batch([negative, positive])], [0, 1])

    def test_entirely_empty_batch(self):
        examples = [self.prepared(self.make_row(str(i), boxes=[])) for i in range(2)]
        yolo, _ = yolo_collate(examples)
        self.assertEqual(tuple(yolo["img"].shape), (2, 3, 32, 32))
        self.assertEqual(tuple(yolo["bboxes"].shape), (0, 4))
        self.assertEqual(tuple(yolo["cls"].shape), (0, 1))
        self.assertEqual(check_batch(examples)[0]["targets"], 0)

    def test_non_target_annotations_stay_separate(self):
        row = self.make_row(boxes=[], category="only_other_labels")
        example = self.prepared(row)
        self.assertEqual(len(example.boxes), 0)
        self.assertEqual(example.record["other_annotations"], row["other_annotations"])

    def test_exif_rotation_is_not_applied(self):
        row = self.make_row(size=(3, 2), boxes=[], orientation=6)
        loaded = load_example(self.root, row)
        self.assertEqual(loaded.rgb.shape, (2, 3, 3))
        np.testing.assert_array_equal(loaded.rgb, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))

    def test_non_rgb_input_converts_to_rgb(self):
        row = self.make_row(mode="L")
        loaded = load_example(self.root, row)
        self.assertEqual(loaded.rgb.shape, (2, 4, 3))
        np.testing.assert_array_equal(loaded.rgb[:, :, 0], loaded.rgb[:, :, 2])

    def test_original_record_is_not_mutated(self):
        row = self.make_row()
        original = copy.deepcopy(row)
        loaded = load_example(self.root, row)
        prepared = prepare_example(loaded, 32)
        prepared.record["targets"][0]["difficult"] = "0"
        prepared.labels[0] = 3
        self.assertEqual(row, original)
        self.assertEqual(loaded.labels.tolist(), [0])

    def test_corrupted_image_rejected(self):
        row = self.make_row()
        with (self.root / row["image"]).open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Image checksum"):
            load_example(self.root, row)

    def test_corrupted_xml_rejected(self):
        row = self.make_row()
        (self.root / row["xml"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "XML checksum"):
            load_example(self.root, row)

    def test_changed_rgb_hash_rejected(self):
        row = self.make_row()
        row["rgb_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "RGB checksum"):
            load_example(self.root, row)

    def test_changed_stored_dimensions_rejected(self):
        row = self.make_row()
        row["width"] = 5
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_example(self.root, row)

    def test_invalid_boxes_and_labels_rejected(self):
        original = self.make_row()
        for box in ([1, 0, 1, 1], [-1, 0, 2, 1], [0, 0, float("nan"), 1]):
            row = copy.deepcopy(original)
            row["targets"][0]["box_xyxy"] = box
            with self.assertRaises(ValueError):
                validate_record(row)
        original["targets"][0]["class_index"] = 1
        with self.assertRaisesRegex(ValueError, "class index"):
            validate_record(original)

    def test_path_escape_rejected(self):
        for value in ("../outside.jpg", "/absolute.jpg", "C:/outside.jpg", "a\\outside.jpg"):
            with self.assertRaises(ValueError):
                inside(self.root, value)

    def test_manifest_hash_and_held_out_rows_rejected(self):
        row = self.make_row()
        path, digest = self.manifest([row])
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_manifest(path, "0" * 64)
        row["split"] = "test"
        path, digest = self.manifest([row])
        with self.assertRaisesRegex(ValueError, "Unexpected split"):
            read_manifest(path, digest)

    def test_map_dataset_and_missing_selection(self):
        rows = [self.make_row("positive"), self.make_row("negative", boxes=[])]
        path, digest = self.manifest(rows)
        dataset = TrainingDataset(self.root, path, digest, size=32)
        batches = list(DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=list))
        self.assertEqual([r["targets"] for r in check_batch(batches[0])], [1, 0])
        with self.assertRaisesRegex(ValueError, "outside"):
            TrainingDataset(self.root, path, digest, ["not_in_training.png"])

    def test_duplicate_manifest_or_selection_rejected(self):
        row = self.make_row()
        path, digest = self.manifest([row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            read_manifest(path, digest)
        path, digest = self.manifest([row])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            TrainingDataset(self.root, path, digest, [row["image"], row["image"]])

    def test_selection_is_order_independent_and_preserves_coverage(self):
        rows = [self.make_row("positive", labels=[3]),
                self.make_row("negative", boxes=[]),
                self.make_row("rotated", boxes=[], orientation=8)]
        first = make_selection(rows, "test-hash")
        self.assertEqual(first, make_selection(list(reversed(rows)), "test-hash"))
        features = {tag for entry in first["selected"] for tag in entry["features"]}
        self.assertIn("class:D40", features)
        self.assertIn("orientation:8", features)
        self.assertIn("source:Source:no_target", features)

    def cli_fixture(self):
        base = self.root
        self.root = base / "raw"
        self.root.mkdir()
        rows = [self.make_row("positive"), self.make_row("negative", boxes=[])]
        manifest, digest = self.manifest(rows)
        config = manifest.with_name("config.json")
        config.write_text("{}", encoding="utf-8")
        package = base / "package"
        package.mkdir()
        for name in ("rdd_data.py", "select_training_cases.py", "verify_training_loader.py"):
            shutil.copyfile(Path(__file__).resolve().parents[1] / name, package / name)
        reference = {
            "manifest_name": "train.jsonl", "manifest_sha256": digest,
            "config_sha256": sha256(config.read_bytes()),
            "expected_training_images": 2, "expected_training_targets": 1,
            "expected_environment": sys.prefix, "versions": {}, "image_size": 32, "batch_size": 2,
        }
        (package / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
        (package / "selection.json").write_text(json.dumps(make_selection(rows, digest)), encoding="utf-8")
        checksums = "".join(f"{sha256(path.read_bytes())}  {path.name}\n" for path in sorted(package.iterdir()))
        (package / "SHA256SUMS").write_text(checksums, encoding="utf-8")
        out = base / "result"
        argv = [str(package / "verify_training_loader.py"), "--root", str(self.root),
                "--splits", str(manifest.parent), "--out", str(out)]
        return package, out, argv, rows

    def invoke_fixture(self, package, argv):
        with patch.object(runner, "__file__", str(package / "verify_training_loader.py")), \
             patch.object(sys, "argv", argv), \
             patch.dict(os.environ, {"PBS_JOBID": "SYNTHETIC_UNIT_TEST"}), \
             patch.object(runner.socket, "gethostname", return_value="synthetic-cpu"), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main()

    def test_cli_complete_archive_and_no_overwrite(self):
        package, out, argv, _ = self.cli_fixture()
        self.invoke_fixture(package, argv)
        report_bytes = (out / "run.json").read_bytes()
        report = json.loads(report_bytes)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["images_checked"], 2)
        self.assertFalse(report["held_out_data_read"])
        self.assertFalse(report["detector_execution"])
        archive = out.with_name(out.name + ".tar.gz")
        with tarfile.open(archive, "r:gz") as bundle:
            self.assertIn("result/run.json", bundle.getnames())
            self.assertIn("result/code/rdd_data.py", bundle.getnames())
        with self.assertRaises(FileExistsError):
            self.invoke_fixture(package, argv)
        self.assertEqual(report_bytes, (out / "run.json").read_bytes())

    def test_cli_records_image_failure_without_success_archive(self):
        package, out, argv, rows = self.cli_fixture()
        with (self.root / rows[0]["image"]).open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Image checksum"):
            self.invoke_fixture(package, argv)
        report = json.loads((out / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertFalse(out.with_name(out.name + ".tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
