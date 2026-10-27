import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from audit_detection_data import PairLimitReached, audit, digest, input_paths, target_boxes
from screen_near_duplicates import screen
from validate_images import audit as audit_images


def fixture_hash(picture):
    # Two target images are two bits apart. An empty-XML image is one bit from
    # each, creating a new bridge at radius 1 that the target-only screen missed.
    return {20: 0, 30: 3, 40: 1, 50: 65535}[picture.getpixel((0, 0))[0]]


class DetectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / "tmp"
        self.scratch.mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="detection_audit_test_", dir=self.scratch))
        self.raw = self.work / "raw"
        self.image_folder = self.raw / "A/train/images"
        self.xml_folder = self.raw / "A/train/annotations/xmls"
        self.image_folder.mkdir(parents=True)
        self.xml_folder.mkdir(parents=True)
        for name, gray, label in (("t_train", 20, "D00"), ("t_train_copy", 20, "D00"),
                                  ("t_test", 30, "D40"), ("n_empty", 40, None),
                                  ("n_other", 50, "D50")):
            with Image.new("RGB", (16, 16), (gray, gray, gray)) as picture:
                picture.save(self.image_folder / (name + ".png"))
            obj = ("<object><name>%s</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
                   "<xmax>14</xmax><ymax>14</ymax></bndbox></object>" % label) if label else ""
            (self.xml_folder / (name + ".xml")).write_text(
                "<annotation><size><width>16</width><height>16</height></size>" + obj + "</annotation>",
                encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            audit_images(self.raw, self.work / "image_audit")
            with patch("screen_near_duplicates.dhash128", side_effect=fixture_hash):
                screen(self.raw, self.work / "image_audit/inventory.jsonl", self.work / "previous_review", radius=0)
        self.legacy_dir = self.work / "legacy"
        self.legacy_dir.mkdir()
        self.previous_images = self.legacy_dir / "images.csv"
        with self.previous_images.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image", "source", "split", "group_id", "crops"])
            writer.writeheader()
            for name, split in (("t_train", "train"), ("t_test", "test")):
                image = "A/train/images/" + name + ".png"
                writer.writerow({"image": image, "source": "A", "split": split, "group_id": image, "crops": 1})
        self.inventory = self.work / "image_audit/inventory.jsonl"
        self.review = self.work / "previous_review"
        self.reference_dir = self.work / "reference"
        self.reference_dir.mkdir()
        self.reference = self.reference_dir / "inputs.json"
        paths = input_paths(self.inventory, self.review, self.previous_images)
        self.reference.write_text(json.dumps({"sha256": {name: digest(p) for name, p in paths.items()}}), encoding="utf-8")
        self.original_bytes = {p: p.read_bytes() for p in paths.values()}

    def tearDown(self):
        target = self.work.resolve()
        target.relative_to(self.scratch.resolve())
        if target == self.scratch.resolve() or not target.name.startswith("detection_audit_test_"):
            raise RuntimeError("Unsafe fixture cleanup path")
        shutil.rmtree(target)

    def run_audit(self, out="detection_review", **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), patch("audit_detection_data.dhash128", side_effect=fixture_hash):
            return audit(self.raw, self.inventory, self.review, self.previous_images,
                         self.reference, self.work / out, radius=1, **kwargs)

    def test_negative_bridge_is_reported_and_no_split_is_changed(self):
        result = self.run_audit()
        self.assertTrue(result["screen_complete"])
        self.assertFalse(result["new_split_created"])
        self.assertEqual(result["images_checked"], 5)
        self.assertEqual(result["counts"]["empty_xml"], 1)
        self.assertEqual(result["counts"]["only_other_labels"], 1)
        self.assertEqual(result["counts"]["historical_exact_copy_images"], 1)
        self.assertEqual(result["other_annotation_labels"], {"D50": 1})
        self.assertEqual(result["previous_fingerprints_compared"], 2)
        self.assertEqual(result["previous_fingerprints_changed"], 0)
        self.assertEqual(result["component_statistics"]["source_clusters"], 2)
        self.assertEqual(result["component_statistics"]["clusters_crossing_previous_splits"], 1)
        self.assertEqual(result["component_statistics"]["previously_assigned_images_in_conflicts"], {"test": 1, "train": 1})
        conflicts = [json.loads(x) for x in (self.work / "detection_review/previous_split_conflicts.jsonl").read_text().splitlines()]
        self.assertIn("A/train/images/n_empty.png", conflicts[0]["members"])
        self.assertEqual(conflicts[0]["newly_considered_images"], 2)  # negative plus recorded exact copy
        self.assertTrue((self.work / "detection_review.tar.gz").is_file())
        with (self.work / "detection_review/source_clusters.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(not r["previous_split"] for r in rows), 3)
        self.assertFalse(any(self.work.joinpath("detection_review").glob("*.jpg")))
        for path, before in self.original_bytes.items():
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(json.loads((self.work / "detection_review/run.json").read_text())["status"], "complete")

    def test_pair_cap_prevents_complete_status_and_cluster_export(self):
        with self.assertRaises(PairLimitReached):
            self.run_audit(max_pairs=1)
        status = json.loads((self.work / "detection_review/run.json").read_text())
        self.assertEqual(status["status"], "incomplete_candidate_limit")
        self.assertFalse((self.work / "detection_review/source_clusters.csv").exists())
        self.assertFalse((self.work / "detection_review.tar.gz").exists())

    def test_modified_image_is_rejected_and_failure_recorded(self):
        with Image.new("RGB", (16, 16), "red") as picture:
            picture.save(self.image_folder / "n_empty.png")
        with self.assertRaisesRegex(ValueError, "Image changed"):
            self.run_audit()
        self.assertEqual(json.loads((self.work / "detection_review/run.json").read_text())["status"], "failed")

    def test_modified_xml_is_rejected(self):
        path = self.xml_folder / "n_empty.xml"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "XML changed"):
            self.run_audit()

    def test_modified_reference_input_is_rejected_before_creating_output(self):
        path = self.review / "annotations.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.run_audit()
        self.assertFalse((self.work / "detection_review").exists())

    def test_unexpected_raw_file_is_rejected(self):
        shutil.copyfile(self.image_folder / "n_empty.png", self.image_folder / "extra.png")
        with self.assertRaisesRegex(ValueError, "file membership"):
            self.run_audit()

    def test_output_cannot_replace_inputs_or_previous_results(self):
        with self.assertRaisesRegex(ValueError, "separate from raw inputs"):
            self.run_audit(out="raw/review")
        self.run_audit()
        before = (self.work / "detection_review/run.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_audit()
        self.assertEqual((self.work / "detection_review/run.json").read_bytes(), before)


class DetectionBoxTests(unittest.TestCase):
    def test_fractional_small_boxes_survive_and_invalid_boxes_are_reported(self):
        record = {"image": "A/train/images/test.png", "size": [16, 16], "objects": [
            {"index": 0, "label": "D00", "box": [1.25, 2.5, 1.75, 3.5], "difficult": "1"},
            {"index": 1, "label": "D10", "box": [3, 4, 3, 8]},
            {"index": 2, "label": "D20", "box": [-1, 4.5, 18, 10]},
            {"index": 3, "label": "D0w0", "box": [1, 1, 2, 2]},
        ]}
        targets, issues = target_boxes(record)
        self.assertEqual([t["label"] for t in targets], ["D00", "D20"])
        self.assertEqual(targets[0]["box_xyxy"], [1.25, 2.5, 1.75, 3.5])
        self.assertEqual(targets[0]["difficult"], "1")
        self.assertEqual(targets[1]["box_xyxy"], [0.0, 4.5, 16.0, 10])
        self.assertEqual([r["issue"] for r in issues], ["nonpositive_target_area", "target_clipped_to_image"])


if __name__ == "__main__":
    unittest.main()
