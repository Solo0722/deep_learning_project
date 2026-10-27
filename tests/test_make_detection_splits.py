import contextlib
import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import make_detection_splits as split


class DetectionSplitTests(unittest.TestCase):
    def setUp(self):
        self.parent = (Path(__file__).resolve().parents[1] / "tmp").resolve()
        self.parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="detection_splits_test_", dir=self.parent)
        self.root = Path(self.temp.name).resolve()
        self.names = {k: "A/train/images/" + k + ".jpg" for k in (
            "a_train", "b_val_conflict", "c_val", "d_test", "e_background",
            "f_test_background", "g_new_background", "h_other", "i_ambiguous", "j_duplicate", "k_zero")}
        self.rows = []
        old = {"a_train": "train", "b_val_conflict": "val", "c_val": "val", "d_test": "test", "k_zero": "train"}
        self.prior_rows = [{"image": self.names[k], "source": "A", "split": s, "group_id": self.names[k], "crops": "1"}
                           for k, s in old.items()]
        for key, image in self.names.items():
            objects = []
            if key in old or key == "j_duplicate":
                objects = [{"index": 0, "label": "D00", "box": [1., 1., 1.25, 1.5], "difficult": "1", "truncated": "0"}]
            if key == "k_zero":
                objects.append({"index": 1, "label": "D20", "box": [4., 4., 4., 5.], "difficult": "0", "truncated": "1"})
            if key in ("h_other", "i_ambiguous"):
                objects = [{"index": 0, "label": "Repair" if key == "h_other" else "D0w0", "box": [2., 2., 7., 7.]}]
            targets = [{"object_index": o["index"], "label": o["label"], "box_xyxy": o["box"],
                        "difficult": o.get("difficult"), "truncated": o.get("truncated")}
                       for o in objects if o["label"] == "D00"]
            row = {"image": image, "xml": image.replace("/images/", "/annotations/xmls/").replace(".jpg", ".xml"),
                   "source": "A", "size": [10, 10], "image_sha256": split.sha(image.encode()),
                   "xml_sha256": split.sha((image + "xml").encode()), "rgb_sha256": split.sha((image + "rgb").encode()),
                   "objects": objects, "targets": targets, "annotation_category": "target_bearing" if targets else
                   "only_other_labels" if objects else "empty_xml", "exif_orientation": 1}
            if key == "j_duplicate":
                canonical = next(r for r in self.rows if r["image"] == self.names["a_train"])
                row.update(duplicate_of=canonical["image"], rgb_sha256=canonical["rgb_sha256"], image_sha256=canonical["image_sha256"])
            self.rows.append(row)
        self.group = {n: n for n in self.names.values()}
        for k in ("b_val_conflict", "e_background", "j_duplicate"):
            self.group[self.names[k]] = self.names["a_train"]
        self.group[self.names["f_test_background"]] = self.names["d_test"]
        self.edges = [("a_train", "e_background"), ("e_background", "b_val_conflict"), ("a_train", "j_duplicate"),
                      ("d_test", "f_test_background")]
        self.previous = self.root / "legacy" / "images.csv"
        self.previous.parent.mkdir()
        self.previous.write_bytes(self.csv_bytes(self.prior_rows))
        self.bundle = self.root / "audit.tar.gz"
        self.policy_path = self.root / "policy.json"
        self.policy = {"seed": 42, "previous_images_sha256": split.sha(self.previous.read_bytes()),
                       "audit_code_sha256": {"fixture": "synthetic"}, "expected_val_to_train_images": [self.names["b_val_conflict"]],
                       "retained_non_target_labels": ["Repair"], "excluded_images": [
                           {"image": self.names["j_duplicate"], "reason": "historical_exact_duplicate", "canonical": self.names["a_train"]},
                           {"image": self.names["i_ambiguous"], "reason": "ambiguous_annotation_label", "expected_labels": ["D0w0"]}],
                       "excluded_objects": [{"image": self.names["k_zero"], "object_index": 1, "reason": "nonpositive_target_area"}]}
        self.write_inputs()

    @staticmethod
    def csv_bytes(rows):
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return text.getvalue().encode()

    def write_inputs(self):
        prior = {r["image"]: r for r in self.prior_rows}
        clusters = [{"image": r["image"], "source": "A", "source_cluster_id": self.group[r["image"]],
                     "previous_split": prior.get(r["image"], {}).get("split", ""),
                     "previous_group_id": prior.get(r["image"], {}).get("group_id", ""),
                     "annotation_category": r["annotation_category"], "valid_targets": len(r["targets"]),
                     "historical_duplicate_of": r.get("duplicate_of", "")} for r in self.rows]
        run = {"status": "complete", "images_checked": len(self.rows), "images_expected": len(self.rows),
               "input_sha256": {"previous_images.csv": self.policy["previous_images_sha256"]},
               "code_sha256": self.policy["audit_code_sha256"], "root": "/synthetic/raw", "limitations": ["Synthetic fixture"]}
        files = {"run.json": split.json_bytes(run),
                 "summary.json": split.json_bytes({"screen_complete": True, "images_checked": len(self.rows), "candidate_pairs": len(self.edges)}),
                 "validated_annotations.jsonl": b"".join((json.dumps(r) + "\n").encode() for r in self.rows),
                 "source_clusters.csv": self.csv_bytes(clusters),
                 "candidate_pairs.jsonl": b"".join((json.dumps({"a": self.names[a], "b": self.names[b], "distance": 1}) + "\n").encode()
                                                    for a, b in self.edges),
                 "previous_split_conflicts.jsonl": (json.dumps({"source_cluster_id": self.names["a_train"]}) + "\n").encode()}
        files["checksums.json"] = split.json_bytes({n: split.sha(b) for n, b in files.items()})
        with tarfile.open(self.bundle, "w:gz") as archive:
            for name, data in files.items():
                info = tarfile.TarInfo("audit/" + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        self.policy["audit_bundle_sha256"] = split.sha(self.bundle.read_bytes())
        self.policy_path.write_bytes(split.json_bytes(self.policy))

    def run_build(self, name="out"):
        with contextlib.redirect_stdout(io.StringIO()):
            return split.build(self.bundle, self.previous, self.policy_path, self.root / name)

    def tearDown(self):
        self.assertEqual(self.root.parent, self.parent)
        self.assertTrue(self.root.name.startswith("detection_splits_test_"))
        self.temp.cleanup()

    def test_exposure_groups_exclusions_and_small_targets(self):
        original = {p: split.sha(p.read_bytes()) for p in (self.previous, self.bundle, self.policy_path)}
        summary = self.run_build()
        records = {r["image"]: r for s in split.SPLITS for r in split.jsonl((self.root / "out" / (s + ".jsonl")).read_bytes())}
        self.assertEqual(len(records), 9)
        for k in ("a_train", "b_val_conflict", "e_background"):
            self.assertEqual(records[self.names[k]]["split"], "train")
        for k in ("d_test", "f_test_background"):
            self.assertEqual(records[self.names[k]]["split"], "test")
        self.assertEqual(records[self.names["c_val"]]["split"], "val")
        self.assertEqual(records[self.names["e_background"]]["targets"], [])
        self.assertEqual(len(records[self.names["k_zero"]]["targets"]), 1)
        target = records[self.names["k_zero"]]["targets"][0]
        self.assertEqual(target["box_xyxy"], [1., 1., 1.25, 1.5])
        self.assertEqual(target["difficult"], "1")
        self.assertEqual(target["class_index"], 0)
        self.assertEqual(summary["historical_test_images_preserved"], 1)
        self.assertEqual(original, {p: split.sha(p.read_bytes()) for p in original})
        self.assertNotIn(b"\r", (self.root / "out" / "SHA256SUMS").read_bytes())
        for name, expected in json.loads((self.root / "out" / "checksums.json").read_bytes()).items():
            self.assertEqual(split.sha((self.root / "out" / name).read_bytes()), expected)

    def test_reproducible_manifests(self):
        self.run_build("one")
        self.run_build("two")
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "images.csv", "summary.json"):
            self.assertEqual((self.root / "one" / name).read_bytes(), (self.root / "two" / name).read_bytes())

    def test_historical_test_bridge_rejected(self):
        for k in ("d_test", "f_test_background"):
            self.group[self.names[k]] = self.names["a_train"]
        self.edges.append(("a_train", "d_test"))
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "historical test"):
            self.run_build()
        self.assertFalse((self.root / "out").exists())

    def test_changed_audit_archive_rejected(self):
        with self.bundle.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "archive checksum"):
            self.run_build()
        self.assertFalse((self.root / "out").exists())

    def test_unreviewed_assignment_rejected(self):
        self.policy["expected_val_to_train_images"] = []
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Unreviewed historical"):
            self.run_build()

    def test_candidate_edge_crossing_group_rejected(self):
        self.edges.append(("a_train", "h_other"))
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Candidate edge crosses"):
            self.run_build()

    def test_positive_box_cannot_silently_be_dropped(self):
        self.rows[0]["targets"] = []
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Positive-area target omitted"):
            self.run_build()

    def test_existing_outputs_and_unreviewed_labels_rejected(self):
        self.run_build()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_build()
        self.policy["retained_non_target_labels"] = []
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Unreviewed annotation label"):
            self.run_build("other")


if __name__ == "__main__":
    unittest.main()
