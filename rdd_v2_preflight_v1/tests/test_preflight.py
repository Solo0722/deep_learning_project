import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller as c
import geometry as g
import geometry_check as gc
import rdd_data
import replay

CFG = json.loads((Path(__file__).resolve().parents[1]/"protocol.json").read_text())
SETTINGS = CFG["controller"]


def unit(i, cls=0, labels=None, retention=1., detected=True, cluster=None):
    return {"image": f"source/train/images/{i}.jpg", "object_index": i,
            "cluster": str(i) if cluster is None else cluster, "source": "source" if i % 2 else "other",
            "class": cls, "class_set": (cls,) if labels is None else labels,
            "reference_detected": detected, "retained": [retention]*len(c.ACTIONS)}


def row():
    return {"image": "S/train/images/test.png", "source": "S", "split": "train", "width": 640, "height": 640,
            "targets": [{"class_index": 0, "label": "D00", "object_index": 0, "box_xyxy": [10.25, 20., 30.75, 80.], "difficult": "1", "truncated": "1"}],
            "other_annotations": [{"index": 1, "label": "Repair", "box": [400., 300., 640., 320.], "difficult": "0", "truncated": "1"}]}


def candidate(flip=False, m=640, tx=0, ty=0):
    return next(v for v in g.bank(CFG["geometry"]) if (v["flip"], v["resized_size"], v["tx"], v["ty"]) == (flip, m, tx, ty))


class ControllerTests(unittest.TestCase):
    def test_minimum_support_counts_independent_clusters(self):
        units = [unit(i, cluster=str(i % 4)) for i in range(16)]
        self.assertFalse(c.class_stats(units, SETTINGS)["supported"])
        units = [unit(i, cluster=str(i % 8)) for i in range(16)]
        self.assertTrue(c.class_stats(units, SETTINGS)["supported"])

    def test_reference_misses_cannot_inflate_support(self):
        units = [unit(i, detected=(i < 15)) for i in range(100)]
        result = c.class_stats(units, SETTINGS)
        self.assertEqual(result["detected_objects"], 15)
        self.assertEqual(result["reference_recall"], .15)
        self.assertFalse(any(result["allowed"]))

    def test_each_strength_is_gated_independently(self):
        units = [unit(i) for i in range(16)]
        for u in units:
            u["retained"][:3] = [1., 0., 1.]
        allowed = c.class_stats(units, SETTINGS)["allowed"]
        self.assertEqual(allowed[:3], [True, False, True])
        probability = c.balanced_distribution(allowed)
        self.assertEqual(probability[0], probability[2])
        self.assertEqual(probability[1], 0)

    def test_balancing_does_not_weight_six_photo_operations_sixfold(self):
        probability = c.balanced_distribution([True]*24)
        self.assertAlmostEqual(sum(probability[:18]), .5)
        self.assertAlmostEqual(sum(probability[18:24]), .5)
        self.assertAlmostEqual(sum(probability), 1.)

    def test_no_support_routes_to_explicit_identity(self):
        state = c.build_policy([], SETTINGS)
        self.assertEqual(state["shared_distribution"][-1], 1.)
        self.assertTrue(all(v["distribution"][-1] == 1 for v in state["routed"].values()))

    def test_pooled_evidence_can_cross_mixtures_but_is_marked_borrowed(self):
        units = [unit(i, cls=0, labels=(0,) if i < 8 else (0, 1)) for i in range(16)]
        state = c.build_policy(units, SETTINGS)
        self.assertTrue(state["pooled"]["0"]["supported"])
        self.assertTrue(any(state["routed"]["0"]["allowed"]))
        self.assertTrue(state["routed"]["0"]["borrowed_evidence"])
        self.assertFalse(any(state["routed"]["0,1"]["allowed"]))

    def test_supported_mixed_failure_vetoes_pooled_success(self):
        units = [unit(i, cls=0) for i in range(100)]
        units += [unit(i+100, cls=1) for i in range(100)]
        units += [unit(i+200, cls=0, labels=(0, 1), retention=0.) for i in range(16)]
        state = c.build_policy(units, SETTINGS)
        self.assertTrue(all(state["pooled"]["0"]["allowed"]))
        self.assertFalse(any(state["routed"]["0,1"]["allowed"]))
        self.assertEqual(state["routed"]["0,1"]["vetoed_actions"], 24)

    def test_shared_policy_and_class_routing_differ_only_by_gate(self):
        units = [unit(i+cls*100, cls=cls) for cls in range(4) for i in range(20)]
        state = c.build_policy(units, SETTINGS)
        for route in state["routed"].values():
            self.assertEqual(route["distribution"], state["shared_distribution"])
        for u in units:
            if u["class"] == 3:
                u["retained"][0] = 0.
        state = c.build_policy(units, SETTINGS)
        self.assertFalse(state["shared_allowed"][0])
        self.assertTrue(state["routed"]["0"]["allowed"][0])
        self.assertFalse(state["routed"]["0,3"]["allowed"][0])

    def test_leave_one_cluster_exposes_support_boundary(self):
        units = [unit(i, cluster=str(i//2)) for i in range(16)]
        result = c.stability(units, SETTINGS, 42, 16)["0"]
        self.assertTrue(all(result["baseline"]["allowed"]))
        self.assertTrue(all(result["leave_one_cluster_changed_actions"]))
        self.assertEqual(result["conditional_bootstrap_retention_pass_frequency"], [1.]*24)
        self.assertIsNone(result["half_action_agreement"])

    def test_keyed_sampling_preserves_global_rng_and_is_reproducible(self):
        py_state, np_state = random.getstate(), np.random.get_state()
        a = c.draw_action([True]*24, 42, 3, "x", 0)
        self.assertEqual(a, c.draw_action([True]*24, 42, 3, "x", 0))
        self.assertEqual(py_state, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertIsNone(c.draw_action([False]*24, 42, 3, "x", 0))

    def test_probe_parser_noise_draws_count_one_object(self):
        r = row(); r["source"] = "S"; r["source_cluster_id"] = "cluster"
        p = {"image": r["image"], "classes": [0], "object_indices": [0], "source_cluster_id": "cluster",
             "target_positions": [0], "reference": {"detected": [True]}, "views": {}}
        for op, level in c.ACTIONS:
            for d in range(2 if op == "gaussian_noise" else 1):
                p["views"][f"{op}/{level}/{d}"] = {"detected": [d == 0]}
        units = c.units_from_panel([p], {r["image"]: r})
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["retained"][-1], .5)
        for mutation in ("duplicate", "missing_view", "held_out", "invalid_bool", "object_id"):
            q, r2 = copy.deepcopy(p), copy.deepcopy(r)
            panel = [q]
            if mutation == "duplicate": panel += [q]
            if mutation == "missing_view": q["views"].pop(next(iter(q["views"])))
            if mutation == "held_out": r2["split"] = "test"
            if mutation == "invalid_bool": q["reference"]["detected"] = [1]
            if mutation == "object_id": q["object_indices"] = [99]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c.units_from_panel(panel, {r2["image"]: r2})


class GeometryTests(unittest.TestCase):
    def test_flip_fractional_coordinates_and_non_target_boxes(self):
        annotations = g.prepared_annotations(row(), 640)
        mapped = g.forward_boxes([r["box_xyxy"] for r in annotations], candidate(flip=True), 640)
        np.testing.assert_array_equal(mapped[0], [609.25, 20., 629.75, 80.])
        np.testing.assert_array_equal(mapped[1], [0., 300., 240., 320.])

    def test_cropped_other_annotation_rejects_even_when_target_fits(self):
        annotations = g.prepared_annotations(row(), 640)
        result = g.evaluate(annotations, candidate(tx=16), 640)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["cropped_ids"], [1])
        rgb = np.zeros((640, 640, 3), dtype=np.uint8)
        pixels, output, _ = g.apply(rgb, annotations, candidate(tx=16))
        np.testing.assert_array_equal(pixels, rgb)
        self.assertEqual(output, annotations)

    def test_real_pixels_and_labels_follow_same_horizontal_flip(self):
        r = row(); r["targets"][0]["box_xyxy"] = [10., 20., 30., 80.]
        annotations = g.prepared_annotations(r, 640)
        rgb = np.zeros((640, 640, 3), dtype=np.uint8)
        rgb[20:80, 10:30] = 255
        pixels, output, result = g.apply(rgb, annotations, candidate(flip=True))
        self.assertTrue(result["accepted"])
        self.assertTrue((pixels[20:80, 610:630] == 255).all())
        self.assertEqual(output[0]["box_xyxy"], [610., 20., 630., 80.])
        self.assertEqual(output[0]["original"], annotations[0]["original"])
        np.testing.assert_array_equal(pixels, rgb[:, ::-1])

    def test_all_classes_and_flags_are_preserved_without_rotation(self):
        r = row(); r["other_annotations"] = []
        r["targets"] = [{**r["targets"][0], "object_index": i, "label": label, "class_index": i} for i, label in enumerate(rdd_data.CLASSES)]
        ann = g.prepared_annotations(r, 640)
        _, out, _ = g.apply(np.zeros((640, 640, 3), np.uint8), ann, candidate(flip=True))
        self.assertEqual([a["original"] for a in out], r["targets"])

    def test_uniform_scale_translation_has_known_edge_mapping(self):
        boxes = [[100., 200., 300., 400.]]
        v = candidate(m=608, tx=-16, ty=16)
        np.testing.assert_allclose(g.forward_boxes(boxes, v, 640), [[95., 222., 285., 412.]])
        np.testing.assert_allclose(g.inverse_boxes([[95., 222., 285., 412.]], v, 640), boxes)

    def test_fractional_subpixel_and_empty_images_survive(self):
        v = candidate(m=608)
        box = [[100.1, 200.1, 100.2, 200.2]]
        transformed = g.forward_boxes(box, v, 640)
        self.assertTrue((transformed[:, 2:] > transformed[:, :2]).all())
        pixels, out, result = g.apply(np.zeros((640, 640, 3), np.uint8), [], v)
        self.assertTrue(result["accepted"])
        self.assertEqual(out, [])
        self.assertTrue((pixels[:16] == 114).all())

    def test_invalid_other_annotation_is_not_silently_removed(self):
        r = row(); r["other_annotations"][0]["box"] = [1., 1., 0., 2.]
        with self.assertRaises(g.InvalidAnnotation):
            g.prepared_annotations(r, 640)

    def test_letterbox_actual_axis_scales_match_reader(self):
        r = row(); r["width"], r["height"] = 1001, 333; r["other_annotations"] = []
        ex = rdd_data.Example(r, 0, np.zeros((333, 1001, 3), np.uint8), np.array([r["targets"][0]["box_xyxy"]]), np.array([0]))
        prepared = rdd_data.prepare_example(ex)
        np.testing.assert_array_equal(prepared.boxes, [a["box_xyxy"] for a in g.prepared_annotations(r, 640)])

    def test_every_bank_candidate_inverts_and_sampling_is_train_only(self):
        boxes = [[220.1, 300.2, 420.3, 440.4]]
        self.assertEqual(len(g.bank(CFG["geometry"])), 54)
        for v in g.bank(CFG["geometry"]):
            np.testing.assert_allclose(g.inverse_boxes(g.forward_boxes(boxes, v, 640), v, 640), boxes, atol=1e-10)
        r = row()
        self.assertEqual(g.chosen_candidate(r, CFG["geometry"], 42, 1), g.chosen_candidate(r, CFG["geometry"], 42, 1))
        r["split"] = "val"
        with self.assertRaises(ValueError): g.chosen_candidate(r, CFG["geometry"], 42, 1)

    def test_pixel_runner_reads_hashes_and_records_54_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"raw"; out = Path(d)/"out"; out.mkdir()
            r = row(); r.update(source="S", annotation_category="target_bearing", exif_orientation=1, xml="S/train/annotations/xmls/test.xml", source_cluster_id="s")
            image_path = root/r["image"]; image_path.parent.mkdir(parents=True)
            xml_path = root/r["xml"]; xml_path.parent.mkdir(parents=True)
            pixels = np.zeros((640, 640, 3), np.uint8); pixels[20:80, 10:31] = 255
            Image.fromarray(pixels).save(image_path); xml_path.write_bytes(b"<annotation/>")
            r["image_sha256"] = gc.file_hash(image_path); r["xml_sha256"] = gc.file_hash(xml_path)
            r["rgb_sha256"] = rdd_data.sha256(b"RGB:640:640:"+pixels.tobytes())
            with patch.object(gc, "selection", return_value=[r]):
                result = gc.pixel_check(root, [r], CFG["geometry"], out)
            self.assertEqual(result["rendered_candidates"], 54)
            records = json.loads((out/"pixel_records.json").read_text())
            self.assertEqual(len(records[0]["candidates"]), 54)
            self.assertEqual(result["visual_review"], "pending")
            image_path.write_bytes(b"modified")
            with self.assertRaises(ValueError): rdd_data.load_example(root, r)


class OutputTests(unittest.TestCase):
    def small_run(self, root, out, fail=False):
        r = row()
        args = ["geometry_check.py", "--splits", str(root/"splits"), "--out", str(out), "--manifest-only"]
        tiny_cfg = copy.deepcopy(CFG); tiny_cfg["expected_train_images"] = 1
        real_loads = json.loads
        def substitute(data, *a, **kw):
            value = real_loads(data, *a, **kw)
            return tiny_cfg if isinstance(value, dict) and value.get("version") == CFG["version"] else value
        with patch.object(sys, "argv", args), patch.object(gc, "read_manifest", return_value=[r]), \
                patch.object(gc, "selection", return_value=[r]), patch.object(json, "loads", side_effect=substitute):
            if fail:
                with patch.object(gc, "census", side_effect=ValueError("injected annotation failure")):
                    gc.main()
            else:
                gc.main()

    def test_complete_archive_and_refusal_to_replace_prior_result(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); out = root/"results"
            self.small_run(root, out)
            self.assertTrue(out.with_suffix(".tar.gz").is_file())
            before = gc.file_hash(out/"result.json")
            with self.assertRaises(ValueError): self.small_run(root, out)
            self.assertEqual(before, gc.file_hash(out/"result.json"))
            self.assertEqual(json.loads((out/"result.json").read_text())["status"], "annotations_complete_pixels_not_checked")

    def test_failure_saved_without_success_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); out = root/"results"
            with self.assertRaises(ValueError): self.small_run(root, out, fail=True)
            self.assertTrue((out/"failure.json").is_file())
            self.assertFalse(out.with_suffix(".tar.gz").exists())

    def test_changed_manifest_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); splits = root/"splits"; splits.mkdir()
            (splits/"train.jsonl").write_text("{}\n")
            out = root/"results"
            with patch.object(sys, "argv", ["check", "--splits", str(splits), "--out", str(out), "--manifest-only"]):
                with self.assertRaises(ValueError): gc.main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
