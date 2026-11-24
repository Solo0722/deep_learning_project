import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import diagnostic as d
import geometry
import local_augmentation as local
import rdd_data
import run_check


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"protocol.json").read_text())


def row(number=0, classes=(0, 1), cluster=None):
    boxes = ([10., 20., 110., 120.], [80., 70., 170., 160.], [350., 350., 400., 380.], [200., 200., 210., 260.])
    return {"image": f"source/train/images/{number}.jpg", "source": "source", "split": "train",
            "source_cluster_id": str(number) if cluster is None else cluster,
            "width": 640, "height": 640, "other_annotations": [],
            "targets": [{"object_index": i, "class_index": c, "label": rdd_data.CLASSES[c],
                         "box_xyxy": list(boxes[i % 4]), "difficult": 0, "truncated": 0} for i, c in enumerate(classes)]}


def example(r):
    rgb = np.random.default_rng(7).integers(10, 230, (640, 640, 3), dtype=np.uint8)
    ex = rdd_data.Example(r, 0, rgb, local.prepared_boxes(r, 640),
                         np.array([t["class_index"] for t in r["targets"]]), local.prepared_geometry(r, 640))
    return ex


def selection(r):
    group = local.groups(r, 640)[0]
    return {"image": r["image"], "group_index": 0, "object_indices": list(group[0])}


class PanelTests(unittest.TestCase):
    def test_balanced_disjoint_clusters_and_order_independence(self):
        cfg = config()
        cfg["panels"]["primary_groups_per_class"] = 3
        rows = [row(100*c+i, (c,), cluster=f"{c}-{i//2}") for c in range(4) for i in range(20)]
        a, b = d.plan_panels(rows, cfg), d.plan_panels(list(reversed(rows)), cfg)
        self.assertEqual(a, b)
        all_records = sum(a["panels"].values(), [])
        self.assertEqual(len(all_records), 24)
        self.assertEqual(len({r["source_cluster_id"] for r in all_records}), 24)
        for name in ("A", "B"):
            self.assertEqual([sum(p["anchor_class"] == c for p in a["panels"][name]) for c in range(4)], [3]*4)

    def test_insufficient_support_population_and_heldout_rejected(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            d.plan_panels([row()], config())
        bad = row()
        bad["split"] = "test"
        with self.assertRaisesRegex(ValueError, "Non-training"):
            d.plan_panels([bad], config())


class RendererTests(unittest.TestCase):
    def test_paired_geometry_all_actions_preserve_metadata_and_reference(self):
        cfg, r = config(), row()
        ex = example(r)
        before = ex.rgb.copy()
        original = copy.deepcopy(r)
        views = list(d.action_views(ex, selection(r), cfg))
        self.assertEqual(len(views), 29)
        self.assertEqual(len({v[0] for v in views}), 29)
        reference = views[1][2]
        for key, pixels, meta in views[2:]:
            self.assertEqual(meta["boxes"], reference["boxes"])
            self.assertEqual(meta["labels"], reference["labels"])
            self.assertEqual(pixels.shape, before.shape)
        self.assertEqual(r, original)
        self.assertTrue(np.array_equal(ex.rgb, before))

    def test_identity_geometry_exactly_reuses_v1_local_renderer(self):
        cfg, r = config(), row()
        cfg["geometry"].update(horizontal_flip=[False], resized_canvas_sizes=[640], translations_pixels=[0], bank_size=1)
        ex, chosen = example(r), selection(r)
        for key, pixels, meta in d.action_views(ex, chosen, cfg):
            if key in ("clean", "reference"):
                self.assertTrue(np.array_equal(pixels, ex.rgb))
            else:
                legacy = local.apply_example(ex, meta["decision"])
                self.assertTrue(np.array_equal(pixels, legacy.rgb))
                self.assertEqual(meta["boxes"], ex.boxes.tolist())

    def test_rejected_geometry_preserves_appearance_and_all_annotations(self):
        cfg, r = config(), row()
        r["other_annotations"] = [{"index": 9, "label": "D50", "box": [0., 0., 3., 3.]}]
        cfg["geometry"].update(horizontal_flip=[False], resized_canvas_sizes=[672], translations_pixels=[0], bank_size=1)
        ex = example(r)
        views = list(d.action_views(ex, selection(r), cfg))
        self.assertFalse(views[1][2]["accepted"])
        self.assertTrue(np.array_equal(views[1][1], ex.rgb))
        self.assertFalse(np.array_equal(views[2][1], ex.rgb))
        self.assertEqual(len(views[1][2]["annotations"]), 3)

    def test_noise_repeats_deterministically_without_global_rng_change(self):
        cfg, r = config(), row()
        before = np.random.get_state()
        def noise():
            return [(k, rdd_data.sha256(p.tobytes())) for k, p, m in d.action_views(example(r), selection(r), cfg)
                    if k.startswith("gaussian_noise")]
        a, b = noise(), noise()
        self.assertEqual(a, b)
        self.assertNotEqual(a[0][1], a[1][1])
        after = np.random.get_state()
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])


class MeasurementTests(unittest.TestCase):
    def test_matching_one_to_one_class_correct_and_confidence_ordered(self):
        settings = config()["controller"]
        p = {"boxes": [[1, 1, 20, 20]]*4, "scores": [.9, .8, .7, .1], "labels": [0, 0, 1, 2]}
        result = d.match_prediction(p, [[1, 1, 20, 20]]*3, [0, 1, 2], settings)
        self.assertEqual(result["detected"], [True, True, False])
        self.assertEqual(result["false_positives"], 1)

    def test_empty_and_invalid_predictions(self):
        cfg = config()["controller"]
        result = d.match_prediction({"boxes": [], "scores": [], "labels": []}, [[1, 1, 2, 2]], [3], cfg)
        self.assertEqual(result["detected"], [False])
        for labels, score, box in (([.5], .8, [1, 1, 2, 2]), ([1], float("nan"), [1, 1, 2, 2]), ([1], .8, [3, 3, 2, 2])):
            with self.assertRaises(ValueError):
                d.match_prediction({"boxes": [box], "scores": [score], "labels": labels}, [[1, 1, 2, 2]], [1], cfg)

    def test_repeat_parity_detects_count_and_coordinate_changes(self):
        p = [{"boxes": [[1., 2., 3., 4.]], "scores": [.5], "labels": [0]}]
        self.assertEqual(d.check_prediction_parity(p, p, 1e-5), 0.)
        q = copy.deepcopy(p)
        q[0]["boxes"][0][0] += .01
        with self.assertRaises(ValueError):
            d.check_prediction_parity(p, q, 1e-5)

    def test_noise_draws_are_averaged_not_extra_support(self):
        r = row(classes=(3,))
        views = {f"{op}/{level}/{draw}": {"detected": [draw == 0]} for op, level in controller.ACTIONS
                 for draw in range(2 if op == "gaussian_noise" else 1)}
        p = {"image": r["image"], "source_cluster_id": r["source_cluster_id"], "classes": [3],
             "object_indices": [0], "target_positions": [0], "reference": {"detected": [True]}, "views": views}
        units = controller.units_from_panel([p], {r["image"]: r})
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["retained"][-3:], [.5]*3)
        self.assertEqual(controller.class_stats(units, config()["controller"])["detected_objects"], 1)

    def test_support_failure_does_not_become_action_agreement(self):
        cfg, r = config(), row(classes=(3,))
        p = {"image": r["image"], "source_cluster_id": r["source_cluster_id"], "classes": [3], "object_indices": [0],
             "target_positions": [0], "clean": {"detected": [True]}, "reference": {"detected": [False]},
             "geometry": {"reason": "accepted"},
             "views": {f"{op}/{level}/{draw}": {"detected": [False]} for op, level in controller.ACTIONS
                       for draw in range(2 if op == "gaussian_noise" else 1)}}
        other = row(number=1, classes=(3,))
        q = copy.deepcopy(p)
        q.update(image=other["image"], source_cluster_id=other["source_cluster_id"])
        result = d.analyze({"A": [p], "B": [q]}, [r, other], cfg)
        self.assertFalse(result["support_in_both_panels"])
        self.assertIsNone(result["between_panels"]["3"]["action_agreement_if_supported"])
        self.assertEqual(result["panels"]["A"]["geometry_recall_diagnostic"]["3"]["clean_found_geometry_missed"], 1)
        with self.assertRaisesRegex(ValueError, "reused across panels"):
            d.analyze({"A": [p], "B": [p]}, [r], cfg)

    def test_shared_collapse_and_balanced_strength_not_maximum(self):
        units = [{"image": str(i), "object_index": 0, "cluster": str(i), "source": "x", "class": c,
                  "class_set": (c,), "reference_detected": True, "retained": [1.]*24}
                 for c in range(4) for i in range(16)]
        policy = controller.build_policy(units, config()["controller"])
        self.assertEqual(policy["routed"]["3"]["distribution"], policy["shared_distribution"])
        self.assertTrue(all(p > 0 for p in policy["shared_distribution"][:-1]))
        self.assertEqual(policy["shared_distribution"][0:3], [1/36]*3)


class OutputTests(unittest.TestCase):
    def test_collection_budget_pairing_and_saved_observations(self):
        import time
        cfg = config()
        rows = [row(0, (0,)), row(1, (3,))]
        plan = {"panels": {name: [{**selection(r), "panel": name, "anchor_class": r["targets"][0]["class_index"],
                                  "target_positions": [0]}] for name, r in zip(("A", "B"), rows)}}
        class FakeDetector:
            images_forwarded = 0
            def predict(self, images):
                self.images_forwarded += len(images)
                return [{"boxes": [], "scores": [], "labels": []} for _ in images]
        detector = FakeDetector()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/"observations").mkdir()
            (out/"previews").mkdir()
            with patch.object(rdd_data, "load_example", side_effect=lambda root, r: example(r)), \
                 patch.object(rdd_data, "prepare_example", side_effect=lambda ex, size: ex):
                records, previews, parity = run_check.collect(detector, out, rows, plan, cfg, out, time.monotonic()+90)
            self.assertEqual(detector.images_forwarded, 2*29+4)
            self.assertEqual(len(previews), 2)
            self.assertEqual(parity["max_abs_error"], 0.)
            for name in ("A", "B"):
                saved = json.loads((out/"observations"/f"{name}_001.json").read_text())
                self.assertEqual(saved, records[name][0])
                self.assertEqual(len(saved["views"]), 27)
                controller.units_from_panel([saved], {r["image"]: r for r in rows})
            self.assertFalse(d.analyze(records, rows, cfg)["support_in_both_panels"])

    def test_budget_stop_leaves_no_complete_archive(self):
        cfg, r = config(), row()
        p = {**selection(r), "panel": "A", "anchor_class": 0, "target_positions": [0, 1]}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                run_check.collect(None, out, [r], {"panels": {"A": [p], "B": []}}, cfg, out, -1)
            self.assertEqual(list(out.iterdir()), [])

    def test_output_overlap_existing_and_archive_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root/"inputs"
            inputs.mkdir()
            for bad in (inputs, inputs/"nested", root):
                with self.assertRaises(ValueError):
                    run_check.output_paths(bad, [inputs])
            out, archive = run_check.output_paths(root/"out", [inputs])
            out.mkdir()
            d.write_json(out/"x.json", {"value": 1})
            checksum = run_check.archive_results(out, archive)
            with self.assertRaises(ValueError):
                run_check.archive_results(out, archive)
            self.assertEqual(d.file_hash(archive), checksum)

    def test_package_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"code.py").write_text("x=1\n")
            (root/"SHA256SUMS").write_text(d.file_hash(root/"code.py")+"  code.py\n")
            d.verify_package(root)
            (root/"code.py").write_text("x=2\n")
            with self.assertRaises(ValueError):
                d.verify_package(root)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch absent locally; execute on CCAST")
class ObserverTests(unittest.TestCase):
    def test_frozen_context_accepts_no_change_and_rejects_mutation(self):
        import torch
        from detector import FrozenDetector
        observer = FrozenDetector.__new__(FrozenDetector)
        observer.model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        with observer.frozen():
            observer.model(torch.ones(1, 2))
        self.assertTrue(observer.audit["unchanged"])
        with self.assertRaisesRegex(ValueError, "state mutated"):
            with observer.frozen():
                observer.model.weight.add_(1)

    def test_frozen_context_rejects_global_rng_change(self):
        import torch
        from detector import FrozenDetector
        observer = FrozenDetector.__new__(FrozenDetector)
        observer.model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "RNG changed"):
            with observer.frozen():
                torch.rand(1)


if __name__ == "__main__":
    unittest.main()
