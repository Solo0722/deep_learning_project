import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as b
import checkpoints
import local_augmentation as aug
import shared_augmentation as shared
from rdd_data import CLASSES, Example, prepare_example
from smoke_augmentation import forced_choice


class LocalAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((b.PACKAGE/'local_protocol.json').read_text())
        self.cfg['image_size'] = 32
        self.policy = copy.deepcopy(self.cfg['augmentation_policy'])
        self.policy.update(probability=1., warmup_epochs=0)

    def example(self, boxes, width=32, height=32, pixels=None):
        row = {'image': 'Fixture/train/images/geometry.png', 'split': 'train', 'width': width, 'height': height,
               'targets': [{'box_xyxy': box, 'object_index': i, 'class_index': i % 4, 'label': CLASSES[i % 4]}
                           for i, box in enumerate(boxes)]}
        if pixels is None:
            pixels = np.random.default_rng(99).integers(0, 256, (height, width, 3), dtype=np.uint8)
        original = Example(row, 9, pixels, np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                           np.arange(len(boxes), dtype=np.int64) % 4)
        return prepare_example(original, 32)

    def test_transitive_components_and_isolated_group(self):
        e = self.example([[1, 1, 8, 8], [7, 7, 12, 12], [11, 11, 14, 14], [22, 22, 29, 29]])
        self.assertEqual([g[0] for g in aug.groups(e.record, 32)], [(0, 1, 2), (3,)])
        reversed_row = copy.deepcopy(e.record)
        reversed_row['targets'].reverse()
        for t in reversed_row['targets']:
            t.update(class_index=3, label='D40')
        self.assertEqual(aug.groups(e.record, 32), aug.groups(reversed_row, 32))
        self.assertEqual(aug.decision(e.record, self.policy, 42, 7, 32), aug.decision(reversed_row, self.policy, 42, 7, 32))

    def test_raster_overlap_differs_from_edge_touch_and_continuous_gap(self):
        touching = self.example([[1, 1, 3, 3], [3, 1, 4, 3]])
        self.assertEqual(len(aug.groups(touching.record, 32)), 2)
        gap = self.example([[1.1, 1.1, 2.1, 2.1], [2.2, 1.1, 3.2, 2.1]])
        self.assertEqual([g[0] for g in aug.groups(gap.record, 32)], [(0, 1)])
        thin = self.example([[4.01, 4.01, 4.02, 4.02]])
        self.assertEqual(aug.groups(thin.record, 32)[0][2:], (1, 1))

    def test_exact_taper_and_union_hole(self):
        bounds, weights = aug.union_weights(((2, 2, 7, 7),))
        expected = np.ones((5, 5), dtype=np.uint8)
        expected[1:4, 1:4] = 2
        self.assertEqual(bounds, (2, 2, 7, 7))
        np.testing.assert_array_equal(weights, expected)
        bounds, weights = aug.union_weights(((2, 2, 8, 4), (2, 2, 4, 8)))
        self.assertEqual(int(np.count_nonzero(weights)), 20)
        np.testing.assert_array_equal(weights[2:, 2:], 0)
        # A diagonal neighbour is outside this L-shaped union, so the corner is tapered.
        self.assertEqual(int(weights[1, 1]), 1)

    def test_overlap_is_composited_once_and_other_group_untouched(self):
        e = self.example([[2, 2, 12, 12], [7, 7, 17, 17], [22, 22, 29, 29]], pixels=np.full((32, 32, 3), 100, dtype=np.uint8))
        choice = forced_choice(e, self.cfg, 'brightness_up', 0)
        result = aug.apply_example(e, choice)
        np.testing.assert_array_equal(result.rgb[8, 8], [120]*3)  # Not 144 from two edits.
        np.testing.assert_array_equal(result.rgb[2, 2], [110]*3)
        np.testing.assert_array_equal(result.rgb[22:29, 22:29], 100)
        self.assertEqual(choice['local']['union_pixels'], 175)
        np.testing.assert_array_equal(result.boxes, e.boxes)
        np.testing.assert_array_equal(result.labels, e.labels)
        self.assertEqual(result.record, e.record)
        np.testing.assert_array_equal(e.rgb, 100)

    def test_duplicate_boxes_do_not_double_area_or_edit(self):
        e = self.example([[2, 2, 7, 7], [2, 2, 7, 7]], pixels=np.full((32, 32, 3), 100, dtype=np.uint8))
        choice = forced_choice(e, self.cfg, 'brightness_up', 0)
        self.assertEqual(choice['local']['union_pixels'], 25)
        self.assertEqual(choice['local']['alpha_units'], 34)
        np.testing.assert_array_equal(aug.apply_example(e, choice).rgb[3:6, 3:6], 120)

    def test_all_operations_preserve_exact_outside_union_and_use_full_reference(self):
        e = self.example([[2, 2, 12, 7], [2, 2, 7, 9], [16, 2, 18, 7]], 19, 11)
        before = copy.deepcopy(e)
        mask = np.zeros((32, 32), dtype=bool)
        # Independent raster mask from the two selected prepared boxes.
        for box in e.boxes[:2]:
            x1, y1, x2, y2 = np.r_[np.floor(box[:2]), np.ceil(box[2:])].astype(int)
            mask[y1:y2, x1:x2] = True
        g = e.geometry
        x, y, w, h = g['pad_left'], g['pad_top'], g['resized_width'], g['resized_height']
        for name in aug.OPERATIONS:
            choice = forced_choice(e, self.cfg, name, 0)
            result = aug.apply_example(e, choice)
            np.testing.assert_array_equal(result.rgb[~mask], before.rgb[~mask])
            np.testing.assert_array_equal(result.boxes, before.boxes)
            np.testing.assert_array_equal(result.labels, before.labels)
            self.assertEqual(result.record, before.record)
            self.assertEqual(result.geometry, before.geometry)
            candidate = shared.transform_content(e.rgb[y:y+h, x:x+w], name, choice['parameters'], choice['noise_seed'])
            # This location is fully inside the union. Shared blur/contrast/noise context must agree exactly.
            np.testing.assert_array_equal(result.rgb[y+5, x+5], candidate[5, 5])
            self.assertTrue(result.augmentation_changed)
            self.assertEqual(result.augmentation_pixel_stats['changed_pixels'], int(np.any(result.rgb != e.rgb, axis=2).sum()))
        np.testing.assert_array_equal(e.rgb, before.rgb)

    def test_rounded_axis_geometry_and_fractional_edge_support(self):
        e = self.example([[18.9, 10.9, 19., 11.]], 19, 11)
        self.assertEqual(aug.prepared_geometry(e.record, 32), e.geometry)
        np.testing.assert_array_equal(aug.prepared_boxes(e.record, 32), e.boxes)
        self.assertEqual(aug.raster_rectangles(e.record, 32), ((0, 31, 24, 32, 25),))
        self.assertEqual(aug.groups(e.record, 32)[0][2:], (1, 1))

    def test_blur_reads_full_image_context_but_writes_only_local_pixel(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[15, 14] = 255
        e = self.example([[15., 15., 16., 16.]], pixels=rgb)
        result = aug.apply_example(e, forced_choice(e, self.cfg, 'gaussian_blur', 0))
        self.assertGreater(int(result.rgb[15, 15, 0]), 0)
        self.assertEqual(np.count_nonzero(np.any(result.rgb != e.rgb, axis=2)), 1)
        np.testing.assert_array_equal(result.rgb[15, 14], 255)

    def test_local_decisions_preserve_appearance_draws_and_global_rng(self):
        e = self.example([[2, 2, 7, 7], [12, 12, 18, 18], [23, 23, 29, 29]])
        before = checkpoints.rng_state()
        selected_groups = []
        for epoch in range(600):
            choice = aug.decision(e.record, self.policy, 42, epoch, 32)
            self.assertEqual({k: v for k, v in choice.items() if k != 'local'}, shared.decision(e.record, self.policy, 42, epoch))
            selected_groups.append(choice['local']['group_index'])
            aug.apply_example(e, choice)
        counts = np.bincount(selected_groups, minlength=3)
        self.assertTrue(((counts > 140) & (counts < 260)).all(), counts)
        checkpoints.assert_same(before, checkpoints.rng_state())

    def test_identity_warmup_negatives_and_validation_have_no_group(self):
        e = self.example([[2, 2, 7, 7]])
        for row, policy, epoch in [(e.record, None, 3), (e.record, {**self.policy, 'warmup_epochs': 1}, 0),
                                    (e.record, {**self.policy, 'probability': 0.}, 2),
                                    ({**e.record, 'split': 'val'}, self.policy, 3),
                                    ({**e.record, 'targets': []}, self.policy, 3)]:
            choice = aug.decision(row, policy, 42, epoch, 32)
            self.assertIsNone(choice['local'])
            self.assertEqual(choice['operation'], 'identity')
        for bad in (self.example([]), copy.deepcopy(e)):
            if bad.record['targets']:
                bad.record['split'] = 'val'
            with self.assertRaises(ValueError):
                forced_choice(bad, self.cfg, 'brightness_up', 0)

    def test_changed_local_metadata_prepared_geometry_or_boxes_rejected(self):
        e = self.example([[2, 2, 7, 7]])
        choice = forced_choice(e, self.cfg, 'brightness_up', 0)
        for key, value in [('union_pixels', 999), ('taper_pixels', 3), ('rectangles', [[1, 1, 8, 8]]), ('group_index', -1)]:
            changed = copy.deepcopy(choice)
            changed['local'][key] = value
            with self.assertRaises(ValueError):
                aug.apply_example(e, changed)
        bad = copy.deepcopy(e)
        bad.boxes[0, 0] += .25
        with self.assertRaises(ValueError):
            aug.apply_example(bad, choice)
        bad = copy.deepcopy(e)
        bad.geometry['pad_top'] += 1
        with self.assertRaises(ValueError):
            aug.apply_example(bad, choice)

    def test_invalid_boxes_and_duplicate_indices_rejected(self):
        e = self.example([[2, 2, 7, 7]])
        for box in ([2, 2, 2, 7], [2, 2, float('nan'), 7], [40, 40, 50, 50]):
            row = copy.deepcopy(e.record)
            row['targets'][0]['box_xyxy'] = box
            with self.assertRaises(ValueError):
                aug.groups(row, 32)
        row = copy.deepcopy(e.record)
        row['targets'].append(copy.deepcopy(row['targets'][0]))
        with self.assertRaises(ValueError):
            aug.groups(row, 32)

    def test_trace_matches_shared_decisions_and_counts_but_records_local_area(self):
        e = self.example([[2, 2, 7, 7], [12, 12, 18, 18]])
        actual, expected = aug.empty_trace(), shared.empty_trace()
        for epoch in range(12):
            choice = aug.decision(e.record, self.policy, 42, epoch, 32)
            result = aug.apply_example(e, choice)
            aug.record_decision(actual, choice, result.augmentation_changed, result.augmentation_pixel_stats)
            shared.record_decision(expected, shared.decision(e.record, self.policy, 42, epoch))
        aug.verify_shared_exposure([{'epoch': 1, 'augmentation': actual}], {'epochs': [expected]})
        self.assertEqual(actual['counts']['selected_groups'], 12)
        self.assertGreater(actual['local_changed_pixels'], 0)
        self.assertLessEqual(actual['local_changed_pixels'], actual['counts']['local_union_pixels'])
        self.assertNotEqual(actual['decision_sha256'], actual['shared_decision_sha256'])
        expected['decision_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            aug.verify_shared_exposure([{'epoch': 1, 'augmentation': actual}], {'epochs': [expected]})

    def test_pairing_rejects_training_appearance_and_taper_changes(self):
        cfg = json.loads((b.PACKAGE/'local_protocol.json').read_text())
        identity = json.loads((b.PACKAGE/'identity_protocol.json').read_text())
        image_level = json.loads((b.PACKAGE/'shared_protocol.json').read_text())
        aug.verify_pairing(cfg, identity, image_level)
        for section, key, value in [(None, 'epochs', 21), ('augmentation_policy', 'probability', .6),
                                    ('augmentation_policy', 'version', 'new_random_draws'), ('local_policy', 'taper_pixels', 3)]:
            changed = copy.deepcopy(cfg)
            (changed if section is None else changed[section])[key] = value
            with self.assertRaises(ValueError):
                aug.verify_pairing(changed, identity, image_level)


if __name__ == '__main__':
    unittest.main()
