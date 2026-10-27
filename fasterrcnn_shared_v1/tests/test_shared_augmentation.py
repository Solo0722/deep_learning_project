import copy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as b
import checkpoints
import shared_augmentation as aug
from rdd_data import Example, prepare_example, sha256
from smoke_augmentation import check_operations
import test_baseline as fixtures


class PixelDetector(fixtures.ToyDetector):
    def forward(self, images, targets=None):
        if self.training:
            pixel_mean = torch.stack([image.mean() for image in images]).mean()
            jitter = torch.rand(()) + random.random() + float(np.random.rand())
            return {"loss_classifier": (self.weight-pixel_mean-.1*jitter).square() + .01}
        return super().forward(images, targets)


class SharedAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BaselineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = copy.deepcopy(self.fixture.cfg)
        self.policy = self.cfg["augmentation_policy"]
        self.policy.update(probability=1., warmup_epochs=0)

    def forced(self, example, operation):
        params = self.policy["operations"][operation]
        return {"image": example.record["image"], "epoch": 1, "target_bearing": True,
                "operation": operation, "family": params["family"], "reason": "applied",
                "parameters": {k: v for k, v in params.items() if k != "family"},
                "noise_seed": 42 if operation == "gaussian_noise" else None}

    def test_known_photometric_values_and_clipping(self):
        pixels = np.repeat(np.array([[[0], [128], [255]]], dtype=np.uint8), 3, axis=2)
        result = aug.transform_content(pixels, "brightness_up", {"factor": 1.2})
        np.testing.assert_array_equal(result[0, :, 0], [0, 154, 255])
        result = aug.transform_content(pixels, "gamma_darken", {"gamma": 2.})
        np.testing.assert_array_equal(result[0, :, 0], [0, 64, 255])
        result = aug.transform_content(pixels, "gamma_brighten", {"gamma": .5})
        np.testing.assert_array_equal(result[0, :, 0], [0, 181, 255])
        extremes = pixels[:, [0, 2]]
        reduced = aug.transform_content(extremes, "contrast_down", {"factor": .8})
        np.testing.assert_allclose(reduced[0, :, 0], [26, 230], atol=1)
        increased = aug.transform_content(extremes, "contrast_up", {"factor": 1.2})
        np.testing.assert_array_equal(increased, extremes)

    def test_blur_constant_and_impulse(self):
        constant = np.full((25, 25, 3), 128, dtype=np.uint8)
        np.testing.assert_array_equal(aug.transform_content(constant, "gaussian_blur", {"sigma_pixels": .8}), constant)
        impulse = np.zeros_like(constant)
        impulse[12, 12] = 255
        result = aug.transform_content(impulse, "gaussian_blur", {"sigma_pixels": .8})
        self.assertTrue(0 < result[12, 12, 0] < 255)
        self.assertGreater(result[12, 11, 0], 0)
        np.testing.assert_array_equal(result, result[::-1, ::-1])

    def test_noise_scale_replay_and_explicit_seed(self):
        image = np.full((256, 256, 3), 128, dtype=np.uint8)
        first = aug.transform_content(image, "gaussian_noise", {"sigma_unit": .02}, 42)
        repeated = aug.transform_content(image, "gaussian_noise", {"sigma_unit": .02}, 42)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, aug.transform_content(image, "gaussian_noise", {"sigma_unit": .02}, 43)))
        difference = first.astype(float)-128
        self.assertLess(abs(difference.mean()), .1)
        self.assertAlmostEqual(difference.std(), 5.1, delta=.15)
        with self.assertRaises(ValueError):
            aug.transform_content(image, "gaussian_noise", {"sigma_unit": .02})

    def test_all_operations_preserve_targets_padding_and_reference(self):
        row = copy.deepcopy(self.fixture.train_rows[0])
        row.update(width=19, height=11)
        row["targets"][0]["box_xyxy"] = [3., 2., 10., 8.]
        rgb = np.arange(19*11*3, dtype=np.uint16).reshape(11, 19, 3).astype(np.uint8)
        example = prepare_example(Example(row, 0, rgb, np.array([[3., 2., 10., 8.]]), np.array([0])), 32)
        before = copy.deepcopy(example)
        mask = np.zeros((32, 32), dtype=bool)
        g = example.geometry
        mask[g["pad_top"]:g["pad_top"]+g["resized_height"], g["pad_left"]:g["pad_left"]+g["resized_width"]] = True
        for operation in aug.OPERATIONS:
            after = aug.apply_example(example, self.forced(example, operation))
            np.testing.assert_array_equal(after.boxes, before.boxes)
            np.testing.assert_array_equal(after.labels, before.labels)
            np.testing.assert_array_equal(after.rgb[~mask], before.rgb[~mask])
            np.testing.assert_array_equal(example.rgb, before.rgb)
            self.assertEqual(after.record, before.record)
            self.assertEqual(after.geometry, before.geometry)
            self.assertTrue(after.augmentation_changed)
        after = aug.apply_example(example, self.forced(example, "brightness_down"))
        # Shared application includes content outside the target box.
        self.assertFalse(np.array_equal(after.rgb[g['pad_top'], 1:10], before.rgb[g['pad_top'], 1:10]))

    def test_class_independent_decisions_and_distinct_epochs(self):
        row = copy.deepcopy(self.fixture.train_rows[0])
        original = aug.decision(row, self.policy, 42, 2)
        row['targets'][0].update(class_index=3, label='D40')
        self.assertEqual(original, aug.decision(row, self.policy, 42, 2))
        results = [aug.decision(row, self.policy, 42, epoch) for epoch in range(20)]
        self.assertGreater(len({r['operation'] for r in results}), 1)
        for result in results:
            self.assertEqual(result, aug.decision(row, self.policy, 42, result['epoch']-1))

    def test_probability_warmup_negatives_and_validation(self):
        f = self.fixture
        self.policy.update(probability=0.)
        self.assertEqual(aug.decision(f.train_rows[0], self.policy, 42, 2)['reason'], 'probability')
        self.policy.update(probability=1., warmup_epochs=1)
        with patch.object(aug, 'random_word', return_value=2**64-1):
            self.assertEqual(aug.decision(f.train_rows[0], self.policy, 42, 2)['reason'], 'applied')
        self.assertEqual(aug.decision(f.train_rows[0], self.policy, 42, 0)['reason'], 'warmup')
        self.assertEqual(aug.decision(f.train_rows[4], self.policy, 42, 2)['reason'], 'no_target')
        self.assertEqual(aug.decision(f.val_rows[0], self.policy, 42, 2)['reason'], 'validation')
        for dataset, indices, validation in [(f.train, [[4]], False), (f.val, [[0, 1, 2, 3, 4]], True)]:
            references = {i: dataset[i].rgb.copy() for i in indices[0]}
            _, _, examples = next(iter(b.loader(dataset, indices, self.cfg, 2, validation)))
            for index, example in zip(indices[0], examples):
                np.testing.assert_array_equal(example.rgb, references[index])
                self.assertFalse(example.augmentation_changed)
        negative = f.train[4]
        with self.assertRaises(ValueError):
            aug.apply_example(negative, self.forced(negative, 'brightness_up'))
        held_out = f.val[0]
        with self.assertRaises(ValueError):
            aug.apply_example(held_out, self.forced(held_out, 'brightness_up'))

    def test_augmentation_does_not_consume_global_rng(self):
        example = self.fixture.train[0]
        before = checkpoints.rng_state()
        for epoch in range(10):
            aug.apply_example(example, aug.decision(example.record, self.policy, 42, epoch))
        aug.apply_example(example, self.forced(example, 'gaussian_noise'))
        checkpoints.assert_same(before, checkpoints.rng_state())

    def test_worker_count_and_access_order_do_not_change_pixels_or_decisions(self):
        def collect(workers, order):
            cfg = {**self.cfg, 'workers': workers}
            dataset = b.Dataset(self.fixture.root, self.fixture.train_rows, 'train', 32)
            result = {}
            for _, _, examples in b.loader(dataset, [[i] for i in order], cfg, 3):
                for example in examples:
                    result[example.image_id] = (sha256(example.rgb.tobytes()), example.augmentation_decision)
            return result
        expected = collect(0, list(range(5)))
        self.assertEqual(expected, collect(0, list(reversed(range(5)))))
        self.assertEqual(expected, collect(2, [4, 2, 0, 3, 1]))

    def test_pixel_dependent_mid_epoch_resume_and_plan_digest(self):
        f = self.fixture
        self.policy['warmup_epochs'] = 1
        def fresh():
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            model = PixelDetector()
            return model, torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        whole, continued = f.output('aug_whole'), f.output('aug_resume')
        model, optimizer = fresh()
        complete = b.initial_progress()
        self.assertEqual(b.fit(model, optimizer, f.train, f.val, self.cfg, 'cpu', whole, f.context, complete), 'complete')
        expected_model, expected_optimizer, expected_rng = checkpoints.cpu_tree(model.state_dict()), checkpoints.cpu_tree(optimizer.state_dict()), checkpoints.rng_state()
        model, optimizer = fresh()
        progress = b.initial_progress()
        self.assertEqual(b.fit(model, optimizer, f.train, f.val, self.cfg, 'cpu', continued, f.context, progress, lambda: progress['step'] >= 4), 'budget_stopped')
        self.assertEqual(progress['epoch'], 1)
        self.assertEqual(progress['next_batch'], 1)
        model, optimizer = fresh()
        progress = b.restore(continued, model, optimizer, f.context)
        self.assertEqual(b.fit(model, optimizer, f.train, f.val, self.cfg, 'cpu', continued, f.context, progress), 'complete')
        checkpoints.assert_same(model.state_dict(), expected_model)
        checkpoints.assert_same(optimizer.state_dict(), expected_optimizer)
        checkpoints.assert_same(checkpoints.rng_state(), expected_rng)
        for actual, expected in zip(progress['history'], complete['history']):
            for key in ('training_loss', 'validation', 'augmentation'):
                self.assertEqual(actual[key], expected[key])
        self.assertEqual(progress['history'][0]['augmentation']['counts'].get('applied_images', 0), 0)
        self.assertEqual(progress['history'][1]['augmentation']['counts']['applied_images'], 4)

    def test_tampered_exposure_trace_is_rejected(self):
        f = self.fixture
        cfg = {**self.cfg, 'epochs': 1}
        original = b.Dataset.__getitem__
        def wrong(dataset, index):
            example = original(dataset, index)
            example.augmentation_decision['epoch'] = 999
            return example
        model, optimizer = f.fresh()
        with patch.object(b.Dataset, '__getitem__', wrong):
            with self.assertRaisesRegex(ValueError, 'exposure/order'):
                b.fit(model, optimizer, f.train, f.val, cfg, 'cpu', f.output('bad_trace'), f.context, b.initial_progress())

    def test_paired_recipe_guard_rejects_unrelated_changes(self):
        cfg = json.loads((b.PACKAGE/'protocol.json').read_text())
        reference = json.loads((b.PACKAGE/'identity_protocol.json').read_text())
        aug.verify_pairing(cfg, reference)
        for key, value in [('learning_rate', .02), ('epochs', 30), ('batch_size', 2), ('seed', 99)]:
            changed = copy.deepcopy(cfg); changed[key] = value
            with self.assertRaises(ValueError):
                aug.verify_pairing(changed, reference)
        for value in (float('nan'), -1., 2.):
            bad = copy.deepcopy(self.policy); bad['probability'] = value
            with self.assertRaises(ValueError):
                aug.validate_policy(bad)

    def test_all_operation_smoke_preview_records_eight_checks(self):
        before = checkpoints.rng_state()
        out = self.fixture.output('preview')
        report = check_operations(self.fixture.train, self.cfg, out)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(len(report['operations']), 8)
        self.assertTrue((out/'augmentation_preview.png').is_file())
        checkpoints.assert_same(before, checkpoints.rng_state())


if __name__ == '__main__':
    unittest.main()
