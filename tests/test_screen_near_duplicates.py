import contextlib
import io
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from screen_near_duplicates import HashIndex, screen
from validate_images import audit


class HashIndexTests(unittest.TestCase):
    def test_radius_search_matches_exhaustive_comparison(self):
        rng = random.Random(41)
        values = [rng.getrandbits(128) for _ in range(80)]
        values.extend([values[0], values[0] ^ 3, values[1] ^ 255])
        tree = HashIndex()
        for index, value in enumerate(values):
            tree.add(value, index)
        for query in values[::7] + [values[0], values[1]]:
            for radius in (0, 2, 8):
                expected = []
                for index, value in enumerate(values):
                    distance = sum(a != b for a, b in zip(format(query, '0128b'), format(value, '0128b')))
                    if distance <= radius:
                        expected.append((distance, index))
                self.assertEqual(sorted(tree.query(query, radius)), sorted(expected))


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / 'tmp'
        self.scratch.mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix='near_duplicate_test_', dir=str(self.scratch)))
        self.raw = self.work / 'raw'
        for source, name, gray, has_target in [('A', 'one', 40, True), ('A', 'two', 41, True),
                                              ('A', 'three', 42, True), ('B', 'copy', 40, True),
                                              ('C', 'negative', 43, False)]:
            folder = self.raw / source / 'train'
            images, xmls = folder / 'images', folder / 'annotations/xmls'
            images.mkdir(parents=True, exist_ok=True)
            xmls.mkdir(parents=True, exist_ok=True)
            with Image.new('RGB', (16, 16), (gray, gray, gray)) as pic:
                pic.save(images / (name + '.png'))
            obj = ('<object><name>D00</name><bndbox><xmin>1</xmin><ymin>1</ymin>'
                   '<xmax>14</xmax><ymax>14</ymax></bndbox></object>') if has_target else ''
            (xmls / (name + '.xml')).write_text('<annotation><size><width>16</width>'
                                              '<height>16</height></size>' + obj + '</annotation>')
        with contextlib.redirect_stdout(io.StringIO()):
            audit(self.raw, self.work / 'audit')
        self.inventory = self.work / 'audit/inventory.jsonl'

    def tearDown(self):
        target = self.work.resolve()
        target.relative_to(self.scratch.resolve())
        if target == self.scratch.resolve() or not target.name.startswith('near_duplicate_test_'):
            raise RuntimeError('Unexpected fixture cleanup path')
        shutil.rmtree(target)

    def test_excludes_negatives_and_exact_copies_but_keeps_hash_collisions_as_candidates(self):
        # Different flat images deliberately collide: candidates must not be called duplicates.
        with contextlib.redirect_stdout(io.StringIO()):
            result = screen(self.raw, self.inventory, self.work / 'review')
        self.assertTrue(result['screen_complete'])
        self.assertEqual(result['counts']['images_without_target_objects'], 1)
        self.assertEqual(result['counts']['exact_duplicate_copies_excluded'], 1)
        self.assertEqual(result['images_screened'], 3)
        self.assertEqual(result['candidate_pairs'], 3)
        self.assertEqual(result['excluded_exact_copies'][0]['duplicate_of'], 'A/train/images/one.png')
        self.assertTrue((self.work / 'review.tar.gz').is_file())
        self.assertTrue((self.work / 'review/candidates_01.jpg').is_file())
        lines = (self.work / 'review/annotations.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 5)
        # Stored object indices and boxes survive serialization for subsequent split preparation.
        record = next(json.loads(line) for line in lines if 'A/train/images/one.png' in line)
        self.assertEqual(record['objects'][0]['box'], [1, 1, 14, 14])

    def test_pair_limit_never_reports_a_complete_screen(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = screen(self.raw, self.inventory, self.work / 'capped', max_pairs=1)
        self.assertFalse(result['screen_complete'])
        self.assertTrue(result['candidate_limit_reached'])
        self.assertEqual(result['candidate_pairs'], 1)
        meta = json.loads((self.work / 'capped/run.json').read_text())
        self.assertEqual(meta['status'], 'incomplete_candidate_limit')

    def test_changed_image_is_rejected(self):
        with Image.new('RGB', (16, 16), 'red') as picture:
            picture.save(self.raw / 'A/train/images/one.png')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Image changed'):
            screen(self.raw, self.inventory, self.work / 'changed')


if __name__ == '__main__':
    unittest.main()
