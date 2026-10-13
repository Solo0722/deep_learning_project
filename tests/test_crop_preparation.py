import contextlib
import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from make_splits import allocate, crop_bounds
from materialize_crops import materialize


class SplitRuleTests(unittest.TestCase):
    def test_fractional_and_border_coordinates_and_degenerate_boxes(self):
        self.assertEqual(crop_bounds([0, 1.25, 10.75, 8], [12, 8]), (0, 1, 11, 8))
        self.assertIsNone(crop_bounds([198, 474, 198, 475], [600, 600]))
        self.assertIsNone(crop_bounds([1.2, 2, 2.9, 8], [12, 10]))
        with self.assertRaises(ValueError):
            crop_bounds([0, 0, 20, 10], [12, 10])

    def test_deterministic_group_allocation_preserves_class_coverage(self):
        vectors = {str(i): [1, 1, 1, 1, 1, 1, 1, 1, 1] for i in range(40)}
        vectors['large_related_group'] = [5, 10, 5, 5, 5, 5, 5, 5, 5]
        first = allocate(vectors, 42)
        self.assertEqual(first, allocate(dict(reversed(list(vectors.items()))), 42))
        self.assertEqual(set(first), set(vectors))
        self.assertEqual(set(first.values()), {'train', 'val', 'test'})
        for split in ('train', 'val', 'test'):
            self.assertTrue(all(sum(vectors[k][feature] for k in first if first[k] == split) > 0
                                for feature in range(1, 9)))


class CropWriterTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / 'tmp'
        self.scratch.mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix='crop_writer_test_', dir=str(self.scratch)))
        self.raw = self.work / 'raw'
        self.raw.mkdir()
        with Image.new('RGB', (64, 32), 'red') as image:
            image.paste('blue', (32, 0, 64, 32))
            image.save(self.raw / 'source.png')
        (self.raw / 'source.xml').write_text('<annotation/>')
        self.original = (self.raw / 'source.png').read_bytes()
        image_hash = hashlib.sha256(self.original).hexdigest()
        xml_hash = hashlib.sha256((self.raw / 'source.xml').read_bytes()).hexdigest()
        self.rows = []
        for index, (label, bounds) in enumerate([('D00', (0, 0, 64, 32)), ('D10', (32, 0, 64, 32))]):
            self.rows.append({'sample_id': 's' + str(index), 'split': 'train', 'source': 'Demo',
                              'image': 'source.png', 'xml': 'source.xml', 'object_index': index,
                              'label': label, 'label_id': index, 'group_id': 'group',
                              'left': bounds[0], 'top': bounds[1], 'right': bounds[2], 'bottom': bounds[3],
                              'original_width': 64, 'original_height': 32,
                              'file_sha256': image_hash, 'xml_sha256': xml_hash,
                              'crop_path': 'train/' + label + '/s' + str(index) + '.png'})
        self.manifest = self.work / 'crops.csv'
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        config = {'manifest_sha256': hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                  'labels': {'0': 'D00', '1': 'D10', '2': 'D20', '3': 'D40'},
                  'crop_size': 224, 'pixel_preprocessing': 'test letterbox'}
        (self.work / 'config.json').write_text(json.dumps(config))

    def tearDown(self):
        target = self.work.resolve()
        target.relative_to(self.scratch.resolve())
        if target == self.scratch.resolve() or not target.name.startswith('crop_writer_test_'):
            raise RuntimeError('Unexpected cleanup path')
        shutil.rmtree(target)

    def test_pixels_aspect_ratio_and_source_integrity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = materialize(self.raw, self.manifest, self.work / 'output')
        self.assertEqual(result['crops_written'], 2)
        self.assertEqual(self.original, (self.raw / 'source.png').read_bytes())
        with Image.open(self.work / 'output/train/D00/s0.png') as image:
            self.assertEqual(image.size, (224, 224))
            self.assertEqual(image.getpixel((20, 20)), (127, 127, 127))
            self.assertEqual(image.getpixel((20, 112)), (255, 0, 0))
            self.assertEqual(image.getpixel((200, 112)), (0, 0, 255))
        with Image.open(self.work / 'output/train/D10/s1.png') as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
        self.assertTrue((self.work / 'output_qa.tar.gz').is_file())
        with (self.work / 'output/qa/selected_samples.csv').open() as stream:
            selected = list(csv.DictReader(stream))
        self.assertTrue(all(r['split'] == 'train' for r in selected))

    def test_rejects_cross_split_image_and_modified_sources(self):
        self.rows[1]['split'] = 'test'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'source image spans'):
            materialize(self.raw, self.manifest, self.work / 'leaky')
        self.rows[1]['split'] = 'train'
        self.write_manifest()
        (self.raw / 'source.xml').write_text('<changed/>')
        with self.assertRaisesRegex(ValueError, 'Source XML changed'):
            materialize(self.raw, self.manifest, self.work / 'changed')


if __name__ == '__main__':
    unittest.main()
