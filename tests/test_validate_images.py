"""Small synthetic audit fixtures; no real dataset is required."""
import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from validate_images import audit


class ImageAuditTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / "tmp"
        self.scratch.mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="image_audit_test_", dir=str(self.scratch)))
        self.raw = self.work / "raw"

    def tearDown(self):
        # Validate the generated deletion target stays inside the fixture workspace.
        target = self.work.resolve()
        target.relative_to(self.scratch.resolve())
        if target == self.scratch.resolve() or not target.name.startswith("image_audit_test_"):
            raise RuntimeError("Unexpected fixture cleanup target")
        shutil.rmtree(target)

    def add(self, source, name, data, width=8, height=6):
        train = self.raw / source / "train"
        images, xmls = train / "images", train / "annotations" / "xmls"
        images.mkdir(parents=True, exist_ok=True)
        xmls.mkdir(parents=True, exist_ok=True)
        path = images / name
        path.write_bytes(data)
        (xmls / (path.stem + ".xml")).write_text(
            "<annotation><size><width>%d</width><height>%d</height></size></annotation>"
            % (width, height), encoding="utf-8")
        return path

    @staticmethod
    def png(mode="RGB", color=(15, 80, 130), comment=None):
        buffer = io.BytesIO()
        info = PngImagePlugin.PngInfo()
        if comment:
            info.add_text("comment", comment)
        with Image.new(mode, (8, 6), color) as picture:
            picture.save(buffer, format="PNG", pnginfo=info)
        return buffer.getvalue()

    def test_corruption_dimensions_and_cross_source_duplicates_without_raw_changes(self):
        first = self.png()
        self.add("A", "one.png", first)
        self.add("B", "copy.PNG", first)
        self.add("B", "metadata.png", self.png(comment="different container metadata"))
        self.add("A", "wrong_size.png", self.png(color=(0, 0, 0)), width=9)
        self.add("B", "broken.jpg", b"not a jpeg")
        self.add("A", "gray.png", self.png(mode="L", color=128))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.raw.rglob("*") if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            summary = audit(self.raw, self.work / "report", progress_every=2)
        self.assertEqual(summary["images_checked"], 6)
        self.assertEqual(summary["decoded_images"], 5)
        self.assertEqual(summary["issue_counts_images"],
                         {"dimension_mismatch": 1, "image_error": 1, "non_rgb_mode": 1})
        self.assertEqual(sorted(map(len, summary["byte_identical_groups"])), [2])
        self.assertEqual(sorted(map(len, summary["rgb_identical_groups"])), [3])
        self.assertEqual({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in self.raw.rglob("*") if p.is_file()}, before)
        meta = json.loads((self.work / "report" / "run.json").read_text())
        self.assertEqual(meta["status"], "complete")
        records = [json.loads(line) for line in
                   (self.work / "report" / "inventory.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 6)
        broken = next(r for r in records if r["image"].endswith("broken.jpg"))
        self.assertNotIn("rgb_sha256", broken)
        self.assertIn("file_sha256", broken)

    def test_existing_reports_and_raw_directory_are_not_overwritten(self):
        self.add("A", "one.png", self.png())
        output = self.work / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            audit(self.raw, output)
        with self.assertRaises(ValueError):
            audit(self.raw, self.raw / "report")

    def test_no_sources_is_an_error(self):
        self.raw.mkdir()
        with self.assertRaises(ValueError):
            audit(self.raw, self.work / "report")


if __name__ == "__main__":
    unittest.main()
