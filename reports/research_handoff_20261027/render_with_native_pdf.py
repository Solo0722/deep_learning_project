"""Use the packaged renderer with a fresh native office PDF when LibreOffice is absent."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TMP = OUT / 'temp'
TMP.mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(TMP)
tempfile.tempdir = str(TMP)
skill = Path(r'C:\Users\SOLOMON\.codex\plugins\cache\openai-primary-runtime\documents\26.1020.11618\skills\documents\render_docx.py')
spec = importlib.util.spec_from_file_location('packaged_render_docx', skill)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
pdf = OUT / 'handoff_qa.pdf'
docx = ROOT / 'output/reports/RDD_Research_Handoff_2026-10-27.docx'
reader = PdfReader(pdf)
assert len(reader.pages) > 0

def already_exported(doc_path, user_profile, convert_tmp_dir, stem, verbose=False):
    assert Path(doc_path).resolve() == docx.resolve()
    assert pdf.stat().st_mtime >= docx.stat().st_mtime
    return str(pdf), 'Fresh native Word COM-compatible export; LibreOffice unavailable.'

renderer.convert_to_pdf = already_exported
original_replace = renderer.replace

def retry_replace(source, destination):
    for attempt in range(15):
        try:
            return original_replace(source, destination)
        except PermissionError:
            if attempt == 14:
                raise
            time.sleep(0.3)

renderer.replace = retry_replace
paths = renderer.rasterize(str(docx), str(OUT / 'rendered_release'), 150, False, False)
all_text = '\n\n'.join(page.extract_text() for page in reader.pages)
(OUT/'pdf_text.txt').write_text(all_text, encoding='utf-8')
stats = []
for i, path in enumerate(paths, 1):
    im = Image.open(path).convert('RGB')
    a = np.array(im)
    colored = int(((a[:, :, 0] != a[:, :, 1]) | (a[:, :, 1] != a[:, :, 2])).sum())
    page_text = reader.pages[i-1].extract_text()
    stats.append({'page': i, 'path': str(path), 'width': im.width, 'height': im.height,
                  'non_grayscale_pixels': colored, 'text_characters': len(page_text),
                  'opening': page_text[:180], 'ending': page_text[-180:]})
result = {'pages': len(paths), 'exporter': str(reader.metadata.get('/Creator')),
          'rasterizer': str(skill), 'dpi': 150, 'pages_info': stats, 'visual_inspection': 'pending'}
(OUT/'render_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'pages': len(paths), 'non_grayscale_pixels': sum(x['non_grayscale_pixels'] for x in stats),
                  'exporter': result['exporter'], 'chars': len(all_text)}))
