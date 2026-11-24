from pathlib import Path
import json
import re
import zipfile
import xml.etree.ElementTree as ET

import pypdfium2 as pdfium
from pypdf import PdfReader

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
pdf = WORK / 'render' / 'report.pdf'
reader = PdfReader(str(pdf))
pages = [p.extract_text() for p in reader.pages]
pdfdoc = pdfium.PdfDocument(str(pdf))
for i in range(len(pdfdoc)):
    page = pdfdoc[i]
    bitmap = page.render(scale=1.7)
    pil = bitmap.to_pil()
    pil.save(WORK / 'render' / f'page-{i+1}.png')
    bitmap.close()
    page.close()
pdfdoc.close()
(WORK / 'render' / 'extracted_text.txt').write_text('\n\n'.join(pages), encoding='utf-8')
docpath = ROOT / 'output/reports/Class_Adaptive_Augmentation_Project_Proposal_and_Related_Work_2026-11-03.docx'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(docpath) as z:
    main = ET.fromstring(z.read('word/document.xml'))
    colors = {x.attrib.get('{'+ns['w']+'}val') for x in main.findall('.//w:color', ns)}
    relationships = ET.fromstring(z.read('word/_rels/document.xml.rels'))
    links = [e.attrib['Target'] for e in relationships if e.attrib.get('Type', '').endswith('/hyperlink')]
text = '\n'.join(pages)
required_scores = ['22.8231', '22.8099', '22.5550', '23.0594', '22.5632', '19.8080', '19.6076', '19.6846', '18.4369']
assert all(s in text for s in required_scores)
assert len(links) == 18
assert colors <= {'000000', 'auto'}
assert all(f'[{i}]' in text for i in range(1, 19))
checks = {
    'pages': len(pages),
    'reference_links': len(links),
    'document_colors': sorted(colors),
    'required_scores_present': True,
    'all_reference_entries_present': True,
    'renderer': 'Microsoft Word PDF export then pypdfium2 rasterization',
    'canonical_renderer_limitation': 'LibreOffice soffice.exe is not available on PATH',
    'page_summaries': [
        {'page': i+1, 'words': len(p.split()), 'first_lines': p.splitlines()[:3], 'last_lines': p.splitlines()[-4:]}
        for i,p in enumerate(pages)
    ],
}
(WORK / 'structural_qa.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(checks, indent=2, ensure_ascii=True))
