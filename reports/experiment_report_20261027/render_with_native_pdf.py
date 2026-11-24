"""Use the packaged DOCX rasterizer after the native office PDF export.

LibreOffice is unavailable on this Windows host. The initial unmodified renderer
attempt failed in _resolve_soffice. The installed Word COM-compatible exporter
generated the PDF (Creator metadata: WPS Docs). No skill source is modified.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import json
import time
from PIL import Image
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
tmp=OUT/'temp'
tmp.mkdir(exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(tmp)
tempfile.tempdir=str(tmp)
skill=Path(r'C:\Users\SOLOMON\.codex\plugins\cache\openai-primary-runtime\documents\26.1020.11618\skills\documents\render_docx.py')
spec=importlib.util.spec_from_file_location('packaged_render_docx',skill)
renderer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
pdf=ROOT/'output/reports/RDD_Experiment_Report_2026-10-27.pdf'
docx=pdf.with_suffix('.docx')
reader=PdfReader(pdf)
assert len(reader.pages)>0
def already_exported(doc_path,user_profile,convert_tmp_dir,stem,verbose=False):
    assert Path(doc_path).resolve()==docx.resolve()
    assert pdf.stat().st_mtime>=docx.stat().st_mtime
    return str(pdf),'PDF from installed Word COM-compatible native office exporter; LibreOffice unavailable.'
renderer.convert_to_pdf=already_exported
original_replace=renderer.replace
def retry_local_replace(source,destination):
    # OneDrive may briefly hold a newly generated PNG while indexing it.
    for attempt in range(15):
        try:
            return original_replace(source,destination)
        except PermissionError:
            if attempt==14: raise
            time.sleep(0.3)
renderer.replace=retry_local_replace
paths=renderer.rasterize(str(docx),str(OUT/'rendered_release'),150,False,False)
text='\n\n'.join(p.extract_text() for p in reader.pages)
(OUT/'pdf_text.txt').write_text(text,encoding='utf-8')
statistics=[]
for i,path in enumerate(paths,1):
    image=Image.open(path).convert('RGB')
    # Exact RGB equality checks black-and-white rendering, including antialiasing.
    import numpy as np
    a=np.array(image)
    colored=int(((a[:,:,0]!=a[:,:,1]) | (a[:,:,1]!=a[:,:,2])).sum())
    statistics.append({'page':i,'width':image.width,'height':image.height,'non_grayscale_pixels':colored,'text_characters':len(reader.pages[i-1].extract_text())})
result={'pages':len(paths),'exporter':str(reader.metadata.get('/Creator')),'rasterizer':str(skill),'dpi':150,'pages_info':statistics,'visual_inspection':'pending'}
(OUT/'render_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'pages':len(paths),'non_grayscale_pixels':sum(x['non_grayscale_pixels'] for x in statistics),'exporter':result['exporter']}))
