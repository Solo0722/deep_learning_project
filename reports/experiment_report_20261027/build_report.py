from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from copy import deepcopy

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.graphics.shapes import Drawing, Line, PolyLine, String, Circle, Rect, Polygon
from reportlab.graphics import renderSVG

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSETS = OUT / 'assets'
ASSETS.mkdir(exist_ok=True)
FINAL = ROOT / 'output' / 'reports'
FINAL.mkdir(parents=True, exist_ok=True)
STEM = 'RDD_Experiment_Report_2026-10-27'
R = ROOT / 'reports' / 'fasterrcnn_class_adaptive_v1'
sources = {}

def read_json(path):
    p = ROOT / path
    sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8-sig'))

def read_csv(path):
    p = ROOT / path
    sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

split = read_json('detection_splits_v1/summary.json')
protocol = read_json('fasterrcnn_class_adaptive_v1/protocol.json')
aggregate = read_csv('reports/fasterrcnn_class_adaptive_v1/aggregate_comparison.csv')
perclass = read_csv('reports/fasterrcnn_class_adaptive_v1/per_class_comparison.csv')
negative = read_csv('reports/fasterrcnn_class_adaptive_v1/negative_image_comparison.csv')
runtime = read_csv('reports/fasterrcnn_class_adaptive_v1/runtime_comparison.csv')
curves = read_csv('reports/fasterrcnn_class_adaptive_v1/comparison_curves.csv')
sets = read_csv('reports/fasterrcnn_class_adaptive_v1/class_set_summary.csv')
diagnosis = read_json('reports/fasterrcnn_class_adaptive_v1/strength_support_diagnostics.json')
controller = read_json('reports/fasterrcnn_class_adaptive_v1/controller_verification.json')
audit = read_json('reports/fasterrcnn_class_adaptive_v1/ccast_verification.json')
comparison = read_json('reports/fasterrcnn_class_adaptive_v1/comparison.json')
shared = read_json('reports/fasterrcnn_online_shared_v1/controller_verification.json')
local = read_json('reports/fasterrcnn_local_v1/augmentation_verification.json')
learning = read_csv('reports/detection_learning_v1/metrics.csv')
yolo = read_json('reports/detection_diagnostics_v1/post_diagnostic_review.json')
validation = read_json('fasterrcnn_class_adaptive_v1/LOCAL_VALIDATION.json')

ORDER = ['identity', 'image_level', 'local', 'online_shared', 'class_adaptive']
NAMES = dict(zip(ORDER, ['Identity', 'Fixed image level', 'Fixed local', 'Online shared local', 'Class adaptive local']))
BEST = {r['condition']: r for r in aggregate if r['selection'] == 'best'}
END = {r['condition']: r for r in aggregate if r['selection'] == 'final'}
def f(x, n=3): return f'{float(x):.{n}f}'
def integer(x): return f'{int(x):,}'

# Monochrome vector figure. All series use black ink and distinguishable line patterns.
def curve_figure():
    d = Drawing(468, 304)
    left, bottom, width, height = 42, 70, 410, 210
    xmin, xmax, ymin, ymax = 1, 20, 14, 24
    xx = lambda x: left + (x-xmin)/(xmax-xmin)*width
    yy = lambda y: bottom + (y-ymin)/(ymax-ymin)*height
    from reportlab.lib.colors import black, white
    d.add(Rect(0,0,468,304,fillColor=white,strokeColor=None))
    for y in range(14,25,2):
        d.add(Line(left-4,yy(y),left,yy(y),strokeColor=black,strokeWidth=.6))
        d.add(String(left-8,yy(y)-3,str(y),textAnchor='end',fontName='Times-Roman',fontSize=9))
    for x in [1,4,8,12,16,20]:
        d.add(Line(xx(x),bottom-4,xx(x),bottom,strokeColor=black,strokeWidth=.6))
        d.add(String(xx(x),bottom-15,str(x),textAnchor='middle',fontName='Times-Roman',fontSize=9))
    d.add(Line(left,bottom,left,bottom+height,strokeColor=black,strokeWidth=.8))
    d.add(Line(left,bottom,left+width,bottom,strokeColor=black,strokeWidth=.8))
    d.add(String(left,bottom+height+11,'Validation AP50:95',fontName='Times-Roman',fontSize=10))
    d.add(String(left+width/2,bottom-28,'Epoch',textAnchor='middle',fontName='Times-Roman',fontSize=10))
    patterns = [None, [6,3], [1,2], [8,2,1,2], [3,2]]
    for i,key in enumerate(ORDER):
        points=[]
        for r in curves:
            points.extend([xx(int(r['epoch'])), yy(float(r[key+'_AP50_95']))])
        d.add(PolyLine(points,strokeColor=black,strokeWidth=1.05,strokeDashArray=patterns[i]))
        if i in [0,4]:
            for r in curves[::3]:
                x,y=xx(int(r['epoch'])), yy(float(r[key+'_AP50_95']))
                d.add(Circle(x,y,1.6,strokeColor=black,fillColor=white if i==4 else black,strokeWidth=.7))
        lx = [42,171,309,42,241][i]; ly = [14,14,14,0,0][i]+11
        d.add(PolyLine([lx,ly,lx+19,ly],strokeColor=black,strokeWidth=1.05,strokeDashArray=patterns[i]))
        if i in [0,4]:
            d.add(Circle(lx+9.5,ly,1.6,strokeColor=black,fillColor=white if i==4 else black,strokeWidth=.7))
        d.add(String(lx+24,ly-3,NAMES[key],fontName='Times-Roman',fontSize=8.5))
    renderSVG.drawToFile(d,str(ASSETS/'learning_curves.svg'))

curve_figure()

doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Inches(8.5), Inches(11)
section.top_margin = section.bottom_margin = Inches(.72)
section.left_margin = section.right_margin = Inches(.8)
section.header_distance = section.footer_distance = Inches(.3)
for style in doc.styles:
    if style.type in (1,2):
        style.font.name = 'Times New Roman'
        style.font.color.rgb = RGBColor(0,0,0)
        if style.element.rPr is not None:
            fonts=style.element.rPr.find(qn('w:rFonts'))
            if fonts is not None:
                for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme','csTheme']:
                    fonts.attrib.pop(qn('w:'+attr),None)
            color = style.element.rPr.find(qn('w:color'))
            if color is not None:
                for attr in ['themeColor','themeTint','themeShade']:
                    color.attrib.pop(qn('w:'+attr),None)
normal = doc.styles['Normal']
normal.font.size = Pt(11)
normal.paragraph_format.line_spacing = 1.08
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.widow_control = True
for name,size,before,after in [('Title',18,0,10),('Subtitle',11,0,10),('Heading 1',13,13,7),('Heading 2',11.5,10,5),('Heading 3',11,8,4)]:
    s=doc.styles[name]; s.font.size=Pt(size); s.font.bold=name!='Subtitle'
    s.paragraph_format.space_before=Pt(before); s.paragraph_format.space_after=Pt(after)
    s.paragraph_format.keep_with_next=True
    s.paragraph_format.keep_together=True
    s.paragraph_format.page_break_before=False
    if s.element.pPr is not None:
        for b in list(s.element.pPr.findall(qn('w:pBdr'))): s.element.pPr.remove(b)
doc.styles['Caption'].font.size=Pt(10)
doc.styles['Caption'].font.italic=False
doc.styles['Caption'].paragraph_format.space_before=Pt(7)
doc.styles['Caption'].paragraph_format.space_after=Pt(4)
doc.styles['Caption'].paragraph_format.keep_with_next=True

header = section.header.paragraphs[0]
header.text = 'Road damage augmentation experiment report'
header.style='Normal'; header.runs[0].font.size=Pt(9)
footer = section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=footer.add_run(); r.font.size=Pt(9)
field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); r._r.addnext(field)

blocks=[]
def paragraph(text, style=None):
    blocks.append({'type':'paragraph','text':text})
    return doc.add_paragraph(text, style)
def heading(text, level=1):
    blocks.append({'type':'heading','level':level,'text':text})
    return doc.add_heading(text,level)
def table(caption, headers, rows, widths, font=10):
    blocks.append({'type':'table','caption':caption,'headers':headers,'rows':rows})
    doc.add_paragraph(caption,'Caption')
    t=doc.add_table(rows=1, cols=len(headers))
    t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    pr=t._tbl.tblPr
    borders=OxmlElement('w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
        x=OxmlElement('w:'+side); x.set(qn('w:val'),'single'); x.set(qn('w:sz'),'3'); x.set(qn('w:color'),'000000'); borders.append(x)
    pr.append(borders)
    for i,w in enumerate(widths): t.columns[i].width=Inches(w)
    materialized = [headers]+list(rows)
    for row_index,vals in enumerate(materialized):
        row=t.rows[0] if vals is headers else t.add_row()
        rowpr=row._tr.get_or_add_trPr()
        cant=OxmlElement('w:cantSplit'); rowpr.append(cant)
        if vals is headers:
            repeat=OxmlElement('w:tblHeader'); rowpr.append(repeat)
        for i,v in enumerate(vals):
            cell=row.cells[i]; cell.width=Inches(widths[i]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=cell._tc.get_or_add_tcPr(); margins=OxmlElement('w:tcMar')
            for name,val in [('top',60),('bottom',60),('left',70),('right',70)]:
                z=OxmlElement('w:'+name); z.set(qn('w:w'),str(val)); z.set(qn('w:type'),'dxa'); margins.append(z)
            cp.append(margins)
            p=cell.paragraphs[0]; p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.02
            p.paragraph_format.keep_with_next = vals is headers or (len(rows)<=9 and row_index<len(materialized)-1)
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 or len(str(v))>30 else WD_ALIGN_PARAGRAPH.CENTER
            run=p.add_run(str(v)); run.font.size=Pt(font); run.bold=vals is headers
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
    return t

# Native Word equations using OMML. Verified through the installed Word-compatible PDF exporter.
def mr(text):
    x=OxmlElement('m:r'); props=OxmlElement('m:rPr'); st=OxmlElement('m:sty'); st.set(qn('m:val'),'p'); props.append(st); x.append(props)
    t=OxmlElement('m:t'); t.set(qn('xml:space'),'preserve'); t.text=text.replace('Σ','∑'); x.append(t); return x
def group(tag,items):
    x=OxmlElement(tag)
    for a in items: x.append(mr(a) if isinstance(a,str) else a)
    return x
def sub(base,low): return group('m:sSub',[group('m:e',[base]),group('m:sub',[low])])
def sup(base,high): return group('m:sSup',[group('m:e',[base]),group('m:sup',[high])])
def ss(base,low,high): return group('m:sSubSup',[group('m:e',[base]),group('m:sub',[low]),group('m:sup',[high])])
def frac(a,b): return group('m:f',[group('m:num',a),group('m:den',b)])
def eq(number, items, description):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(5); p.paragraph_format.space_after=Pt(7)
    p.paragraph_format.keep_together=True
    math=group('m:oMath',items); p._p.append(math)
    p.add_run('     ('+str(number)+')')
    blocks.append({'type':'equation','number':number,'description':description})

paragraph('Road Damage Augmentation Experiments and Results','Title')
paragraph('Detailed project report through 27 October 2026','Subtitle')
paragraph('This report documents the implementation, completed experiments, verified results, and unresolved methodological issues in the road-damage augmentation project. It covers the early crop-classification study, detector feasibility and debugging, the five-condition Faster R-CNN comparison, and the subsequent diagnosis of the proposed online class-adaptive controller. Proposed changes are separated from completed work.')
paragraph('The main finding is negative for the proposed accuracy claim. The class-adaptive controller runs and its decisions can be audited, but its selected validation AP50:95 of 22.563 is below online shared local augmentation at 23.059 and identity at 22.823. Each condition has only one training run. These results establish a functioning prototype and a documented comparison, not a reliable generalization advantage.')

heading('1 Research objective and current conclusion')
paragraph('The research question is whether local appearance augmentation can be selected online according to the damage classes present in an overlapping group of annotations, while preserving the detector’s ability to recognize those objects. A common policy may be limited by the most sensitive class. Conditioning decisions on the actual class combination could allow different strengths in different regions. This is the motivating hypothesis; it is not an established property of the completed system.')
paragraph('The proposed contribution is the augmentation controller: overlap-group construction, a shared local edit, detector-based retention constraints, class-balanced difficulty feedback, exact-class-set policy selection, and reproducible online updates. The full-image experiments use an existing Faster R-CNN with a ResNet-50 feature pyramid backbone and COCO initialization. The detector architecture itself is not newly designed. The earlier crop classifier was a separate, custom network trained from scratch.')
paragraph('Five detection conditions have completed the same 20-epoch optimizer budget: identity, fixed image-level augmentation, fixed local augmentation, online shared local augmentation, and online class-adaptive local augmentation. Online shared has the highest observed selected AP50:95; identity has the highest observed selected AP50. Class adaptation is nearly equal numerically to fixed local in this run and lower than its primary online-shared comparator. No statistical equivalence, significant difference, or population-level ranking can be inferred from one run per condition. [S3, S4]')
paragraph('The next justified step is a focused diagnostic of the feedback measurement and probe support, followed by short controlled comparisons if the diagnostic succeeds. Another unchanged full run would estimate variability but would not resolve the identified design weaknesses. No revised controller, repeated-seed study, or final test evaluation is included in the present results.')

heading('2 Development history and experiment scope')
heading('2 1 Earlier crop classification study',2)
paragraph('The project initially used annotated road-damage crops as classification inputs. The completed conventional-augmentation baseline used a randomly initialized CNN with 695,732 trainable parameters, unweighted cross-entropy, AdamW, batch size 64, and a 20-epoch cosine learning-rate schedule. It selected epoch 20 by validation macro-F1. On 8,242 validation crops, 7,461 predictions were correct: accuracy was 90.524% and macro-F1 was 0.891457. [S6]')
table('Table 1  Historical crop classification results', ['Class','Support','Precision','Recall','F1'],[
['D00',3898,'0.9042','0.9472','0.9252'],['D10',1771,'0.9524','0.9368','0.9445'],['D20',1592,'0.8759','0.8021','0.8374'],['D40',981,'0.8686','0.8491','0.8588']], [2.0,1.1,1.2,1.2,1.4])
paragraph('The largest directional error was D20 classified as D00: 223 crops, or 14.01% of D20 examples. A box-aspect-ratio comparator achieved 67.03% accuracy and 0.4439 macro-F1, while the majority-class comparator achieved 47.29% accuracy and 0.1605 macro-F1. The CNN exceeded those controls, but their results also showed why annotation geometry and source cues needed attention. The error pattern alone did not identify its cause.')
paragraph('The full-image formulation was adopted to study augmentation with localization, multiple objects, overlapping boxes, and images without target annotations. Crop accuracy is therefore historical feasibility evidence, not a baseline in the detection AP tables. The detection split was rebuilt with historical exposure and full-pool similarity groups taken into account; the crop and detection validation sets are not interchangeable.')
heading('2 2 Detector engineering checks',2)
paragraph('Before full training, the project verified image and XML integrity, model-specific label adapters, coordinate conversion, target-negative examples, synthetic CUDA execution, real-image forward and backward passes, checkpoint recovery, and metric behavior. Synthetic and small-subset checks were used to detect implementation faults before committing to longer jobs. Passing them establishes software behavior under the tested cases; it does not establish augmentation efficacy. [S7-S10]')
table('Table 2  Completed engineering stages', ['Stage','Observed evidence','Scope'],[
['Data audit','8 focused tests passed','Input integrity and exclusion handling'],
['Training loader','22 tests; 15 real images passed','All four classes, empty targets, geometry'],
['Detector integration','14 tests; both detectors ran on 15 images','Finite loss, gradients, updates and inference'],
['Small learning check','18 tests; 200 updates per detector','Training-only learning and checkpoint checks'],
['YOLO diagnostic','Fixed checkpoints; no optimizer steps','Normalization sensitivity on 15 training images'],
['Class-adaptive package','81 local focused tests passed','Policy, exposure, replay and resume behavior']
], [1.45,2.35,3.1])
paragraph('The 81-test class-adaptive result is a local result, with zero errors, failures, or skips. A separate actual pretrained-model CPU check exercised synthetic 128-pixel inputs and verified initialization parity, probe-state preservation, policy reconstruction, and checkpoint replay. It did not use RDD pixels or establish CUDA behavior or accuracy. Target-host test and smoke terminal logs are not included in the returned class-adaptive archive, so that archive alone cannot independently certify those test counts. The subsequent full CCAST training run is separately recorded as complete. [S11]')
heading('2 3 Small training check and YOLO diagnosis',2)
paragraph('Both detectors were first checked on the same 15 selected training images containing 16 retained boxes. Each received 200 updates. Faster R-CNN improved, while YOLOv8n failed to retain useful detections at the end of this particular custom-loop check. Table 3 reports training-subset measurements on a 0-100 scale. They must not be interpreted as held-out performance or evidence that one architecture is generally superior. [S8]')
table('Table 3  Training subset learning check', ['Detector','Step','AP50','AP50:95'],[[('Faster R-CNN' if r['model'].startswith('faster') else 'YOLOv8n'),r['step'],f(float(r['AP50'])*100),f(float(r['AP50_95'])*100)] for r in learning], [3.0,1.0,1.45,1.45])
paragraph('The Faster R-CNN result did not meet the predeclared small-subset engineering review threshold of final AP50 at least 50 with a gain of at least 10 percentage points. It supplied partial evidence of learning, followed later by the full validation baseline reported here. Reloaded predictions matched before and after checkpoint recovery for both models in the returned records. YOLO showed a classification-loss and parameter-update spike on an all-negative batch near the end of training, but that observation did not establish a unique cause.')
paragraph('The YOLO follow-up held checkpoint weights and input batches fixed and changed only the use of BatchNorm statistics at inference. At step 100, AP50 increased from 0.518 with running statistics to 17.197 with current batch statistics. At step 200, it increased from 0 to 8.077. AP50:95 changed from 0.231 to 4.077 at step 100 and from 0 to 3.777 at step 200. Returning to running statistics reproduced the original predictions. All six saved evaluations were independently recomputed. [S9]')
paragraph('This isolated normalization sensitivity in the tested setup, but it did not repair learning or validate batch-statistic inference. At step 200, 13 of 16 targets had a raw candidate box at IoU at least 0.5 under the intervention, but only three had such a candidate with sufficient correct-class score. Candidate availability is not one-to-one recall. A proposed normalization-only retraining comparison was deferred to keep the main augmentation study focused. No full YOLO augmentation comparison or full-image scratch-initialization comparison has been completed.')

heading('3 Dataset preparation and partitioning')
heading('3 1 Raw inputs and retained labels',2)
paragraph('The audit covered 38,385 image/XML pairs from seven RDD2022 source collections. All discovered training image files had matching XML files and vice versa. The four retained damage categories are D00 longitudinal cracks, D10 transverse cracks, D20 alligator cracks, and D40 potholes. Labels outside this set remain separate from the four-class target list. [S1]')
table('Table 4  Audited raw source collections', ['Source','Image and XML pairs'], [[k,integer(v)] for k,v in [('China Drone',2401),('China MotorBike',1977),('Czech',2829),('India',7706),('Japan',10506),('Norway',8161),('United States',4805)]], [4.7,2.2])
paragraph('Two images were excluded. United_States_003917.jpg duplicated the pixels of a retained image while two annotation boxes differed; annotations were not merged. India_006389.jpg had a sole ambiguous label, D0w0, which was neither guessed into a target class nor treated as a target-negative image. One zero-width D20 box in Japan_001265.jpg was removed while the image and its other three valid targets were retained. The resulting pool contains 38,383 images and 55,002 target boxes.')
paragraph('Boxes use continuous XYXY edges clipped to the stored image bounds without an origin shift or integer rounding in the annotation representation. Valid difficult or truncated targets remain included. Stored image orientation is preserved rather than applying EXIF rotation. Empty target lists mean that none of the four target classes is annotated; they do not establish visually healthy pavement.')
heading('3 2 Frozen splits and leakage safeguards',2)
table('Table 5  Frozen detection partitions', ['Partition','Images','Target bearing','No targets','Target boxes'],[[name,integer(split['statistics'][key]['images']),integer(split['statistics'][key]['categories']['target_bearing']),integer(split['statistics'][key]['images']-split['statistics'][key]['categories']['target_bearing']),integer(split['statistics'][key]['objects'])] for key,name in [('train','Training'),('val','Validation'),('test','Reserved test')]], [1.45,1.25,1.4,1.35,1.45])
table('Table 6  Target box distribution', ['Class','Training','Validation','Reserved test'],[[c,integer(split['statistics']['train']['class_objects'][c]),integer(split['statistics']['val']['class_objects'][c]),integer(split['statistics']['test']['class_objects'][c])] for c in ['D00','D10','D20','D40']], [1.8,1.7,1.7,1.7])
paragraph('The allocation targets a 70/15/15 split and preserves historical anchors. Source-image similarity clusters and candidate links remain within a single partition. Two components with previous training/validation conflicts were assigned to training, including six historically validated images, so known training exposure was not presented as fresh held-out evidence. Historical test target-bearing images were preserved. The frozen split audit recorded zero group crossings and zero candidate-edge crossings.')
paragraph('This is a custom mixed-source detection split, not an official cross-country generalization experiment. Perceptual matches are grouping candidates rather than certified duplicates. The screening procedure cannot establish complete geographic or capture-sequence independence. Historical validation exposure also remains relevant to interpreting development decisions. Test partition counts in this report are preparation metadata, not test performance; no test detector evaluation was conducted.')
heading('3 3 Shared preprocessing and adapters',2)
paragraph('Full images are converted to RGB, resized with aspect ratio preserved, and letterboxed to 640 by 640 pixels. Box scaling uses the actual resized dimensions, including separate axis scales caused by integer size rounding. Detector adapters map the foreground labels consistently; Faster R-CNN reserves label zero for background and uses labels one through four for damage. Target-negative images remain valid training examples. Evaluation transforms predictions back to the original stored coordinates and clips them to the image bounds.')

heading('4 Methodology of the proposed solution')
heading('4 1 Local groups and a single shared edit',2)
paragraph('For each prepared target-bearing image, target boxes are rasterized by flooring minima and ceiling maxima within the valid image content. An overlap graph connects boxes whose raster supports share positive area. Each connected component forms one augmentation group, including singleton boxes. The group class set contains every distinct damage class represented in that component. Four classes allow 15 possible nonempty exact sets. [S2]')
paragraph('At an augmentation opportunity, one group is selected uniformly. A candidate appearance transform is computed from the full resized image content and composited once within the union of the selected group’s boxes. This avoids repeatedly transforming pixels where annotations overlap. The union is a box-based support, not a pixel-level damage segmentation; background inside a box can also change.')
eq(1,['x̃ = x + ',sub('α','G'),' ⊙ [',sub('T','o,m'),'(x) − x]'],'Single masked group edit')
paragraph('In Equation 1, x is the prepared image, T is operation o at strength m, and α is zero outside the group union, one in its one-pixel-eroded interior, and 0.5 on the outer eight-neighbor boundary layer. A one-pixel-wide support retains alpha 0.5. Candidate operations and final composition use the declared clipping and uint8 rounding rules. The labels and coordinates remain fixed. Preserving coordinates does not prove that every edit preserves visually recognizable damage.')
heading('4 2 Candidate augmentation bank',2)
paragraph('The first controller uses a bounded appearance-only bank to keep annotations geometrically valid and make feedback interpretable. The photometric family contains brightness up/down, contrast up/down, and gamma brightening/darkening. The second family contains Gaussian blur and Gaussian noise. A larger bank, including geometric transforms, would require additional label transformation and validity checks and has not been tested here. Restricting this first study is an implementation control, not a claim that other augmentations are ineffective.')
table('Table 7  Candidate strength grid', ['Operation parameter','Level 1','Level 2','Level 3'],[
['Brightness and contrast decrease factor','0.9','0.8','0.7'],['Brightness and contrast increase factor','1.1','1.2','1.3'],['Gamma brightening exponent','0.9','0.8','0.7'],['Gamma darkening exponent','1.1','1.2','1.3'],['Gaussian blur radius in pixels','0.4','0.8','1.2'],['Gaussian noise standard deviation','0.01','0.02','0.03']], [3.6,1.1,1.1,1.1])
paragraph('Noise is independent across pixels and RGB channels in the stored RGB/255 representation. Contrast uses the full-content mean luminance. These operations act on stored color values rather than linear-light intensities. Fixed controls use level 2. They sample the two families with equal probability and sample operations uniformly within the selected family. Therefore, each photometric operation has conditional probability 1/12 and each blur/noise operation has probability 1/4; the eight operations are not globally equiprobable.')
heading('4 3 Paired online probes',2)
paragraph('The controller updates before epochs 2, 4, 6, through 20, after one unaugmented warmup epoch. Policies remain fixed between updates. Each update selects 256 target-bearing training images in total using keyed hash ranking, preferring distinct source clusters, and then selects one group per image uniformly. This is not a budget of 256 images per class or per class set. The shared and class-adaptive online conditions use the same prospective image/group panels.')
paragraph('Every selected image supplies an unmodified reference view and transformed views at three strengths for all eight operations. Gaussian noise uses two independent draws, each paired across strengths. The other seven operations use one view per strength. This gives 28 image forwards per probe image: one reference plus 27 transformed views. Ten updates require 71,680 additional image forwards per online run, plus auxiliary ROI computations.')
paragraph('Probes use training data in evaluation/inference mode and do not take optimizer steps. The implementation preserves parameters, buffers, gradients, and random-number-generator state across probing. Policy decisions and noise have separate keyed random streams. Policies are stored with training checkpoints; resuming a committed update does not run it again. A probe interrupted before commitment is reconstructed from the same checkpoint, with partial work recorded separately.')
heading('4 4 Retention and exact class set support',2)
paragraph('The controller first measures whether damage that the current detector recognizes remains detectable after augmentation. Standard predictions are matched to ground-truth boxes one-to-one, in descending score order, requiring the correct class, confidence at least 0.25, and IoU at least 0.5. Matching is performed against the full image; retention then uses objects in the selected group that were detected in the reference view.')
eq(2,[sub('R','c|s,o,m'),' = ',frac([sub('Σ','j∈J(c,s)'),sup('dⱼ','0'),sup('dⱼ','o,m')],[sub('Σ','j∈J(c,s)'),sup('dⱼ','0')])],'Reference detection retention by class and exact group set')
paragraph('Here s is the exact group class set, c is a class in s, and J(c,s) contains class-c objects in selected groups with exact set s. Each d indicates a class-correct matched detection. Repeated noise draws are averaged within the object before aggregation, so they do not count as independent support. A class that is poorly detected initially can have few eligible denominator objects even when many annotations exist. Retention measures conditional stability of existing detections; it cannot certify initially missed objects or semantic correctness.')
paragraph('Every represented class in an exact set must have at least eight distinct reference-detected group objects spanning at least four source clusters. Each family-strength candidate must then retain at least 80% for every represented class and every operation in that family. This is a joint constraint: one sensitive class or operation can reject the family-strength candidate. The minimum support reduces decisions based on one or two detections, but it is not a statistical precision guarantee.')
eq(3,[sub('M','f,s'),' = {m : ',sub('min','c∈s, o∈f'),' ',sub('R','c|s,o,m'),' ≥ 0.80},    ',ss('m','f,s','*'),' = max ',sub('M','f,s')],'Strongest admissible tested strength after support checks')
paragraph('Equation 3 describes the implemented strength rule after support is sufficient. It selects the strongest passing tested level even if a weaker level fails; no monotonicity assumption is imposed. Missing or insufficiently supported sets and sets without an admissible family receive identity. Version 1 does not borrow support from another set, back off to the shared policy, retain a stale policy, or resample a different group to force an edit.')
heading('4 5 Auxiliary difficulty and family probabilities',2)
paragraph('For difficulty feedback, the detector’s evaluation RPN proposals are assigned to target objects by the ROI matcher. One explicitly self-assigned ground-truth ROI is also included per object so that a missing proposal does not remove that object’s auxiliary observation. These extra ground-truth ROIs are used only for feedback; they are not included as ordinary output detections.')
eq(4,[sub('ℓ','j'),' = ',frac(['1'],[sub('n','j')]),' ',sub('Σ','r∈Pⱼ'),' [',sub('ℓ','CE'),'(r) + ',sub('ℓ','SmoothL1'),'(r)]'],'Auxiliary object loss averaged over assigned positive ROIs')
paragraph('P is the assigned ROI set and n is its size for object j. The per-ROI term combines classification cross-entropy and class-specific SmoothL1 box loss, summed across four coordinates with beta 1/9. The controller excludes background and RPN loss from this difficulty measure. Loss is averaged over ROIs within an object, over repeated draws within an object, and then over all selected-group objects within each class and exact set, including reference misses. Only retention and support restrict to reference-detected objects. Reference and transformed views generate their own proposals in version 1; this later becomes an important measurement limitation.')
eq(5,[sub('A','c|s,o,m'),' = clip(',frac([ss('L̄','c|s','o,m'),' − ',ss('L̄','c|s','0')],[ss('L̄','c|s','0'),' + 0.1']),', 0, 5)'],'Normalized clipped net class mean loss increase')
paragraph('The mean is taken before subtraction and clipping in Equation 5. A harder object and an easier object may offset each other. A zero score therefore means that the normalized net class mean increase did not exceed zero, not that pixels were unchanged or every object became easier. The denominator offset 0.1 limits instability near zero reference loss, and the upper cap is 5.')
eq(6,[sub('D','f,s'),' = ',frac(['1'],['|s| |f|']),' ',sub('Σ','c∈s'),' ',sub('Σ','o∈f'),' ',sub('A','c|s,o,m*')],'Equal weighting of represented classes and family operations')
eq(7,[sub('p','f|s'),' = 0.8 ',frac(['exp(',sub('D','f,s'),'/0.25)'],[sub('Σ','g∈Fₛ'),' exp(',sub('D','g,s'),'/0.25)']),' + ',frac(['0.2'],['|Fₛ|'])],'Family sampling with temperature and uniform exploration')
paragraph('F is the set of eligible families. Equation 7 uses temperature 0.25 and exploration mass 0.2; with one eligible family its probability is one. An operation is then chosen uniformly within the selected family at its selected strength. Crucially, the difficulty score chooses family probabilities, not the strength within a family. Strength is already fixed by the maximum passing level in Equation 3. All numerical settings were engineering choices for this development protocol, not demonstrated optimal values.')
heading('4 6 Training loop and comparator distinction',2)
paragraph('After warmup, each target-bearing training image has an augmentation opportunity with probability 0.5. A selected group receives its current exact-set policy, or identity if the policy cannot act. Images without targets remain unchanged. Keyed decisions use the run seed, epoch, image identifier, and a purpose-specific key independently of global training randomness. Training then uses the normal Faster R-CNN detection loss and optimizer update; the controller does not introduce a separate learned network or backpropagate through its decisions.')
eq(8,[sub('L','det'),' = ',sub('L','RPN cls'),' + ',sub('L','RPN box'),' + ',sub('L','ROI cls'),' + ',sub('L','ROI box')],'Standard detector training objective')
paragraph('The online shared local control uses the same operations, panels, grid, retention threshold, difficulty construction, and update schedule, but pools selected groups across exact class sets into one shared policy with constraints over all four classes. The proposed condition restricts feedback and routing to each exact class set. This comparison changes support and downstream edit exposure as a consequence of conditioning. It does not isolate class routing while holding every realized transformation fixed.')

heading('5 Full experiment design and implementation')
table('Table 8  Five completed detection conditions', ['Condition','Edit region','Policy'],[
['Identity','No edit','No training augmentation'],['Fixed image level','Full resized image content','Level 2; equal families'],['Fixed local','One overlap group union','Same fixed choices; local mask'],['Online shared local','One overlap group union','One feedback-selected policy for all groups'],['Class adaptive local','One overlap group union','Feedback-selected policy per exact class set']], [1.75,2.0,3.15])
paragraph('The sequence separates several questions: whether the fixed appearance policy helps; whether localizing that policy helps; whether online shared selection helps relative to fixed local; and whether exact-class-set conditioning helps relative to online shared. The primary proposed-method contrast is the last pair. The fixed-to-online comparison also introduces access to three strength levels instead of one, so any difference cannot be attributed solely to online feedback.')
paragraph('Every condition starts fresh from the same COCO weight file with a newly initialized five-class ROI predictor. No condition starts from another condition’s best learned checkpoint. The frozen manifests, preprocessing, nominal seed 42, optimizer schedule, validation procedure, and 20-epoch budget match. All backbone stages are trainable; FrozenBatchNorm2d uses epsilon zero. The runs each complete 134,320 optimizer steps and 537,260 image presentations over 26,863 training images. [S2-S4]')
table('Table 9  Shared training and evaluation settings', ['Setting','Value'],[
['Detector','Faster R-CNN ResNet-50 FPN; COCO_V1 initialization'],['Input','640 by 640 letterboxed RGB; stored orientation'],['Classes','Four foreground damage classes plus background'],['Epochs and batch size','20 epochs; training batch 4; validation batch 4'],['Optimizer','SGD; learning rate 0.0025; momentum 0.9'],['Weight decay','0.0005'],['Learning-rate warmup','1,000 steps; start factor 0.001'],['Learning-rate decay','Multiply by 0.1 after epochs 12 and 16'],['Precision','Float32; AMP and TF32 disabled'],['Loader and CPU settings','Two data workers; two CPU threads'],['Validation','Every epoch; unaugmented images'],['Checkpoint selection','Maximum validation AP50:95; earlier epoch on ties'],['Prediction settings','Score floor 0.001; NMS IoU 0.5; at most 100 detections'],['Additional diagnostic','No-target image detections at confidence 0.25']], [2.05,4.85])
paragraph('AP50:95 averages precision over the four target classes and IoU thresholds from 0.50 to 0.95 in increments of 0.05. AP50 and AP75 hold the IoU threshold at 0.50 and 0.75, respectively. AR100 measures average recall with at most 100 detections. All AP and AR values in this report use a 0-100 scale. Differences between them are percentage points. These detection measures combine classification, localization, ranking, and missed/extra detections; they are not simple classification accuracy.')
paragraph('Validation uses 5,760 images and 8,225 target boxes. The same validation set selects each checkpoint and supplies the reported development comparison. The final-epoch values are also shown to make selection effects visible. The reserved test partition is excluded. Repeated consultation of validation results during staged development prevents treating this comparison as a final untouched evaluation.')
heading('5 1 CCAST execution and reproducibility',2)
paragraph('Full experiments ran as PBS batch jobs on CCAST using one NVIDIA A10, four allocated CPU cores, 32 GB requested RAM, and a 24-hour wall-time request. The batch workflow checks package integrity, activates the project environment, runs its validation stages, records training output, and writes recoverable checkpoints. Submission occurs from a login node; the scheduler supplies the GPU allocation. A disconnected SSH monitoring session does not by itself terminate an already submitted batch job.')
paragraph('The recorded full-run software is Python 3.12.3, PyTorch 2.3.0, TorchVision 0.18.0, CUDA build 12.1, NumPy 1.26.4, Pillow 10.3.0, and pycocotools 2.0.10. YOLO development additionally used ultralytics-opencv-headless 8.4.182 and opencv-python-headless 4.11.0.86. Local verification used its separately recorded local versions; those results are not described as target-environment executions.')
paragraph('Code packages, manifests, weights, configurations, and returned archives are pinned by SHA-256. Full checkpoint binaries remain on CCAST, while the downloaded records contain configuration, metrics, predictions, controller observations, decisions, and checkpoint metadata. Saved predictions support independent metric recomputation without rerunning detector inference. Seed matching and deterministic augmentation decisions do not imply bitwise-identical GPU training: full deterministic algorithms were not enabled, and the measured warmup trajectories differ.')

heading('6 Detection results')
heading('6 1 Selected checkpoints and final epochs',2)
table('Table 10  Validation performance at the selected checkpoint', ['Condition','Epoch','AP50:95','AP50','AP75','AR100'],[[NAMES[k],BEST[k]['epoch'],f(BEST[k]['AP50_95']),f(BEST[k]['AP50']),f(BEST[k]['AP75']),f(BEST[k]['AR100'])] for k in ORDER], [2.15,.65,1.1,1.0,1.0,1.0])
paragraph('Online shared local achieves 23.059 AP50:95, 0.236 points above identity. Fixed image-level augmentation is 0.013 points below identity, and fixed local is 0.268 points below identity. The class-adaptive result is 0.496 points below online shared, 0.260 below identity, 0.247 below fixed image level, and only 0.008 above fixed local. The observed primary comparison therefore does not support an accuracy benefit from exact-class-set conditioning. It also does not establish that the method is universally harmful. [S3, S4]')
paragraph('Identity achieves the largest AP50, at 51.142. The fact that online shared leads AP50:95 but not AP50 matters: no single condition dominates every reported accuracy measure. The AP50-to-AP50:95 gap is consistent with substantial sensitivity to tighter localization requirements, but does not identify which object sizes, sources, or annotation properties cause the gap.')
table('Table 11  Final epoch 20 validation performance', ['Condition','AP50:95','AP50','AP75','AR100'],[[NAMES[k],f(END[k]['AP50_95']),f(END[k]['AP50']),f(END[k]['AP75']),f(END[k]['AR100'])] for k in ORDER], [2.3,1.15,1.15,1.15,1.15])
paragraph('Final-epoch AP50:95 ranges from 22.392 to 22.588. The class-adaptive final score is 22.493, slightly above identity at that endpoint, while its selected-checkpoint score remains below identity. This does not justify changing the primary criterion after observing the result. The declared checkpoint-selection rule is applied consistently to every condition.')
heading('6 2 Class level results',2)
table('Table 12  Selected checkpoint AP50:95 by class', ['Class','Identity','Image level','Local','Online shared','Class adaptive'],[[r['class']]+[f(r[k+'_AP50_95']) for k in ORDER] for r in perclass], [1.0,1.15,1.2,1.1,1.25,1.2])
table('Table 13  Selected checkpoint AP50 by class', ['Class','Identity','Image level','Local','Online shared','Class adaptive'],[[r['class']]+[f(r[k+'_AP50']) for k in ORDER] for r in perclass], [1.0,1.15,1.2,1.1,1.25,1.2])
paragraph('Class-adaptive AP50:95 is lower than online shared for every class: D00 by 0.670 points, D10 by 0.604, D20 by 0.093, and D40 by 0.618. D20 has the highest AP50:95 in every condition, while D40 has the lowest. The class distribution and source mixture differ, so these rankings are descriptive and should not be reduced to a single claim about intrinsic class difficulty. D20 AP50 under class adaptation is marginally above online shared, but that isolated change does not overturn the aggregate or per-class AP50:95 comparison.')
heading('6 3 Learning trajectories and selection limits',2)
paragraph('The learning curves show improvements across the training schedule and a pronounced increase after epoch 12, when the learning rate first decays. Similar timing across conditions prevents attributing that change uniquely to the controller. The best checkpoints occur at epoch 14 for identity, image-level, and online shared, and at epoch 18 for fixed local and class adaptive. These runs do not establish convergence beyond the declared schedule.')
# The vector figure is converted to PNG by the separate asset helper before the final build.
if (ASSETS/'learning_curves.png').exists():
    p=doc.add_paragraph(); p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(ASSETS/'learning_curves.png'),width=Inches(6.5))
    cap=doc.add_paragraph('Figure 1  Validation AP50:95 across all 20 epochs. Black line patterns identify conditions. These are individual development trajectories, not mean curves or confidence intervals.','Caption')
    cap.paragraph_format.keep_with_next=False
paragraph('Epoch 1 is unaugmented in every condition, yet AP50:95 already differs: identity 16.537, fixed image level 15.892, fixed local 16.424, online shared 15.733, and class adaptive 16.518. The cause is not established by the saved evidence. These initial differences cannot be subtracted from final scores as a correction, and they are not an estimate of final-run variance. They show why nominal seed matching is insufficient to claim perfectly paired training trajectories.')
heading('6 4 Predictions on images without target annotations',2)
table('Table 14  Selected checkpoint detections on 2193 no-target images', ['Condition','Images with detections','Predicted boxes','Image fraction'],[[NAMES[r['condition']],integer(r['images_with_predictions']),integer(r['prediction_boxes']),f(r['fraction_percent'],2)+'%'] for r in negative], [2.3,1.75,1.4,1.45])
paragraph('These counts use confidence 0.25. Class adaptation produces the smallest no-target image fraction, 36.75%, compared with 40.40% for online shared and 39.12% for identity. This is useful descriptive evidence, but lower false alarms at one threshold can accompany weaker positive-image detection. It does not establish a better operating point at matched recall. The detections are false positives relative to retained annotations; the images were not visually certified to contain no road damage.')
heading('6 5 Runtime and computation',2)
table('Table 15  Recorded full run time and execution identity', ['Condition','Hours','Host','PBS job'],[[NAMES[r['condition']],f(r['hours'],2),r['host'],r['pbs_job']] for r in runtime], [2.2,1.0,1.25,2.45])
paragraph('The five full runs total '+f(sum(float(r['hours']) for r in runtime),2)+' recorded run-hours. These intervals include training, validation, and online probes where applicable, but exclude the preceding tests and smoke checks. They are not end-to-end project labor or complete allocation billing. Differences between runs do not isolate controller overhead because execution conditions and augmentation paths also differ.')
paragraph('Recorded probe time is 56.41 minutes for online shared and 57.19 minutes for class adaptation. Both use 71,680 extra image forwards with no additional detector optimizer steps. Thus the optimizer budgets match while total forward computation does not match the fixed and identity controls. No inference-speed improvement is claimed; the controller operates during training.')

heading('7 Observed controller behavior')
heading('7 1 Online shared policy',2)
paragraph('The shared controller acted on all 158,151 scheduled augmentation opportunities, and 156,819 applications changed pixels. Its class support was sufficient at all ten updates. It selected blur/noise level 1 at every update; photometric strength varied across levels 1, 2, and 3 and was initially ineligible. When both families were eligible, probabilities stayed near one half. [S5]')
paragraph('This behavior provides a concrete explanation for why the online condition is not equivalent to the fixed level-2 local baseline. Most blur/noise edits became milder, while photometric settings changed over training. The observed AP difference cannot distinguish the value of the feedback mechanism from the value of this realized strength distribution. A matched fixed mild or uniform-family control is still needed for attribution.')
table('Table 16  Applied strength distribution in the online conditions', ['Strength','Online shared edits','Online shared share','Class adaptive edits','Class adaptive share'],[[str(i),integer(shared['total']['strength_level_'+str(i)]),f(100*shared['total']['strength_level_'+str(i)]/158151,2)+'%',integer(controller['total']['strength_level_'+str(i)]),f(100*controller['total']['strength_level_'+str(i)]/144568,2)+'%'] for i in [1,2,3]], [.8,1.55,1.5,1.55,1.5])
heading('7 2 Class adaptive exposure and sparse support',2)
paragraph('The class-adaptive condition shared the same 158,151 opportunities but applied 144,568 edits; 13,583 opportunities, or 8.59%, received identity. Of the applied edits, 144,418 changed pixels and 150 did not. The fallback reasons were 1,205 unobserved class-set cases and 12,378 insufficient-support cases. No opportunity fell back because a sufficiently supported set had no admissible family. Equal opportunities therefore did not produce equal actual edit exposure. [S4]')
table('Table 17  Exact class set activation and realized exposure', ['Exact class set','Active updates','Opportunities','Applied','Identity'],[[r['labels'],r['active_updates']+'/10',integer(r['opportunities']),integer(r['applied']),integer(r['identity'])] for r in sets], [2.3,1.05,1.2,1.2,1.15],font=9.5)
paragraph('All singleton sets except D40 were active at every update; D40 and D00+D10 were active at nine. D20+D40 was active at only one update. Nine of the 15 possible sets never activated. That is a serious coverage limitation for mixed groups, but it does not mean that most training opportunities were disabled: common singleton sets dominate the total, and 91.41% of all opportunities still received an edit.')
paragraph('The local fixed and shared-online controls selected 199,490 boxes within 158,151 group applications, including 26,843 applications involving more than one box. The local support changes only a portion of the image content and preserves unselected regions. This locality is part of the intervention, not a claim that every annotated object receives an augmentation in each epoch.')
paragraph('Class adaptation selected level 3 for 76.17% of applied edits, compared with 18.79% under online shared. This is a substantial change in the realized intervention. It is consistent with less restrictive conditioning for common supported sets, but the records do not establish that stronger edits caused the AP decrease. Support, family choices, detector trajectories, and realized exposure also differ.')

heading('8 Retrospective diagnosis of strength and support')
paragraph('After the completed class-adaptive run, saved training observations and frozen training metadata were examined without new optimizer steps, detector inference, or test evaluation. The analysis reconstructs decisions on the observed trajectory. It cannot predict the accuracy of a controller that would have changed that trajectory. [S12]')
heading('8 1 Strongest passing is not the same as most useful',2)
paragraph('Of 95 active family/update/class-set states, 52 had exactly zero clipped net difficulty. Those states accounted for 97,048 of 144,568 applied edits, or 67.13%. In 44 states the selected strength had zero difficulty although another eligible strength had a positive score; those states accounted for 82,914 applications. This follows from the implemented rule: strength maximizes the passing level, while difficulty is used only after strength selection.')
paragraph('For example, the D00 photometric state at epoch 4 selected level 3 with difficulty zero and minimum retention about 0.907; all three levels were eligible and level 1 had a positive difficulty of about 0.0088. This is evidence that the maximum-level rule can ignore the measured difficulty ordering. It is not evidence that level 1 would produce better validation AP.')
paragraph('For 94.26% of applied edits, both family probabilities were within 0.05 of one half. Consequently, much of this run’s adaptation was expressed through admissible strength and fallback behavior rather than sharply different family weights. A zero clipped net score does not imply an unchanged image, no harder object, or no possible regularization benefit. The score’s aggregation and clipping prevent those stronger interpretations.')
paragraph('A leave-one-source-cluster-out sensitivity check found six selected states whose retention decision failed under at least one cluster deletion. They routed 3,186 applications. This is a descriptive sensitivity check, not a confidence interval or a formal significance test; nevertheless, it shows that the minimum support threshold does not guarantee a stable decision.')
heading('8 2 The auxiliary loss compares changing ROI samples',2)
paragraph('Across 20,752 correlated object-operation comparisons at selected states, the number of assigned positive ROIs changed in 86.64% of cases. Auxiliary loss decreased in 77.90%. The mean ROI count rose from 15.05 in the reference view to 16.87 in the transformed view. In zero-score states, mean count rose from 15.91 to 18.19 and loss decreased in 86.49% of comparisons.')
paragraph('The measured association between loss change and ROI-count change was approximately -0.534. These repeated observations are correlated, so the coefficient is descriptive and does not establish causation. More fundamentally, the observer averages over a different proposal sample after the transformation. An apparent loss decrease can combine an appearance effect with changes in which ROIs are measured. It is therefore too strong to interpret this auxiliary difference as an isolated estimate of object difficulty.')
paragraph('The records do not support the specific explanation that harder proposals disappeared and caused the decrease; average proposal counts increased. Saved feedback contains object summaries and ROI counts but not the full per-ROI boxes and logits required to reconstruct a fixed-ROI counterfactual. A new frozen-checkpoint probe is needed to evaluate that measurement change.')
heading('8 3 Sparse panels rather than absent training examples',2)
paragraph('Among 20,709 opportunities for mixed-class groups, 12,119 received identity, a fallback fraction of 58.52%. Annotation availability alone implies that at least 8,798 of the total 13,583 fallbacks could not have been avoided on the old panels even with perfect reference detection. Thus a meaningful part of the support problem is panel allocation, not just detector failure.')
table('Table 18  Training availability for the highest exposure exact sets', ['Exact class set','Groups','Images','Source clusters','Active updates'],[[r['labels'],integer(diagnosis['support']['training_inventory_by_exact_set'][r['class_set']]['groups']),integer(diagnosis['support']['training_inventory_by_exact_set'][r['class_set']]['images']),integer(diagnosis['support']['training_inventory_by_exact_set'][r['class_set']]['source_clusters']),r['active_updates']+'/10'] for r in sets if r['class_set'] in ['0','1','2','3','0,1','0,2','1,2','2,3']], [2.0,1.1,1.1,1.5,1.2])
paragraph('D00+D20 is present in 440 training source clusters but never activated; D10+D20 is present in 371 clusters and also never activated. D20+D40 appears in 637 clusters and activated once. Those examples show that poor panel support is not equivalent to an absence of the group type in the training pool. Availability is only an upper bound on detectable support, and inventories across sets can share source clusters or images.')
paragraph('The four singleton sets plus D00+D10, D00+D20, D10+D20, and D20+D40 account for 156,158 of 158,151 opportunities, or 98.74%. This motivates prioritizing those eight strata within the same probe budget. It does not justify claiming that a stratified panel will necessarily meet detector-based support or improve AP, and it does not remove the need to record unsupported rare sets.')

heading('9 Verification and evidential limits')
paragraph('The latest class-adaptive archive contains 123 members. Member checksums and checksum coverage were verified, archived source bytes match the released package, and configuration/environment records match the declared protocol. The frozen training/validation separation, original-coordinate validation ground truth, epoch histories, prediction coverage and geometry, training budget, and best/final checkpoint metadata were checked. [S4]')
paragraph('Independent COCO evaluation recomputed the selected epoch 18 and final epoch 20, including per-class metrics, with maximum discrepancy zero. All ten controller updates and all 15 exact-set states per update were reconstructed from saved per-object observations using an independent aggregation implementation. Training decisions, sample order, group geometry, fallbacks, and exposure counts were replayed across all 20 epochs. Earlier conditions retain their prior independent metric audits; the comparison also checked their original archive hashes and saved records.')
paragraph('These audits do not constitute independent reruns of the detector or raw-pixel pipeline. Full checkpoint tensors were not re-executed locally, and raw probe boxes/logits were not archived. Reported pixel-change counters and state-preservation flags were checked for consistency with source and records, not reproduced from a fresh live GPU model. Unit tests and successful training are complementary evidence, not substitutes for performance confirmation.')
paragraph('The principal scientific limitations are one seed per condition, checkpoint selection on the same validation set used for reporting, a narrow operation bank, no full YOLO or scratch-initialization replication, no matched fixed-strength realization control, unequal online forward cost, and unequal edit exposure after class-set fallbacks. Warmup trajectory differences limit causal attribution. The observed support and ROI issues identify plausible intervention targets, but do not prove which mechanism caused the AP result.')
paragraph('No held-out test score, cross-country generalization claim, severity-specific result, robustness benchmark, deployment assessment, or repeated-seed uncertainty estimate is reported. The small selected-subset checks, crop-classification results, and full-image validation scores answer different questions and remain separate throughout the report.')

heading('10 Changes recommended before another full experiment')
paragraph('The following changes are proposed, not implemented or validated by the completed experiments. Their purpose is to test the diagnosed mechanisms with less computation before launching another full run. Preserve the version-1 results and thresholds as the historical comparison; a revised controller must be recorded as a new condition.')
heading('10 1 Stabilize the auxiliary observer',2)
paragraph('At a frozen detector checkpoint, obtain reference-view ROIs and their object assignments once, then reuse those ROI coordinates and labels for the reference and transformed feature maps. Continue to use the detector’s ordinary view-specific predictions for retention. This separates a fixed-sample appearance-loss comparison from the ordinary detection-stability measurement. Archive the ROI identities, boxes, labels, logits, and losses needed to reproduce the observer. The new observer may change scores; improved AP remains an untested hypothesis.')
heading('10 2 Allocate probes to exact sets deliberately',2)
paragraph('Keep the total 256-image probe budget and distinct-source-cluster preference, but allocate support explicitly across the four singleton and four common mixed sets. Resolve images or clusters shared by multiple strata deterministically and report the actual sample budget. Retain the minimum eight detected objects and four source clusters, and keep identity fallback for unresolved cases. Do not lower the retention threshold or borrow unsupported class information merely to increase activity. Assess whether coverage gains cost too much support in previously well-measured sets.')
heading('10 3 Test the strength rule separately',2)
paragraph('Compare the weakest admissible tested strength with the existing strongest-admissible rule under the same retention threshold of 0.8. The weakest-passing choice is a diagnostic control, not a declared optimal method. Do not replace the maximum-level rule with maximum difficulty until the observer’s proposal-composition issue has been examined. A fixed mild-strength control and a uniform-family control can then distinguish the benefit of feedback from the realized transformation distribution.')
heading('10 4 Use paired training starts and explicit continuation gates',2)
paragraph('Create a common unaugmented warmup checkpoint for each seed and branch comparison arms from the same model, optimizer, random states, and subsequent sample order. Check repeatability before attributing a small difference to policy. Record GPU/library settings and any remaining nondeterminism. One common checkpoint improves pairing, but repeated seeds are still needed to estimate reliability.')
paragraph('First complete the frozen-checkpoint observer and support diagnostic. Continue only if it produces reproducible measurements, preserves ordinary predictions outside the intended changes, and demonstrates usable support for the prioritized sets at the fixed budget. Next run short paired training comparisons that change observer, panel allocation, and strength rule separately or in an explicitly designed factorial study. Predeclare the accuracy, support, exposure, and compute criteria before those outcomes are seen. No numerical AP improvement threshold is retroactively invented in this report.')
paragraph('If those checks fail, further full-budget training should wait for the identified issue to be resolved. If they succeed, freeze the revised protocol, execute matched seed repetitions, and reserve the test partition for the final settled comparison. The project can currently claim a working and audited class-adaptive prototype with documented negative development evidence. It cannot yet claim that class adaptation improves road-damage detection.')

heading('11 Evidence inventory and reproducibility record')
paragraph('The sources below are local project records used for this report. Earlier review files contain historical pending statuses; this report resolves those statuses using the later completed-run evidence. Numeric tables are generated from saved machine-readable records rather than copied from screenshots. The companion report source manifest records SHA-256 values for the directly loaded inputs. No training job, inference run, test evaluation, or source-controller change was performed to prepare this report.')
paragraph('Project root: C:\\Users\\SOLOMON\\OneDrive - North Dakota University System\\deep_learning_project. Paths below are relative to this root for portability within the report package.')
evidence = [
('S1','Dataset and exclusions','detection_splits_v1/summary.json; config.json; excluded_images.json; excluded_objects.json; reports/detection_split_review_v1/verification.json.'),
('S2','Executed method and training recipe','fasterrcnn_class_adaptive_v1/protocol.json; controller.py; class_set_feedback.py; probe_detector.py; probe_runner.py; local_augmentation.py; shared_augmentation.py; baseline.py.'),
('S3','Five-condition numeric comparison','reports/fasterrcnn_class_adaptive_v1/aggregate_comparison.csv; per_class_comparison.csv; negative_image_comparison.csv; runtime_comparison.csv; comparison_curves.csv; comparison.json.'),
('S4','Completed class-adaptive audit','reports/fasterrcnn_class_adaptive_v1/ccast_verification.json; controller_verification.json; RESULTS.md; class_set_summary.csv; recomputed_metrics.json.'),
('S5','Earlier detector-condition audits','reports/fasterrcnn_baseline_v1/ccast_verification.json; reports/fasterrcnn_shared_v1/ccast_verification.json; reports/fasterrcnn_local_v1/ccast_verification.json; reports/fasterrcnn_online_shared_v1/ccast_verification.json and controller_verification.json.'),
('S6','Historical crop classifier','reports/conventional_seed42_v1/analysis/RESULTS.md; verification.json; per_class_metrics.csv.'),
('S7','Real-image loader and integration','reports/detection_loader_v1/ccast_verification.json; reports/detection_integration_v1/ccast_verification.json; corresponding test outputs retained in the project conversation.'),
('S8','Training-only learning check','reports/detection_learning_v1/metrics.csv; ccast_verification.json; post_learning_review.json.'),
('S9','Completed YOLO normalization diagnostic','reports/detection_diagnostics_v1/post_diagnostic_review.json; ccast_verification.json; metrics.csv.'),
('S10','Preparation and frozen split verification','reports/detection_preparation_v1/verification.json; reports/detection_split_review_v1/verification.json; detection_preparation_v1/tests/test_audit_detection_data.py.'),
('S11','Local class-adaptive implementation checks','fasterrcnn_class_adaptive_v1/LOCAL_VALIDATION.json; tests/test_class_controller.py; tests/test_online_resume.py.'),
('S12','Retrospective strength and support analysis','reports/fasterrcnn_class_adaptive_v1/strength_support_diagnostics.json; strength_selection_diagnostic_cells.csv; class_set_training_inventory.csv; diagnose_strength_support.py.')]
for label,title,path in evidence:
    blocks.append({'type':'paragraph','text':'['+label+'] '+title+'. '+path})
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(7); p.paragraph_format.keep_together=True
    p.add_run('['+label+'] '+title+'.\n')
    p.add_run(path).font.size=Pt(9.5)
heading('11 1 Returned detection result archives',2)
archives = [
('Identity','fasterrcnn_identity_seed42_v1_records_b07d6daa294f.tar.gz','4d16ae384cc8f8663b47a00f84df331ea32c44f3742aebaa71e114b55fc0026f'),
('Fixed image level','fasterrcnn_shared_seed42_v1_records_31a774d9401a.tar.gz','28b28bee01376cc10fe0a60dca7ddfb9cd97ec9b548597fe316d94db164bfd7c'),
('Fixed local','fasterrcnn_local_seed42_v1_records_b047925ba90a.tar.gz','6f1efc305780b1ab018a807bd728460b04582586166f6aeb616e693c6cfdcbac'),
('Online shared local','fasterrcnn_online_shared_seed42_v1_records_437f03a837f3.tar.gz','b30df4e003be14cf8a53059341451c5101261d98d32e58cdc94866c793f0a231'),
('Class adaptive local','fasterrcnn_class_adaptive_seed42_v1_records_26e50b915ed9.tar.gz','c4817b3cb3bc69813776d06662a9070a7415e9552cab7c6b331c5237414d9b74')]
for label,name,sha in archives:
    p=paragraph(label); p.paragraph_format.keep_with_next=True
    p=paragraph(name+'\nSHA-256 '+sha)
    for run in p.runs: run.font.size=Pt(9)
doc.core_properties.title='Road Damage Augmentation Experiments and Results'
doc.core_properties.subject='Detailed experimental report through 27 October 2026'
doc.core_properties.author=''
doc.core_properties.keywords='RDD2022, Faster R-CNN, augmentation, class-adaptive, experimental report'
doc.core_properties.comments='Generated from verified project records; proposed changes are identified as untested.'
doc.save(FINAL/(STEM+'.docx'))
(OUT/'report_content.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'source_manifest.json').write_text(json.dumps({'report_date':'2026-10-27','inputs':sources,'new_experiments':False},indent=2),encoding='utf-8')
print(json.dumps({'docx':str(FINAL/(STEM+'.docx')),'paragraphs':len(doc.paragraphs),'tables':len(doc.tables),'equations':8,'word_count_approx':sum(len(b.get('text','').split()) for b in blocks)},indent=2))
