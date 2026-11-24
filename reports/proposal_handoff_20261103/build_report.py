from pathlib import Path
import csv
import hashlib
import json
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'reports' / 'Class_Adaptive_Augmentation_Project_Proposal_and_Related_Work_2026-11-03.docx'
OUT.parent.mkdir(parents=True, exist_ok=True)
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = Inches(.8)
sec.left_margin = sec.right_margin = Inches(.9)
sec.header_distance = sec.footer_distance = Inches(.35)

for name in ('Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Heading 3', 'Caption', 'List Bullet', 'List Number', 'Header', 'Footer'):
    s = doc.styles[name]
    s.font.name = 'Times New Roman'
    s.font.color.rgb = RGBColor(0, 0, 0)
    s.font.underline = False
    s.font.size = Pt(11.5)
    s.paragraph_format.space_after = Pt(6)
    s.paragraph_format.widow_control = True
    rpr = s.element.get_or_add_rPr()
    for color in rpr.findall(qn('w:color')):
        for a in ('themeColor', 'themeTint', 'themeShade'):
            color.attrib.pop(qn('w:' + a), None)
    ppr = s.element.get_or_add_pPr()
    for border in ppr.findall(qn('w:pBdr')):
        ppr.remove(border)

normal = doc.styles['Normal'].paragraph_format
normal.line_spacing = 1.08
doc.styles['Title'].font.size = Pt(20)
doc.styles['Title'].font.bold = True
doc.styles['Title'].paragraph_format.space_after = Pt(8)
doc.styles['Subtitle'].font.size = Pt(12)
doc.styles['Subtitle'].font.italic = False
for name, size in [('Heading 1', 14), ('Heading 2', 12)]:
    s = doc.styles[name]
    s.font.size = Pt(size)
    s.font.bold = True
    s.paragraph_format.space_before = Pt(12)
    s.paragraph_format.space_after = Pt(5)
    s.paragraph_format.keep_with_next = True
doc.styles['Caption'].font.size = Pt(10.5)
doc.styles['Caption'].font.italic = False
doc.styles['Caption'].paragraph_format.keep_with_next = True

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('Page ').font.size = Pt(9)
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)

all_text = []
def para(text, style=None):
    follows_heading = len(doc.paragraphs) > 0 and doc.paragraphs[-1].style.name.startswith('Heading')
    p = doc.add_paragraph(text, style)
    if follows_heading:
        p.paragraph_format.keep_together = True
    if text.startswith(('Tao et al.', 'An independent reviewer should')):
        p.paragraph_format.keep_together = True
    all_text.append(text)
    return p

def heading(text, level=1):
    all_text.append(text)
    return doc.add_heading(text, level)

def point(label, text):
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(label + ' ').bold = True
    p.add_run(text)
    all_text.append(label + ' ' + text)
    return p

def table(headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    props = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:' + side)
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '4')
        el.set(qn('w:color'), '000000')
        borders.append(el)
    props.append(borders)
    margins = OxmlElement('w:tblCellMar')
    for side, value in [('top', 70), ('bottom', 70), ('left', 90), ('right', 90)]:
        el = OxmlElement('w:' + side)
        el.set(qn('w:w'), str(value))
        el.set(qn('w:type'), 'dxa')
        margins.append(el)
    props.append(margins)
    repeat = OxmlElement('w:tblHeader')
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = h
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    for ri, row in enumerate(t.rows):
        nosplit = OxmlElement('w:cantSplit')
        row._tr.get_or_add_trPr().append(nosplit)
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shading = OxmlElement('w:shd')
            shading.set(qn('w:fill'), 'FFFFFF')
            cell._tc.get_or_add_tcPr().append(shading)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.keep_with_next = ri < len(t.rows) - 1
                if ci > 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(10.5)
                    r.font.color.rgb = RGBColor(0, 0, 0)
                    r.bold = ri == 0
    for ci, w in enumerate(widths):
        t.columns[ci].width = Inches(w)
    all_text.extend([' | '.join(headers)] + [' | '.join(map(str, r)) for r in rows])
    para('').paragraph_format.space_after = Pt(0)
    return t

def hyperlink(p, label, url):
    link = OxmlElement('w:hyperlink')
    link.set(qn('r:id'), p.part.relate_to(url, RT.HYPERLINK, is_external=True))
    run = OxmlElement('w:r')
    rp = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '000000')
    rp.append(color)
    size = OxmlElement('w:sz')
    size.set(qn('w:val'), '21')
    rp.append(size)
    run.append(rp)
    text = OxmlElement('w:t')
    text.text = label
    run.append(text)
    link.append(run)
    p._p.append(link)


para('Class Adaptive Augmentation for Multi Class Object Detection', 'Title')
para('Project proposal and related work limitations', 'Subtitle')
para('Prepared for independent research review and project handoff\n3 November 2026')

heading('1 Project purpose and scope')
para('The proposed project aims to improve multi-class object detection through class-adaptive data augmentation. Its motivating problem is that a transformation can benefit the learning of one class while providing little benefit or harming another class in the same image. The intended contribution is a framework that accounts for these differences when choosing augmentation, with improved detection as the principal goal.')
para('The intended scope extends across application domains. Road damage detection is the existing application and source of preliminary evidence, rather than a restriction on the eventual method. Claims of wider applicability remain prospective until supported by evidence from other domains. The relevant task is full-image object detection, where several labelled objects can coexist, rather than classification of isolated object crops.')
para('This report specifies the research problem, objectives, related-work limitations and current evidence. It deliberately excludes the proposed methodology: no controller architecture, selection rule, equations, implementation plan or new experimental protocol is prescribed. Earlier implementations are historical evidence and do not constrain the next design.')
para('The literature already contains class-adaptive, instance-adaptive, online, local and detection-specific augmentation. The research opportunity therefore cannot rest on any of those labels alone. The defensible candidate gap concerns how class-dependent augmentation benefits are handled jointly in multi-object images, including effects on other classes and on detection performance. Whether a new solution offers a novel and useful advance remains to be established.')

heading('2 Research problem and motivation')
heading('2 1 Different objects can respond differently to the same transformation', 2)
para('Augmentation changes the examples a detector learns from. Its value depends on whether those changes introduce useful variation while retaining the evidence needed to identify and localize each object. Operation type, severity, object size, appearance, context and the current state of learning can all matter. A single policy shared across an image does not explicitly distinguish these factors for each class.')
para('For illustration, blur could remove a thin crack cue while leaving a larger damaged region recognizable. A change that suppresses a background shortcut could help one class yet erase a discriminative cue for another. These examples explain the hypothesis; they are not established causal findings from the project. Published classification research documents uneven class-level effects, but those results do not by themselves establish the same mechanisms in road damage detection. [11]')
heading('2 2 Multi object images introduce coupled decisions', 2)
para('An image can contain several object classes, so a whole-image decision exposes them to the same transformation. Local editing can reduce this coupling, but bounding boxes may overlap, include background, or contain another object. Context outside the edited object can also be useful for detection. Existing work already addresses parts of this spatial problem through object-aware and context-aware augmentation. [6–9]')
para('There is also a learning interaction: all classes share detector parameters. Even spatially separate edits can change the representation learned for other classes. Spatial separation therefore does not guarantee an absence of cross-class harm. The project must distinguish direct changes to another object’s pixels from changes to learning through the shared model. These are related but different concerns.')
heading('2 3 Recognizability does not establish learning benefit', 2)
para('A transformed object remaining recognizable to the current detector is evidence about its immediate response. It does not establish that training on that transformation will improve later detection. Conversely, a difficult transformed example is not automatically harmful. The central unresolved issue is how to distinguish useful difficulty from destruction of relevant information. Prior methods already use validation performance or estimated training gain to guide augmentation, so introducing a benefit signal alone would not establish novelty. [3–5]')
heading('2 4 Augmentation is broader than geometry', 2)
para('Augmentation includes geometric changes, appearance changes and compositional or object-based changes. None of these families should be excluded simply because an earlier implementation did not use it. Eligibility depends on the task and annotation meaning. An operation valid for one domain or label definition may be invalid for another.')
para('Geometry-only was a comparison condition in the existing experiments: it measured whether the particular additional local appearance policies improved detection beyond the geometry already used. It did not define the scope of augmentation or show that geometry is universally sufficient. Its strong result is evidence about those tested recipes, not a reason to narrow the proposed research to geometry.')

heading('3 Research aim and objectives')
para('Primary aim: develop a class-adaptive augmentation framework that improves multi-class object detection by exploiting useful variation while accounting for adverse effects on co-occurring classes.')
point('Class sensitivity.', 'Identify which augmentation effects differ meaningfully across classes and when differences attributed to class are instead explained by size, context, frequency or instance appearance.')
point('Joint scene effects.', 'Address the fact that an augmentation decision can affect several objects, including objects outside the intended target region.')
point('Detection benefit.', 'Connect the adaptation objective to better recognition and localization, rather than treating policy diversity or current-model tolerance as sufficient evidence of success.')
point('General applicability.', 'Seek a formulation that is not tied to road-specific class names, while allowing transformation validity to depend on the domain.')
point('A defensible contribution.', 'Establish what the proposed framework adds beyond existing class-adaptive, instance-adaptive and detection-specific methods and their straightforward combinations.')
heading('3 1 Research questions', 2)
point('RQ1.', 'Under what conditions do classes within the same image benefit from conflicting augmentation choices?')
point('RQ2.', 'Can class-adaptive augmentation improve detection while accounting for effects on other classes and shared context?')
point('RQ3.', 'What does explicit class information add beyond instance appearance, object scale and a strong shared augmentation policy?')
point('RQ4.', 'Does the resulting benefit extend beyond a single dataset, detector or domain, and is it worthwhile relative to the additional computation?')
para('Understanding when and why augmentation helps supports the performance objective: it identifies which problem the framework must actually solve. It is not a substitute goal that permits a claim of improved detection without a measured improvement.')
heading('3 2 Intended contribution and present status', 2)
para('The intended contribution is a detection-oriented treatment of class-dependent augmentation interactions, supported by evidence of useful detection gains. A framework may still involve trade-offs between classes; a universal guarantee that every class always improves would be unjustified. The project should make those trade-offs visible instead of hiding them within a dataset average.')
para('At present, the project has working implementations and completed comparisons, but it has not established the intended detection advantage, cross-domain transfer or superiority over named published methods. The proposal is therefore a research objective supported by an identifiable problem, not a claim that a successful new method already exists.')

heading('4 Related works and their limitations')
para('The discussion below distinguishes a paper’s reported capability from a limitation relative to this project. A difference in task or scope is not proof that the published method performs poorly. Where an absence is inferred from a reviewed formulation, it is a candidate gap requiring confirmation, not an experimentally demonstrated failure.')
heading('4 1 Shared augmentation policies', 2)
para('AutoAugment searches for augmentation policies; RandAugment simplifies policy control through a small set of shared settings; TrivialAugment samples an operation and magnitude without a learned policy. These methods establish strong shared-policy alternatives and show that a complicated controller needs a substantive justification. Their standard policies do not explicitly coordinate separate class-dependent choices for objects within one image. That scope limitation makes them useful background, but they are insufficient by themselves to establish novelty for class adaptation because later work already addresses it. [16–18]')

heading('4 2 Online and class dependent augmentation', 2)
para('CTAugment, introduced with ReMixMatch, learns magnitude-bin preferences from the network’s responses to labelled examples. Its statistics are shared rather than indexed by object class. The reviewed formulation concerns semi-supervised classification and does not specify regional decisions for several labelled objects. Its relevance is online adaptation; its limitation for this project is the absence of explicit multi-object class coordination. Its response-based signal should not be described as a direct measurement of the downstream learning benefit of each transformation. [1]')
para('CUDA already adapts augmentation strength separately for each class. Its class-specific strength influences augmentation count and magnitude, while operations are sampled uniformly. It therefore directly contradicts any claim that online class adaptation itself is new. Its classification formulation does not resolve how different object classes in one image receive compatible decisions, and a single class-strength variable does not independently learn the usefulness of each operation. These are formulation differences, not evidence that adapting CUDA to detection would necessarily fail. [2]')
para('MADAug uses current model features to condition augmentation operation probabilities and magnitudes, with a validation-based learning objective. Such instance conditioning can implicitly capture class differences, so describing it as class-blind would be misleading. Its reviewed formulation operates on classification examples; it does not explicitly assign coordinated local decisions to multiple annotated objects. The remaining question is whether explicit multi-object class treatment adds value beyond this already adaptive, instance-sensitive alternative. [3]')
para('AdaAug explicitly learns class- and instance-adaptive augmentation policies. Yoo and Kang also propose class-adaptive augmentation for image classification. These are necessary neighboring works, not omissions that can be dismissed because the project uses a detector. The primary descriptions establish that class-conditioned augmentation predates this project. Detailed claims about their treatment of local interactions require closer full-method assessment; this report does not infer such limitations from an abstract alone. [14,15]')

heading('4 3 Augmentation selected for learning benefit', 2)
para('BDA estimates an augmentation’s net prediction gain by comparing updates from original and augmented data using validation loss, then uses that information in a bandit formulation. It is a direct precedent for tracking which transformations improve learning. Its reported reward is aggregated rather than an explicit coordination of object-class-specific regional effects within a detection image. A new benefit-tracking module would therefore need to demonstrate an advance beyond BDA’s existing learning-gain idea. [4]')
para('DDAS uses a differentiable search objective based on validation performance and includes object detection experiments on COCO. It cannot be characterized as a classification-only method or as relying solely on model tolerance. Its searched policies can vary over training, while the reported formulation does not explicitly allocate class-conditioned regional decisions among co-occurring objects. The relevant distinction is the proposed handling of class interactions, not merely use of validation feedback or detection augmentation. [5]')

heading('4 4 Object and context aware detection augmentation', 2)
para('Zoph et al. search augmentation strategies specifically for object detection, including image-level and bounding-box-level operations. Local augmentation and detection-driven policy search are therefore established capabilities. The reported policy does not explicitly maintain online semantic-class preferences and reconcile their effects in each mixed-class scene. That leaves a possible extension, but adding bounding-box edits alone would repeat prior work. [6]')
para('Scale-Aware Automatic Augmentation accounts for object scale in image- and box-level augmentation. It includes softened local blending and a scale-balance criterion that penalizes harm to object-size groups. It already addresses unequal effects across groups and some artifacts of local editing. Its grouping concerns scale rather than semantic class; whether class adds useful information beyond scale remains unresolved. A claim that earlier work never protects a disadvantaged group would be false. [7]')
para('Progressive and Selective Instance-Switching uses class-level validation performance to allocate more instance-switching augmentation to weaker classes. It also handles attached objects during switching. It is a direct precedent for class-aware augmentation in object detection. Its control concerns the quantity and selection of instance replacements rather than a general class-dependent choice among operation types and severities. Low class performance also does not identify which alternative transformation would improve learning; that latter observation is an analytical limitation of the signal. [8]')
para('InterAug explicitly uses relationships between objects to construct context-aware augmentation regions, including regions intended to reduce unwanted interference with other objects. It directly weakens a broad claim that previous local augmentation ignores co-occurring objects. The reported augmentation sampling does not itself learn a class-specific benefit policy, but the region construction can be combined with other policies. An adaptive extension of InterAug is therefore a serious alternative to consider, not something this report can rule out by definition. [9]')
para('RangeAugment learns augmentation ranges online and reports applications including object detection. It combines task learning with an image-similarity objective. Its reviewed range parameters are shared rather than explicitly conditioned on each object class, and it does not prescribe local multi-object coordination. Online adaptation across tasks is already established; the potential distinction must concern what class-dependent information changes the decisions and improves detection. [10]')

heading('4 5 Evidence on class effects and road damage precedents', 2)
para('Kirichenko et al. document detrimental class-level effects that can be hidden by aggregate classification accuracy, examine mechanisms and class-conditional interventions, and discuss annotation ambiguity. This supports taking uneven class effects seriously. It does not prove that the project’s road detection losses were caused by conflicting transformations. Annotation quality, class overlap and other explanations must remain possible. It also means that identifying class-level harm is not, by itself, a new contribution. [11]')
para('Tsuchiya et al. evaluate transformations by class and construct augmented road-damage training data using the selected class-relevant operations. Despite the title’s use of “classifying,” the work uses YOLOv3 detection. It is an especially close domain precedent. Its augmentation choices are derived before subsequent training and applied at image level when relevant classes are present, rather than resolving separate regional preferences online in a mixed-class image. Road-specific class-aware augmentation cannot be claimed as new without addressing this work. [12]')
para('Tao et al. investigate class- and modality-dependent augmentation for pavement distress detection with paired intensity and range imagery. Their image-level choice combines class preferences, including a maximum-strength aggregation, and the policy is fixed for subsequent experiments. This already uses the image’s class composition. The possible gap is finer handling of competing regional effects and changing learning needs, rather than awareness that classes differ. Their paired-sensor setting also differs from the project’s RGB data. [13]')

heading('5 What the literature supports as a research gap')
para('The evidence supports a narrower and more demanding problem than “make augmentation class-adaptive.” Existing studies already supply many relevant components. The candidate contribution is whether class-dependent learning effects can be accounted for jointly in full-image detection in a way that improves on those components and their reasonable combinations.')
point('Multi object decisions.', 'Classification policies usually act on an example with a single training label. Detection requires considering several annotated objects and their spatial or contextual relationships. However, InterAug and instance-switching already address parts of that interaction.')
point('Benefit attribution.', 'Current performance, object retention and class weakness provide different information from an augmentation’s effect on future learning. Learning-benefit objectives already exist; the unresolved opportunity is class-specific attribution and coordination in detection.')
point('Additional value of class information.', 'Class may be a useful conditioning variable, but it can also stand in for size, frequency or appearance. A contribution needs evidence that class adaptation adds something beyond existing instance- and scale-aware alternatives.')
point('Broader applicability.', 'A general framework would need to accommodate domain-specific label validity without being encoded around four road-damage categories. Cross-domain usefulness is an ambition that the current evidence does not establish.')
para('These are candidate research gaps inferred from a bounded review, not a verified claim that no previous system combines the relevant ideas. The strongest novelty risk is that the eventual proposal becomes a straightforward combination of an existing adaptive policy, a learning-gain signal and an existing object-aware region rule. A defensible proposal must identify the substantive problem that such a combination still leaves unresolved.')
para('Claims that should not be made include: the first class-adaptive augmentation method; the first online augmentation method for detection; the first local object augmentation; the first method to consider co-occurring objects; or the first method to select augmentation using validation benefit. Each conflicts with prior capabilities summarized above.')

heading('6 Evidence already produced by this project')
para('The following results are included only to preserve the project’s factual starting point. They are single-seed validation results on road damage detection with Faster R-CNN. AP50:95 is reported on a 0–100 scale. The project has not yet demonstrated an improvement against CUDA, CTAugment, MADAug, BDA, DDAS or other named published methods in a common detection comparison.')

para('Table 1  Version 1 results at each condition’s best validation checkpoint', 'Caption')
with (ROOT / 'reports/fasterrcnn_class_adaptive_v1/aggregate_comparison.csv').open(newline='', encoding='utf-8-sig') as f:
    best = [r for r in csv.DictReader(f) if r['selection'] == 'best']
table(['Condition', 'Best epoch', 'AP50:95'], [(r['label'], r['epoch'], f"{float(r['AP50_95']):.4f}") for r in best], [4.15, 1.1, 1.45])
para('All five conditions completed 20 epochs. The class-adaptive condition was 0.4962 AP points below online shared local and 0.2599 points below identity at the selected checkpoints. This implementation did not establish the intended advantage. Sparse support and the strength-selection behavior motivated revisions, but their individual causal contributions were not isolated. [W1]')

para('Table 2  Version 2 results at fixed endpoints', 'Caption')
table(['Comparison', 'Condition', 'AP50:95'], [
    ('Four arm pilot at epoch 4', 'Geometry only', '19.8080'),
    ('Four arm pilot at epoch 4', 'Fixed local', '18.8610'),
    ('Four arm pilot at epoch 4', 'Online shared local', '19.3656'),
    ('Four arm pilot at epoch 4', 'Class adaptive local', '19.6076'),
    ('Strict two arm run at epoch 8', 'Geometry only', '19.6846'),
    ('Strict two arm run at epoch 8', 'Class adaptive local', '18.4369'),
], [2.7, 2.55, 1.45])
para('In the four-arm pilot, class adaptation exceeded fixed and shared local appearance augmentation but remained below geometry-only. A numerical repeatability discrepancy was subsequently investigated. Strict checks matched the short trajectories examined; these checks do not prove the absence of every implementation error. In the later strict two-arm run, class adaptation was 1.2477 AP points below geometry-only at epoch 8. [W2–W4]')
para('These comparisons have different horizons and checkpoint-selection rules, so their absolute scores must not be used to infer a causal improvement or deterioration from one version to another. No held-out test claim, multiple-seed conclusion or cross-domain conclusion follows from them. The results show that the tested class-adaptive recipes did not improve the principal detection metric over the relevant stronger control. They do not establish that class-adaptive augmentation in general cannot help.')
para('The project has also produced data preparation, detector integration, metric checks, checkpoint checks and debugging evidence. Those establish useful engineering capability. They do not substitute for evidence of scientific novelty or better detection, and the completed comparisons did not isolate which transformations caused each class’s learning gain or loss.')

heading('7 Evidence boundaries and independent review brief')
para('The established facts are that augmentation can have uneven class effects in published settings; class-aware and detection-aware augmentation already exist; and the project’s tested recipes have not established its intended advantage. The inference is that joint class-dependent decisions in detection may still offer a useful research opportunity. The unknowns are whether the exact problem is sufficiently prevalent in the intended data, whether a proposed solution is meaningfully distinct from prior work, and whether it improves detection at a reasonable cost.')
para('An independent reviewer should preserve the performance objective and assess the candidate gap before inheriting any previous controller. The review should identify the closest existing method or combination, distinguish measured cross-class conflicts from plausible examples, and explain what capability remains missing. The choice of algorithm and eligible augmentation families is deliberately left open in this document.')
heading('7 1 Literature coverage', 2)
para('This is a targeted review of closely related works, not a systematic survey or a full reproduction audit. Direct primary method text was inspected for ReMixMatch/CTAugment, CUDA, MADAug, DDAS, detection AutoAugment, instance-switching and RangeAugment. The local Tsuchiya paper was inspected, and Kirichenko’s paper and relevant findings were reviewed. BDA, Scale-Aware AutoAugment, InterAug and Tao were assessed using indexed passages from primary papers where direct retrieval was restricted. These passages support the specific claims made here, but they are not a complete implementation audit.')
para('AdaAug and Yoo–Kang are included on the basis of verified primary descriptions; their finer mechanisms have not been comprehensively reassessed for this report. AutoAugment, RandAugment and TrivialAugment are background references whose central policy mechanisms were checked. Any final novelty statement should follow a full-method and code comparison with the closest alternatives, particularly those with incomplete coverage here.')

heading('References')
references = [
('Berthelot et al. ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring. ICLR, 2020. Source of CTAugment.', 'https://arxiv.org/html/1911.09785'),
('Ahn, Ko and Yun. CUDA: Curriculum of Data Augmentation for Long-Tailed Recognition. ICLR, 2023.', 'https://arxiv.org/html/2302.05499'),
('Hou, Zhang and Zhou. When to Learn What: Model-Adaptive Data Augmentation Curriculum. ICCV, 2023.', 'https://arxiv.org/html/2309.04747v2'),
('Lu et al. BDA: Bandit-based Transferable AutoAugment. SIAM International Conference on Data Mining, 2023, pp. 550–558.', 'https://doi.org/10.1137/1.9781611977653.ch62'),
('Liu, Huang, Huang and Wang. Direct Differentiable Augmentation Search. ICCV, 2021.', 'https://arxiv.org/html/2104.04282'),
('Zoph et al. Learning Data Augmentation Strategies for Object Detection. ECCV, 2020.', 'https://arxiv.org/html/1906.11172'),
('Chen et al. Scale-Aware Automatic Augmentation for Object Detection. CVPR, 2021.', 'https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scale-Aware_Automatic_Augmentation_for_Object_Detection_CVPR_2021_paper.pdf'),
('Wang et al. Data Augmentation for Object Detection via Progressive and Selective Instance-Switching. Author preprint, 2019.', 'https://arxiv.org/html/1906.00358'),
('Thopalli, Devi and Thiagarajan. InterAug: A Tuning-Free Augmentation Policy for Data-Efficient and Robust Object Detection. ICCV Workshops, 2023, pp. 253–261.', 'https://openaccess.thecvf.com/content/ICCV2023W/VIPriors/papers/Thopalli_InterAug_A_Tuning-Free_Augmentation_Policy_for_Data-Efficient_and_Robust_Object_ICCVW_2023_paper.pdf'),
('RangeAugment: Efficient Online Augmentation with Range Learning. Author preprint, arXiv:2212.10553, 2022.', 'https://arxiv.org/html/2212.10553'),
('Kirichenko et al. Understanding the Detrimental Class-level Effects of Data Augmentation. NeurIPS, 2023.', 'https://arxiv.org/html/2401.01764'),
('Tsuchiya et al. A Method of Data Augmentation for Classifying Road Damage Considering Influence on Classification Accuracy. Procedia Computer Science 159, 2019, pp. 1449–1458.', 'https://doi.org/10.1016/j.procs.2019.09.315'),
('Tao, Bai, Wang and Tešić. Data-Centric Multimodal Pavement Distress Detection Using Intensity–Range Imagery. ACM ICMR, 2026, pp. 2448–2456.', 'https://doi.org/10.1145/3805622.3810611'),
('Cheung and Yeung. AdaAug: Learning Class- and Instance-adaptive Data Augmentation Policies. ICLR, 2022.', 'https://openreview.net/forum?id=rWXfFogxRJN'),
('Yoo and Kang. Class-Adaptive Data Augmentation for Image Classification. IEEE Access 11, 2023, pp. 26393–26402.', 'https://doi.org/10.1109/ACCESS.2023.3258179'),
('Cubuk et al. AutoAugment: Learning Augmentation Strategies From Data. CVPR, 2019.', 'https://openaccess.thecvf.com/content_CVPR_2019/html/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.html'),
('Cubuk et al. RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. NeurIPS, 2020.', 'https://proceedings.neurips.cc/paper_files/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html'),
('Müller and Hutter. TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. ICCV, 2021.', 'https://openaccess.thecvf.com/content/ICCV2021/papers/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.pdf'),
]
for number, (entry, url) in enumerate(references, 1):
    p = para(f'[{number}] {entry}')
    p.paragraph_format.keep_together = True
    p.paragraph_format.space_after = Pt(8)
    for r in p.runs:
        r.font.size = Pt(10.5)
    p.add_run(' ')
    hyperlink(p, 'Primary source', url)

heading('Project evidence sources', 2)
para('The numerical summaries above were checked against the following local records in the deep_learning_project workspace. These records document this project; they are not external published benchmarks.')
for ref, title, paths in [
    ('W1', 'Version 1 audit and aggregate comparison', ['reports/fasterrcnn_class_adaptive_v1/RESULTS.md', 'reports/fasterrcnn_class_adaptive_v1/aggregate_comparison.csv']),
    ('W2', 'Version 2 four arm pilot audit', ['reports/rdd_v2_pilot_v1/RESULTS.md']),
    ('W3', 'Reproducibility and strict execution audits', ['reports/rdd_v2_repro_v2/RESULTS.md', 'reports/rdd_v2_strict_v1/RESULTS.md']),
    ('W4', 'Strict epoch 8 paired comparison', ['reports/rdd_v2_pair8_v1/RESULTS.md', 'reports/rdd_v2_pair8_v1/endpoint_comparison.csv']),
]:
    p = para(f'[{ref}] {title}.\n' + '\n'.join(paths))
    p.paragraph_format.keep_together = True
    for r in p.runs:
        r.font.size = Pt(10)

doc.core_properties.title = 'Class Adaptive Augmentation for Multi Class Object Detection'
doc.core_properties.subject = 'Project proposal and related work limitations without proposed methodology'
doc.core_properties.author = ''
doc.core_properties.keywords = 'class adaptive augmentation, object detection, proposal, related work'
doc.save(OUT)

text = '\n\n'.join(all_text)
(WORK / 'report_text.txt').write_text(text, encoding='utf-8')
stats = {
    'output': str(OUT),
    'words': len(re.findall(r"\b[\w’-]+\b", text)),
    'references': len(references),
    'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
    'scope': 'Proposal, literature limitations and evidence boundaries; no proposed methodology',
    'verified_result_sources': ['reports/fasterrcnn_class_adaptive_v1/aggregate_comparison.csv', 'reports/rdd_v2_pair8_v1/endpoint_comparison.csv', 'reports/rdd_v2_pilot_v1/RESULTS.md'],
}
(WORK / 'build_record.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
print(json.dumps(stats, indent=2))
