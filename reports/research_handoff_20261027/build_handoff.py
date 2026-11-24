from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'output/reports/RDD_Research_Handoff_2026-10-27.docx'
SOURCE = ROOT / 'output/reports/RDD_Experiment_Report_2026-10-27.docx'
REDESIGN = Path(r'C:\Users\SOLOMON\Downloads\RDD_Methodology_Redesign_and_Execution_Plan.docx')
doc = Document(SOURCE)
body = doc._element.body

# Retain the audited historical report, including its native equations and figures.
# Its previous next-step recommendation is superseded by the later redesign discussion.
cut = next(p for p in doc.paragraphs if p.text == '10 Changes recommended before another full experiment')
removing = False
for element in list(body):
    if element is cut._p:
        removing = True
    if removing and element.tag != qn('w:sectPr'):
        body.remove(element)
for p in list(doc.paragraphs)[:4]:
    body.remove(p._p)

for p in doc.paragraphs:
    if p.text == '1 Research objective and current conclusion':
        p.text = '1 Research objective and version 1 conclusion'
    elif p.text.startswith('4 Methodology'):
        p.text = '4 Implemented version 1 methodology'
    elif p.text.startswith('The next justified step is a focused diagnostic'):
        p.text = ('The version-1 results motivate revising the observation rule, support allocation and action bank. '
                  'The subsequent redesign review and the latest version-2 discussion are documented in Sections 10 and 11. '
                  'Version 2 has not been implemented or evaluated; the older version-1 recommendation is not the current execution specification.')
    elif p.text.startswith('No held-out test score, cross-country generalization claim'):
        p.text = ('No test-set, unseen-country, deployment or repeated-seed result is available. '
                  'Crop, training-subset and detection validation results answer different questions.')
    elif p.text.startswith('All singleton sets except D40 were active'):
        p.text = p.text.replace('Ten of the 15 possible sets never activated.', 'Nine of the 15 possible sets never activated.')

for section in doc.sections:
    section.header.paragraphs[0].text = 'RDD2022 research handoff | Status verified 27 October 2026'
    for r in section.header.paragraphs[0].runs:
        r.font.size = Pt(9)

if 'Handoff Code' not in doc.styles:
    doc.styles.add_style('Handoff Code', 1)
cs = doc.styles['Handoff Code']
cs.font.name = 'Consolas'
cs.font.size = Pt(8.5)
cs.font.color.rgb = RGBColor(0, 0, 0)
cs.paragraph_format.line_spacing = 1.0
cs.paragraph_format.space_before = Pt(3)
cs.paragraph_format.space_after = Pt(5)
cs.paragraph_format.keep_together = True
cs.paragraph_format.widow_control = True

def p(text, style=None):
    return doc.add_paragraph(text, style)

def h(text, level=1, new_page=False):
    x = doc.add_heading(text, level)
    if new_page:
        x.paragraph_format.page_break_before = True
    return x

def code(text):
    return p(text.strip('\n'), 'Handoff Code')

def table(headers, rows, widths=None, caption=None, size=10):
    if caption:
        p(caption, 'Caption')
    if widths is None:
        widths = [6.9 / len(headers)] * len(headers)
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        b = OxmlElement('w:' + side)
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), '3')
        b.set(qn('w:color'), '000000')
        borders.append(b)
    t._tbl.tblPr.append(borders)
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
    for ri, vals in enumerate([headers] + list(rows)):
        row = t.rows[0] if ri == 0 else t.add_row()
        rp = row._tr.get_or_add_trPr()
        rp.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            rp.append(OxmlElement('w:tblHeader'))
        for i, val in enumerate(vals):
            c = row.cells[i]
            c.width = Inches(widths[i])
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            mar = OxmlElement('w:tcMar')
            for k, v in [('top', 65), ('bottom', 65), ('left', 75), ('right', 75)]:
                e = OxmlElement('w:' + k)
                e.set(qn('w:w'), str(v))
                e.set(qn('w:type'), 'dxa')
                mar.append(e)
            c._tc.get_or_add_tcPr().append(mar)
            cp = c.paragraphs[0]
            cp.paragraph_format.space_after = Pt(0)
            cp.paragraph_format.line_spacing = 1.03
            cp.paragraph_format.keep_with_next = ri == 0
            r = cp.add_run(str(val))
            r.font.size = Pt(size)
            r.bold = ri == 0
    p('').paragraph_format.space_after = Pt(0)
    return t

# Front matter is inserted before the retained historical sections.
front_start = len(body)
p('Road Damage Augmentation Research Handoff', 'Title')
p('Comprehensive transfer record for a receiving AI | 27 October 2026', 'Subtitle')
p('Researcher: Solomon Owusu Ansah. This document consolidates the project history, implemented algorithms, audited results, '
  'latest proposed revision, execution environment and source locations. It is a handoff record, not a claim that the proposed method succeeded. '
  'No new training experiment was run while preparing it.')
h('Read this first', 1)
p('The implemented class-adaptive controller did not improve the primary validation metric over its online shared comparator. '
  'At each run’s selected checkpoint, AP50:95 was 22.563 for class adaptive, 23.059 for online shared, and 22.823 for identity on a 0–100 scale. '
  'The adaptive difference was −0.496 percentage points versus online shared and −0.260 versus identity. These are descriptive results from one seed, not a significance test.')
table(['Workstream', 'Status at handoff'], [
    ['Data and engineering', 'Frozen full-image splits, checked loaders, GPU integration, metrics, checkpoints and resumable PBS training exist.'],
    ['Earlier crop study', 'One conventional-augmentation custom-CNN baseline completed. It answers a different task from full-image detection.'],
    ['Version 1 detection', 'All five 20-epoch Faster R-CNN conditions completed and were audited. The test partition remains unevaluated.'],
    ['Version 1 diagnosis', 'Strength-selection behavior, changing ROI samples and sparse exact class-set support were measured. Their causal contribution to AP is not established.'],
    ['Version 2', 'A revised direction was discussed: whole-image geometry, local appearance, controlled group copy-paste and per-class admissibility. No version-2 code, package or experiment is present.'],
    ['Immediate next work', 'Resolve the explicit protocol gaps, implement a new versioned package, run preflight checks and a matched short pilot before any full experiment.'],
], [1.55, 5.35])
p('The receiving AI should start with Sections 10–11 and the source map in Section 13, then inspect the audited version-1 code and reports. '
  'Do not restart completed jobs, treat the old manuscript as a version-2 specification, or describe the revised method as validated.')

h('How to use this document', 1, new_page=True)
table(['Sections', 'Purpose'], [
    ['1–3', 'Research question, chronological development, data preparation and frozen splits.'],
    ['4–5', 'Exact implemented version-1 controller and controlled training design, including equations and parameter choices.'],
    ['6–9', 'Numerical comparisons, controller exposure, diagnosis, audit scope and limits of the evidence.'],
    ['10–11', 'Review of the external redesign document, latest version-2 direction and unresolved implementation decisions.'],
    ['12–13', 'CCAST operation and recovery, hashes, code and evidence locations, GitHub status and transfer checklist.'],
    ['14–15', 'Manuscript/course context, collaboration requirements and a prompt to start the next AI session.'],
], [1.0, 5.9])
p('Evidence labels used throughout: completed means an observed execution; verified means checked against saved artifacts within the stated audit scope; '
  'inference means a plausible explanation that has not been isolated experimentally; proposed means a future design. '
  'Unit tests, synthetic data and same-training-image learning checks are software or feasibility evidence, not generalization evidence.')
p('Metric convention: AP values in comparison tables are percentages on a 0–100 scale unless a table explicitly shows fractions. '
  'A difference of 0.5 points means 0.5 percentage points, not a 50% relative gain. The chosen checkpoint maximizes validation AP50:95, with the earliest epoch winning a tie.')
table(['Name in discussion', 'Meaning and package'], [
    ['Identity', 'No study augmentation; fasterrcnn_baseline_v1. Standard resize/letterbox still occurs.'],
    ['Fixed image level', 'Earlier name: fixed shared augmentation; fasterrcnn_shared_v1. The entire image content receives a fixed-bank appearance edit.'],
    ['Fixed local', 'Same fixed operation bank, applied to one overlap group; fasterrcnn_local_v1.'],
    ['Online shared local', 'One feedback-selected local policy shared across classes; fasterrcnn_online_shared_v1.'],
    ['Online class adaptive local', 'Version 1 routes by the exact set of classes in a group; fasterrcnn_class_adaptive_v1. Version 2 proposes pooling evidence per class instead.'],
], [1.6, 5.3])
p('Current local project root: ' + str(ROOT))
p('Current repository: https://github.com/Solo0722/deep_learning_project. At this handoff, local and remote main point to '
  '8ed2567c6f5a26960949e12116f4fa1c0b8a403b. Completed class-adaptive audit files and newer reports are still untracked locally; the repository link alone does not contain this complete handoff state.')
p('Snapshot scope: filesystem and repository status were checked for this document. Remote CCAST paths are supported by saved run evidence and user transcripts; no new SSH inspection of current cluster storage or job state was performed.')
doc.add_page_break()
front_nodes = [e for e in list(body)[front_start - 1:] if e.tag != qn('w:sectPr')]
# A section-properties element is always the final body child; new paragraphs precede it.
for e in front_nodes:
    body.remove(e)
for i, e in enumerate(front_nodes):
    body.insert(i, e)

h('10 Review of the redesign and reasons for changing direction', new_page=True)
p('Source: RDD_Methodology_Redesign_and_Execution_Plan.docx in the user’s Downloads folder. The document is a proposal to assess, '
  'not an instruction that automatically overrides experimental evidence. It was read and compared with the completed audits. '
  'Its two routes were Approach A, context-decoupled local synthesis, and Approach B, feature-space augmentation with an IoU curriculum. '
  'Neither route has been executed in this project.')
h('10 1 Which criticisms are supported', 2)
table(['Issue', 'Evidence and verdict'], [
    ['Exact class-set sparsity', 'Supported. Nine of fifteen possible sets never obtained an active policy. Mixed-class opportunities fell back to identity 58.52% of the time. Training examples do exist for several affected sets; the probe support was the limitation.'],
    ['Strength-selection rule', 'Supported as a design limitation. The strongest passing level was selected even when it had zero clipped difficulty or a lower level had a positive score. Stronger appearance change is not automatically more useful training.'],
    ['Auxiliary loss comparability', 'Supported measurement concern. Reference and edited views used different positive proposals. ROI counts changed in 86.64% of the examined pairs, so loss changes cannot be interpreted purely as the effect on the same object samples.'],
    ['Narrow action bank', 'Fact. Version 1 used six photometric operations plus blur and noise. It did not test context relocation, copy-paste or a broad geometric bank. The missed benefit of those alternatives is still a hypothesis.'],
    ['Mask boundaries caused failure', 'Unproven. A narrow taper could introduce artifacts, but no renderer ablation isolated its effect on accuracy. Smooth blending is a proposed intervention, not an established cure.'],
    ['Class imbalance caused failure', 'Unproven causal claim. Class frequencies differ, but the five runs did not isolate imbalance from other mechanisms. Sampling changes need their own controls.'],
    ['CUDA nondeterminism explains scores', 'Not established. Pre-augmentation epoch-1 trajectories differed, but neither repeated-run variance nor the precise source of those differences was measured.'],
], [1.75, 5.15])
p('Three corrections to older summaries: nine of fifteen class sets never activated (six did); '
  '94.26% counts augmentation applications with near-uniform family probabilities, not policy states; '
  'and the dataset contains seven collections from six countries.')

h('10 2 What Approach A proposed and what survived review', 2)
p('The external document proposed smooth distance-based masks, Fourier-amplitude perturbations, copy-paste with Poisson or color-space matching, '
  'factorization into four class-level policies, and an EMA-based GT-ROI observer. Its useful direction was to address both support fragmentation and '
  'the limited set of appearance-only interventions. Its technical guarantees and budget estimates were too strong.')
table(['Original assertion or suggestion', 'Assessment carried into the revision'], [
    ['Preserving Fourier phase preserves crack semantics', 'Not a guarantee. Amplitude changes can alter fine-edge visibility and contrast; blur is a simple counterexample to phase alone being sufficient. Fourier augmentation was not retained in the initial version-2 bank.'],
    ['GT ROIs and no_grad make feedback zero cost', 'False. Clean and transformed features still require backbone work unless explicitly reused in a valid way. Version 2 must record additional image forwards, ROI work and wall time.'],
    ['Four class estimates solve mixed-set behavior', 'Pooling improves sample availability, but marginal evidence does not prove safety for every overlap combination. Version 2 adds joint admissibility and a supported mixed-group veto.'],
    ['Use a simple minimum of maximum safe levels', 'Unsafe without monotonicity. Each tested action/strength must retain its own admissibility result; a failing lower level cannot be assumed safe because a higher one passed.'],
    ['Copy individual defects from RGB boxes', 'The data has boxes, not verified instance masks or crack skeletons. An RGB box contains pavement/context. Version 2 proposes copying an annotated group with context and explicit label closure, not pretending it is a segmented object.'],
    ['Pasting into no-target images gives clean background', 'No-target annotations do not establish healthy pavement or a valid road region. Recipient areas need review; background and camera compatibility cannot be assumed.'],
    ['Fixed speedup and success thresholds', 'The proposed 3–5× minority exposure, 95 GPU-hour plan, arbitrary strength caps and early-epoch AP gates were not demonstrated or accepted as evidence. Measure cost and freeze justified gates before running.'],
], [2.2, 4.7])
p('The revision therefore retains smooth local rendering, class-pooled evidence and controlled context relocation as hypotheses to test. '
  'It removes the claims of zero-cost feedback, guaranteed semantic preservation, automatically correct mixtures and predetermined improvement.')

h('10 3 Why Approach B is not the next experiment', 2)
p('Approach B would add ROI feature-style mixing, DropBlock and a rising IoU threshold, approximately 0.5 to 0.65. '
  'These changes move the intervention inside the detector and change supervision or feature distributions at the same time. '
  'They would make it harder to identify whether local augmentation policy is responsible for any change. '
  'Higher IoU thresholds can also reduce positive assignments when proposal quality is insufficient. '
  'The latest direction keeps this route deferred; it is not an implemented fallback or a second active experiment.')
p('Relevant prior-art anchors discussed during review include Copy-Paste (arXiv:2012.07177), Fourier Domain Adaptation (arXiv:2004.05498), '
  'MixStyle (arXiv:2104.02008) and Cascade R-CNN (arXiv:1712.00726). Their established ingredients are not new contributions of this project. '
  'A new claim must concern a demonstrated residual capability of the complete, controlled method, not a renamed combination of known operations.')

h('11 Latest version 2 methodology and unresolved specification', new_page=True)
p('Status: conversational design only. The working description is class-adaptive group augmentation with whole-image geometry and controlled group copy-paste. '
  'The latest discussion supersedes the earlier suggestion to keep exact-set policies and merely switch to the weakest admissible strength. '
  'It does not supersede the recorded version-1 results. No version-2 experiment can be reported until a new protocol and implementation exist.')
h('11 1 Research question and invariant training setup', 2)
p('The revised question is whether class-conditioned admissibility improves training when the action bank includes context and geometric variation, '
  'while avoiding independent edits to overlapping damage labels. The primary comparison is class-adaptive versus shared online selection under the same expanded bank. '
  'The proposed detector remains Faster R-CNN with ResNet-50 FPN and COCO initialization, using the existing frozen data splits, 640-pixel letterbox and 20-epoch training recipe. '
  'Architecture novelty is not claimed. A retention gate is a measured detection-stability rule, not proof of unchanged human semantics.')

h('11 2 Three augmentation components', 2)
table(['Component', 'Proposed behavior and annotation rule'], [
    ['Shared whole-image geometry', 'Horizontal reflection, modest isotropic scaling and translation on a fixed canvas. Transform every box with the entire image. Reject transformations that crop annotated damage under the declared policy. Use the same geometry distribution in every new arm.'],
    ['In-place local appearance', 'Retain the current appearance families as candidates, render one connected overlap group once, and replace the very narrow taper with a specified smooth blend. Original target boxes remain valid because objects do not move.'],
    ['Controlled group copy-paste', 'Copy a complete annotated overlap group with surrounding context to a reviewed compatible road region. Transfer every included annotation, maintain orientation, and preserve the original donor. This is a context-patch intervention with boxes, not mask-based instance transplantation.'],
], [1.65, 5.25])
p('Horizontal flipping does not convert a transverse crack into a longitudinal crack by itself. A left-right reflection maps a horizontal line to a horizontal line '
  'and a vertical line to a vertical line; applied to the whole scene, it also reflects the road direction. The intended crack category is relative to the road, '
  'not merely to the x or y axis of the image. A 90-degree rotation of an isolated crack patch can change that relation and is excluded. '
  'Vertical flips, arbitrary local rotations and anisotropic stretching are also excluded from the initial bank. '
  'Even label-preserving whole-scene geometry still needs visual and annotation checks for realistic use.')
p('For a continuous XYXY box in an image of width W, horizontal reflection maps [xmin, ymin, xmax, ymax] to '
  '[W − xmax, ymin, W − xmin, ymax]. Scale and translation must use the same affine transform for pixels and boxes. '
  'The implementation must follow the project’s continuous-coordinate convention rather than mix it with inclusive integer-pixel formulas.')

h('11 3 Controlled group copy-paste', 2)
p('The proposed donor bank uses training images only. Sample a target class uniformly, then an eligible donor group containing that class; '
  'the exact handling of groups eligible for multiple classes still needs specification. Proposed uniform donor scales are 0.9, 1.0 and 1.1. '
  'Keep the group’s orientation and use compatible road/camera views rather than freely exchanging vehicle-camera and drone patches.')
p('The patch must include the complete selected group, a defined context margin, and all annotations intersecting the copied region, or the candidate must be rejected. '
  'Its opaque interior must contain all copied target boxes. Feathering occurs outside that interior so that the blend does not erase the labeled damage. '
  'The source image is not modified. The recipient gains transformed labels for the copied objects. '
  'Non-target annotated objects also require explicit handling; they cannot silently become unlabeled artifacts.')
p('Recipients are proposed to come from training images with no target boxes, using reviewed pasteable road regions. '
  'Reject placements that overlap known annotations or violate road/camera compatibility. An empty target list is not proof that the area contains no damage. '
  'The proposed copy-paste probability is 0.25 on eligible opportunities, leaving most negative training presentations unchanged. '
  'The donor bank, recipient bank, nominal opportunity schedule and rejection bookkeeping must be identical across the augmented comparators.')
p('These banks and masks do not yet exist as a reviewed version-2 artifact. Building them is substantive dataset work, not a minor coding detail. '
  'If sufficiently reliable recipient regions cannot be obtained within the available time, copy-paste must be explicitly deferred and the scientific claim narrowed; '
  'it must not be replaced silently by arbitrary rectangle pasting.')

h('11 4 Per-class admissibility and group decisions', 2)
p('Replace fifteen exact class-set policy tables with evidence pooled per damage class and candidate action. '
  'For each tested operation/strength, measure how many reference-detected objects remain correctly detected in the edited view. '
  'The proposed reference detection rule remains confidence at least 0.25 and IoU at least 0.5, with a retention threshold of 0.8. '
  'The proposed minimum support increases to sixteen reference-detected objects from eight distinct source clusters per class and action. '
  'These are engineering choices awaiting implementation and sensitivity assessment, not validated statistical guarantees.')
p('A group containing several classes can use an action only when it is admissible for every constituent class. '
  'Sufficiently supported direct observations on that mixed group can veto an action that the pooled marginal estimates would admit. '
  'When a decision borrows per-class evidence because the exact mixture has little support, record that fact. '
  'When even a constituent class/action lacks support, retain identity. The exact mixed-group veto threshold has not yet been frozen.')
p('The latest proposal does not maximize the clipped loss-increase score and does not automatically choose the strongest passing level. '
  'It uses balanced random sampling over eligible families, operations and tested strengths, with the precise hierarchy to be fixed in the configuration. '
  'The auxiliary loss remains a diagnostic. Thus version 2 is an adaptive admissibility controller, not a claim to directly optimize downstream learning gain.')
p('This remains distinct from online shared local augmentation: the shared comparator uses one common admissibility policy across classes, '
  'whereas the adaptive controller routes a group through the intersection of its constituent class policies. '
  'If the resulting class policies are identical in practice, the two methods may behave similarly; logs must expose that outcome rather than assume differentiation.')

h('11 5 Probe observer and update loop', 2)
p('The proposed update schedule remains before epochs 2, 4, 6, …, 20, after one common unaugmented warmup epoch. '
  'Probe the frozen detector on training-only panels without optimizer, gradient, parameter, normalization-buffer or global-RNG changes. '
  'Ordinary detector predictions provide the retention measurement. Use matched GT boxes and deterministic nearby ROIs, fixed or geometrically mapped across views, '
  'for a comparable auxiliary loss diagnostic; GT ROIs must never be inserted into inference predictions.')
p('For an in-place edit, compare the same objects before and after the edit. For copy-paste, compare donor objects with their transformed copies and inspect '
  'recipient detections before and after pasting, including false positives outside the pasted region. '
  'The mapping of boxes, retained support, source clusters and recipient context must be archived. '
  'Feedback is not zero cost: record backbone image forwards, auxiliary ROI work, wall time and failed/restarted probes. '
  'The exact panel allocation and forward budget are still unresolved and must be matched between the two online arms.')
p('At each update: select the frozen probe panel, render candidate views, collect detection and diagnostic observations, validate support, '
  'build the admissible-action sets, commit the controller together with model/optimizer/RNG state, then resume ordinary detector training. '
  'During inference and validation, use the detector alone without augmentation or ground-truth-dependent controller routing.')

h('11 6 New comparison arms and experimental sequence', 2)
table(['New arm', 'Role'], [
    ['Standard geometry only', 'Controls for the shared whole-image geometric bank. It is a new baseline, not the historical identity run.'],
    ['Fixed expanded augmentation', 'Uses the same local appearance and copy-paste renderer/banks with a fixed declared sampling distribution.'],
    ['Online shared expanded augmentation', 'Uses feedback to define one common admissible-action distribution. Controls for online selection without class routing.'],
    ['Online class-adaptive expanded augmentation', 'Uses per-class pooled evidence and joint group admissibility. Primary comparator: the online shared expanded arm.'],
], [2.1, 4.8])
p('Create a common unaugmented warmup checkpoint for each seed, then branch the four arms from the same model, optimizer, scheduler and random states. '
  'Use the same subsequent image order and separate keyed augmentation streams. Check actual repeatability on the target CUDA stack; '
  'warn_only=True permits nondeterministic operations and is not a determinism guarantee. Record any remaining limits.')
p('Execution order: one consolidated renderer/annotation/controller preflight; a matched short pilot; then the full 20-epoch comparison only if the pipeline '
  'has usable support, correct annotations, reproducible checkpoint recovery and acceptable measured cost. '
  'After the protocol is frozen, use matched repeated seeds and the necessary component ablations. '
  'Reserve the test partition for a final settled comparison. Do not repeatedly tune against it.')
p('Primary endpoint: selected validation AP50:95 under the frozen selection rule. Secondary diagnostics include AP50, AP75, per-class AP, '
  'negative-image false alarms, localization errors, final-epoch metrics, applied/changed/rejected/fallback counts, policy differences across classes and total compute. '
  'A copy-paste-removal ablation is needed for a claim about the context intervention. A renderer ablation is needed if smoother masks are claimed to cause a gain. '
  'The old version-1 runs remain historical evidence; they cannot serve as matched controls for a changed operation bank.')

h('11 7 Decisions that must be frozen before implementation', 2)
table(['Unresolved item', 'Minimum decision needed'], [
    ['Whole-image geometry', 'Flip probability, scale and translation bounds, canvas padding, interpolation, clipping rules and precise rejection of cropped annotations.'],
    ['Appearance and blending', 'Version-2 candidate magnitudes, application probability, context margin, mask function/width and handling of adjacent annotations. Do not assume every version-1 number was already accepted for version 2.'],
    ['Donor and recipient banks', 'Review procedure, saved region format, source/camera compatibility, annotation closure, donor reuse limits and bank size feasible within the project deadline.'],
    ['Sampling and exposure', 'How multi-class donor groups are weighted, the probability hierarchy, rejection caps/fallbacks and how nominal opportunities and realized exposure are compared.'],
    ['Observer', 'Deterministic nearby-ROI geometry and assignments, loss aggregation, transformed-ROI clipping and the exact mixed-group veto/support rule.'],
    ['Probe and compute budget', 'Panel composition, number of candidate views, repeated noise draws, matched online forward count and a measured pilot timing budget.'],
    ['Experimental decision rule', 'Pilot length, predeclared continuation criteria, repeated-seed count, uncertainty analysis and ablations justified by the claims. Three seeds alone do not guarantee significance.'],
    ['Course scope', 'Whether using a standardized COCO-pretrained detector satisfies the original course requirement for a custom CNN trained from scratch. Instructor approval is not established in the records.'],
], [1.55, 5.35])
p('Recommendation for the receiving AI: consolidate these items into one reviewable protocol/configuration before writing the new controller. '
  'Do not declare the design finished merely because the overall concept has been named. '
  'Keep the class-pooled admissibility direction separate from the still-unvalidated semantic and engineering assumptions of context relocation.')

h('12 CCAST environment, execution and recovery')
h('12 1 Locations and pinned environment', 2)
p('SSH account and host: solomon.owusu.1@prime.ccast.ndsu.edu. Login-node names vary, including login0003 and login0004. '
  'The project root on CCAST is:')
code('/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research')
table(['Path relative to remote project root', 'Purpose'], [
    ['dataset/raw', 'Original country/source images and Pascal VOC XML annotations.'],
    ['dataset/detection_splits_v1', 'Frozen train.jsonl, val.jsonl, test.jsonl and split provenance.'],
    ['weights/detection_v1', 'Official COCO Faster R-CNN and YOLO checkpoints plus SHA256SUMS.'],
    ['.venvs/rdd_detection_v1', 'Project virtual environment layered on the shared conda pytorch environment.'],
    ['runs', 'Training outputs, controller records, full checkpoints and exported records archives.'],
    ['runs/baseline_job_logs', 'Live PBS logs named <PBS_JOBID>.log.'],
], [2.8, 4.1])
table(['Dependency', 'Verified CCAST version'], [
    ['Python / CUDA build', '3.12.3 / PyTorch CUDA 12.1'],
    ['torch / torchvision', '2.3.0 / 0.18.0'],
    ['NumPy / Pillow', '1.26.4 / 10.3.0'],
    ['ultralytics-opencv-headless', '8.4.182'],
    ['opencv-python-headless / pycocotools', '4.11.0.86 / 2.0.10'],
    ['Hardware used', 'One NVIDIA A10 per GPU job, approximately 23,028 MiB reported memory.'],
], [2.8, 4.1])
p('The shared conda environment is /mmfs1/apps/pyenvs/anaconda3-2022.05/envs/pytorch. '
  'pip check passed after installing the project additions. The frozen environment listing is rdd_detection_v1_environment.txt. '
  'Local Windows tests used a different Python/PyTorch environment and do not replace target-environment checks.')

h('12 2 Batch jobs and what disconnection means', 2)
p('The production wrapper requests queue gpus, one A10, four CPU cores, 32 GB RAM and 24 hours under group x-ccast-prj-aaboah. '
  'submit.sh is run on a login node; the PBS job activates conda and the project environment itself. '
  'No interactive GPU allocation or manual environment activation is required merely to submit it. '
  'GPU training and heavy tests run inside the allocation, never directly on the login node.')
code('''# Historical production PBS resource pattern
#PBS -q gpus
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:glist=a10
#PBS -l walltime=24:00:00
#PBS -W group_list=x-ccast-prj-aaboah
#PBS -j oe''')
p('A submitted PBS batch job continues after SSH disconnection. An interactive session may not. '
  'After reconnecting, inspect qstat and the saved log before allocating another GPU or submitting a duplicate. '
  'No active job is assumed at this handoff; all five recorded version-1 full runs are complete.')
code('''# Windows PowerShell: reconnect
ssh solomon.owusu.1@prime.ccast.ndsu.edu''')
code('''# CCAST Bash: inspect only
qstat -u solomon.owusu.1
project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
tail -n 80 "$project/runs/baseline_job_logs/796544.bright04.log"''')
table(['Completed condition', 'PBS job ID'], [
    ['Identity', '794364.bright04'], ['Fixed image level', '795553.bright04'],
    ['Fixed local', '795787.bright04'], ['Online shared local', '796146.bright04'],
    ['Online class-adaptive local', '796544.bright04'],
], [3.5, 3.4])
p('The version-1 wrapper checks package hashes, runs unit tests, executes a separate real-data GPU smoke, validates its gate and launches full training in a fresh process. '
  'Start refuses existing output or submission records. Full .pt checkpoints stay on CCAST. '
  'If an unfinished run reaches its time budget, wait for the old job to end, inspect its saved status and then explicitly use bash submit.sh resume in that verified package. '
  'Do not delete a run to force a retry. There is no automatic resubmission.')

h('12 3 Interactive diagnostics when genuinely needed', 2)
p('For a new targeted interactive check, the historical allocation command was interact -c 4 -g 1 -m 16 -t 1. '
  'Run it on the login node and wait for the allocated-node prompt before the activation block. '
  'The two steps cannot simply be pasted as if the new shell were already running. Batch scripts are preferable for long work.')
code('interact -c 4 -g 1 -m 16 -t 1')
code('''# Run only after the allocation has opened its compute-node shell
source /mmfs1/apps/pyenvs/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate pytorch
project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
source "$project/.venvs/rdd_detection_v1/bin/activate"''')
p('Before GPU computation, check PBS_JOBID, the expected sys.prefix, torch.cuda.is_available() and torch.cuda.device_count()==1. '
  'Respect the scheduler’s CUDA_VISIBLE_DEVICES. nvidia-smi may list all four physical GPUs even when only one is assigned. '
  'CUDA NMS and RoI Align forward/backward already passed in the project environment; do not rerun them routinely without a changed environment or a relevant failure.')

h('12 4 Package and result transfer discipline', 2)
p('Once the specification and checks exist, create a uniquely named version-2 package. Record its hash, upload with scp, compare SHA-256 on CCAST, '
  'refuse an existing extraction directory, extract with tar --keep-old-files, check SHA256SUMS and submit through the wrapper. '
  'No version-2 package or valid launch command exists yet.')
p('For a completed run, download the emitted records archive and verify its SHA-256. Inspect run status, source/config hashes, checkpoint metadata, '
  'metrics and controller records before the next experiment. A completed log alone does not establish scientific validity. '
  'Records archives do not include full model checkpoints unless their manifests explicitly say so.')
p('An early SHA256SUMS file used Windows CRLF endings. This non-mutating workaround made its checks pass; '
  'do not change hashed source files to fix the manifest:')
code("tr -d '\\r' < SHA256SUMS | sha256sum -c -")

h('13 Repository, artifact inventory and provenance', new_page=True)
h('13 1 What is and is not on GitHub', 2)
p('Repository: https://github.com/Solo0722/deep_learning_project. The working branch is main. '
  'Local HEAD and the GitHub main branch were both verified at commit 8ed2567c6f5a26960949e12116f4fa1c0b8a403b, '
  'whose subject is “Add main manuscript and references for RDD Prototype Submission.” '
  'The earlier memory that GitHub contained only an initial bootstrap is outdated.')
p('Before this handoff was generated, git status showed no tracked-file modifications but did show untracked additions. '
  'These include the completed class-adaptive audit/analysis files, output/, and reports/experiment_report_20261027/. '
  'This handoff and its build/QA files are additional local artifacts. No commit, push or publication was performed for this request. '
  'Sharing the repository URL alone omits these local additions, raw data, remote checkpoints and the external redesign document.')
p('Treat saved run protocols and verified audit outputs as the authority for completed experiments. '
  'The latest conversation defines proposed changes; this handoff records them with unresolved items. '
  'PROJECT_PLAN.md and some older review JSON files are historical snapshots and may say that work is pending even though later runs completed it. '
  'research_paper/main.tex is the current manuscript source but is not a frozen version-2 implementation specification.')

h('13 2 Source map', 2)
p('All relative paths in the following map are relative to the local project root printed on page 2. '
  'The same package names appear under the remote project root. The receiving AI should inspect actual files and dates before editing.')
table(['Location', 'What to read or reuse'], [
    ['PROJECT_PLAN.md; DATA_AUDIT.md; METHOD_REVIEW.md', 'Original scope, crop-era decisions and historical methodological review. Read chronologically, not as current run status.'],
    ['dataset_splits_v1/', 'Earlier crop-classification split package. Do not substitute it for the full-image detector manifests.'],
    ['detection_splits_v1/', 'Frozen full-image manifests; README, summary.json, config.json, reviewed_policy.json, excluded_images.json and excluded_objects.json document construction.'],
    ['audit_dataset.py; audit_detection_data.py; make_splits.py; make_detection_splits.py', 'Data audit and split generation. Regeneration must use the frozen policy and new output paths.'],
    ['materialize_crops.py; train_baseline.py; BASELINE_RUNBOOK.md', 'Earlier crop extraction and custom-CNN baseline implementation.'],
    ['pilot_runner.py; PILOT_RUNBOOK.md; reports/pilot_verification_v1/', 'Earlier crop-controller development and local synthetic verification. Not a completed adaptive crop efficacy experiment.'],
    ['detection_loader_v1/; detection_integration_v1/', 'Image/annotation adapters, representative training selection and two-detector real-image integration checks.'],
    ['detection_learning_v1/; detection_diagnostics_v1/', 'Small training-only learning check and YOLO normalization diagnostic.'],
    ['fasterrcnn_baseline_v1/; fasterrcnn_shared_v1/; fasterrcnn_local_v1/', 'Three fixed baseline training packages. Each protocol.json and README explains its exact condition.'],
    ['fasterrcnn_online_shared_v1/', 'Online shared controller and frozen probe implementation used as the principal adaptive comparator.'],
    ['fasterrcnn_class_adaptive_v1/', 'Completed exact-set controller package, including controller.py, class_set_feedback.py, shared_controller.py, probe_runner.py and probe_detector.py.'],
    ['reports/conventional_seed42_v1/analysis/', 'Crop baseline RESULTS.md, verification.json, curves, confusion matrix and per-class metrics.'],
    ['reports/detection_learning_v1/; reports/detection_diagnostics_v1/', 'Saved learning metrics and YOLO diagnostic review; later status supersedes their old prospective next-step fields.'],
    ['reports/fasterrcnn_baseline_v1/; reports/fasterrcnn_shared_v1/; reports/fasterrcnn_local_v1/; reports/fasterrcnn_online_shared_v1/', 'Per-condition result audits and comparisons. Confirm each file name from the folder before relying on it.'],
    ['reports/fasterrcnn_class_adaptive_v1/', 'Latest consolidated five-arm audit, comparisons, support/strength diagnosis and replay records; important local untracked material.'],
    ['research_paper/main.tex; research_paper/references.bib', 'Current manuscript and bibliography. Keep the existing editor/source when asked to edit; do not silently switch to an alternate draft.'],
    ['output/reports/RDD_Experiment_Report_2026-10-27.docx', 'Earlier detailed black-and-white experimental report; Sections 10–11 of this handoff supersede its prospective next steps.'],
    ['reports/research_handoff_20261027/', 'This handoff’s builder, source snapshot manifest and rendering/verification records.'],
], [2.75, 4.15], size=9.5)
p('External redesign source, outside the repository: ' + str(REDESIGN))
p('Also preserve the original proposal “Road Defect CNN – Class Adaptive Augmentation.docx” and Deep_Learning_Project_Deadlines.pdf. '
  'The prototype submission ZIP from 26 October is a time-specific submission snapshot, not a guarantee that all later audit files are included.')

h('13 3 Start with these exact audit files', 2)
table(['File in reports/fasterrcnn_class_adaptive_v1/', 'Use'], [
    ['RESULTS.md', 'Human-readable conclusion and caveats for the completed adaptive run.'],
    ['aggregate_comparison.csv; per_class_comparison.csv', 'Selected and final metrics for all five conditions and each damage class.'],
    ['negative_image_comparison.csv; runtime_comparison.csv', 'False-alarm diagnostics and measured training/probe runtime.'],
    ['comparison_curves.csv; comparison.json', 'Epoch trajectories and consolidated comparison provenance.'],
    ['ccast_verification.json; recomputed_metrics.json', 'Archive/source/run checks and independent metric recomputation from saved predictions.'],
    ['controller_verification.json; controller_class_set_rounds.csv', 'Decision replay and per-update exact-set policy behavior.'],
    ['strength_support_diagnostics.json; strength_selection_diagnostic_cells.csv', 'Strongest-admissible selection, zero-score activity and support sensitivity.'],
    ['class_set_training_inventory.csv; class_set_summary.csv; controller_class_set_exposure.csv', 'Training availability versus actual panel support and applied/fallback opportunities.'],
    ['audit_ccast_results.py; audit_class_controller.py; diagnose_strength_support.py', 'Audit implementation. Inspect dependencies and invocation before reproducing analyses.'],
], [3.1, 3.8], size=9.5)

h('13 4 Completed result archives', 2)
p('These five archives were downloaded to the local project root and their hashes were verified during the recorded result audits. '
  'The handoff reproduces those recorded hashes; it does not imply a fresh rehash of every large archive. '
  'Each archive is also under the remote runs/ directory. Reverify it before transfer or new extraction.')
archives = [
    ('Identity', 'fasterrcnn_identity_seed42_v1_records_b07d6daa294f.tar.gz', '4d16ae384cc8f8663b47a00f84df331ea32c44f3742aebaa71e114b55fc0026f'),
    ('Fixed image level', 'fasterrcnn_shared_seed42_v1_records_31a774d9401a.tar.gz', '28b28bee01376cc10fe0a60dca7ddfb9cd97ec9b548597fe316d94db164bfd7c'),
    ('Fixed local', 'fasterrcnn_local_seed42_v1_records_b047925ba90a.tar.gz', '6f1efc305780b1ab018a807bd728460b04582586166f6aeb616e693c6cfdcbac'),
    ('Online shared local', 'fasterrcnn_online_shared_seed42_v1_records_437f03a837f3.tar.gz', 'b30df4e003be14cf8a53059341451c5101261d98d32e58cdc94866c793f0a231'),
    ('Online class-adaptive local', 'fasterrcnn_class_adaptive_seed42_v1_records_26e50b915ed9.tar.gz', 'c4817b3cb3bc69813776d06662a9070a7415e9552cab7c6b331c5237414d9b74'),
]
for name, filename, sha in archives:
    x = p(name)
    x.paragraph_format.keep_with_next = True
    x.runs[0].bold = True
    code(filename + '\nSHA-256: ' + sha)
p('The complete records archive for the adaptive run contained 123 audited members. Saved predictions enabled selected/final COCO metric recomputation, '
  'and observations enabled replay of the controller decisions. This did not independently reconstruct unsaved raw probe logits/ROIs or rerun GPU training from raw images.')

h('13 5 Other reproducibility hashes and minimum transfer set', 2)
for name, sha in [
    ('rdd_detection_v1_environment.txt', 'f07b523b024d61f46fe407df674fa2361a2f9332ed489f794f7b8b20024d77c8'),
    ('weights/detection_v1/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth', '258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'),
    ('weights/detection_v1/yolov8n.pt', 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'),
    ('detection_learning_check_v1.tar.gz', '0ab14951c60bf566e2585ea66648d3bc743e2c427558b1f85b762ed50dd9693a'),
    ('detection_diagnostic_check_v1.tar.gz', '14dd16aad80c0dbf546994aa4c2aad80c6dfb215ca5d7f7a222e218b5557193f'),
]:
    code(name + '\nSHA-256: ' + sha)
p('For reasoning-only continuity, give the next AI this handoff, the external redesign DOCX and access to the current manuscript, '
  'fasterrcnn_class_adaptive_v1/, fasterrcnn_online_shared_v1/ and the complete reports/fasterrcnn_class_adaptive_v1/ folder. '
  'For implementation, also provide the frozen split package, all baseline packages and tests, and the environment listing. '
  'For independent metric re-audit, provide the relevant records archives. For training/resume, access to CCAST raw data, weights and full checkpoints is additionally required. '
  'This Word document does not embed those assets.')

h('13 6 Evidence keys used in the historical sections', 2, new_page=True)
p('Bracketed S identifiers in Sections 1–9 refer to the source groups below. Within each group, subsequent bare filenames belong to the named folder unless another path is given.')
historical_blocks = json.loads((ROOT/'reports/experiment_report_20261027/report_content.json').read_text(encoding='utf-8'))
for block in historical_blocks:
    if block.get('type') == 'paragraph' and re.match(r'^\[S\d+\]', block.get('text', '')):
        srcp = p(block['text'])
        srcp.paragraph_format.space_after = Pt(7)
        srcp.paragraph_format.line_spacing = 1.05
        for r in srcp.runs:
            r.font.size = Pt(10)

h('14 Writing status, course context and collaboration requirements', new_page=True)
h('14 1 Manuscript and contribution boundaries', 2)
p('The project’s active LaTeX document is research_paper/main.tex, with references.bib in the same directory. '
  'The user previously requested preservation of the methodology structure and equations while adding a submission-style experimental setup. '
  'Do not replace the current file or update scientific claims merely because a new method has been discussed. '
  'When asked to revise it, inspect the actual source, edit in place, preserve the open editor and compile using the built-in document compiler.')
p('The manuscript must distinguish the earlier custom crop CNN from the standardized full-image Faster R-CNN detector. '
  'The augmentation controller is the proposed methodological contribution; neither Faster R-CNN nor YOLO becomes a custom architecture by changing its output classes. '
  'The original plan required training a custom CNN from scratch. The user later chose full-image detection and authorized a pretrained Faster R-CNN baseline, '
  'but the record does not show instructor confirmation that this change satisfies the original requirement. '
  'Do not conceal this scope question or invent permission.')
p('Current defensible contribution statement: a working, audited overlap-group class-adaptive augmentation prototype and a controlled single-seed development comparison '
  'that documents negative accuracy evidence and concrete support/observer limitations. '
  'Not defensible: state-of-the-art performance, proven semantic safety, causal explanations of failure, class-adaptive superiority, cross-country generalization, '
  'test-set performance or successful version-2 results.')
p('Related-work notes and audits include RELATED_WORKS_RESEARCH.md, research_paper/METHOD_PROTOCOL.md and research_paper/SOURCE_AUDIT.md. '
  'Prior comparisons discussed PSIS, RangeAugment, CUDA, MADAug, CTAugment, TeachAugment, BDA and InterAug. '
  'Before a novelty claim, verify the primary sources and identify what the nearest method already does. '
  'Neither adaptive magnitude selection, class-conditioned augmentation, overlap grouping, copy-paste nor style mixing alone establishes novelty here.')
p('Primary reference entry points for the receiving AI: the RDD repository at https://github.com/sekilab/RoadDamageDetector; '
  'the TorchVision detection tutorial at https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html; '
  'the papers listed in Section 10; and PyTorch’s reproducibility documentation at https://docs.pytorch.org/docs/stable/notes/randomness.html. '
  'This handoff is not a new exhaustive literature review. Recheck citations and current documentation when they become material to implementation or submission.')

h('14 2 Course milestones from the supplied schedule', 2)
table(['Date in 2026', 'Required deliverable'], [
    ['26 October', 'Working proposed-method prototype, baseline comparison or documented debugging evidence, updated methods section.'],
    ['2 November', 'Preliminary comparative results, curves/error analysis, draft results and remaining experiment plan.'],
    ['9 November', 'Main comparison and ablations, repeat-run or suitable robustness evidence, revised results.'],
    ['16 November', 'Complete manuscript draft and remaining experiment list.'],
    ['23 November', 'Revised manuscript, essential remaining experiments and response to feedback.'],
    ['4 December', 'Final manuscript, reproducible code, data documentation, slides and response to feedback.'],
], [1.35, 5.55])
p('Source: Deep_Learning_Project_Deadlines.pdf. Written deliverables after 5 October are due at 11:59 PM Central Time in that schedule; '
  'presentation arrangements are separate. This is the supplied course schedule, not a live check for later instructor changes. '
  'The 26 October prototype package was prepared, but this handoff does not verify submission acceptance or a grade.')

h('14 3 Working agreement for the next AI', 2)
p('Be direct and evidence-bounded. Separate measured failure, missing measurement, inference and proposed intervention. '
  'Do not agree reflexively, present speculation as fact, or promise that a new operation will fix an unisolated cause. '
  'Reason through crack labels, road context, annotation transforms, overlap and evaluation consequences before sending a training command.')
p('The user prefers a concise plain-language explanation before formal detail, then concrete work. '
  'For CCAST guidance, safe blocks of commands with checksum and error guards are now preferred when commands can truly run together; '
  'otherwise give one command and wait for its output. Label Windows PowerShell versus CCAST Bash. '
  'Do not make the user repeatedly reactivate environments for batch submissions that already do it.')
p('Keep the main research objective in view. The YOLO diagnostic was a legitimate investigation but was deferred when it distracted from the Faster R-CNN comparison. '
  'Avoid repeated broad testing after relevant checks pass, additional full experiments without an interpretable comparison, and unbounded diagnostic branches. '
  'Before broad implementation, provide one concrete design and its material assumptions for review; do not turn every routine reversible action into a permission request.')
p('Use plain black-and-white reports without decorative formatting. Write manuscripts as papers for submission, not chronological progress notes. '
  'Verify local and remote repository contents before telling the user that a link contains the latest work. '
  'Do not publish, send messages, overwrite completed experiments, or launch remote training merely because it appears in this handoff.')
p('The user’s explicit response rules require the standalone sentence “You are wrong about this.” when correcting a demonstrably false user claim, '
  'followed by the specific correction. Do not use it for an uncertain question, a preference, or correction of the AI’s own mistake. '
  'When essential missing context makes a request impossible to answer usefully, use “This is too vague to answer usefully. Here\'s what you need to define first.” '
  'and identify the minimum missing information. Do not force either phrase into an otherwise straightforward request.')

h('15 Ready-to-use prompt for the receiving AI', new_page=True)
p('The following prompt can accompany this document. It is an explicit starting brief; access to the listed files is still necessary.')
p('I am continuing an RDD2022 road-damage detection project. Read this handoff completely and inspect the source files that support your next action. '
  'Five version-1 Faster R-CNN augmentation conditions are complete. The online class-adaptive run did not improve the primary validation score over online shared. '
  'The latest direction is a proposed version 2, not implemented or tested. Do not claim an efficacy result for it.')
p('My immediate goal is to make the version-2 methodology fully specified and reviewable before starting its experiments. '
  'Preserve the frozen full-image split and version-1 evidence. Start from Sections 10 and 11: shared whole-image geometry, smooth group-local appearance, '
  'controlled annotation-aware group copy-paste, pooled per-class admissibility and balanced sampling of admissible actions. '
  'Auxiliary loss is diagnostic; do not silently restore the old strongest-passing or maximum-hardness rule.')
p('First inspect fasterrcnn_class_adaptive_v1/protocol.json, controller.py, class_set_feedback.py, probe_detector.py, probe_runner.py and local_augmentation.py; '
  'then inspect reports/fasterrcnn_class_adaptive_v1/RESULTS.md, strength_support_diagnostics.json, controller_verification.json and the comparison CSVs. '
  'Read RDD_Methodology_Redesign_and_Execution_Plan.docx as a proposal with the corrections in this handoff. '
  'Verify Git status because the latest audit/report files were untracked at handoff.')
p('Return one concrete protocol covering the unresolved items in Section 11.7, including annotation invariants, reviewed donor/recipient banks, mixed-group veto, '
  'exact sampling, matched compute and pilot continuation criteria. Flag only decisions that genuinely require my input; use evidence and reasonable defaults for routine choices. '
  'Do not prepare a full GPU run until the preflight and short-pilot plan is defined. Do not resume YOLO or Approach B without a reason connected to the main question.')
p('After the protocol is settled, implement in a new versioned package, preserve old outputs, test the material failure modes, and give CCAST commands in safe copy-pasteable blocks. '
  'Use common warmup checkpoints and matched order/random streams for comparisons. Keep the test set untouched until the final settled evaluation. '
  'Report what is measured, what is inferred and what remains unknown. Be direct; avoid speculative assurances and unnecessary detours.')
p('This handoff is complete enough to resume reasoning and locate the evidence. It is not evidence that all files have been shared, '
  'that a current CCAST allocation exists, or that the remaining version-2 engineering choices are already resolved.')

# Normalize all explicit and theme-based colors to black, including inherited hyperlinks.
for s in doc.styles:
    if s.type in (1, 2):
        s.font.color.rgb = RGBColor(0, 0, 0)
for color in doc._element.xpath('.//w:color'):
    color.set(qn('w:val'), '000000')
    for a in ('themeColor', 'themeTint', 'themeShade'):
        color.attrib.pop(qn('w:' + a), None)
for p0 in doc.paragraphs:
    p0.paragraph_format.widow_control = True

doc.core_properties.title = 'Road Damage Augmentation Research Handoff'
doc.core_properties.subject = 'Verified research history, results and proposed version 2 through 27 October 2026'
doc.core_properties.author = 'Solomon Owusu Ansah'
doc.core_properties.keywords = 'RDD2022, Faster R-CNN, augmentation, research handoff, CCAST'
doc.core_properties.comments = 'Prepared from project artifacts and the recorded methodology discussion. Version 2 is proposed, not implemented.'
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)

# A small source snapshot makes the assembled narrative auditable without rewriting research artifacts.
sources = [SOURCE, REDESIGN, ROOT/'detection_splits_v1/summary.json',
    ROOT/'fasterrcnn_class_adaptive_v1/protocol.json',
    ROOT/'reports/fasterrcnn_class_adaptive_v1/RESULTS.md',
    ROOT/'reports/fasterrcnn_class_adaptive_v1/strength_support_diagnostics.json',
    ROOT/'reports/fasterrcnn_class_adaptive_v1/controller_verification.json',
    ROOT/'reports/fasterrcnn_class_adaptive_v1/ccast_verification.json',
    ROOT/'reports/fasterrcnn_class_adaptive_v1/aggregate_comparison.csv',
    ROOT/'reports/fasterrcnn_class_adaptive_v1/per_class_comparison.csv',
    ROOT/'reports/fasterrcnn_class_adaptive_v1/negative_image_comparison.csv',
    ROOT/'reports/fasterrcnn_class_adaptive_v1/runtime_comparison.csv',
    ROOT/'research_paper/main.tex', ROOT/'research_paper/references.bib',
    ROOT/'Deep_Learning_Project_Deadlines.pdf']
manifest = {
    'date': '2026-10-27',
    'output': str(OUT),
    'output_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
    'local_and_remote_main': '8ed2567c6f5a26960949e12116f4fa1c0b8a403b',
    'repo': 'https://github.com/Solo0722/deep_learning_project',
    'new_experiments': False,
    'version_2_status': 'Proposed in discussion; not implemented or run',
    'source_sha256': {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources if s.exists()},
    'historical_result_archives': [{'condition': a, 'filename': b, 'previously_verified_sha256': c} for a,b,c in archives],
}
(WORK/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
texts = [p0.text for p0 in doc.paragraphs]
for t in doc.tables:
    texts.extend(' | '.join(c.text for c in row.cells) for row in t.rows)
(WORK/'document_text.txt').write_text('\n'.join(texts), encoding='utf-8')
with ZipFile(OUT) as z:
    xml = z.read('word/document.xml').decode()
checks = {
    'paragraphs': len(doc.paragraphs), 'tables': len(doc.tables),
    'native_equations': xml.count('<m:oMath>'),
    'word_count_approx': len(' '.join(texts).split()),
    'superseded_heading_removed': '10 Changes recommended before another full experiment' not in xml,
    'has_version_2_status': 'No version-2 code, package or experiment is present.' in xml,
    'output_bytes': OUT.stat().st_size,
}
assert checks['superseded_heading_removed']
assert checks['has_version_2_status']
assert checks['native_equations'] == 8, checks
(WORK/'structural_verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps({'output': str(OUT), **checks}, indent=2))
