import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
ROOT = r"C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project"
OUT = os.path.join(ROOT, "output", "reports")
FIG = os.path.join(OUT, "nov9_figs")
os.makedirs(FIG, exist_ok=True)

# ---------------- Figure: method diagram ----------------
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 110); ax.set_ylim(0, 52); ax.axis("off")


def box(x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec="#333333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, text=None, dy=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", fontsize=8, color="#333333")


blue, green, orange, grey, red = "#dbe8f5", "#dff0e0", "#fbe6cf", "#eeeeee", "#f6d6db"
# training loop (top)
box(1, 36, 17, 10, "Training image\nwith classes S\n(e.g. D00 + D40)", grey)
box(24, 36, 20, 10, "Pick edit o for S\n(identity or one of\n~13 menu edits)", orange, bold=True)
box(50, 36, 17, 10, "Base geometry\n(flip, scale)\n+ edit o", grey)
box(73, 36, 17, 10, "Detector\n(Faster R-CNN /\nRetinaNet)", blue)
box(95, 36, 13, 10, "SGD\nupdate", blue)
arrow(18.5, 41, 23.5, 41)
arrow(44.5, 41, 49.5, 41)
arrow(67.5, 41, 72.5, 41)
arrow(90.5, 41, 94.5, 41)

# estimator (bottom)
box(1, 6, 21, 14, "Every K = 200 steps:\nsmall batch B,\nunedited and with\neach edit o", grey)
box(28, 6, 22, 14, "Gradients on the\ndetector head:\ng0 (unedited), go (edited)", blue)
box(56, 6, 22, 14, "Per-class reference\nsets Rc (held out):\ngradient hc of\nclass-c loss", green)
box(84, 6, 24, 14, "Benefit\nu(o,c) = η<go - g0, hc>\n/ (Lc + δ)\nsmoothed, with std. error", green, bold=True)
arrow(22.5, 13, 27.5, 13)
arrow(50.5, 13, 55.5, 13)
arrow(78.5, 13, 83.5, 13)
# policy box
box(84, 24.5, 24, 7.5, "Benefit table:\nedits x classes", red, fs=8.5)
arrow(96, 20.5, 96, 24)
ax.annotate("", xy=(34, 35.5), xytext=(84, 28.5),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#b33a50",
                            connectionstyle="arc3,rad=0"))
ax.text(50, 25.3, "score(o) = mean over c in S of benefit;  block o if it reliably hurts any c in S;\n"
        "sample with softmax + 10% exploration  (shared variant: one score for all classes)",
        ha="center", fontsize=8, color="#b33a50")
ax.text(1, 49.5, "Training loop", fontsize=10, fontweight="bold")
ax.text(1, 21.6, "Class-utility estimator (online)", fontsize=10, fontweight="bold")
plt.tight_layout()
fdiag = os.path.join(FIG, "method_diagram.png")
plt.savefig(fdiag, dpi=200)
plt.close()

# ---------------- Figure: ablation ladder ----------------
labels = ["No aug.", "Fixed\nimage-level", "Fixed local", "Online\nshared", "Online\nclass-adaptive"]
vals = [22.823, 22.810, 22.555, 23.059, 22.563]
labels2 = ["Geometry\nonly", "+ fixed\nlocal", "+ online\nshared", "+ class-\nadaptive"]
vals2 = [19.808, 18.861, 19.366, 19.608]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, lab, v, ref, title in [(axs[0], labels, vals, 22.823, "20-epoch runs: difference from no augmentation"),
                               (axs[1], labels2, vals2, 19.808, "Epoch-4 pilot: difference from geometry only")]:
    d = [x - ref for x in v]
    ax.bar(range(len(d)), d, color=["#888888" if x == 0 else ("#13804f" if x > 0 else "#b33a50") for x in d])
    for i, x in enumerate(d):
        ax.text(i, x + (0.03 if x >= 0 else -0.08), f"{x:+.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=8.5)
    ax.set_ylabel("AP50:95 difference (points)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-1.15, 0.45)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fabl = os.path.join(FIG, "ablation_ladder.png")
plt.savefig(fabl, dpi=200)
plt.close()

f_curves = os.path.join(ROOT, "reports", "rdd_v2_pair8_v1", "epoch8_comparison.png")
f_perclass = os.path.join(OUT, "nov2_figs", "per_class_ap.png")

# ---------------- Document ----------------
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, m, Inches(1))
st = doc.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
for name, size in (("Heading 1", 14), ("Heading 2", 12)):
    h = doc.styles[name]
    h.font.name = "Calibri"; h.font.size = Pt(size); h.font.bold = True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.paragraph_format.space_before = Pt(12); h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True


def para(text, bold_lead=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    r = p.add_run(text); r.italic = italic
    return p


def bullet(text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(2)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), hexcolor)
    tcPr.append(s)


def table(caption, header, rows, widths, bold_row=None, italic_col=None):
    c = doc.add_paragraph(); c.add_run(caption).bold = True
    c.paragraph_format.keep_with_next = True; c.paragraph_format.space_after = Pt(2)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]; cell.text = ""
        r = cell.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(9.5)
        shade(cell, "DCE6F0")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(9.5)
            if bold_row is not None and ri == bold_row:
                r.bold = True
            if italic_col is not None and i == italic_col:
                r.italic = True
    for row in t.rows:
        for i, wd in enumerate(widths):
            row.cells[i].width = Inches(wd)
            row.cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(path, caption, width=6.5):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    p.paragraph_format.keep_with_next = True
    c = doc.add_paragraph(); r = c.add_run(caption); r.italic = True; r.font.size = Pt(9.5)


t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Class-Aware Augmentation Selection for Multi-Class Object Detection"); r.bold = True; r.font.size = Pt(16)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Project Milestone: Main Results and Validation (due November 9, 2026)").font.size = Pt(12)
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Solomon Owusu Ansah  |  Student ID 1641802\nCE 738 Deep Learning for Engineers (Drs. Armstrong Aboah and Trung Le)")

doc.add_heading("1. Summary", level=1)
para("The preliminary results (November 2) showed that the first class-adaptive controller, which picks "
     "augmentations by how well the detector tolerates them, does not improve road damage detection. This "
     "milestone does three things:")
bullet(" the complete main comparison and ablations of that controller, with repeat-run and robustness evidence "
       "(Sections 2-3);", "Closes the first method:")
bullet(" a new method, class-utility augmentation, that chooses edits by their measured training benefit for "
       "each class (Section 4);", "Introduces the revised method:")
bullet(" a larger study design (three datasets, two detectors, three seeds) with the code finished and tested. "
       "The first experiments are running on the CCAST GPU cluster now; their results will be added as they "
       "arrive (Sections 5-6).", "Starts the new experiments:")
para("Every ablation of the tolerance-based controller points the same way. Online control helps a little "
     "(+0.24 AP50:95 for the shared policy), but adding class-specific tolerance rules does not help (-0.50). Over "
     "a longer fixed horizon it falls 1.25 AP50:95 below geometry-only training, with all four classes lower. "
     "The pipeline itself is verified: repeated training is exactly reproducible, and every metric was "
     "recomputed independently with zero difference. The failure is in what the controller measures, which is "
     "what the new method changes.", bold_lead="Main result. ")

doc.add_heading("2. Main Comparison and Ablation Results (tolerance-based controller)", level=1)
para("All runs: RDD2022 (4 damage classes, 26,863 training and 5,760 validation images), Faster R-CNN "
     "ResNet-50-FPN with COCO weights, seed 42, validation set only (test set kept closed). The conditions were "
     "chosen so that each step adds one component, which makes the main comparison an ablation ladder.")
table("Table 1. Main comparison and component ablation, 20 epochs (best validation checkpoint).",
      ["Condition", "Component added", "AP50:95", "AP50", "vs previous step"],
      [["No augmentation", "-", "22.82", "51.14", "-"],
       ["Fixed image-level edits", "appearance edits", "22.81", "50.69", "-0.01"],
       ["Fixed local edits", "edits only inside boxes", "22.56", "50.16", "-0.26"],
       ["Online shared policy", "online tolerance check", "23.06", "51.03", "+0.50"],
       ["Online class-adaptive (proposed v1)", "per-class tolerance rules", "22.56", "50.18", "-0.50"]],
      [2.2, 1.7, 0.75, 0.7, 1.0], bold_row=3)
table("Table 2. Ablation on top of label-safe geometric augmentation (common epoch-1 start).",
      ["Arm", "AP50:95 epoch 4", "AP50 epoch 4", "AP50:95 epoch 8", "AP50 epoch 8"],
      [["Geometry only", "19.81", "45.05", "19.68", "44.33"],
       ["+ fixed local edits", "18.86", "43.44", "not run", "-"],
       ["+ online shared policy", "19.37", "43.87", "not run", "-"],
       ["+ class-adaptive policy", "19.61", "44.76", "18.44", "41.97"]],
      [2.0, 1.15, 1.1, 1.15, 1.1])
figure(fabl, "Figure 1. Ablation results as differences from the reference condition. Left: 20-epoch runs "
             "(Table 1). Right: epoch-4 pilot with geometric augmentation (Table 2).")
para("Online control is the only component with a positive effect (+0.50 over fixed local edits, +0.24 over no "
     "augmentation). Adding per-class tolerance rules removes this gain. With geometric augmentation, the "
     "class-adaptive policy beats the fixed and shared policies at epoch 4 but stays below geometry only. The "
     "paired extension to epoch 8 widens the gap to -1.25 AP50:95 (Figure 2).", bold_lead="What the ablations show. ")
figure(f_curves, "Figure 2. Paired run to epoch 8. Left: validation AP50:95. Right: class-adaptive minus "
                 "geometry-only per epoch. The class-adaptive arm is behind at every epoch.")
figure(f_perclass, "Figure 3. Per-class AP50:95. Left: 20-epoch runs. Right: epoch-8 paired run, where all four "
                   "classes decline, most for alligator cracks (D20, -2.06).")

doc.add_heading("3. Repeat-Run and Robustness Evidence", level=1)
table("Table 3. Validation evidence for the reported results.",
      ["Check", "What was done", "Outcome"],
      [["Repeat-run determinism", "Same training repeated, after a controller probe, and after a save/restore, "
        "under strict GPU settings (88 recorded comparisons)", "All matched exactly"],
       ["Metric recomputation", "All endpoint metrics recomputed from saved predictions with official COCO "
        "evaluation, independent code", "Zero difference"],
       ["Controller replay", "29,696 controller views re-matched; gates and routing rebuilt with separate code",
        "Exact match"],
       ["Gate sensitivity", "Leave-one-cluster-out recomputation of 96 class/edit gates", "1-2 gates change"],
       ["Consistency over time", "Class-adaptive minus geometry-only at epochs 2-8", "Negative at all 7 epochs"],
       ["Run-to-run spread", "Epoch-1 AP50:95 across five nominally identical starts", "15.73-16.54 (0.8 range)"]],
      [1.5, 3.4, 1.6])
para("The pipeline and the reported numbers are reliable. The main limit is that each condition has one "
     "training seed. The 0.8-point spread at epoch 1 means differences under about 0.5 AP50:95 (all of Table 1) "
     "should be treated as unresolved. The 1.25-point epoch-8 loss is larger and consistent in direction, but "
     "still a single seed. The new study therefore uses three paired seeds for every comparison, with confidence "
     "intervals.", bold_lead="Interpretation. ")

doc.add_heading("4. Revised Method: Class-Utility Augmentation", level=1)
para("The analysis of the first controller found three problems (Nov 2 report, Section 5.3):")
bullet(" it allowed any edit the detector could tolerate, without asking whether training on the edit helps;",
       "Wrong signal:")
bullet(" leaving the image unchanged was never compared as a choice;", "No 'do nothing' option:")
bullet(" too few examples for many class combinations.", "Sparse evidence:")
para("The revised method fixes all three. During training it estimates, for every edit o and class c, how much "
     "one training step on the edited images would reduce the loss on a small held-out reference set of class c, "
     "compared with the unedited images:")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("u(o, c) = η · <g_o − g_0, h_c> / (L_c(R_c) + δ)"); r.italic = True; r.font.size = Pt(12)
para("Here g_o and g_0 are the gradients of the detector-head loss on a batch with and without edit o, and h_c "
     "is the gradient of the class-c loss on its reference set R_c (64 images). The estimate is smoothed over "
     "time and carries a standard error. For a training image containing classes S, the policy then:")
bullet(" averages the benefit over the classes present, with 'no edit' always available at benefit 0;",
       "Scores each edit:")
bullet(" removes any edit that reliably hurts one of the classes present (95% one-sided);",
       "Blocks harmful edits:")
bullet(" draws an edit from a softmax over scores, plus 10% exploration.", "Samples:")
para("A shared-utility version uses identical machinery but averages benefit over all classes, so every image "
     "gets the same preferences. Comparing the two isolates the value of class information.")
figure(fdiag, "Figure 4. Class-utility augmentation. Top: the training loop picks an edit for each image from the "
              "classes it contains. Bottom: every 200 steps the online estimator updates the edit x class "
              "benefit table from gradient agreement with held-out per-class reference sets.")

doc.add_heading("5. Experimental Design", level=1)
table("Table 4. Datasets.",
      ["Domain", "Dataset", "Why it is included"],
      [["Road damage", "RDD2022 (D00, D10, D20, D40)", "Thin cracks beside blob-like potholes; 35% of images "
        "contain 2+ classes"],
       ["Bridge damage", "dacl10k (damage classes, polygons to boxes)", "Colour-defined classes (rust, "
        "efflorescence) beside shape-defined cracks"],
       ["General objects", "PASCAL VOC 2007+2012", "Standard benchmark; checks the method beyond damage"]],
      [1.2, 2.3, 3.0])
para("Every arm uses flips and scale jitter. The adaptive menu has 13 edits at one moderate strength each: "
     "translation, rotation (±10°), vertical flip (excluded for RDD and VOC), brightness, contrast, saturation, "
     "hue, Gaussian blur, Gaussian noise, JPEG compression, sharpening, random erasing and mosaic.",
     bold_lead="Augmentations. ")
table("Table 5. Experiments and the question each answers.",
      ["Experiment", "Design", "Question"],
      [["Single-edit study", "Shared 3-epoch start, then base vs base + one edit for 3 epochs; 3 datasets x "
        "2 detectors x 3 seeds (258 runs)", "Do different classes benefit from different edits?"],
       ["Proxy check", "4 edits trained on the full schedule", "Do short runs predict full-run effects?"],
       ["Class-targeted study", "Edit applied only inside one class's boxes", "Does an edit for one class affect "
        "the others?"],
       ["Main comparison", "No extra edits, uniform random edits, policy from the single-edit table, shared-"
        "utility, class-utility; full training, 3 seeds", "Does class information improve on the same method "
        "without it?"],
       ["Ablations", "No harm block; no 'do nothing'; inverse-AP class weights; the v1 tolerance controller",
        "Which parts matter?"]],
      [1.4, 3.1, 2.0])
para("Main metric is test macro AP50, with AP50:95, per-class AP and false alarms as secondary. Arms share "
     "seeds. Each difference gets a paired image-level bootstrap interval and a seed-level paired t interval; "
     "a claim needs both to exclude zero. The test set is opened once, after all settings are frozen.",
     bold_lead="Statistics. ")

doc.add_heading("6. Current Status: Implementation and Running Experiments", level=1)
para("The full pipeline was built as a new, tested package:")
bullet(" common record format and loaders for RDD2022, PASCAL VOC and dacl10k, with data checks "
       "(class co-occurrence, near-duplicate screen between splits, reference-set split);", "Data:")
bullet(" all 13 edits with box-correct geometry and label-safe rejection, plus mosaic;", "Augmentation:")
bullet(" Faster R-CNN and RetinaNet, trainer with checkpoint resume and a deadline stop for cluster time "
       "limits, COCO evaluation with fast per-class AP50 bootstrap;", "Training and evaluation:")
bullet(" automatic PBS job generation for CCAST and the single-edit analysis (edit x class table, heatmap, "
       "interaction test, proxy agreement, training-length check).", "Cluster and analysis:")
para("The code passes 344 automated tests, run before every cluster job. Experiments are running now on CCAST "
     "(NVIDIA A10): data preparation and smoke tests for RDD2022 first, then the single-edit study and the "
     "training-length check for RDD2022 with Faster R-CNN. VOC and dacl10k follow once downloaded and checked. "
     "No results from the new study are reported here yet; Table 6 shows what will be filled in.",
     bold_lead="Experiments in progress. ")
table("Table 6. Single-edit study, RDD2022 + Faster R-CNN: change in validation AP50 vs base (mean of 3 seeds). "
      "Running.",
      ["Edit", "D00", "D10", "D20", "D40", "Macro"],
      [[e, "running", "running", "running", "running", "running"] for e in
       ["Translation", "Rotation", "Brightness", "Contrast", "Saturation", "Hue", "Blur", "Noise",
        "JPEG", "Sharpen", "Random erasing", "Mosaic"]],
      [1.6, 0.95, 0.95, 0.95, 0.95, 0.95], italic_col=None)

doc.add_heading("7. Plan to the Final Package", level=1)
table("Table 7. Remaining schedule.",
      ["Date", "Milestone", "Planned content"],
      [["Nov 16", "Complete manuscript draft", "Single-edit results for RDD (and VOC/dacl10k if ready); full "
        "draft with method diagram; remaining experiment list"],
       ["Nov 23", "Revised manuscript", "Main comparison on RDD with 3 seeds; ablations; proxy check; "
        "revised discussion and limitations"],
       ["Dec 4", "Final package", "Remaining datasets/detector as compute allows; final manuscript, "
        "code, data documentation, slides"]],
      [0.8, 1.6, 4.1])
para("If compute runs short, the order is: single-edit study with Faster R-CNN on all datasets, then the main "
     "comparison with Faster R-CNN, then the proxy check, the class-targeted study, RetinaNet, and the "
     "ablations. The v1 results above stay in the paper as the tolerance-based baseline and as a documented "
     "negative result.")

out = os.path.join(OUT, "CE738_Main_Results_and_Validation_2026-11-09.docx")
doc.save(out)
print(out)
