import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = r"C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project"
OUT = os.path.join(ROOT, "output", "reports")
FIG = os.path.join(OUT, "nov2_figs")
os.makedirs(FIG, exist_ok=True)

# ---------- Figure 3: per-class AP ----------
classes = ["D00\nlongitudinal", "D10\ntransverse", "D20\nalligator", "D40\npothole"]
v1 = {"Identity": [24.744, 20.206, 29.975, 16.367],
      "Online shared": [24.905, 20.050, 30.162, 17.120],
      "Class-adaptive": [24.235, 19.446, 30.069, 16.502]}
p8 = {"Geometry only": [22.3915, 16.5069, 24.7822, 15.0576],
      "Class-adaptive": [21.5166, 15.7426, 22.7198, 13.7686]}
cols = ["#1b6b8a", "#13804f", "#b33a50"]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
w = 0.26
for ax, data, title, cc in [
        (axs[0], v1, "Study 1: 20-epoch runs (selected checkpoint)", cols),
        (axs[1], p8, "Study 2: paired runs at epoch 8", ["#555555", "#b33a50"])]:
    n = len(data)
    for i, (k, v) in enumerate(data.items()):
        xs = [j + (i - (n - 1) / 2) * w for j in range(4)]
        ax.bar(xs, v, w, label=k, color=cc[i])
    ax.set_xticks(range(4))
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("Validation AP50:95 (%)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(10, 32)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
f3 = os.path.join(FIG, "per_class_ap.png")
plt.savefig(f3, dpi=200)

f1 = os.path.join(ROOT, "reports", "fasterrcnn_class_adaptive_v1", "comparison.png")
f2 = os.path.join(ROOT, "reports", "rdd_v2_pair8_v1", "epoch8_comparison.png")

# ---------- Document ----------
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, m, Inches(1))
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
for name, size in (("Heading 1", 14), ("Heading 2", 12)):
    h = doc.styles[name]
    h.font.name = "Calibri"
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True


def para(text, bold_lead=None, italic=False, align=None, size=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullet(text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(2)
    return p


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear")
    s.set(qn("w:color"), "auto")
    s.set(qn("w:fill"), hexcolor)
    tcPr.append(s)


def table(caption, header, rows, widths, bold_row=None):
    c = doc.add_paragraph()
    c.add_run(caption).bold = True
    c.paragraph_format.keep_with_next = True
    c.paragraph_format.space_after = Pt(2)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        shade(cell, "DCE6F0")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9.5)
            if bold_row is not None and ri == bold_row:
                r.bold = True
            if i > 0:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in t.rows:
        for i, wd in enumerate(widths):
            row.cells[i].width = Inches(wd)
            row.cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(path, caption, width=6.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    p.paragraph_format.keep_with_next = True
    c = doc.add_paragraph()
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9.5)


# Title block
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Class-Adaptive Data Augmentation for Multi-Class Road Damage Detection")
r.bold = True
r.font.size = Pt(16)
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Project Milestone: Preliminary Results (due November 2, 2026)")
r.font.size = Pt(12)
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Solomon Owusu Ansah  |  Student ID 1641802\nCE 738 Deep Learning for Engineers (Drs. Armstrong Aboah and Trung Le)")

doc.add_heading("1. Summary", level=1)
para("This milestone reports the first comparative results for the proposed class-adaptive augmentation method on the "
     "RDD2022 road damage dataset. The method chooses local appearance edits (brightness, contrast, gamma, blur, noise) "
     "separately for each damage class, using an online check of how well the current detector still finds that class "
     "after the edit. Three studies were completed:")
bullet(" five 20-epoch training runs comparing no augmentation, fixed augmentation, a shared online policy and the "
       "class-adaptive policy;", "Study 1:")
bullet(" a four-arm pilot and a longer two-arm paired run (to epoch 8) with a stronger common geometry baseline;", "Study 2:")
bullet(" a determinism check showing that repeated training is exactly reproducible under strict GPU settings.", "Study 3:")
para("The prototype works and every result was independently re-checked, but the class-adaptive policy did not "
     "improve accuracy. It scored 22.56 AP50:95 against 23.06 for the shared online policy (Study 1) and "
     "18.44 against 19.68 for geometry-only training at epoch 8 (Study 2). The error analysis points to one main "
     "cause: the controller measures whether the detector tolerates an edit, not whether training on the edit "
     "helps. Section 6 sets out the plan that follows from this.", bold_lead="Main finding. ")

doc.add_heading("2. Experimental Setup", level=1)
para("RDD2022 images were split by source cluster into 26,863 training and 5,760 validation images (8,225 validation "
     "boxes; 2,193 validation images contain no target damage). Four classes are used: D00 longitudinal cracks, "
     "D10 transverse cracks, D20 alligator cracks and D40 potholes. The test split was kept closed.", bold_lead="Data. ")
para("Faster R-CNN with a ResNet-50 FPN backbone, initialised from COCO weights, 640-pixel letterboxed input, "
     "batch size 4, SGD (learning rate 0.0025, momentum 0.9, weight decay 0.0005), seed 42, trained on one NVIDIA A10 "
     "GPU on CCAST. Metrics are COCO AP50:95 and AP50 on the validation set, plus per-class AP and false alarms on "
     "images with no damage.", bold_lead="Detector and training. ")
para("Before an update, the controller runs the current detector on a fixed panel of 256 training images, with and "
     "without each of 24 candidate edits (8 operations x 3 strengths). An edit is allowed for a class if the detector "
     "still finds at least 80% of the objects of that class it found before (with at least 16 objects from 8 "
     "clusters as support). The shared policy uses one allowed set for all classes; the class-adaptive policy uses a "
     "separate allowed set per class or class combination and applies the edit only inside the damage boxes.",
     bold_lead="Controller. ")

doc.add_heading("3. Initial Comparative Results", level=1)
doc.add_heading("3.1 Study 1: five conditions, 20 epochs", level=2)
para("All five conditions used the same initialisation, data order, 134,320 optimizer steps and evaluation. The "
     "checkpoint with the highest validation AP50:95 was selected for each run.")
table("Table 1. Study 1 results (validation, one run per condition).",
      ["Condition", "Best epoch", "AP50:95", "AP50", "Final-epoch AP50:95", "Run time (h)"],
      [["No augmentation (identity)", 14, "22.82", "51.14", "22.39", "11.1"],
       ["Fixed image-level edits", 14, "22.81", "50.69", "22.58", "11.5"],
       ["Fixed local edits", 18, "22.56", "50.16", "22.49", "13.3"],
       ["Online shared local policy", 14, "23.06", "51.03", "22.59", "14.1"],
       ["Online class-adaptive local (proposed)", 18, "22.56", "50.18", "22.49", "14.5"]],
      [2.3, 0.8, 0.8, 0.7, 1.2, 0.8], bold_row=3)
para("The shared online policy was best (+0.24 AP over no augmentation). The class-adaptive policy was 0.50 AP below "
     "it and 0.26 AP below no augmentation. All differences are under 0.6 AP and come from one run each, so they "
     "are within the range expected from training noise (see Section 5.4).")
figure(f1, "Figure 1. Validation AP50:95 per epoch (left) and selected-checkpoint AP50:95 (right) for the five "
           "Study 1 conditions. The jump after epoch 12 is the scheduled learning-rate drop and appears in all runs.")

doc.add_heading("3.2 Study 2: stronger geometry baseline", level=1 if False else 2)
para("Study 2 added label-safe geometric augmentation (flips, small resizes and shifts) to every arm, so local "
     "appearance edits are tested on top of a stronger baseline. All arms started from the same epoch-1 checkpoint.")
table("Table 2. Four-arm pilot, epoch 4 (validation AP50:95).",
      ["Arm", "AP50:95", "AP50", "Difference vs geometry only"],
      [["Geometry only", "19.81", "45.05", "0.00"],
       ["+ fixed local edits", "18.86", "43.44", "-0.95"],
       ["+ online shared policy", "19.37", "43.87", "-0.44"],
       ["+ class-adaptive policy", "19.61", "44.76", "-0.20"]],
      [2.4, 1.0, 1.0, 2.0])
para("Class adaptation recovered most of the loss caused by fixed edits and beat the shared policy by 0.24 AP, but "
     "did not beat geometry-only training. Because the class-adaptive arm was still rising at epoch 4, it was "
     "extended to a fixed epoch-8 endpoint in a paired run against geometry-only.")
table("Table 3. Paired epoch-8 comparison (validation, fixed endpoint, no checkpoint selection).",
      ["Metric (%)", "Geometry only", "Class-adaptive", "Difference"],
      [["AP50:95", "19.68", "18.44", "-1.25"], ["AP50", "44.33", "41.97", "-2.36"],
       ["AP75", "14.56", "13.42", "-1.14"], ["AR100", "43.70", "43.01", "-0.70"],
       ["AP small", "8.11", "7.40", "-0.71"], ["AP medium", "16.26", "14.96", "-1.31"],
       ["AP large", "21.18", "20.06", "-1.11"]],
      [1.6, 1.4, 1.4, 1.2])
figure(f2, "Figure 2. Training curves of the paired epoch-8 run (left) and the class-adaptive minus geometry-only "
           "difference per epoch (right). The gap widens after epoch 4 instead of closing.")

doc.add_heading("4. Draft Results Section", level=1)
para("Under a matched training budget, the class-adaptive controller did not improve detection accuracy on RDD2022. "
     "In 20-epoch training without geometric augmentation (Table 1), the best validation AP50:95 was obtained by the "
     "shared online policy (23.06), followed by no augmentation (22.82) and the class-adaptive policy (22.56). With "
     "label-safe geometric augmentation as a common base (Tables 2 and 3), class-adaptive local edits outperformed "
     "fixed local edits and the shared online policy at epoch 4, but remained below geometry-only training. Over a "
     "fixed eight-epoch horizon the gap widened to 1.25 AP50:95, and all four classes declined (Figure 3). The "
     "class-adaptive detector produced fewer false alarms on damage-free images (2,890 versus 3,371 boxes) at the "
     "cost of lower recall (64.7% versus 67.0%), so the policy shifted the precision-recall operating point rather "
     "than improving overall detection. These are single-seed development results on the validation set; the "
     "determinism study (Section 5.4) shows that repeated training with identical inputs is exactly reproducible "
     "under strict GPU settings, but variation across random seeds remains to be measured.")

doc.add_heading("5. Error Analysis", level=1)
doc.add_heading("5.1 Per-class behaviour", level=2)
figure(f3, "Figure 3. Per-class validation AP50:95. Left: Study 1. Right: Study 2 at epoch 8.")
para("In Study 1 the class-adaptive policy lost about 0.6 AP on D00, D10 and D40 relative to the shared policy and "
     "matched it on D20. Potholes (D40), the smallest class, gained the most from online control in general "
     "(+0.75 AP for the shared policy over no augmentation). In Study 2 every class declined, with the largest loss on "
     "alligator cracks (D20, -2.06 AP). The losses are not limited to rare classes, which suggests a policy-level "
     "problem rather than a single weak class.")

doc.add_heading("5.2 False alarms and recall", level=2)
table("Table 4. Detection errors at confidence 0.25, IoU 0.5 (Study 2, epoch 8).",
      ["Measure", "Geometry only", "Class-adaptive"],
      [["Correctly detected targets", "5,512", "5,323"], ["Recall (%)", "67.0", "64.7"],
       ["Precision (%)", "20.4", "21.2"], ["False-alarm boxes on damage-free images", "3,371", "2,890"],
       ["Damage-free images with any detection (of 2,193)", "1,239", "1,147"]],
      [3.2, 1.4, 1.4])
para("The class-adaptive detector is more conservative: about 14% fewer false-alarm boxes but 189 fewer detected "
     "targets. The damage-free images were not visually re-checked; some of these false alarms may be unlabelled "
     "damage, so these counts are an upper bound on true errors.")

doc.add_heading("5.3 Why the controller did not help", level=2)
bullet(" The controller allows any edit the detector can tolerate (at least 80% of objects still found). Tolerating "
       "an edit does not mean that training on it improves the detector. In Study 1, the strongest strength was "
       "used for 76% of edits.", "It measures tolerance, not benefit.")
bullet(" Whenever an edit was allowed it was applied; leaving the image unchanged was never compared as an option. "
       "In Study 2, blur or noise made up 50% of applied edits even though their value was never tested.",
       "No 'do nothing' option.")
bullet(" Most combinations of classes in one image had too few examples for a reliable decision. In Study 1, 9 of "
       "15 class combinations were never active; 8.6% of opportunities fell back to no edit. In Study 2, 77% of "
       "pothole opportunities at epoch 8 borrowed evidence from pooled data.", "Sparse class-combination evidence.")
bullet(" Each update costs about 6 minutes of extra GPU time (about 1 hour per 20-epoch run), with no accuracy "
       "return so far.", "Extra cost.")

doc.add_heading("5.4 Reproducibility check", level=2)
para("The Study 1 runs already differed at epoch 1, before any augmentation was applied (16.54, 15.89, 16.42, 15.73 "
     "and 16.52 AP50:95), and an early smoke test showed diverging trajectories for arms with identical treatment. "
     "A dedicated diagnostic on CCAST (2026-11-02) enabled strict deterministic GPU execution (deterministic "
     "algorithms and cuDNN, TF32 off) and compared repeated training, training after a controller probe and training "
     "after a checkpoint save and restore. All 88 recorded comparisons, covering inputs, model, optimizer, gradients and "
     "random-number states over seven updates, matched exactly. This points to GPU numerical non-determinism as the "
     "likely cause of the earlier divergence, with no state leak found in the probe or resume paths, and the "
     "epoch-8 run in Study 2 used these strict settings. Seed-to-seed variation has not yet been measured. All reported metrics were also "
     "recomputed independently from saved predictions with official COCO evaluation, with zero difference.")

doc.add_heading("6. Plan for Remaining Experiments", level=1)
para("The results so far show that choosing augmentations by detector tolerance does not improve accuracy. The "
     "next stage therefore changes the signal the method uses: from 'can the detector tolerate this edit for this "
     "class?' to 'does training on this edit improve this class?'. The planned work:")
bullet(" Train short runs that each add a single edit, and measure the change in per-class AP against a no-edit "
       "run. This tests directly whether different classes benefit from different edits.",
       "Measure per-class benefit first.")
bullet(" Estimate each edit's benefit for each class during training from gradient agreement with a small "
       "per-class reference set, always include 'no edit' as a choice, and block edits that measurably hurt a class.",
       "Rebuild the controller around benefit.")
bullet(" Compare against no augmentation, uniform augmentation, a policy built from the measured effects, and the "
       "same controller with one shared score for all classes (to isolate the value of class-specific selection).",
       "Fair comparisons.")
bullet(" Repeat runs over 3 seeds, test on two detectors (Faster R-CNN and RetinaNet) and add two more datasets "
       "(dacl10k bridge damage and PASCAL VOC) so the conclusion does not rest on one dataset.",
       "Robustness.")
bullet(" Keep the current class-adaptive controller as an ablation, and report the negative result honestly.",
       "Retain Study 1-2 evidence.")
para("Compute is planned on CCAST A10 GPUs (about 840 GPU-hours in total). The single-edit measurement runs come "
     "first, because the rest of the plan depends on finding per-class differences in benefit.")

out = os.path.join(OUT, "CE738_Preliminary_Results_2026-11-02.docx")
doc.save(out)
print(out)
