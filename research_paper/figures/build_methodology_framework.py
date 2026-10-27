"""Regenerate the vector framework figure with ReportLab."""

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas


OUTPUT = Path(__file__).with_name("methodology_framework.pdf")
WIDTH, HEIGHT = 516, 154
BOX_W, BOX_H = 148, 48
INK = HexColor("#263b4a")
FILL = HexColor("#f4f6f7")


def box(pdf, x, y, number, title, first_line, second_line):
    pdf.setStrokeColor(INK)
    pdf.setFillColor(FILL)
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, BOX_W, BOX_H, 4, stroke=1, fill=1)
    pdf.setFillColor(INK)
    pdf.setFont("Times-Bold", 10)
    pdf.drawCentredString(x + BOX_W / 2, y + 33, f"{number}. {title}")
    pdf.setFont("Times-Roman", 9)
    pdf.drawCentredString(x + BOX_W / 2, y + 20, first_line)
    pdf.drawCentredString(x + BOX_W / 2, y + 9, second_line)


def arrow(pdf, x1, y1, x2, y2):
    pdf.setStrokeColor(INK)
    pdf.setFillColor(INK)
    pdf.setLineWidth(0.85)
    pdf.line(x1, y1, x2, y2)
    dx, dy = x2 - x1, y2 - y1
    distance = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / distance, dy / distance
    tip = pdf.beginPath()
    tip.moveTo(x2, y2)
    tip.lineTo(x2 - 5 * ux + 2.5 * uy, y2 - 5 * uy - 2.5 * ux)
    tip.lineTo(x2 - 5 * ux - 2.5 * uy, y2 - 5 * uy + 2.5 * ux)
    tip.close()
    pdf.drawPath(tip, stroke=0, fill=1)


def main():
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(WIDTH, HEIGHT))
    pdf.setTitle("Online class-adaptive augmentation with overlap-group control")
    pdf.setSubject("Class-conditioned local augmentation with overlap-group control")
    pdf.setFillColor(white)
    pdf.rect(0, 0, WIDTH, HEIGHT, stroke=0, fill=1)
    top, bottom = 96, 10
    xs = (10, 184, 358)
    box(pdf, xs[0], top, 1, "Current detector", "After common warm-up", "Freeze state during probes")
    box(pdf, xs[1], top, 2, "Paired group probes", "Edit the actual box union", "Pool by group class set")
    box(pdf, xs[2], top, 3, "Joint retention gate", "Check each class and operation", "Require enough probe support")
    box(pdf, xs[2], bottom, 4, "Select group policy", "Strongest tested passing level", "Balanced difficulty + exploration")
    box(pdf, xs[1], bottom, 5, "Apply one group edit", "Sample family and operation", "Edit shared pixels once")
    box(pdf, xs[0], bottom, 6, "Optimize detector", "Full image and all annotations", "Full detection loss")
    arrow(pdf, xs[0] + BOX_W, top + 24, xs[1], top + 24)
    arrow(pdf, xs[1] + BOX_W, top + 24, xs[2], top + 24)
    arrow(pdf, xs[2] + 74, top, xs[2] + 74, bottom + BOX_H)
    arrow(pdf, xs[2], bottom + 24, xs[1] + BOX_W, bottom + 24)
    arrow(pdf, xs[1], bottom + 24, xs[0] + BOX_W, bottom + 24)
    arrow(pdf, xs[0] + 74, bottom + BOX_H, xs[0] + 74, top)
    pdf.setFillColor(INK)
    pdf.setFont("Times-Italic", 8.5)
    pdf.drawString(xs[0] + 81, 75, "periodic update")
    pdf.save()
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
