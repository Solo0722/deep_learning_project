"""Plain black-and-white Word formatting shared by both progress reports."""
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

BLACK = RGBColor(0, 0, 0)


class Report:
    def __init__(self, title, milestone):
        self.doc = d = Document()
        s = d.sections[0]
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(s, m, Inches(1))
        n = d.styles["Normal"]
        n.font.name = "Times New Roman"
        n.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        n.font.size = Pt(12)
        n.paragraph_format.space_after = Pt(6)
        n.paragraph_format.line_spacing = 1.15
        for name, size in (("Heading 1", 13), ("Heading 2", 12)):
            h = d.styles[name]
            h.font.name = "Times New Roman"
            h.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.italic = False
            h.font.color.rgb = BLACK
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(4)
            h.paragraph_format.keep_with_next = True
        self.centre(title, bold=True, size=14)
        self.centre(milestone)
        self.centre("Solomon Owusu Ansah, Student ID 1641802")
        self.centre("CE 738 Deep Learning for Engineers")
        self.fig_n = 0
        self.tab_n = 0

    def centre(self, text, bold=False, size=12):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(size)

    def h1(self, text):
        self.doc.add_heading(text, level=1)

    def h2(self, text):
        self.doc.add_heading(text, level=2)

    def p(self, text):
        return self.doc.add_paragraph(text)

    def bullets(self, items):
        for it in items:
            q = self.doc.add_paragraph(it, style="List Bullet")
            q.paragraph_format.space_after = Pt(2)

    def table(self, caption, header, rows, widths):
        self.tab_n += 1
        c = self.doc.add_paragraph()
        c.add_run(f"Table {self.tab_n}. ").bold = True
        c.add_run(caption)
        c.paragraph_format.keep_with_next = True
        c.paragraph_format.space_after = Pt(3)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            r = cell.paragraphs[0].add_run(h)
            r.bold = True
            r.font.size = Pt(10)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(10)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
                pf = row.cells[i].paragraphs[0].paragraph_format
                pf.space_after = Pt(0)
                pf.line_spacing = 1.0
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return self.tab_n

    def figure(self, path, caption, width=6.3):
        self.fig_n += 1
        q = self.doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.keep_with_next = True
        q.add_run().add_picture(path, width=Inches(width))
        c = self.doc.add_paragraph()
        c.add_run(f"Figure {self.fig_n}. ").bold = True
        c.add_run(caption)
        return self.fig_n

    def save(self, path):
        self.doc.save(path)
        print(path)
