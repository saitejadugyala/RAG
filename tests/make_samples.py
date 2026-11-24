"""Generate small edge-case files (good and deliberately broken) used by the tests.

    .venv/bin/python tests/make_samples.py <folder>
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PARA = (
    "Retrieval augmented generation combines a search step with a language model. "
    "Documents are split into chunks, embedded into vectors and stored in a vector database. "
    "At query time the most similar chunks are retrieved and passed to the model as context. "
)


def text_image(lines: list[str], size=(1600, 600)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=42)
    y = 40
    for ln in lines:
        d.text((40, y), ln, fill="black", font=font)
        y += 70
    return img


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def make_text_pdf(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    styles = getSampleStyleSheet()
    story = []
    for page in range(1, 4):
        story.append(Paragraph(f"Chapter {page}: Vector Search Basics", styles["Heading1"]))
        for _ in range(4):
            story.append(Paragraph(PARA * 2, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if page == 2:
            t = Table([["Model", "Dimensions", "Max tokens"],
                       ["text-embedding-3-small", "1536", "8191"],
                       ["text-embedding-3-large", "3072", "8191"]])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
            story.append(t)
            img = text_image(["Quarterly revenue grew 12 percent",
                              "Customer churn fell to 3 percent"])
            story.append(RLImage(io.BytesIO(png_bytes(img)), width=400, height=150))
        story.append(PageBreak())

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 20, "ACME Corp Confidential")
        canvas.drawString(500, 20, f"Page {doc.page}")

    SimpleDocTemplate(str(path), pagesize=A4).build(story, onFirstPage=footer, onLaterPages=footer)


def make_scanned_pdf(path: Path) -> None:
    import pymupdf as fitz

    img = text_image([
        "SCANNED INVOICE NUMBER 4471",
        "The customer agreed to pay within thirty days.",
        "Late payments incur a two percent monthly fee.",
        "Please contact billing for any questions.",
    ], size=(1700, 900))
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(20, 20, 575, 320), stream=png_bytes(img))
    doc.save(str(path))


def make_docx(path: Path) -> None:
    import docx
    from docx.shared import Inches

    d = docx.Document()
    d.add_heading("Employee Handbook", level=1)
    d.add_paragraph("This handbook describes company policies for all employees. " * 3)
    d.add_heading("Leave Policy", level=2)
    d.add_paragraph("Employees receive twenty days of paid leave per year. " * 4)
    d.add_paragraph("Unused leave can be carried over to the next year.", style="List Bullet")
    t = d.add_table(rows=3, cols=3)
    for r, row in enumerate([["Leave type", "Days", "Paid"], ["Annual", "20", "Yes"],
                             ["Sick", "10", "Yes"]]):
        for c, val in enumerate(row):
            t.cell(r, c).text = val
    img = text_image(["Remote work allowed two days per week",
                      "Manager approval required in advance"])
    d.add_picture(io.BytesIO(png_bytes(img)), width=Inches(5))
    d.save(str(path))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    make_text_pdf(out / "guide.pdf")
    make_scanned_pdf(out / "scanned_invoice.pdf")
    make_docx(out / "handbook.docx")
    (out / "notes.txt").write_text(("Chunk overlap preserves context across boundaries. " * 30 + "\n\n") * 3)
    (out / "legacy_cp1252.txt").write_bytes(("Café résumé naïve – smart “quotes”. " * 20).encode("cp1252"))
    # ---- files that must fail cleanly ----
    (out / "empty.txt").write_text("")
    (out / "corrupt.pdf").write_bytes(b"%PDF-1.4\n garbage garbage")
    (out / "fake.docx").write_text("this is not a docx")
    (out / "data.csv").write_text("a,b\n1,2\n")
    (out / "tiny.txt").write_text("hi")
    print(f"samples written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples"))
