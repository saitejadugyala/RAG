"""STEP 2 - Extract text, tables and images from the file.

    PDF   native text, tables (-> Markdown), OCR of embedded images,
          OCR of whole scanned pages
          headings found from font size/bold are marked with "#"
    DOCX  paragraphs in order (headings -> "#", lists -> "-"),
          tables (-> Markdown), OCR of embedded images where they appear
    TXT   text, with the encoding detected automatically and likely
          headings (short standalone lines) marked with "#"

Tables are written as Markdown so their rows and columns survive chunking.
A failure on one page or one image is recorded as a warning and the rest
of the file is still extracted.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from .errors import ExtractionError, OCRError
from .models import Block, Document
from .ocr import OCR
from .settings import Settings


def extract(path: Path, file_type: str, fingerprint: str, settings: Settings, ocr: OCR) -> Document:
    doc = Document(path=str(path.resolve()), name=path.name, file_type=file_type,
                   fingerprint=fingerprint)
    if file_type == "pdf":
        _extract_pdf(path, doc, settings, ocr)
    elif file_type == "docx":
        _extract_docx(path, doc, ocr)
    else:
        _extract_txt(path, doc)
    return doc


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """[["Name", "Age"], ["Ann", "31"]]  ->  | Name | Age |\\n| --- | --- |\\n| Ann | 31 |"""
    cells = [[re.sub(r"-\n(?=\S)", "-", c or "").replace("\n", " ").replace("|", "/").strip()
              for c in row] for row in rows]
    cells = [row for row in cells if any(row)]
    if not cells:
        return ""
    width = max(len(row) for row in cells)
    cells = [row + [""] * (width - len(row)) for row in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in cells[1:]]
    return "\n".join(lines)


# ================================================================== PDF
def _extract_pdf(path: Path, doc: Document, settings: Settings, ocr: OCR) -> None:
    import pymupdf

    try:
        pdf = pymupdf.open(path)
    except Exception as e:
        raise ExtractionError(f"Cannot open PDF: {e}") from e
    if pdf.needs_pass and not pdf.authenticate(""):
        raise ExtractionError("PDF is password-protected")
    if pdf.page_count == 0:
        raise ExtractionError("PDF has no pages")

    doc.title = (pdf.metadata or {}).get("title") or None
    doc.stats["pages"] = pdf.page_count
    doc.stats["body_font_size"] = _pdf_body_font_size(pdf)
    images_done: set[int] = set()  # a logo on every page is only OCR'd once
    failed_pages = 0

    for number, page in enumerate(pdf, start=1):
        try:
            _extract_pdf_page(pdf, page, number, doc, settings, ocr, images_done)
        except Exception as e:
            failed_pages += 1
            doc.warnings.append(f"page {number}: could not extract ({e})")
    pdf.close()

    if failed_pages == doc.stats["pages"]:
        raise ExtractionError("No page could be extracted")


def _extract_pdf_page(pdf, page, number, doc, settings, ocr, images_done) -> None:
    """Extract one page. Text, tables and image text are collected with their
    vertical position and added in reading order (top to bottom), so a table
    or chart stays under the heading it belongs to."""
    import pymupdf

    items: list[tuple[float, float | None, float | None, Block]] = []  # (top, bottom, line height, block)

    # 1) Tables
    table_areas = []
    try:
        for table in page.find_tables().tables:
            markdown = table_to_markdown(_pdf_table_rows(table, page))
            if markdown:
                items.append((table.bbox[1], None, None, Block(markdown, kind="table", page=number)))
                table_areas.append(pymupdf.Rect(table.bbox))
    except Exception as e:
        doc.warnings.append(f"page {number}: table detection failed ({e})")

    # 2) Normal text, skipping areas already captured as tables.
    #    Lines printed clearly bigger (or bold and short) than the body text are
    #    marked as headings with "#", so structure-aware chunking can use them.
    text_items = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:  # 1 = image block
            continue
        text = _pdf_block_text(block, doc.stats["body_font_size"], table_areas)
        if text:
            top, bottom = block["bbox"][1], block["bbox"][3]
            line_height = (bottom - top) / max(1, len(block["lines"]))
            text_items.append((top, bottom, line_height, text))
    page_text = "".join(t for *_, t in text_items)

    # 3) Hardly any text? It's a scanned page - render it and OCR the whole page
    if len(page_text) < settings.scanned_page_chars and not table_areas:
        if not ocr.available:
            doc.warnings.append(f"page {number}: looks scanned but OCR is not available")
            return
        picture = page.get_pixmap(dpi=settings.ocr_dpi)
        result = ocr.read_image(Image.open(io.BytesIO(picture.tobytes("png"))))
        if ocr.is_good(result):
            doc.blocks.append(Block(result.text, kind="ocr_page", page=number,
                                    ocr_confidence=round(result.confidence, 1)))
        else:
            doc.warnings.append(f"page {number}: scanned page unreadable "
                                f"(confidence {result.confidence:.0f})")
        return

    items += [(top, bottom, lh, Block(t, kind="text", page=number)) for top, bottom, lh, t in text_items]

    # 4) Images on the page (charts, screenshots, pictures of tables)
    if ocr.available:
        for image_info in page.get_images(full=True):
            xref = image_info[0]
            if xref in images_done:
                continue
            images_done.add(xref)
            try:
                image = pdf.extract_image(xref)
                result = ocr.read_image_bytes(image["image"], is_table=True)
                if ocr.is_good(result):
                    rects = page.get_image_rects(xref)
                    top = rects[0].y0 if rects else page.rect.height
                    items.append((top, None, None, Block(result.text, kind="ocr_image", page=number,
                                                         ocr_confidence=round(result.confidence, 1))))
            except OCRError as e:
                doc.warnings.append(f"page {number}, image {xref}: {e}")

    # 5) Add everything in reading order. Neighbouring text blocks are joined:
    #    as the SAME paragraph when the gap between them is smaller than a line
    #    (some PDFs store every printed line as its own block), otherwise as a
    #    new paragraph. Headings always stay a paragraph of their own.
    last_bottom = last_line_height = None
    for top, bottom, line_height, block in sorted(items, key=lambda item: item[0]):
        previous = doc.blocks[-1] if doc.blocks else None
        if (block.kind == "text" and previous is not None and previous.kind == "text"
                and previous.page == number and last_bottom is not None):
            same_paragraph = (top - last_bottom < 0.8 * last_line_height
                              and not block.text.startswith("#")
                              and not previous.text.rsplit("\n\n", 1)[-1].startswith("#"))
            previous.text += ("\n" if same_paragraph else "\n\n") + block.text
        else:
            doc.blocks.append(block)
        last_bottom, last_line_height = bottom, line_height


def _pdf_body_font_size(pdf) -> float:
    """The font size used for most of the characters = normal body text."""
    sizes: Counter = Counter()
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _pdf_block_text(block: dict, body_size: float, table_areas: list) -> str:
    """Text of one PDF text block, with heading lines marked "#", "##" or "###".

    Works line by line, because PyMuPDF sometimes puts a heading (or the last
    row of a table) in the same block as the text next to it:
      * lines inside a table area are skipped - the table was already captured
      * font >= 1.6 x body size  -> "# "    font >= 1.15 x body size -> "## "
      * a short, all-bold block that isn't a sentence -> "### "
    """
    import pymupdf

    lines = []
    for line in block["lines"]:
        centre = pymupdf.Rect(line["bbox"])
        centre = pymupdf.Point((centre.x0 + centre.x1) / 2, (centre.y0 + centre.y1) / 2)
        if any(centre in area for area in table_areas):
            continue
        spans = [sp for sp in line["spans"] if sp["text"].strip()]
        text = "".join(sp["text"] for sp in line["spans"]).strip()
        if spans:
            lines.append((text, min(sp["size"] for sp in spans),
                          all(sp["flags"] & 16 for sp in spans)))
    if not lines:
        return ""

    def size_marker(text: str, size: float) -> str:
        if not body_size or len(text) > 120:
            return ""
        return "# " if size >= body_size * 1.6 else "## " if size >= body_size * 1.15 else ""

    whole = " ".join(t for t, _, _ in lines)
    smallest = min(size for _, size, _ in lines)
    if (body_size and len(lines) <= 2 and all(bold for _, _, bold in lines) and len(whole) <= 80
            and not whole.endswith(".") and not size_marker(whole, smallest)
            and smallest >= body_size):
        return "### " + whole

    parts: list[str] = []  # each heading becomes its own paragraph
    for text, size, _ in lines:
        marker = size_marker(text, size)
        if marker:
            if parts and parts[-1].startswith(marker):  # a heading wrapped onto 2 lines
                parts[-1] += " " + text
            else:
                parts.append(marker + text)
        elif parts and not parts[-1].startswith("#"):
            parts[-1] += "\n" + text
        else:
            parts.append(text)
    return "\n\n".join(parts)


def _pdf_table_rows(table, page) -> list[list[str]]:
    """Read every cell's text from its exact position on the page.

    PyMuPDF's own extract()/to_markdown() can drop the spaces inside cells
    ("15 USD" -> "15USD", "What happens" -> "Whathappens"); reading the text
    inside each cell's box keeps them. Falls back to extract() if that fails."""
    import pymupdf

    try:
        rows = [[page.get_textbox(pymupdf.Rect(cell)).strip() if cell else "" for cell in row.cells]
                for row in table.rows]
        if any(any(cell for cell in row) for row in rows):
            return rows
    except Exception:
        pass
    return table.extract()


# ================================================================= DOCX
def _extract_docx(path: Path, doc: Document, ocr: OCR) -> None:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        word = docx.Document(str(path))
    except Exception as e:
        raise ExtractionError(f"Cannot open DOCX: {e}") from e
    doc.title = word.core_properties.title or None

    lines: list[str] = []  # paragraphs collected until the next table or image
    images_done: set[str] = set()

    def save_lines() -> None:
        if lines:
            doc.blocks.append(Block("\n".join(lines)))
            lines.clear()

    def ocr_image(rel_id: str) -> None:
        """OCR one embedded image and add it at the current position."""
        if rel_id in images_done or rel_id not in word.part.related_parts:
            return
        images_done.add(rel_id)
        if not ocr.available:
            doc.warnings.append(f"image {rel_id} skipped: OCR is not available")
            return
        try:
            result = ocr.read_image_bytes(word.part.related_parts[rel_id].blob, is_table=True)
            if ocr.is_good(result):
                save_lines()  # keep the image text after the paragraphs before it
                doc.blocks.append(Block(result.text, kind="ocr_image",
                                        ocr_confidence=round(result.confidence, 1)))
        except Exception as e:
            doc.warnings.append(f"image {rel_id}: {e}")

    # Walk the body in order, so tables and images stay next to the text around them
    for element in word.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        try:
            if tag == "p":
                line = _docx_paragraph(Paragraph(element, word))
                if line.startswith("#"):
                    save_lines()  # a heading starts a new block
                if line:
                    lines.append(line)
                for rel_id in _docx_image_ids(element):
                    ocr_image(rel_id)
            elif tag == "tbl":
                save_lines()
                markdown = _docx_table(Table(element, word))
                if markdown:
                    doc.blocks.append(Block(markdown, kind="table"))
                for rel_id in _docx_image_ids(element):  # images inside table cells
                    ocr_image(rel_id)
        except Exception as e:
            doc.warnings.append(f"skipped a <{tag}> element ({e})")
    save_lines()

    # Images not found in the body (e.g. floating shapes) are added at the end
    for rel in word.part.rels.values():
        if "image" in rel.reltype and not rel.is_external:
            ocr_image(rel.rId)


def _docx_image_ids(element) -> list[str]:
    """Relationship ids of the pictures inside a paragraph or table."""
    embed = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
    blips = element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
    return [b.get(embed) for b in blips if b.get(embed)]


def _docx_paragraph(paragraph) -> str:
    """Return the paragraph as text, marking headings with # and lists with -."""
    text = paragraph.text.strip()
    if not text:
        return ""
    style = (paragraph.style.name if paragraph.style is not None else "").lower()
    if style == "title":
        return "# " + text
    if style.startswith("heading"):
        level = int("".join(ch for ch in style if ch.isdigit()) or 1)
        return "#" * min(level + 1, 6) + " " + text  # Title is #, Heading 1 is ##
    if style.startswith("list"):
        return "- " + text
    return text


def _docx_table(table) -> str:
    rows = []
    for row in table.rows:
        cells, previous = [], None
        for cell in row.cells:
            if cell._tc is previous:  # merged cells are repeated by python-docx
                continue
            previous = cell._tc
            cells.append(cell.text)
        rows.append(cells)
    return table_to_markdown(rows)


# ================================================================== TXT
def _extract_txt(path: Path, doc: Document) -> None:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings = ["utf-16"]
    else:
        encodings = ["utf-8-sig", "cp1252", "latin-1"]  # latin-1 never fails

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ExtractionError("Could not decode the text file")

    doc.stats["encoding"] = encoding
    if encoding in ("cp1252", "latin-1"):
        doc.warnings.append(f"not UTF-8; read as {encoding}")
    doc.blocks.append(Block(_mark_txt_headings(text)))


def _mark_txt_headings(text: str) -> str:
    """Plain text has no styles, so guess headings: the first line of the file,
    or a short line that starts a new paragraph (blank line before it, text right after it),
    starts with a capital or a number like "1.", and doesn't end like a sentence.
    Questions ("How do I cancel?") count, so each FAQ answer gets its own section."""
    lines = text.splitlines()
    first_line = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    for i, line in enumerate(lines):
        stripped = line.strip()
        before_blank = i == 0 or not lines[i - 1].strip()
        after_text = i + 1 < len(lines) and bool(lines[i + 1].strip())
        if i == first_line:  # the first line of a file is usually its title
            before_blank, after_text = True, True
        looks_like_heading = (
            3 <= len(stripped) <= 80
            and len(stripped.split()) <= 10
            and (stripped[0].isupper() or stripped[0].isdigit())
            and not stripped.endswith((".", ",", ";", ":"))
            and ": " not in stripped  # "Date: 10 November" is a field, not a heading
            and not stripped.startswith(("#", "-", "*", "|"))
        )
        if before_blank and after_text and looks_like_heading:
            lines[i] = "## " + stripped
    return "\n".join(lines)
