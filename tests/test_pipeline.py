"""Tests for the whole pipeline and each step. Fake embeddings: no API key needed.

    .venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_samples  # noqa: E402

from ingestion import IngestionPipeline, Settings  # noqa: E402
from ingestion.errors import EmbeddingError  # noqa: E402
from ingestion.models import Block, Document  # noqa: E402
from ingestion.step3_clean import clean_document  # noqa: E402
from ingestion.step4_chunk import chunk_document  # noqa: E402
from ingestion.step5_check_chunks import check_chunks  # noqa: E402

HAS_TESSERACT = shutil.which("tesseract") is not None


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    folder = tmp_path_factory.mktemp("samples")
    make_samples.main(folder)
    return folder


def new_pipeline(tmp_path, **settings):
    return IngestionPipeline(Settings(data_dir=tmp_path / "data", **settings), fake_embeddings=True)


def results_by_name(summary):
    return {Path(r["file"]).name: r for r in summary["results"]}


def stored_text(pipeline):
    return " ".join(pipeline.store.collection.get(include=["documents"])["documents"]).lower()


# ------------------------------------------------------------- whole pipeline
def test_good_files_are_stored_and_bad_files_quarantined(samples, tmp_path):
    pipeline = new_pipeline(tmp_path)
    summary = pipeline.run([samples])
    results = results_by_name(summary)

    for good in ("guide.pdf", "handbook.docx", "notes.txt", "legacy_cp1252.txt"):
        assert results[good]["status"] == "success", results[good]
        assert results[good]["stored"] > 0

    for bad, step in [("empty.txt", "1 check file"), ("data.csv", "1 check file"),
                      ("fake.docx", "1 check file"), ("corrupt.pdf", "2 extract"),
                      ("tiny.txt", "3 clean")]:
        assert results[bad]["status"] == "quarantined", results[bad]
        assert results[bad]["failed_step"] == step, results[bad]
        assert (pipeline.settings.quarantine_dir / f"{bad}.error.json").exists()

    assert results["guide.pdf"]["tables"] >= 1
    assert summary["vectors_in_collection"] == summary["chunks_stored"]
    assert Path(summary["report_file"]).exists()

    stored = pipeline.store.collection.get(limit=1, include=["embeddings", "metadatas"])
    assert len(stored["embeddings"][0]) == 1536
    assert {"source", "chunk_index", "quality_flags", "file_hash"} <= set(stored["metadatas"][0])


@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract is not installed")
def test_ocr_reads_scanned_pages_and_images(samples, tmp_path):
    pipeline = new_pipeline(tmp_path)
    results = results_by_name(pipeline.run(
        [samples / "scanned_invoice.pdf", samples / "handbook.docx", samples / "guide.pdf"]))
    assert results["scanned_invoice.pdf"]["ocr_blocks"] == 1  # whole scanned page
    assert results["handbook.docx"]["ocr_blocks"] == 1  # image in a DOCX
    assert results["guide.pdf"]["ocr_blocks"] == 1  # image in a PDF

    text = stored_text(pipeline)
    for phrase in ("invoice", "thirty days", "remote work", "revenue"):
        assert phrase in text


def test_unchanged_files_are_skipped_and_changed_files_replaced(samples, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    shutil.copy(samples / "notes.txt", folder / "notes.txt")
    pipeline = new_pipeline(tmp_path)

    assert pipeline.run([folder])["success"] == 1
    count = pipeline.store.count()

    assert pipeline.run([folder])["skipped"] == 1
    assert pipeline.store.count() == count  # no duplicates

    (folder / "notes.txt").write_text("A completely different document about llamas. " * 10)
    assert pipeline.run([folder])["success"] == 1
    assert all("llamas" in d for d in pipeline.store.collection.get()["documents"])  # old chunks gone


def test_outage_marks_file_failed_and_next_run_retries(samples, tmp_path, monkeypatch):
    pipeline = new_pipeline(tmp_path)

    def outage(texts):
        raise EmbeddingError("simulated OpenAI outage", fatal=True)

    monkeypatch.setattr(pipeline.embedder, "embed", outage)
    result = results_by_name(pipeline.run([samples / "notes.txt"]))["notes.txt"]
    assert result["status"] == "failed" and result["failed_step"] == "6 embed"
    assert not (pipeline.settings.quarantine_dir / "notes.txt.error.json").exists()

    monkeypatch.undo()
    assert results_by_name(pipeline.run([samples / "notes.txt"]))["notes.txt"]["status"] == "success"


def test_one_bad_chunk_does_not_fail_the_file(samples, tmp_path, monkeypatch):
    pipeline = new_pipeline(tmp_path)
    real_embed = pipeline.embedder.embed

    def picky(texts):
        if len(texts) > 1:
            raise ValueError("batch rejected")
        if "Unused leave" in texts[0]:
            raise ValueError("bad input")
        return real_embed(texts)

    monkeypatch.setattr(pipeline.embedder, "embed", picky)
    result = results_by_name(pipeline.run([samples / "handbook.docx"]))["handbook.docx"]
    assert result["status"] == "success"


def test_remove_deleted_files(samples, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    shutil.copy(samples / "notes.txt", folder / "notes.txt")
    pipeline = new_pipeline(tmp_path)
    pipeline.run([folder])
    (folder / "notes.txt").unlink()
    assert len(pipeline.remove_deleted_files()) == 1
    assert pipeline.store.count() == 0


# ---------------------------------------------------------------- single steps
def test_cleaning_removes_noise():
    settings = Settings()
    invisible = "​"
    body = f"The ﬁnal re-\nport is ready for{invisible} review today.\nContents ........ 4\n"
    topics = ["alpha", "beta", "gamma", "delta", "omega"]
    blocks = [Block(f"ACME Confidential\nThis page covers {t} in depth.\nIt starts with {t}.\n"
                    f"{body}More about {t} follows.\nThe {t} section ends here.\nPage {i} of 5",
                    page=i)
              for i, t in enumerate(topics, start=1)]
    blocks += [Block("Duplicate disclaimer block.", page=5)] * 2
    doc = clean_document(Document("x", "x.pdf", "pdf", "h", blocks=blocks), settings)
    text = "\n".join(b.text for b in doc.blocks)

    assert "ACME Confidential" not in text  # header
    assert "Page 1" not in text  # footer / page number
    assert "....." not in text  # table-of-contents leader
    assert "final report is ready for review today" in text  # ligature + hyphen + line join
    assert invisible not in text
    assert "This page covers gamma in depth." in text
    assert doc.stats["duplicate_blocks_removed"] == 1


@pytest.mark.parametrize("strategy", ["recursive", "sentence", "fixed"])
def test_every_chunk_strategy_respects_the_size(strategy):
    settings = Settings(chunk_strategy=strategy, chunk_size=120, chunk_overlap=20)
    table = "| a | b |\n| --- | --- |\n" + "\n".join(f"| row{i} | val{i} |" for i in range(80))
    doc = Document("x", "x.txt", "txt", "h" * 64,
                   blocks=[Block(make_samples.PARA * 20, page=1), Block(table, kind="table", page=2)])
    chunks = chunk_document(clean_document(doc, settings), settings)

    assert len(chunks) > 3
    assert all(c.tokens <= 130 for c in chunks)
    table_chunks = [c for c in chunks if c.metadata["has_table"]]
    assert len(table_chunks) > 1
    assert all(c.text.startswith("| a | b |") for c in table_chunks)  # header repeated


def test_chunk_checks_reject_and_flag():
    settings = Settings()
    good = chunk_document(Document("x", "x.txt", "txt", "h", blocks=[Block("useful text " * 10)]),
                          settings)[0]

    def variant(index, text=None, tokens=None):
        c = copy.deepcopy(good)
        c.index = index
        if text is not None:
            c.text, c.tokens = text, tokens if tokens is not None else c.tokens
        return c

    duplicate = variant(1)
    symbols = variant(2, "#$%^&*()_+{}|:<>?~" * 5)
    tiny = variant(3, "ok", tokens=1)
    repetitive = variant(4, "spam " * 100, tokens=100)

    summary = check_chunks([good, duplicate, symbols, tiny, repetitive], settings)
    assert not good.rejected
    assert duplicate.flags == ["reject:duplicate"]
    assert symbols.rejected and tiny.rejected
    assert not repetitive.rejected and "flag:repetitive" in repetitive.flags
    assert summary["rejected"] == 3


# ------------------------------------------------------- fixes from the review
def test_changed_chunk_settings_trigger_reingest(samples, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    shutil.copy(samples / "notes.txt", folder / "notes.txt")
    data = tmp_path / "data"

    first = IngestionPipeline(Settings(data_dir=data, chunk_size=500), fake_embeddings=True)
    assert first.run([folder])["success"] == 1
    assert first.run([folder])["skipped"] == 1  # nothing changed

    smaller = IngestionPipeline(Settings(data_dir=data, chunk_size=200, chunk_overlap=30),
                                fake_embeddings=True)
    assert smaller.run([folder])["success"] == 1  # same file, new settings -> redone


def test_unknown_env_names_are_reported(monkeypatch):
    from ingestion.settings import unknown_env_names

    monkeypatch.setenv("RAG_WORK_DIR", "./data")  # old name
    monkeypatch.setenv("RAG_CHUNK_SIZE", "400")  # valid name
    assert "RAG_WORK_DIR" in unknown_env_names()
    assert "RAG_CHUNK_SIZE" not in unknown_env_names()


def test_file_refused_by_openai_is_quarantined_not_retried_forever(samples, tmp_path, monkeypatch):
    pipeline = new_pipeline(tmp_path)

    def refuse_everything(texts):
        raise ValueError("400 invalid input")

    monkeypatch.setattr(pipeline.embedder, "embed", refuse_everything)
    result = results_by_name(pipeline.run([samples / "notes.txt"]))["notes.txt"]
    assert result["status"] == "quarantined" and result["failed_step"] == "6 embed"


def test_txt_headings_are_detected():
    from ingestion.step2_extract import _mark_txt_headings

    text = ("Company FAQ\n\nHow do I cancel?\nAny time from settings.\n\n"
            "Date: 10 November\nNot a heading because it is a field.\n\n"
            "This is a normal sentence that ends with a full stop.\nMore text.")
    marked = _mark_txt_headings(text).splitlines()
    assert marked[0] == "## Company FAQ"  # first line = title
    assert "## How do I cancel?" in marked
    assert "Date: 10 November" in marked
    assert not any(ln.startswith("## This is a normal") for ln in marked)


def test_pdf_headings_and_tables_stay_in_their_section(tmp_path):
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table

    styles = getSampleStyleSheet()
    pdf = tmp_path / "sections.pdf"
    SimpleDocTemplate(str(pdf)).build([
        Paragraph("Pricing Plans", styles["Heading1"]),
        Paragraph("Each plan includes a number of dashboards and storage. " * 3, styles["BodyText"]),
        Table([["Plan", "Price"], ["Starter", "15 USD"], ["Business", "40 USD"]],
              style=[("GRID", (0, 0), (-1, -1), 0.5, (0, 0, 0))]),
        Paragraph("Customer Support", styles["Heading1"]),
        Paragraph("Support answers within four business hours on weekdays. " * 3, styles["BodyText"]),
    ])
    pipeline = new_pipeline(tmp_path)
    assert results_by_name(pipeline.run([pdf]))["sections.pdf"]["status"] == "success"

    stored = pipeline.store.collection.get(include=["documents", "metadatas"])
    sections = {m["section"] for m in stored["metadatas"] if "section" in m}
    assert {"Pricing Plans", "Customer Support"} <= sections
    table_meta = next(m for m in stored["metadatas"] if m["has_table"])
    assert table_meta["section"] == "Pricing Plans"  # table is under its own heading


@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract is not installed")
def test_docx_image_text_stays_in_its_section(tmp_path):
    import io

    import docx
    from docx.shared import Inches

    word = docx.Document()
    word.add_heading("Expenses", level=1)
    word.add_paragraph("Expense limits approved by the finance committee are shown below.")
    word.add_picture(io.BytesIO(make_samples.png_bytes(make_samples.text_image(
        ["Hotel limit is 220 USD per night", "Meals limit is 60 USD per day"]))), width=Inches(5))
    word.add_heading("Security", level=1)
    word.add_paragraph("Laptops must use full disk encryption and lock after five minutes.")
    path = tmp_path / "inline.docx"
    word.save(str(path))

    pipeline = new_pipeline(tmp_path)
    pipeline.run([path])
    stored = pipeline.store.collection.get(include=["documents", "metadatas"])
    hotel = next(m for d, m in zip(stored["documents"], stored["metadatas"]) if "Hotel limit" in d)
    assert hotel["section"] == "Expenses"


# ------------------------------------------- structure chunking: main sections
def _doc(*paragraphs):
    return Document("x", "x.pdf", "pdf", "h" * 64, blocks=[Block("\n\n".join(paragraphs), page=1)])


def test_main_section_level_ignores_a_single_title():
    from ingestion.step4_chunk import main_section_level

    assert main_section_level(_doc("# Title", "## A", "text", "## B", "text")) == 2
    assert main_section_level(_doc("# Topic 1", "## Idea", "x", "# Topic 2", "## Idea", "y")) == 1
    assert main_section_level(_doc("plain text only")) is None


def test_small_sub_sections_are_packed_inside_their_section():
    settings = Settings(chunk_strategy="structure")
    sub = "This sub-section explains one small idea in a single short paragraph of text."
    doc = _doc("# Topic one", "## Core idea", sub, "## Example", sub + " Example.", "## Remember", sub + " Note.",
               "# Topic two", "## Core idea", sub + " Two.", "## Example", sub + " Two example.")
    chunks = chunk_document(doc, settings)
    assert len(chunks) == 2  # one per main topic, not one per sub-heading
    assert chunks[0].metadata["section"] == "Topic one"
    assert chunks[0].metadata["subsections"] == "Core idea | Example | Remember"
    assert "## Example" in chunks[0].text and "Topic two" not in chunks[0].text


def test_a_chunk_never_ends_with_a_lonely_heading():
    settings = Settings(chunk_strategy="structure", chunk_size=60, chunk_overlap=10)
    long = "Each of these sentences adds some words so that the section grows past the limit."
    doc = _doc("# Topic", "## First", long, long, "## Second", long, long, "## Third", long)
    for chunk in chunk_document(doc, settings):
        last_paragraph = chunk.text.split("\n\n")[-1]
        assert not last_paragraph.startswith("#"), chunk.text
