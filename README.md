# RAG Ingestion Pipeline

Turns PDF, TXT and DOCX files into clean, checked chunks, embeds them with OpenAI
(`text-embedding-3-small`, 1536 dimensions) and stores them in ChromaDB.

```
 docs/*.pdf, *.docx, *.txt
        │
  1  check file      is it readable, the right type, not empty, not corrupt?
  2  extract         text, tables (→ Markdown), OCR for images and scanned pages
  3  clean           remove headers/footers, page numbers, OCR garbage, duplicates
  4  chunk           split into token-sized chunks with overlap
  5  check chunks    reject bad chunks, flag weak ones
  6  embed           OpenAI → 1536 numbers per chunk
  7  store           ChromaDB (data/chroma)
```

![Ingestion flow](ingestion-flow.svg)

The full flow, including every skip, quarantine and retry path, is in
[`ingestion-flow.svg`](ingestion-flow.svg).

## Quick start

```bash
brew install tesseract                                   # OCR engine
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                                     # then put your OPENAI_API_KEY in it

.venv/bin/python ingest.py                               # ingest everything in docs/
.venv/bin/python serve.py                                # web app: http://127.0.0.1:8000
```

Or from the terminal: `python ask.py "What chunk size does ingestion use?"`

No API key yet? Test the whole pipeline offline (the vectors are fake, so search
results are meaningless, but every step runs):

```bash
python ingest.py --fake-embeddings --collection test
```

## Project layout

```
ingest.py                   run the ingestion pipeline   (python ingest.py --help)
ask.py                      ask a question in the terminal (python ask.py --help)
serve.py                    start the web app (FastAPI + web page)
evaluate.py                 measure retrieval and answer quality
search.py                   plain vector search, no LLM (quick check)

ingestion/
  settings.py               every setting in one place (override in .env)
  models.py                 Block, Document, Chunk, FileResult
  errors.py                 one error type per step

  step1_check_file.py       STEP 1  file checks + fingerprint
  step2_extract.py          STEP 2  PDF / DOCX / TXT extraction
  ocr.py                            Tesseract helper used by step 2
  step3_clean.py            STEP 3  cleaning and noise removal
  step4_chunk.py            STEP 4  chunking strategies
  step5_check_chunks.py     STEP 5  reject / flag rules
  step6_embed.py            STEP 6  OpenAI embeddings with retries
  step7_store.py            STEP 7  ChromaDB storage

  pipeline.py               runs steps 1-7 for each file + failure handling
  tracking.py               manifest, quarantine, reports

retrieval/                  question -> hybrid search -> top 3 -> cited answer
api/main.py                 FastAPI backend (/api/ask, /api/search, /api/files, /api/health)
api/auth.py                 GitHub sign-in + roles (viewer, editor, admin)
web/                        frontend: index.html, style.css, app.js (no build step)
evaluation/  eval/          metrics and the test question set

docs/                       the knowledge base: 3 RAG-concept documents
                            (scripts/make_rag_docs.py recreates them)
scripts/make_docs.py        sample company documents, used by the tests
tests/                      pytest suite (no API key needed)
data/                       created on first run: chroma/, reports/, quarantine/, ingest.log
```

To understand the code, read `ingestion/pipeline.py` → `ingest_file()` first.
It calls the seven steps in order, and each step lives in its own file.

## What each step does

**1. Check file**: the file exists and is readable; its type is `.pdf`, `.docx` or `.txt`;
it isn't empty or over 100 MB; its content matches its extension (catches renamed or
corrupt files and old `.doc` files); Office `~$` lock files are skipped. Then it records a SHA-256
fingerprint so unchanged files are skipped next time.

**2. Extract**:
- **PDF**: text, tables found by PyMuPDF and turned into Markdown, OCR of embedded images
  (each image only once, icons skipped), and OCR of whole scanned pages (rendered at 300 DPI).
  Password-protected PDFs are refused.
- **DOCX**: paragraphs in order, headings become `#`, lists become `-`, tables become Markdown,
  and embedded images go through OCR.
- **TXT**: the encoding is detected automatically (UTF-8, UTF-16, Windows cp1252).

Content is added in reading order, so a table or image stays under the heading it belongs to.
Headings are marked with `#` for structure chunking: Word heading styles, PDF lines in a
larger font than the body text (or short all-bold lines), and in TXT the first line or a short
standalone line such as a question. OCR tries several Tesseract layouts and discards
low-confidence words and images.

**3. Clean**: fixes ligatures (`ﬁ` → `fi`), invisible characters and curly quotes. Removes
page numbers, `.....` leaders, symbol-only lines, OCR garbage, and headers and footers
(short lines repeated at the top or bottom of most pages). Rejoins `re-\nport` → `report` and
joins hard-wrapped lines. Removes duplicate blocks. Rejects the document if almost nothing is left.

**4. Chunk**: four strategies, all counted in tokens with overlap:

| Strategy | How it splits | Use when |
|----------|---------------|----------|
| `structure` (default) | keeps heading sections and tables separate; recursively splits oversized sections by paragraph → line → sentence → word | structured documents |
| `recursive` | paragraph → line → sentence → word | general text without section boundaries |
| `sentence` | whole sentences only | Q&A over prose |
| `fixed` | fixed token windows | very large, unstructured text |

Each table gets its own chunk, and big tables are split by rows with the header repeated.
Tiny leftovers are merged into the previous chunk. Every chunk stores its file, pages,
section heading, content types and OCR confidence.

**5. Check chunks**:
- **Rejected** (not stored): empty, too short, too long for OpenAI, no words, mostly symbols, duplicate.
- **Flagged** (stored, noted in `quality_flags`): low OCR confidence, repetitive, short, oversized,
  mostly numbers, many symbols, starts mid-sentence.

**6. Embed**: sends chunks in batches (by count and by tokens), with `dimensions=1536`.
Every vector is checked.

**7. Store**: ChromaDB with cosine similarity. A collection refuses vectors from a different
model. When a file changes, its new chunks are written before the old ones are deleted.

## Failure handling

| What went wrong | What happens |
|-----------------|--------------|
| The file is bad (corrupt, empty, unsupported, nothing readable) | **Quarantined**: copied to `data/quarantine/` with `<name>.error.json` saying which step failed and why. Your original stays in place. |
| OpenAI rate limit, timeout or outage | Retried with growing waits. If it still fails, the file is marked **failed** and retried automatically next run. |
| OpenAI rejects one chunk | Only that chunk is dropped; the rest of the file is stored. |
| OpenAI rejects every chunk of a file | **Quarantined**: the content is the problem, so it isn't retried forever. |
| Missing or wrong API key | Stops immediately with a clear message (exit code 2). |
| ChromaDB write error | Retried 3 times, then **failed**. The previous version of the file stays searchable. |
| One bad page, image or table | Shown as a warning; the rest of the file is ingested. |
| A bug in the code | That file is **failed**, the traceback goes to `data/ingest.log`, and the run continues. |
| The run crashes | `data/manifest.json` is saved after every file, so the next run picks up where it stopped. |

After each run you get:
- `data/reports/run-<id>.json`: what happened to each file
- `data/reports/chunks-<id>.jsonl`: rejected and flagged chunks to review
- `data/ingest.log`: the full log

The exit code is 1 if any file failed, so cron jobs and CI notice.

## Web app (FastAPI + HTML/CSS/JS)

```bash
python serve.py            # open http://127.0.0.1:8000
python serve.py --reload   # restart automatically while editing code
```

Type a question (or click an example) and press **Ask** or `⌘ ↵`. You get:

- the **answer** with citation chips: click `[1]` to jump to that source
- the **sources** that were sent to the model: file, section, page, similarity, cited or not,
  and the exact text the model saw
- **search details**: every chunk the hybrid search found with its cosine similarity, BM25
  keyword score and ranks, and whether it passed the relevance check
- **filters** on the left: document, file type, section, PDF page, tables only, skip flagged
  chunks, and how many sources to send to the model (1-6)
- a **history** of this session's questions

"No information" answers are shown in purple. If the model fails, the page shows the passages
the search found instead. The server only listens on this computer (127.0.0.1).

### Set up GitHub sign-in

The web app requires signing in with GitHub, and only usernames you list get in.

1. **Create a GitHub OAuth app.** On GitHub: *Settings → Developer settings → OAuth Apps →
   New OAuth App*:
   - Application name: `RAG Knowledge Assistant`
   - Homepage URL: `http://127.0.0.1:8000`
   - Authorization callback URL: `http://127.0.0.1:8000/auth/callback`

   Click *Register application*, then *Generate a new client secret*.
2. **Make a session secret** (signs the login cookie):
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
3. **Add to `.env`:**
   ```
   RAG_GITHUB_CLIENT_ID=<Client ID from step 1>
   RAG_GITHUB_CLIENT_SECRET=<client secret from step 1>
   RAG_SESSION_SECRET=<output of step 2>
   RAG_ADMINS=your-github-username
   ```
4. Restart `python serve.py`, open **http://127.0.0.1:8000** (use `127.0.0.1`, not
   `localhost`) and click *Sign in with GitHub*.

**Roles**, by GitHub username (comma separated, case doesn't matter):

| Role | `.env` list | Can |
|------|-------------|-----|
| viewer | `RAG_VIEWERS` | ask questions, search, see the file list |
| editor | `RAG_EDITORS` | viewer + upload/replace documents (coming next) |
| admin | `RAG_ADMINS` | editor + delete, re-ingest, logs and reports (coming next) |

Anyone not listed is refused (`RAG_DEFAULT_ROLE=none`); set `RAG_DEFAULT_ROLE=viewer` to let any
GitHub user ask questions. Roles are checked on every request, so removing a name takes effect
at the next server restart. Sessions last 8 hours (`RAG_SESSION_HOURS`).

**How it is kept safe:** a random `state` value blocks forged logins; only the public profile is
requested (`read:user` scope); the GitHub token is used once and never stored; the session
cookie is signed and HttpOnly; every sign-in, refusal and question is logged with the GitHub
username in `data/reports/access.jsonl`. With `RAG_AUTH=off` there is no sign-in at all - use
that only on your own computer.

**Backend endpoints** (try them at `http://127.0.0.1:8000/docs`; all need a signed-in user):

| Method | Path | Does |
|--------|------|------|
| GET | `/api/me` | the signed-in user and their role |
| GET | `/auth/login` · `/auth/callback` · POST `/auth/logout` | GitHub sign-in and sign-out |
| GET | `/api/health` | ready? number of chunks, models in use |
| GET | `/api/files` | ingested files with their sections (fills the filter menus) |
| POST | `/api/ask` | `{"question", "filters", "top_k"}` → answer, citations, sources, search scores |
| POST | `/api/search` | same input → ranked chunks only, no LLM call |

Errors: `401` not signed in, `403` not allowed (role too low or removed), `400` bad question,
`422` invalid request, `503` not ready (sign-in not configured, empty database or missing key;
the server recovers by itself after `python ingest.py`), `502` the LLM failed (the
response still contains the passages found). New chunks from `ingest.py` are picked up without
restarting the server.

## Asking questions (retrieval + generation)

![Retrieval flow](retrieval-flow.svg)

The full retrieval flow, including the no-information and error paths, is in
[`retrieval-flow.svg`](retrieval-flow.svg).

```bash
.venv/bin/python ask.py "What chunk size and overlap does ingestion use?"
.venv/bin/python ask.py                       # interactive: several questions in a row
```

```
 question
    │
  1  prepare question   vector (same embedding model as ingestion) + keywords
  2  hybrid search      semantic search (cosine, ChromaDB) + keyword search (BM25)
                        on the chunks that pass the metadata filters,
                        merged with Reciprocal Rank Fusion
  3  select sources     relevant = similarity ≥ 0.30 OR keyword score ≥ 4.0
                        AND similarity ≥ 0.6 × the best similarity
                            (strong keyword matches are exempt)
                        keep the top 3          nothing relevant? ──► "no information"
  4  generate answer    LLM answers ONLY from those 3 sources, cites them [1][2][3]
                        sources don't answer it? ──────────────────► "no information"
                        citations checked: numbers that don't exist are removed
```

Example output:

```
Answer:
  Ingestion uses chunks of 500 tokens with an overlap of 75 tokens [1].

Sources:
  [1] rag_ingestion_guide.pdf · section "5. Chunking strategies" · page 2   (cited)
```

When nothing in the documents is related, the answer is
"I couldn't find any information related to this topic in the documents."

**Metadata filters** (combine as many as you like):

| Option | Keeps only chunks… |
|--------|--------------------|
| `--file rag_ingestion_guide.pdf` | from that file |
| `--type pdf` | of that file type (`pdf`, `docx`, `txt`) |
| `--section expenses` | whose section heading contains the text |
| `--page 2` | from that PDF page |
| `--tables-only` | that contain a table |
| `--exclude-flagged` | without quality flags (e.g. low OCR confidence) |

Other options: `--top-k 5`, `--show-search` (every hit with its cosine and keyword
score), `--json`. Every question and answer is logged to `data/reports/questions.jsonl`.

The code is in `retrieval/`, one file per step like ingestion:
`step1_prepare_question.py`, `step2_hybrid_search.py`, `step3_select_sources.py`,
`step4_generate_answer.py`, tied together by `retrieval/pipeline.py`.

**Failure handling:** an empty database, an empty question or a missing API key gives a clear
message. OpenAI rate limits and timeouts are retried. If the LLM still fails or replies with
something unreadable, `ask.py` shows the most relevant passages instead of crashing.

**Relative cut-off:** a source must also score at least 60% of the best match for that question.
With a best match of 0.66 the limit is 0.40, so a weak 0.30 source is dropped even though it
passed the fixed 0.30 cut-off. For a vague question where everything scores lower, the limit
drops with it. Strong keyword matches (exact codes, numbers, names) are exempt.

**Tuning:** if real answers are being refused, lower `RAG_MIN_SIMILARITY` (try 0.25); if unrelated
chunks get through, raise it. `RAG_MIN_RELATIVE_SCORE` controls the relative cut-off (higher =
stricter, `0` turns it off). `ask.py --show-search` shows the scores to decide.

## Evaluating quality (retrieval + generation)

```bash
.venv/bin/python evaluate.py                    # retrieval + generation
.venv/bin/python evaluate.py --retrieval-only   # search quality only, no LLM calls
.venv/bin/python evaluate.py --tag ocr_image    # one kind of question
```

It runs every question in `eval/dataset.jsonl` (26 questions about `docs/`: 21 answerable,
5 the documents can't answer) through the real pipeline and scores it:

| Step | Metric | What it tells you |
|------|--------|-------------------|
| Retrieval | Hit rate@3 | was the chunk with the answer among the 3 sources sent to the LLM? |
| | MRR | how high it ranked (1st = 1.0, 2nd = 0.5, 3rd = 0.33) |
| | Precision@3 | share of the sent sources that are actually correct |
| | Context recall | share of the key facts present in the sources |
| | Miss diagnosis | `not_found_by_search` / `dropped_by_relevance` / `ranked_below_top_k` |
| Generation | Fact recall | key facts (e.g. "25", "31 March") present in the answer |
| | Citation valid / precision | answer cites real sources, and the right ones |
| | False refusal rate | "no information" given for a question that has an answer |
| | Refusal accuracy | "no information" given for a question that has none |

Each run writes `data/reports/eval-<run>.md` (readable report: scores, change since the previous
run, scores per question type, and every failing question with its answer and sources) and a
`.json` with all the data. The run fails (exit code 1) when a score is below its threshold;
thresholds are at the top of `evaluation/runner.py`.

**Adding questions:** append a line to `eval/dataset.jsonl`; the format is described at the top
of `evaluation/dataset.py`. Add real questions your users ask, especially ones that went wrong.

**Use it to tune:** change one setting (`RAG_MIN_SIMILARITY`, `RAG_TOP_K`, chunk size, ...),
run `evaluate.py` again and compare with the "vs previous" column.

## Common commands

```bash
.venv/bin/python ingest.py                         # ingest docs/
.venv/bin/python ingest.py reports/ contract.pdf   # specific folders or files
.venv/bin/python ingest.py --force                 # re-ingest even unchanged files
.venv/bin/python ingest.py --strategy structure --chunk-size 400 --chunk-overlap 60
.venv/bin/python ingest.py --strategy sentence --chunk-size 400 --chunk-overlap 60
.venv/bin/python ingest.py --no-ocr
.venv/bin/python ingest.py --remove-deleted        # drop vectors of files deleted from disk
.venv/bin/python search.py "pricing plans" -k 3     # plain vector search, no LLM
.venv/bin/python ask.py "pricing plans" --type pdf  # cited answer
.venv/bin/python evaluate.py                      # measure retrieval and answer quality
.venv/bin/python -m pytest tests -q                # run the tests
```

Settings such as chunk size, OCR language and the collection name can also be set in `.env`
(see `.env.example`). A `RAG_*` name that isn't recognised (a typo or an old name) is reported
as a warning at start-up.

A file is skipped only when the file **and** the settings that shape its chunks (strategy, chunk
size, overlap, OCR on/off, vector size) are unchanged since its last successful run. Change
any of them and the next run re-ingests the affected files automatically.

## The knowledge base in `docs/`

Three documents about RAG itself, one per stage, each in a different format so every
ingestion path is used:

| File | Content | What it exercises |
|------|---------|-------------------|
| `rag_ingestion_guide.pdf` | file checks, extraction, OCR, cleaning, chunking strategies, embeddings, failure handling | 4 pages, 2 tables, a chart image (OCR), headers/footers to strip, a scanned checklist page (full-page OCR) |
| `rag_retrieval_notes.txt` | query embedding, semantic and BM25 search, RRF, metadata filters, thresholds, top-k, re-ranking, common failures | plain text with headings and lists |
| `rag_generation_guide.docx` | grounded prompting, citations, "no information" behaviour, model settings, quality metrics, prompt injection | headings, numbered list, a table, an example image (OCR) |

Recreate them with `.venv/bin/python scripts/make_rag_docs.py docs`. Any PDF, DOCX or TXT you
drop into `docs/` is picked up by the next `python ingest.py`.
