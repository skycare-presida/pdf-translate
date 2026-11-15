# SkyCare PDF Medical Translation Service

An async Flask microservice that translates medical PDFs for repatriation
missions. It is called from Presida (the FileMaker-based case management
system) when a case requires a foreign-language discharge summary, imaging
report or similar to be made readable for the receiving clinical team.

All inference runs on EU-hosted endpoints (OVHcloud AI Endpoints: Mistral +
Llama) so that patient data never leaves the European Economic Area, keeping
the pipeline inside the organisation's GDPR posture. No document ever touches
US-hosted model APIs.

The service is deliberately opinionated about what it will and will not do.
Several design choices below exist because a wrong medical translation is
worse than a failed one — every fidelity guard is there to prevent plausible-
looking output silently replacing real clinical content.

## Pipeline

A document arrives and is auto-classified into one of three modes:

**Mode A — text-layer PDFs.** The born-digital case: a hospital information
system has produced a PDF that contains an actual text layer. The service
extracts that text with PyMuPDF (preserving the source's own line breaks and
blank lines), translates it in large chunks with an explicit instruction to
keep line and list structure, renders a clean English PDF, and appends the
untouched original pages behind it. There is no in-place block replacement —
earlier versions tried to whiteout and rewrite the original pages block by
block, and the complexity was unsustainable for watermarked, multi-column,
or list-dense documents. The current design treats "translate" and "preserve
layout" as separate problems: the translation is a cleanly-set new document,
the original is the reference appendix.

**Mode B — clean scans.** No usable text layer, but the scan itself is clean
enough to read. The service rasterises each page, sends them to Mistral for
OCR with a structured language-assessment prompt (it returns both the text
and a confidence/language JSON), then translates with the appropriate model.
Translated text is laid out as a new section followed by the original page
images as an appendix.

**Mode C — phone photos and worse scans.** Same pipeline as B, with rotation
correction and a lower threshold for escalating to the stronger translation
model. In practice the distinction between B and C is modest — they share
most of the code path — but C's pre-handling catches phone-camera-specific
problems B doesn't need to worry about.

## Model routing

Two models, selected per document:

- **Mistral-Small-3.2-24B-Instruct-2506** (multimodal): does OCR for scans
  and photos, and handles translation for languages it is strong on.
- **Meta-Llama-3.3-70B-Instruct** (text only): handles translation for
  languages where the 24B model has demonstrated real failures.

The escalation list was built empirically from observed problems and now
covers Arabic, Farsi/Persian, Hebrew, Pashto, Urdu, Thai, Bengali, Tamil,
Telugu, Kannada, Malayalam, Sinhala, Burmese, Khmer, Lao, Tibetan, Mongolian,
Amharic, Georgian, Armenian, Turkish, and Azerbaijani. Turkish was added
after Mistral rendered "Taburculuk Tanıları" (discharge diagnoses) as
"Tuberculosis Diagnoses" — a phonetic guess that invents a disease, exactly
the kind of failure this escalation list exists to prevent. Languages not
on the list are escalated when Mistral's own confidence drops below 0.75,
or by explicit `model=llama` override.

## Safety guards

These are the parts of the service that exist because specific real failures
happened once.

**Truncation is fatal.** Every model response is checked for
`finish_reason=length`. If the output overflowed the token cap, the job
fails rather than returning a silently-truncated medical translation. A
partial translation that looks complete and carries an attestation page is
worse than a failed job — the operator gets an error, retries or splits
the document, and nothing fabricated reaches the clinician.

**Poor-scan fast-fail.** Vision models presented with illegible scans don't
fail cleanly — they hallucinate plausible-looking clinical text at length.
One real case produced 129,503 characters from a creased two-page form that
actually held ~3,000 characters of real content. The service guards against
this with a per-page character ceiling (default 8,000), a hard OCR call
timeout (default 5 minutes, below the HTTP client's silent 10-minute
default), and a fast-fail path that returns an `UnreadableScan` with an
explanation rather than pretending to have read the page.

**Cyrillic Tesseract fallback.** When Mistral over-produces on a scan it
has identified as Cyrillic script, the service retries the OCR step
locally with Tesseract (`ukr+rus+bul+srp+mkd+bel+kaz+eng`) before giving
up. Tesseract returns what is actually on the page rather than fabricating,
and on printed Cyrillic — which Mistral-24B handles poorly — it is both
more accurate and immune to the hallucination failure mode. Runs entirely
on-box, so no GDPR consideration.

**OpenCV scan preprocessing.** Per page, applies deskew + CLAHE contrast
equalisation + light denoise, measures the confident-word yield before and
after, and keeps the enhanced image **only if it improves yield**. A clean
page keeps its raw pixels untouched. The adaptive decision matters: in
testing, blanket preprocessing that rescued a creased page by taking its
OCR yield from 0 to 30 words also degraded a clean page from 193 words to
125, because contrast equalisation blurs detail that was already sharp.
A fixed pipeline would have made working documents worse.

**QA verification pass.** An optional second-model review of the translated
draft against the source for clinically material discrepancies — mistranslated
diagnoses or section headings, altered drug/dose/route/frequency, laboratory
values or units that differ, reversed negations, content present in the
source but absent from the translation or invented in it. Any findings are
printed on the attestation page. Exempt by default for French, German,
Spanish, Italian and English (reliably handled, the pass buys nothing); on
by default for everything else; overridable per request with `qa=on|off|auto`.

**Error PDF.** Every failure writes a one-page PDF explaining what happened
and what the operator can do about it (typically: get a clearer scan, or
arrange manual translation). The job status still returns `failed` via the
status endpoint — so Presida's existing `Exit Loop If failed` branch is
untouched — but the result endpoint serves the error PDF rather than a bare
HTTP 500, so the result field contains a readable document rather than
nothing.

## API

Called asynchronously because translation of a long scan can exceed
FileMaker's HTTP timeout:

```
POST /translate/submit        → { job_id }
GET  /translate/status/<id>   → { status: queued|processing|complete|failed, ... }
GET  /translate/result/<id>   → PDF (success) or error PDF (failure)
```

A synchronous `POST /translate` endpoint also exists for testing.

Form fields accepted by `/submit`:
- `file`           — the PDF or image to translate (required)
- `target_lang`    — defaults to `English`
- `mode`           — `A`, `B`, `C`, or `AUTO` (default)
- `model`          — `mistral`, `llama`, or empty for auto-selection
- `qa`             — `auto` (default), `on`, `off`
- `source_hint`    — free text describing the document type (optional)
- `jobID` / `job_id` — client-supplied ID (Presida sends `Get(UUID)`)

Authentication: `Authorization: Bearer <TRANSLATE_API_KEY>`.

## Repository layout

```
├── README.md                       — this file
├── install.sh                      — idempotent installer (run as root on Ancilla)
├── .env.example                    — template for the two API keys
├── service/
│   ├── pdf_translate_v3_service.py — the service itself (~2700 lines)
│   └── requirements.txt            — Python dependencies
└── scripts/
    └── skycare-pdf-translate-v3.service  — systemd unit
```

## Deployment

Installed at `/opt/skycare/pdf-translate-v3/` on Ancilla (Ubuntu 24.04),
running as `www-data` under systemd as `skycare-pdf-translate-v3`, listening
on port 5107. Fronted by nginx with a UFW rule limiting source to the
Presida server's IP.

### Fresh install

Clone and run the installer as root:

```bash
git clone https://github.com/skycare-presida/pdf-translate.git
cd pdf-translate
sudo ./install.sh
```

The installer is idempotent and safe to re-run. It:

- apt-installs Tesseract and the Cyrillic language packs
- checks port 5107 is free (prompts for an alternative if not)
- reuses the v1 or v2 venv by symlink if present (saves ~400 MB and ~2 min)
- inherits API keys from an existing `/opt/skycare/pdf-translate*/env` file
  if it exists, otherwise prompts for the OVHcloud key and generates a new
  inbound `TRANSLATE_API_KEY`
- creates `/opt/skycare/pdf-translate-v3/`, `/var/log/skycare/`, and
  `/tmp/skycare-jobs/` with the right ownership and modes
- deploys the service file, `.env` (mode 600), systemd unit, and starts it
- adds a UFW rule limiting port 5107 to the Presida IP
- verifies the OpenCV / numpy / pytesseract imports actually succeed —
  these degrade silently if missing, which bit us in Nov 2026 when
  preprocessing was packaged but the venv had no cv2
- runs a health check and prints FileMaker-ready curl examples with the
  real API key

### Updating the service file only

The common case during iteration — swap the Python file, keep everything
else untouched. Deployment on Ancilla:

```bash
sudo cp service/pdf_translate_v3_service.py /opt/skycare/pdf-translate-v3/
sudo chown www-data:www-data /opt/skycare/pdf-translate-v3/pdf_translate_v3_service.py
sudo systemctl restart skycare-pdf-translate-v3
```

### Ancilla systemd gotchas

These are the specific configuration points that have caused silent
failures on this box, now encoded in `scripts/skycare-pdf-translate-v3.service`:

- **www-data home is `/var/www` and unwritable** — Gunicorn needs
  `--worker-tmp-dir /tmp/skycare-jobs` or it exits 255.
- **`ProtectSystem=strict`** requires `/tmp` explicitly in `ReadWritePaths`
  or Flask uploads fail with "No usable temporary directory".
- **`PrivateTmp=false`** — the SQLite job store at `/tmp/skycare-jobs/jobs.db`
  must be visible across both Gunicorn workers; `PrivateTmp=true` gives each
  its own `/tmp` and jobs silently disappear.

## History

The service has gone through several substantial redesigns as failure modes
surfaced. The commit log is the authoritative record; a brief timeline of
the larger shifts:

- **v1** (single-model, in-place block replacement) — lost lists, dropped
  blocks, destroyed watermark bboxes. Retired.
- **v2** (two-stage adaptive Mistral/Llama) — added language escalation.
- **v3** (async job queue) — current. Added SQLite job store, FileMaker-
  friendly submit/poll/fetch endpoints, truncation guards.
- **Simplified Mode A** (Sept 2026) — abandoned block-by-block whiteout in
  favour of text-layer extraction → clean English PDF → original appendix.
  Deleted ~370 lines.
- **Fast-fail scan handling + Cyrillic Tesseract + error PDF** (Nov 2026).
- **Adaptive OpenCV preprocessing** (Nov 2026) — per-page, measured, kept
  only when it improves yield.

## Related services

Part of a small suite of Flask microservices on Ancilla; see the
infrastructure notes for the full inventory. Not open-sourced: a medical
service with documented EU inference endpoints and specific attack surfaces.
