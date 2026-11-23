# SkyCare PDF Medical Translation Service

An async Flask microservice that translates medical PDFs for repatriation
missions. It is called from Presida (the FileMaker-based case management
system) when a case requires a foreign-language discharge summary, imaging
report or similar to be made readable for the receiving clinical team.

Translation is done by DeepL (API Pro), with the OVHcloud models as an
automatic fallback. OCR, language assessment and the QA pass run on EU-hosted
OVHcloud AI Endpoints (Mistral + Llama). DeepL SE is a German company; with a
Pro key the text is processed for the translation only and not retained —
confirm the data-processing terms on the DeepL account before relying on
that. Do not use a DeepL API Free key (`...:fx`) for patient documents.

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

## Translation engine

**DeepL first.** Once the text has been extracted (text layer or OCR), it is
translated by DeepL. The request keeps the source's line structure (hard-
wrapped prose is re-joined first so sentences are translated whole; lists,
headings and table rows stay one per line), is split into 10,000-character
requests to stay inside DeepL's 128 KiB limit, and relies on the SDK's
backoff for rate limits (HTTP 429) and transient errors.

**Text PDFs keep their original layout.** For a PDF with a real text layer
(Mode A) the service first asks DeepL to translate the PDF itself, which is
what the DeepL website does: the result is a PDF with the original fonts,
tables and logos, in the target language. The service adds a cover page
(provenance, a one-line DeepL statement, QA result) in front and the untouched original
pages behind. The translated text is read back out of DeepL's PDF so the
untranslated-output guard and the QA pass still check it against the source.
If whole-document translation fails, or loses most of the text, the job
drops to translating the extracted text instead (below). DeepL bills a PDF
at a minimum of 50,000 characters; `DEEPL_DOCUMENT_TRANSLATION=off` turns
this off and `DEEPL_DOCUMENT_MAX_PAGES=N` limits it to shorter documents.
The upload is named `document.pdf`, so the patient's file name is not sent.
Scans and photos (Modes B/C) cannot use this — DeepL does not OCR.

**Text translation keeps lines and paragraphs.** Where the text is translated
rather than the file, paragraph breaks are taken from the vertical gaps on
the source page and every line of DeepL's output is rendered as its own line,
so a well-laid-out document does not come back as one block of text.

**Fallback to the OVHcloud models.** If DeepL is not configured, errors
(outage, bad key, quota exhausted), does not support the source or target
language, returns nothing, or returns text that still looks like the source,
the job carries on with the original Mistral/Llama path described below —
it does not fail. The reason is logged and printed on the PDF's provenance
line ("DeepL not used: ..."), and after an outage/auth/quota error DeepL is
skipped for 10 minutes so a long document does not wait on every batch.

`model=mistral|llama` on a request skips DeepL for that job, and
`TRANSLATE_ENGINE=llm` in `.env` switches it off entirely. `/health` reports
which engine is active.

What DeepL does not do: it cannot follow the fidelity prompt, so the
translator's brace notes (`{sic}`, INN expansions, kept-original
institution names) only appear on fallback translations. `[ILLEGIBLE]`
markers from OCR are carried through and counted in the note on scanned documents. The QA
pass is unchanged and now checks DeepL's output with an independent model.

## Model routing

These rules pick the fallback translation model (and Mistral always does OCR).

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

**Local OCR fallback for any runaway page.** If the vision model runs to its
output cap or times out on a single page — it is rambling, not transcribing —
that page is read locally with Tesseract instead of failing the document
(script detected per page; Latin, Cyrillic, Greek, Arabic and Hebrew packs
are installed). A page on which Tesseract finds almost no words is taken to
be a photo, image or blank page: the translation carries a visible "no
machine-readable text — see the original" marker for it and the cover line
says so in capitals. The job still fails as `UnreadableScan` if no page in
the document is readable or if Tesseract also over-produces. OCR calls are
budgeted at 6,000 output tokens per page, so a runaway page is caught in
about a third of the time it used to take. The cover line names any pages
read by local OCR.

**A stronger model reads scans first.** Scans and photographs are read by
`Qwen3.5-397B-A17B`, a larger vision model on OVHcloud AI Endpoints, because
Mistral-Small gives up on poor images a person can still read. If that model
errors, times out, loops or returns nothing, the same pages go to
Mistral-Small as before, and then to local OCR. `OCR_MODEL` in `.env` picks
another OVHcloud vision model; `OCR_MODEL=Mistral-Small-3.2-24B-Instruct-2506`
restores the previous behaviour. The cover line and `/health` name the model
that did the reading.

**Runaway OCR is abandoned as it happens.** The OCR response is streamed and
checked every 1,500 characters. If the model starts looping on one line, or
produces more text than the pages can hold, the call is dropped at once and
the batch is split or the page is read locally — instead of waiting 2-5
minutes for the model to reach its output cap. `OCR_STREAMING=off` in `.env`
restores the single blocking call.

**Lists stay lists.** The OCR prompt asks for every bullet, numbered item and
table row on its own line. Before DeepL, wrapped lines are re-joined only
when the next line plainly continues the sentence (starts lower-case, or
follows a comma or hyphen), so list items — which start with a capital, a
bullet or a number — are never merged. Scan translations are rendered line
for line.

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
├── .env.example                    — template for the API keys
├── service/
│   ├── pdf_translate_v3_service.py — the service itself (~2700 lines)
│   └── requirements.txt            — Python dependencies
├── scripts/
│   └── skycare-pdf-translate-v3.service  — systemd unit
└── docs/
    ├── history.md                  — why the service looks the way it does
    └── filemaker-integration.md    — calling the service from Presida
tests/
└── test_deepl_engine.py            — DeepL engine + fallback tests (pytest, no network)
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
  inbound `TRANSLATE_API_KEY`; inherits `DEEPL_AUTH_KEY` the same way or
  prompts for it (Enter skips — the service then uses the OVHcloud models)
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

If `requirements.txt` changed (e.g. the `deepl` package), also run
`sudo -u www-data /opt/skycare/pdf-translate-v3/venv/bin/pip install -r service/requirements.txt`
before the restart. Without the package the service still runs, on the
fallback models.

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

Several substantial redesigns in response to specific real failures.
[`docs/history.md`](docs/history.md) has the full account of each pivot
and the reasoning — worth reading before changing any of the safety
guards, because most of them exist to prevent a specific class of silent
failure that cost real debugging time to find.

## FileMaker integration

The service is called from Presida with a submit/poll/fetch loop.
[`docs/filemaker-integration.md`](docs/filemaker-integration.md) has the
script pattern, form fields, the cURL container-variable quirk, and the
`jobID` / `job_id` dual-field convention.

## Related services

Part of a small suite of Flask microservices on Ancilla; see the
infrastructure notes for the full inventory. Not open-sourced: a medical
service with documented EU inference endpoints and specific attack surfaces.
