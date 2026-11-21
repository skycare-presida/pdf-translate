# Design history

The service has gone through several substantial redesigns as failure modes
surfaced. Each pivot was in response to a specific class of real failure,
not a planned refactor — this document captures the *why* so future-you (or
a successor) doesn't repeat the dead ends. The commit log holds the *what*.

## v1 → v2: adaptive two-stage model routing

**v1** used a single model (Mistral-Small-3.2-24B-Instruct) for both OCR
and translation. It worked acceptably for Latin-script European languages
and failed quietly for everything else — not by crashing, but by producing
confident-looking mistranslations.

**v2** split the pipeline into two stages. Stage 1 asks Mistral to OCR and
return a JSON self-assessment of language and confidence alongside the
text. Stage 2 picks the translation model based on that assessment:
Mistral if confidence is high and the language isn't on the always-escalate
list; Llama-3.3-70B otherwise. The escalate list was built empirically as
documents surfaced — starting with Arabic, Farsi, Hebrew and Thai; later
adding Turkish after Mistral rendered *Taburculuk Tanıları* (discharge
diagnoses) as "Tuberculosis Diagnoses" on a real Acıbadem document.
Current list: Arabic, Farsi/Persian, Hebrew, Pashto, Urdu, Thai, Bengali,
Tamil, Telugu, Kannada, Malayalam, Sinhala, Burmese, Khmer, Lao, Tibetan,
Mongolian, Amharic, Georgian, Armenian, Turkish, Azerbaijani.

## v2 → v3: async job queue

**v2** was synchronous: `POST /translate` returned the translated PDF in
the same HTTP response. On documents longer than a few pages this reliably
exceeded FileMaker's HTTP timeout (~5 minutes via cURL), killing the
request even though the translation was still running on the server. The
job completed, the result went nowhere.

**v3** introduced a submit/poll/fetch pattern with a SQLite-backed job
store at `/tmp/skycare-jobs/jobs.db`. FileMaker now submits, gets a job
ID, polls the status endpoint on a 15-second interval, and fetches the
PDF when `complete`. The HTTP calls are all short; the long work runs
on a thread pool that the worker reaps independently. No FileMaker
timeout can kill a translation. The old synchronous `POST /translate`
remains as a testing convenience.

One gotcha discovered during this redesign: the job store has to be
SQLite in a shared `/tmp`, not an in-memory dict. Gunicorn's two workers
would otherwise each have their own dict, and whichever worker didn't
create a job would return 404 on the status call ~50% of the time.
This is why the systemd unit has `PrivateTmp=false` — `PrivateTmp=true`
gives each worker its own `/tmp` and reintroduces the same bug.

## Mode A: in-place block replacement → text-layer extraction

**Original Mode A** tried to preserve the source document's visual layout
by extracting each text block with PyMuPDF, translating it in isolation,
then writing the translation back into the same bounding box with the
original text whited out. It worked for simple documents and failed in a
dozen interesting ways on real ones:

- **Collapsed lists.** PyMuPDF groups a run of lines into one block. A
  60-line drug chart arrived as a single block, got translated as one
  unit, and the layout engine reflowed it into a justified paragraph
  with the line breaks gone — losing which dose attached to which drug.
- **Dropped blocks.** Translation used a numbered-list protocol
  (`1|text`, `2|text`…) asking the model to return the translations in
  the same shape. On long lists the model silently dropped trailing
  indices — one real case failed on block 25 of 25, consistently.
- **Watermark destruction.** A diagonal "ONAYSIZ" (UNAPPROVED) watermark
  on a Turkish document had each character as a separate rotated text
  object. The block extractor found seven tiny rotated fragments,
  translated each one, and wrote the translations back at the original
  positions — producing seven whited-out rectangles with gibberish
  inside, destroying the watermark without preserving its meaning.
- **Mid-sentence prose fragments.** Fixing the "collapsed lists" bug by
  splitting blocks per visual line created a new problem: flowing prose
  paragraphs that the PDF had wrapped across many lines were fed to the
  model as isolated fragments like `"halinde acil kranial BT önerilir.
  Sepsis ve eşlik eden metabolik/sistemik bozukl..."`. The model
  correctly refused to translate an incoherent fragment, returning
  nothing, and the job failed.

Each fix added complexity (list-vs-prose heuristic, sentinel-delimiter
protocol, watermark aggregation, per-line bbox handling) and the service
grew to 2,738 lines. The complexity was unsustainable: every new
document shape exposed a corner case.

**The current Mode A** abandoned in-place rewriting. It extracts the text
layer with `page.get_text("text")` (which preserves the source's own line
breaks and blank lines), translates it in large chunks with an explicit
instruction to keep line and list structure, renders a clean English PDF,
and appends the untouched original pages behind it. The watermark problem
disappears because we're no longer editing the original page — ONAYSIZ
sits on the appendix, visible to any clinician who needs to see it. The
drug-list problem disappears because `get_text` already knows where the
lines are, and the Markdown-ish render preserves them. The mid-sentence
fragment problem disappears because we never fragment the source at all.
~370 lines deleted.

The lesson: trying to preserve visual layout *and* translate was two
problems glued together. Separating them — clean translation as one
document, original as appendix — made both tractable. A reader who needs
to check a `{sic}` value or a preserved institutional name has the
original right there.

## Poor-scan fast-fail + Cyrillic Tesseract fallback

A creased Ukrainian oncology discharge form (Zhytomyr Regional Cancer
Centre) demonstrated a failure mode we hadn't guarded against: on
illegible scans, vision models don't fail — they hallucinate. Mistral
produced 129,503 characters of plausible-looking clinical text from two
pages that held maybe 3,000 characters of real content. The service
accepted this as valid OCR, tried to translate it, and the whole job
ran for 25 minutes before timing out.

Two guards now catch this:

- **Per-page character ceiling** (`MAX_OCR_CHARS_PER_PAGE = 8000`). If
  OCR produces more than this per page, the service treats it as
  pathological output rather than valid text. Raises `UnreadableScan`.
- **Hard OCR timeout** (`OCR_TIMEOUT_SECONDS = 300`). Below the OpenAI
  client's silent 10-minute default; a dense OCR that takes longer than
  5 minutes is almost certainly in a fabrication loop.

**Then** the Cyrillic-specific recovery: when the fabricating scan was
identified as Cyrillic script, the service now retries the OCR step
locally with Tesseract (`ukr+rus+bul+srp+mkd+bel+kaz+eng`) before
giving up. Tesseract returns what's actually on the page rather than
fabricating, and on printed Cyrillic — which Mistral-Small-24B handles
poorly — it's more accurate too. All on-box; no GDPR consideration.

In testing on the Zhytomyr document, Tesseract read the diagnosis,
patient name (Гончарова Ірина Вікторівна), all dates, the radiotherapy
dose (34 Гр / 17 fractions), and the full discharge drug list cleanly.
A read Mistral had turned into 129k characters of garbage, Tesseract
handled in seconds.

## Adaptive OpenCV preprocessing

Poor scans happen for reasons other than the OCR engine being wrong for
the script — bad lighting, creases, low contrast. Standard preprocessing
(deskew, contrast normalisation, denoise) helps these. The temptation was
to apply a fixed preprocessing pipeline to every page.

Testing on the Zhytomyr document — one clean form page and one creased
page — demonstrated why that would have been wrong:

| Page | Raw OCR (confident words) | Enhanced |
|------|---------------------------|----------|
| 1 (clean dense form) | 193 | 125 |
| 2 (creased, mostly blank) | 0 | 30 |

Blanket enhancement *degraded* the clean page by a third. CLAHE and
denoising blur detail that was already sharp. A fixed pipeline would
have made working documents worse while rescuing broken ones.

The solution: **adaptive per page**. Enhance a copy, measure its
confident-word yield via Tesseract, keep the enhanced version only if
yield improves by >5%. The clean page keeps raw pixels; the creased
page gets the full treatment. Preprocessing can only help, never harm,
because it measures before committing.

Two design notes worth remembering if this is ever tuned:

- **Yield, not mean confidence.** An early version used mean confidence
  as the quality metric and made the wrong call on the clean page —
  enhancement raised its mean (48 vs 45) by *dropping* the hard words
  it could no longer read. Mean confidence rewarded that; word yield
  caught it.
- **No adaptive thresholding.** Experimented with
  `cv2.adaptiveThreshold` in the pipeline. It helps very poor scans
  but destroys detail on borderline ones. Dropped in favour of CLAHE
  + light denoise only.

## QA verification pass

Translation quality is set by the model, not the pipeline. Mistral and
Llama on OVHcloud are competent but not perfect — and a single real
error on a discharge summary (the "Tuberculosis Diagnoses" case) could
affect patient care. The QA pass addresses this: after translation, a
second-model call reviews the draft against the source for clinically
material discrepancies (mistranslated diagnoses, altered drug/dose/
route/frequency, lab values or units that differ, reversed negations,
content present in the source but absent from the translation or
invented in it). Any findings are printed on the attestation page.

Checking is easier than translating, so the pass catches slips even
when the same model did both. Exempt by default for French, German,
Spanish, Italian and English (reliably handled by both models, the
pass buys nothing); on by default for everything else.

## Error PDF

Jobs failed by writing an error string to the status endpoint and
returning HTTP 500 at `/translate/result/<id>`. FileMaker's existing
error branch worked on the status, but the result field held nothing —
just a 500. For a user-facing document pipeline this was wrong: the
operator deserves a readable explanation of what happened.

The service now writes a one-page PDF on failure (red "Translation
not completed" heading, source filename, what happened, remedy
suggestion) and serves it from the result endpoint. The status
endpoint still returns `failed` so the FileMaker branch is untouched,
but the fetch returns a readable document instead of a bare 500.

Two failure classes get distinct error PDFs: `UnreadableScan`
("Scan quality too low for OCR — a clearer scan, or manual
translation, is needed") and everything else ("Translation could
not be completed").

## DeepL as the translation engine (Nov 2026)

Translation moved from the OVHcloud models to DeepL, which is stronger and
more consistent on the European languages that make up most of the
workload. Everything else stayed: Mistral still does OCR and language
assessment, the QA pass still runs, and the PDF builders are untouched.

The old Mistral/Llama translation path was kept as an automatic fallback
rather than removed. A DeepL outage, an exhausted quota or a language
DeepL does not cover must not turn into a failed job when the previous
engine could have translated the document. Every fallback is named on
the PDF's provenance line so the reader knows which engine produced the
text in front of them.

## Local OCR fallback for runaway pages (Nov 2026)

A clean five-page scan failed after eight minutes: the vision model ran to
the 16,384-token cap on pages 1-2, the batch was split, and page 1 alone
(152 readable words) ran to the cap again, which was fatal. The existing
fabrication guard never fired because the truncation check comes first, and
the Tesseract fallback only covered Cyrillic.

A single page that overflows or times out is now read with Tesseract,
whatever the script, and OCR calls are budgeted per page so the overflow
shows up sooner. A page with no words at all gets a marker rather than
failing the document — a photo page in the middle of a discharge letter
should not cost the reader the other four pages.
