"""
SkyCare PDF Translation Microservice  v3 — Async Job Queue
============================================================
Extends v2 (two-stage Mistral/Llama adaptive pipeline) with an asynchronous
job queue to eliminate FileMaker's 5-minute curl timeout risk.

Architecture:
  POST /translate/submit  — accepts file, returns job_id immediately (<1s)
  GET  /translate/status/<job_id>  — poll for pending/processing/complete/failed
  GET  /translate/result/<job_id>  — fetch completed PDF (auto-cleans job)

  Translation runs in a background ThreadPoolExecutor (2 workers, matching
  the Gunicorn worker count). Jobs are stored in memory with a 2-hour TTL;
  result PDFs are written to /tmp/skycare-jobs/ and cleaned up on fetch
  or on TTL expiry.

  The original synchronous POST /translate endpoint is retained for
  direct testing and non-FileMaker callers.

LONG DOCUMENTS AND TRUNCATION
  Every model call is bounded so its output fits inside MAX_OUTPUT_TOKENS:
    - OCR runs MAX_IMAGES_PER_CALL (2) pages per call, so a 30-page document
      becomes 15 sequential calls rather than one call that overflows. If a
      batch still overflows, it is halved and retried automatically, down to
      a single page.
    - Translation splits text into MAX_TRANSLATE_CHARS (20k) chunks on
      paragraph -> line -> word boundaries, so no chunk can overflow.
      Only the final chunk is asked to append the attestation.
  Every response is then checked for finish_reason == "length". If any call
  was cut off, the job FAILS with TranslationTruncated rather than returning
  a partial document. A truncated medical translation is more dangerous than
  no translation: it looks complete and carries an attestation while silently
  omitting clinical content.
  Failures surface via GET /translate/status/<job_id> with the error text.

FileMaker script pattern:
  1. Submit PDF → receive job_id
  2. Loop: wait 10s → GET status → break when complete or failed
  3. GET result → store PDF in container field
  Note: check for status == "failed" and surface the "error" field — a failed
  job means the document was NOT fully translated and must not be used.

Port: 5107 (v1=5105, v2=5106)

Setup:
    pip install flask pymupdf pdfplumber reportlab openai pillow
    export OVHCLOUD_API_KEY="your_key_here"
    python pdf_translate_v3_service.py

Endpoints:
    POST /translate/submit
        file        — PDF or image (jpg/png/tiff/webp/bmp)
        target_lang — default "English"
        source_hint — optional hint
        mode        — optional "auto" (default), "A", "B", or "C"
        model       — optional "mistral" or "llama" (bypasses auto-selection)
        job_id      — optional client-supplied UUID (also accepted as jobID)
        Returns: {"job_id": "...", "status": "pending", "poll_url": "..."}

    GET /translate/status/<job_id>
        Returns: {"job_id": "...", "status": "pending|processing|complete|failed",
                  "filename": "...", "elapsed_seconds": 12.3}
        On failure also includes: {"error": "..."}

    GET /translate/result/<job_id>
        Returns: translated PDF (application/pdf) — job cleaned up after fetch
        Returns 404 if job unknown or expired, 409 if not yet complete

    POST /translate
        Synchronous endpoint retained for testing.
        Same parameters as /translate/submit but waits and returns PDF directly.

    GET /health
        Returns JSON status including queue depth and worker count.
"""

import base64
import io
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf as fitz
import pdfplumber
from flask import Flask, jsonify, request, send_file
from openai import OpenAI

try:                      # DeepL is the primary translation engine, but the
    import deepl          # service must still start (and fall back to the
except ImportError:       # OVHcloud models) if the package is missing.
    deepl = None
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Frame,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OVHCLOUD_BASE_URL  = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
MODEL_MISTRAL      = "Mistral-Small-3.2-24B-Instruct-2506"
MODEL_LLAMA        = "Meta-Llama-3_3-70B-Instruct"

RENDER_DPI           = 150
SKEW_THRESHOLD_DEG   = 3.0
MAX_CHARS_PER_CHUNK  = 60_000   # Mode A block batching (input-side only)

# --- OCR quality / poor-scan handling ---------------------------------------
# A poor-quality scan does not make OCR fail cleanly — it makes the vision model
# GUESS and ramble, emitting far more (hallucinated) text than the page holds.
# One real case: a creased Ukrainian discharge form produced 129,000 characters
# from two pages before hitting the token cap, after a 7-minute call. Two guards
# below bound that failure so it surfaces fast and honestly rather than hanging
# or shipping fabricated clinical text:
#   OCR_TIMEOUT_SECONDS caps how long any single OCR call may run.
#   MAX_OCR_CHARS_PER_PAGE is the sanity ceiling on real text per A4 page — a
#   dense discharge page holds ~3-4k chars; well above that means the model is
#   fabricating, so we treat it as an unreadable scan and fail with a reason.
OCR_TIMEOUT_SECONDS    = 300      # 5 min hard cap per OCR call (was silently 600)
MAX_OCR_CHARS_PER_PAGE = 8_000    # above this per page = pathological OCR

# --- Output-token budgeting -------------------------------------------------
# The failure mode we are guarding against is SILENT TRUNCATION: the model hits
# its output cap mid-document, returns finish_reason="length", and we happily
# build a PDF from the partial text. Every call below is bounded so that the
# expected output fits comfortably inside MAX_OUTPUT_TOKENS, and every response
# is checked for finish_reason="length" — which raises rather than truncating.
MAX_OUTPUT_TOKENS    = 16_384   # Per-call output cap (model context is 128k)

# OCR batch size. Dense scanned forms produce far more text per page than the
# estimate above suggests, so this is deliberately conservative: 2 pages per
# call. If a batch overflows anyway, ocr_batch_adaptive() halves it and retries
# rather than failing the job, so this value is a starting point rather than a
# hard constraint — lower means fewer retries, higher means fewer calls.
MAX_IMAGES_PER_CALL  = 2

# Translation: output is roughly the same length as input, so the input chunk
# must be well under the output cap. ~4 chars/token → 16,384 tokens ≈ 65k chars;
# 20k chars in gives ~3x headroom for languages that expand when translated.
MAX_TRANSLATE_CHARS  = 20_000

# Translation verification (QA pass). After translation, a second call reviews
# the draft against the source for clinically material discrepancies. Exempt
# languages skip it: high-resource European languages are translated reliably
# and the second call buys nothing there.
QA_EXEMPT_LANGUAGES = {"french", "german", "spanish", "italian", "english"}
QA_MAX_FINDINGS_SHOWN = 15

# --- DeepL (primary translation engine) --------------------------------------
# DeepL translates the extracted text; OCR, language assessment and the QA pass
# stay on the OVHcloud models. If DeepL cannot be used for any reason (no key,
# quota, outage, unsupported language, output that looks untranslated) the job
# falls back to the original Mistral/Llama translation path — a DeepL problem
# must never fail a job that the old engine could have completed.
#   TRANSLATE_ENGINE   "deepl" (default) or "llm" to switch DeepL off entirely.
#   DEEPL_AUTH_KEY     the key. Free keys end ":fx"; the SDK picks the free or
#                      pro endpoint from that, so there is no URL to configure.
DEEPL_ENGINE_NAME        = "DeepL"
TRANSLATE_ENGINE         = os.environ.get("TRANSLATE_ENGINE", "deepl").strip().lower()
# DeepL caps a request body at 128 KiB. Non-Latin text can cost 6 bytes per
# character once JSON-escaped, so 10k characters per request stays well inside.
DEEPL_MAX_CHARS          = 10_000
DEEPL_TIMEOUT_SECONDS    = 60
DEEPL_MAX_RETRIES        = 3        # SDK retries 429 / 5xx with backoff
DEEPL_COOLDOWN_SECONDS   = 600      # after an outage/auth/quota failure, skip DeepL this long
DEEPL_MODEL_TYPE         = os.environ.get("DEEPL_MODEL_TYPE", "prefer_quality_optimized")
DEEPL_ENGLISH_VARIANT    = os.environ.get("DEEPL_ENGLISH_VARIANT", "EN-GB").upper()

# target_lang arrives as a language NAME ("English"); DeepL wants a code.
DEEPL_TARGET_CODES = {
    "english": DEEPL_ENGLISH_VARIANT, "british english": "EN-GB",
    "american english": "EN-US", "french": "FR", "german": "DE", "spanish": "ES",
    "italian": "IT", "dutch": "NL", "polish": "PL", "portuguese": "PT-PT",
    "brazilian portuguese": "PT-BR", "russian": "RU", "ukrainian": "UK",
    "turkish": "TR", "greek": "EL", "arabic": "AR", "hebrew": "HE",
    "chinese": "ZH", "japanese": "JA", "korean": "KO", "czech": "CS",
    "slovak": "SK", "slovenian": "SL", "hungarian": "HU", "romanian": "RO",
    "bulgarian": "BG", "danish": "DA", "swedish": "SV", "norwegian": "NB",
    "finnish": "FI", "estonian": "ET", "latvian": "LV", "lithuanian": "LT",
    "indonesian": "ID", "thai": "TH", "vietnamese": "VI",
}
# Names the OCR assessment may use that differ from DeepL's own language names.
DEEPL_SOURCE_ALIASES = {"farsi": "persian", "mandarin": "chinese",
                        "cantonese": "chinese", "castilian": "spanish"}


PORT = int(os.environ.get("TRANSLATE_PORT", 5107))

ACCEPTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}

ALWAYS_ESCALATE_LANGUAGES = {
    "arabic", "farsi", "persian", "hebrew", "pashto", "urdu",
    "thai", "bengali", "tamil", "telugu", "kannada", "malayalam",
    "sinhala", "burmese", "khmer", "lao", "tibetan", "mongolian",
    "amharic", "georgian", "armenian",
    # Turkish added after observed Mistral failures on a real discharge
    # summary: "Taburculuk Tanıları" (Discharge Diagnoses) rendered as
    # "Tuberculosis Diagnoses" — phonetic guessing that invents a disease.
    # Agglutinative morphology makes Turkish harder than its resource level
    # suggests; the 70B model is measurably stronger on it.
    "turkish", "azerbaijani",
}

ESCALATION_CONFIDENCE_THRESHOLD = 0.75

# Job queue settings
MAX_WORKERS   = 2                    # Match Gunicorn worker count
JOB_TTL_SECS  = 7200                 # 2 hours — clean up uncollected jobs
JOB_DIR       = Path("/tmp/skycare-jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Truncation guard
# ---------------------------------------------------------------------------

class TranslationTruncated(RuntimeError):
    """
    Raised when a model response was cut off at the output-token limit.

    This is deliberately fatal to the job. A truncated medical translation is
    worse than no translation: it looks complete, carries an attestation, and
    silently omits clinical content. The job fails, the error surfaces via
    /translate/status, and the operator can retry or split the document.
    """


class UnreadableScan(RuntimeError):
    """
    Raised when a scan is too poor for reliable OCR.

    Distinct from TranslationTruncated: this is not a token-budget problem but
    an image-quality one. A poor scan makes the vision model fabricate text
    rather than fail, so we detect the symptom (implausible output length, or a
    call that runs past the timeout) and stop. The message is written for the
    operator: it names the likely cause (scan quality) and the remedy (a
    clearer scan or manual translation), and it is what gets rendered onto the
    error PDF returned to FileMaker.
    """


def _looks_repetitive(text: str) -> tuple[bool, str]:
    """
    Detect the degenerate loop a vision model falls into when it cannot read an
    image: it emits the same line or phrase over and over until it hits the
    output cap. Distinguishing this from a genuinely long page matters, because
    the remedies are opposite — a loop needs a clearer image or a simpler task,
    a long page needs a bigger budget.

    Returns (is_repetitive, human-readable reason).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 12:
        return False, ""

    # Look at the tail, where a loop will have taken hold
    tail = lines[-40:]
    unique = len(set(tail))
    if unique <= max(2, len(tail) // 10):
        return True, (f"only {unique} distinct line(s) in the last {len(tail)} — "
                      f"most repeated: {max(set(tail), key=tail.count)!r}")

    # A single line repeated many times anywhere
    for line in set(tail):
        if len(line) > 8 and tail.count(line) >= 10:
            return True, f"line repeated {tail.count(line)} times: {line!r}"

    return False, ""


def _check_not_truncated(response, stage: str, job_context: str = "") -> None:
    """
    Inspect an OpenAI-compatible response for finish_reason == "length".

    finish_reason values: "stop" (model finished naturally), "length" (hit the
    max_tokens cap — output is incomplete), "content_filter", "tool_calls".
    Anything other than a clean stop on a translation call means we cannot
    trust the output to be complete.
    """
    try:
        finish_reason = response.choices[0].finish_reason
    except (AttributeError, IndexError):
        log.warning("%s: could not read finish_reason — proceeding", stage)
        return

    if finish_reason == "length":
        # Diagnose before raising: a loop and a genuinely long page look
        # identical from the finish_reason alone.
        try:
            partial = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            partial = ""

        repetitive, why = _looks_repetitive(partial)
        if repetitive:
            log.warning("%s hit the output cap and the output is REPETITIVE (%s) — "
                        "the model is looping, which usually means it cannot read "
                        "the image clearly", stage, why)
        elif partial:
            log.warning("%s hit the output cap after %d chars; output does not look "
                        "repetitive, so the page may genuinely be very dense. "
                        "Tail: %r", stage, len(partial), partial[-160:])

        where = f" ({job_context})" if job_context else ""
        why = ("; the output was repetitive, which indicates the model could not "
               "read the image clearly rather than the page being unusually long"
               ) if repetitive else ""
        raise TranslationTruncated(
            f"{stage} output was truncated at the {MAX_OUTPUT_TOKENS}-token "
            f"limit{where}{why}. This job has been failed rather than returning "
            "an incomplete translation."
        )

    if finish_reason not in ("stop", None):
        log.warning("%s: unexpected finish_reason=%r", stage, finish_reason)

api_key = os.environ.get("OVHCLOUD_API_KEY", "")
if not api_key:
    log.warning("OVHCLOUD_API_KEY not set — using free rate-limited tier")

ovh_client = OpenAI(api_key=api_key or "anonymous", base_url=OVHCLOUD_BASE_URL)
# Both models share the same OVHcloud endpoint — same client, different model strings

DEEPL_AUTH_KEY = os.environ.get("DEEPL_AUTH_KEY", "").strip()
deepl_client = None
if TRANSLATE_ENGINE != "deepl":
    log.info("TRANSLATE_ENGINE=%s — DeepL disabled, translating with OVHcloud models",
             TRANSLATE_ENGINE)
elif deepl is None:
    log.warning("deepl package not installed — translating with OVHcloud models "
                "(pip install deepl to enable DeepL)")
elif not DEEPL_AUTH_KEY:
    log.warning("DEEPL_AUTH_KEY not set — translating with OVHcloud models")
else:
    deepl_client = deepl.DeepLClient(
        DEEPL_AUTH_KEY, timeout_s=DEEPL_TIMEOUT_SECONDS, max_retries=DEEPL_MAX_RETRIES,
        send_platform_info=False,
    )
    if DEEPL_AUTH_KEY.endswith(":fx"):
        log.warning("DEEPL_AUTH_KEY is a DeepL API *Free* key. Free-plan text may be "
                    "retained by DeepL; use a Pro key for patient documents.")
    log.info("DeepL enabled (%s plan) — OVHcloud models are the fallback",
             "free" if DEEPL_AUTH_KEY.endswith(":fx") else "pro")

# Static API key for inbound request authentication
SERVICE_API_KEY = os.environ.get("TRANSLATE_API_KEY", "")
if not SERVICE_API_KEY:
    log.warning("TRANSLATE_API_KEY not set — inbound authentication is DISABLED")

# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------
# Job store — SQLite (shared across all Gunicorn workers via filesystem)
# ---------------------------------------------------------------------------
#
# All workers read/write the same database file. SQLite's WAL mode gives
# safe concurrent access without external locking. Each worker also runs
# its own ThreadPoolExecutor for the actual translation work.

DB_PATH = JOB_DIR / "jobs.db"

# Per-worker thread pool — 2 workers × 2 threads = max 4 concurrent translations,
# but Gunicorn's sync workers mean only 1 request is active per worker at a time,
# so effective concurrency is MAX_WORKERS.
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="translate")


def _db() -> sqlite3.Connection:
    """
    Open a per-call SQLite connection with WAL mode and a 10s busy timeout.
    WAL allows concurrent readers + one writer without blocking.
    Connections are not shared across threads — each call gets its own.
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _init_db() -> None:
    """Create the jobs table if it doesn't exist. Called at module load."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id       TEXT PRIMARY KEY,
                status       TEXT NOT NULL DEFAULT 'pending',
                filename     TEXT NOT NULL,
                submitted_at REAL NOT NULL,
                completed_at REAL,
                result_path  TEXT NOT NULL,
                mode_used    TEXT,
                error        TEXT
            )
        """)
        conn.commit()
    log.info("Job database initialised at %s", DB_PATH)


def _job_result_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.pdf"


def _create_job(filename: str, job_id: str | None = None) -> str:
    job_id = job_id or str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            """INSERT INTO jobs
               (job_id, status, filename, submitted_at, result_path)
               VALUES (?, 'pending', ?, ?, ?)""",
            (job_id, filename, time.time(), str(_job_result_path(job_id)))
        )
        conn.commit()
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    if not kwargs:
        return
    allowed = {"status", "completed_at", "mode_used", "error"}
    fields  = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values     = list(fields.values()) + [job_id]
    with _db() as conn:
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
        conn.commit()


def _get_job(job_id: str) -> dict | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def _cleanup_expired_jobs() -> None:
    """Remove jobs and result files older than JOB_TTL_SECS."""
    cutoff = time.time() - JOB_TTL_SECS
    with _db() as conn:
        rows = conn.execute(
            "SELECT job_id, result_path FROM jobs WHERE submitted_at < ?", (cutoff,)
        ).fetchall()
        if rows:
            for row in rows:
                Path(row["result_path"]).unlink(missing_ok=True)
            conn.execute("DELETE FROM jobs WHERE submitted_at < ?", (cutoff,))
            conn.commit()
    if rows:
        log.info("Cleaned up %d expired jobs", len(rows))


def _start_cleanup_thread() -> None:
    """Background thread: sweep expired jobs every 15 minutes."""
    def loop():
        while True:
            time.sleep(900)
            try:
                _cleanup_expired_jobs()
            except Exception as e:
                log.warning("Job cleanup error: %s", e)
    t = threading.Thread(target=loop, daemon=True, name="job-cleanup")
    t.start()


def _build_error_pdf(source_filename: str, detail: str, headline: str) -> bytes:
    """
    Render a one-page PDF explaining why a job failed, so FileMaker's result
    field holds a readable document rather than nothing. The job status is still
    "failed" and the error text is still on the status endpoint; this is the
    human-facing companion to that.
    """
    from reportlab.platypus import SimpleDocTemplate
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    margin = 22 * mm
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin,
                            title="Translation could not be completed")

    title_style = ParagraphStyle("ErrTitle", parent=styles["Title"], fontSize=16,
                                 textColor=colors.HexColor("#b00020"), spaceAfter=6)
    sub_style = ParagraphStyle("ErrSub", parent=styles["Normal"], fontSize=11,
                               textColor=colors.HexColor("#333333"), spaceAfter=14)
    body_style = ParagraphStyle("ErrBody", parent=styles["Normal"], fontSize=10,
                                leading=15, spaceAfter=10)
    label_style = ParagraphStyle("ErrLabel", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.HexColor("#777777"), spaceAfter=2)

    story = [
        Paragraph("Translation not completed", title_style),
        Paragraph(_xml(headline), sub_style),
        Paragraph("Source document", label_style),
        Paragraph(_xml(source_filename), body_style),
        Paragraph("What happened", label_style),
        Paragraph(_xml(detail), body_style),
        Spacer(1, 6 * mm),
        Paragraph(
            "This document has been flagged as failed in the translation system. "
            "No translated content is provided, deliberately: returning an "
            "incomplete or fabricated medical translation would be unsafe. Please "
            "obtain a clearer scan where possible, or arrange manual translation.",
            body_style),
        Spacer(1, 8 * mm),
        Paragraph(
            f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · SkyCare PDF "
            "Translation Service", label_style),
    ]
    doc.build(story)
    return buf.getvalue()


def _run_job(job_id: str, file_bytes: bytes, ext: str, target_lang: str,
             source_hint: str, forced_mode: str, model_override: str,
             uploaded_filename: str, qa_mode: str = "auto") -> None:
    """Worker function — runs in thread pool, updates SQLite on completion."""
    _update_job(job_id, status="processing")
    log.info("Job %s started: file=%s", job_id, uploaded_filename)
    try:
        output_pdf, mode_used = run_pipeline(
            file_bytes, ext, target_lang, source_hint,
            forced_mode, model_override, uploaded_filename, qa_mode,
        )
        result_path = _job_result_path(job_id)
        result_path.write_bytes(output_pdf)
        _update_job(job_id,
                    status="complete",
                    completed_at=time.time(),
                    mode_used=mode_used)
        log.info("Job %s complete: mode=%s size=%d bytes", job_id, mode_used, len(output_pdf))
    except UnreadableScan as e:
        # Expected failure on a poor scan — not a bug. Fail the job (so any
        # FileMaker branch on "failed" still fires) but also write an error PDF
        # so the operator gets a readable explanation in the result field.
        log.warning("Job %s: unreadable scan — %s", job_id, e)
        try:
            err_pdf = _build_error_pdf(uploaded_filename, str(e),
                                       "Scan quality too low for OCR")
            _job_result_path(job_id).write_bytes(err_pdf)
        except Exception as pe:
            log.warning("Job %s: could not write error PDF: %s", job_id, pe)
        _update_job(job_id, status="failed", error=str(e), completed_at=time.time())
    except Exception as e:
        log.exception("Job %s failed", job_id)
        try:
            err_pdf = _build_error_pdf(uploaded_filename, str(e),
                                       "Translation could not be completed")
            _job_result_path(job_id).write_bytes(err_pdf)
        except Exception as pe:
            log.warning("Job %s: could not write error PDF: %s", job_id, pe)
        _update_job(job_id, status="failed", error=str(e), completed_at=time.time())


# Initialise DB and start cleanup thread at module load
_init_db()
_start_cleanup_thread()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Shared translation system prompt — used by both Mistral and Llama
SYSTEM_PROMPT = """You are a professional medical translator, fluent in many languages. \
You are acting solely as a linguistic translator — not as a clinician. \
For the purposes of this task, you have no knowledge of what medications, treatments, \
or findings are typical or expected for any given condition or patient. \
Your only source of information is the document in front of you.

You have been tasked with translating the attached medical document into {target_lang}. \
Complete accuracy is paramount. There could be severe medicolegal consequences if you \
summarise, guess, interpolate, or add any information not explicitly present in the source document.

Strict fidelity rules:
- Translate only what is present on the page. Do not infer, add, or complete anything that is not there.
- Drug names, dosages, frequencies, routes of administration, laboratory values, reference ranges, \
and diagnostic codes must be transcribed exactly as they appear in the source.
- If any text is partially illegible or unclear, mark it [ILLEGIBLE] or [UNCLEAR] rather than guessing.
- Never substitute a drug name, value, or clinical detail based on what might be typical or expected \
for a patient with a given condition.
- Where drug names use mixed capitalisation to indicate the INN (International Non-proprietary Name) \
within a brand name, preserve this formatting and note the full INN in curly braces only if you can \
identify it with certainty from the text itself. If uncertain, transcribe the name exactly as printed \
and do not expand it.
- Preserve numeric formatting exactly as printed, including decimal separators: if the source \
writes "0,17" with a decimal comma, the translation writes "0,17" — never silently convert \
between comma and point.
- Unit abbreviations: translate to the standard {target_lang} clinical abbreviation and be \
consistent throughout (e.g. Turkish "dk" is minutes → "/min"). Never substitute a different \
unit or measure than the one printed, and never leave the same source unit translated in one \
place and untranslated in another.
- If the source contains an apparent error — an implausible dose, a misspelling, a malformed \
number — transcribe it EXACTLY as printed and append {{sic — as written in source}}. Never \
correct it, and never leave it unmarked: the reader must know the source itself says this.
- Institutional names, blood-product designations, and administrative terms you cannot \
translate with certainty must be kept in the original language with a brace clarification of \
their apparent meaning, e.g. "Kızılay {{Turkish Red Crescent blood service}}". A literal \
word-by-word rendering of such terms is worse than leaving them untranslated.
- Section headings are high-stakes: a misread heading misleads the reader about everything \
under it. Translate headings using the surrounding document for context, and if a heading is \
ambiguous, keep the original in braces after your translation.
- Handwritten annotations must be clearly identified as such: {{handwritten: annotation text}}.
- Any translator's clarification or expansion must appear in curly braces {{like this}} and must be \
clearly distinguishable from the translated text.
- Preserve all section headings, table structures, and document layout as closely as possible.
- Mark each page with the header: "AI Translation of original document in [detected language] into {target_lang}"
- If pages appear to be missing or out of order, note this at the end of the document.

Fidelity attestation: At the end of the document, append the following statement, completed honestly:
"Translator's attestation: This translation contains no information beyond that present in the \
source document. All drug names, dosages, and clinical values have been transcribed directly from \
the source. Items marked [ILLEGIBLE] or [UNCLEAR] indicate text that could not be read with \
certainty. Number of such items: ___."
"""

# Stage 1 OCR prompt — asks Mistral to extract text AND self-assess language confidence.
# Deliberately does NOT ask for "AI Translation of..." page headers — those are added
# by the PDF builder. Including them in the OCR output confuses Stage 2 into thinking
# the document has already been translated.
OCR_ASSESSMENT_PROMPT = """You are performing two tasks on the document image(s) provided:

TASK 1 — TEXT EXTRACTION
Extract all text exactly as it appears on the page(s). Preserve layout, headings, and structure.
Mark handwritten text as {handwritten: text}. Mark illegible text as [ILLEGIBLE].
Separate pages with a line containing only '---PAGE BREAK---'.
Do NOT add any headers, labels, or commentary of your own — output only the document text.

TASK 2 — LANGUAGE ASSESSMENT
After all extracted text, output a JSON block on its own line in this exact format:
```json
{"language": "Spanish", "script": "Latin", "confidence": 0.95, "escalate": false, "reason": "High confidence Spanish medical document"}
```

confidence is your honest estimate (0.0–1.0) of how well you can translate this language accurately.
Set escalate to true if you believe another model would produce a significantly better translation.
Be honest — it is better to escalate than to produce a poor translation of a medical document.

Output the extracted text first, then the JSON block on its final line. Nothing else."""


# ---------------------------------------------------------------------------
# Rotation correction  — runs before mode detection and rendering
# ---------------------------------------------------------------------------

# Pillow EXIF orientation tag values → degrees to rotate counter-clockwise
_EXIF_ORIENTATION_TO_DEGREES = {
    1: 0,    # Normal
    2: 0,    # Mirrored horizontal (we don't handle mirroring)
    3: 180,
    4: 0,    # Mirrored vertical
    5: 90,
    6: 270,  # Rotated 90 CW → rotate 270 CCW to correct
    7: 270,
    8: 90,   # Rotated 90 CCW → rotate 90 CCW to correct
}


def correct_pdf_rotation(pdf_bytes: bytes) -> bytes:
    """
    Ensure pages render upright before they are sent to the model.

    A PDF's /Rotate attribute is not a defect to be corrected — it is the
    instruction that makes a page display the right way up. Phone-scanner apps
    routinely store the raw sensor image sideways and set /Rotate to compensate,
    which is why a document can look perfectly upright in a viewer while its
    embedded image is on its side. pymupdf's get_pixmap() honours /Rotate, so
    such a page already renders correctly and must be left alone.

    (An earlier version of this function called set_rotation(0) on any page with
    a non-zero /Rotate, believing it was straightening the page. It was doing
    the opposite: discarding the correction and handing the model a sideways
    image. On a dense table the model then stopped transcribing and began
    repeating itself until it hit the output token cap.)

    So the only case worth acting on is a page whose /Rotate is already 0 but
    whose content is nonetheless sideways — a raw image with no compensating
    attribute. Detection for that is text-direction based where there is a text
    layer, and aspect-ratio based where there is not.

    Returns corrected PDF bytes, or the input unchanged if nothing needed doing.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    changed = False

    for page in doc:
        if page.rotation != 0:
            # /Rotate is doing its job — get_pixmap() will render this upright.
            log.info("Page %d: /Rotate=%d present, page renders upright — "
                     "leaving as is", page.number + 1, page.rotation)
            continue

        content_rot = _detect_content_rotation(page)
        if content_rot != 0:
            log.info("Page %d: content is sideways with no /Rotate to compensate "
                     "— rotating %d°", page.number + 1, content_rot)
            _rotate_page_content(page, content_rot)
            changed = True

    if not changed:
        doc.close()
        return pdf_bytes

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _detect_content_rotation(page: fitz.Page) -> int:
    """
    Return the clockwise degrees by which the content appears rotated
    (i.e. how much we need to rotate counter-clockwise to correct it).
    Returns 0, 90, 180, or 270.

    Strategy:
    - Text-based page: use the median direction vector of text spans.
      A span with dir≈(0,-1) means text runs downward → page rotated 90° CW.
    - Image-only page: compare page aspect ratio to image aspect ratio.
      If a portrait-shaped page contains a landscape-shaped image, the image
      was likely taken sideways.
    """
    # Try text-based detection first
    try:
        blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        dirs = []
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    if len(span.get("text", "").strip()) < 3:
                        continue
                    d = span.get("dir", (1, 0))
                    dirs.append(d)
        if len(dirs) >= 3:
            # Median x and y components
            median_x = sorted(d[0] for d in dirs)[len(dirs) // 2]
            median_y = sorted(d[1] for d in dirs)[len(dirs) // 2]
            angle = math.degrees(math.atan2(-median_y, median_x))  # PDF y-axis is inverted

            # Snap to nearest 90°
            snapped = round(angle / 90) * 90 % 360
            if snapped == 0:
                return 0
            # Content angle is how much it's rotated FROM upright;
            # we need to correct by the inverse
            correction = (360 - snapped) % 360
            if correction in (90, 180, 270):
                return correction
    except Exception:
        pass

    # Image-only page heuristic — ONLY run if there is no extractable text on the page.
    # If a page has text, it is already upright; embedded images (logos, stamps, charts)
    # may be landscape-shaped without implying the page itself is rotated.
    try:
        page_text = page.get_text("text").strip()
        if page_text:
            return 0
    except Exception:
        pass

    try:
        page_w = page.rect.width
        page_h = page.rect.height
        page_is_portrait = page_h >= page_w

        images = page.get_images(full=True)
        if images:
            xref = images[0][0]
            img_info = page.parent.extract_image(xref)
            img_w = img_info["width"]
            img_h = img_info["height"]
            img_is_portrait = img_h >= img_w

            if page_is_portrait and not img_is_portrait:
                log.info("Page %d: landscape image in portrait page → assuming 90° content rotation", page.number)
                return 90
            elif not page_is_portrait and img_is_portrait:
                log.info("Page %d: portrait image in landscape page → assuming 270° content rotation", page.number)
                return 270
    except Exception:
        pass

    return 0


def _rotate_page_content(page: fitz.Page, degrees: int) -> None:
    """
    Rotate the content of a page by `degrees` counter-clockwise in-place.
    Adjusts the page mediabox to match the new orientation if needed.
    """
    w = page.rect.width
    h = page.rect.height

    # pymupdf >= 1.23 removed page.transform(); use set_rotation() for right-angle corrections.
    # Map content rotation degrees → PDF rotation attribute (inverse of content rotation)
    pdf_rotation = {
        90:  270,
        180: 180,
        270: 90,
    }.get(degrees)

    if pdf_rotation is not None:
        page.set_rotation(pdf_rotation)


def correct_image_rotation(image_bytes: bytes) -> bytes:
    """
    Correct rotation in a raw image file (jpg/png/tiff etc.).

    1. Reads EXIF orientation tag and applies the corresponding rotation.
    2. If no EXIF data, falls back to aspect-ratio heuristic: images that
       are wider than tall are assumed to be landscape phone photos and are
       rotated 90° CCW to portrait.

    Returns corrected PNG bytes.
    """
    img = Image.open(io.BytesIO(image_bytes))
    rotated = False

    # EXIF orientation
    try:
        exif = img._getexif()  # Returns None for PNGs
        if exif:
            orientation = exif.get(274)  # Tag 274 = Orientation
            degrees = _EXIF_ORIENTATION_TO_DEGREES.get(orientation, 0)
            if degrees:
                log.info("Image: correcting EXIF orientation tag %d → rotate %d°", orientation, degrees)
                img = img.rotate(degrees, expand=True)
                rotated = True
    except (AttributeError, Exception):
        pass

    # Aspect-ratio heuristic if no EXIF correction applied
    if not rotated:
        w, h = img.size
        if w > h * 1.2:  # Clearly landscape — likely a phone photo held sideways
            log.info("Image: landscape aspect ratio (%.0f×%.0f) → rotating 90° CCW", w, h)
            img = img.rotate(90, expand=True)

    buf = io.BytesIO()
    # Always save as PNG to avoid EXIF-in-JPEG confusion on re-read
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def detect_mode(pdf_bytes: bytes, forced: str = "auto") -> str:
    """
    Return "A", "B", or "C".
    A = text-based PDF
    B = clean scanned PDF (image-only, pages reasonably straight)
    C = smartphone photo / heavily skewed / poor quality
    """
    if forced in ("A", "B", "C"):
        return forced

    # Check for selectable text on first page
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if pdf.pages:
            text = pdf.pages[0].extract_text() or ""
            if len(text.strip()) > 50:
                log.info("Mode A: text-based PDF")
                return "A"

    # Image-only PDF — assess skew
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    skew = _estimate_skew(doc[0])
    doc.close()

    if abs(skew) > SKEW_THRESHOLD_DEG:
        log.info("Mode C: scanned/photo PDF, skew=%.1f°", skew)
        return "C"

    log.info("Mode B: clean scanned PDF, skew=%.1f°", skew)
    return "B"


def detect_mode_for_image() -> str:
    """Images uploaded directly are always mode C."""
    return "C"


def _estimate_skew(page: fitz.Page) -> float:
    """
    Rough skew estimate from the angle of text spans on the page.
    Returns degrees; 0 = straight, positive = clockwise tilt.
    """
    try:
        blocks = page.get_text("rawdict")["blocks"]
        angles = []
        for b in blocks:
            for line in b.get("lines", []):
                spans = line.get("spans", [])
                if spans:
                    # The wmode/dir fields give writing direction
                    d = spans[0].get("dir", (1, 0))
                    angle = math.degrees(math.atan2(d[1], d[0]))
                    angles.append(angle)
        if not angles:
            return 0.0
        return sum(angles) / len(angles)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Text extraction with bounding boxes (Mode A)
# ---------------------------------------------------------------------------





def _qa_attestation_paragraphs(assessment: dict | None, body_style) -> list:
    """
    ReportLab paragraphs describing the verification outcome. Always says
    something when QA ran or failed; silent only when skipped as exempt.
    """
    qa = (assessment or {}).get("qa")
    if not qa:
        return []

    out = [Spacer(1, 4 * mm)]
    if qa["status"] == "clean":
        out.append(Paragraph(
            "<b>AI verification:</b> a second model compared this translation "
            "against the source for clinically material discrepancies "
            "(diagnoses, headings, drugs, doses, laboratory values, negations, "
            "omissions) and found none. This supplements but does not replace "
            "human review.", body_style))
    elif qa["status"] == "findings":
        out.append(Paragraph(
            f"<b>AI verification — {_xml(qa['note'])}.</b> A second model "
            "compared this translation against the source and reported the "
            "following. Each item should be checked against the original "
            "before the affected content is relied upon:", body_style))
        for f in qa["findings"][:QA_MAX_FINDINGS_SHOWN]:
            sev = "CRITICAL" if f["severity"] == "critical" else "minor"
            out.append(Paragraph(
                f"&bull; <b>[{sev}]</b> {_xml(f['issue'])} "
                f"(source: \u201c{_xml(f['source_excerpt'])}\u201d &rarr; "
                f"translation: \u201c{_xml(f['translation_excerpt'])}\u201d)",
                body_style))
        if len(qa["findings"]) > QA_MAX_FINDINGS_SHOWN:
            out.append(Paragraph(
                f"&bull; ... and {len(qa['findings']) - QA_MAX_FINDINGS_SHOWN} "
                "further finding(s) — see service logs.", body_style))
    elif qa["status"] == "skipped":
        out.append(Paragraph(
            f"<b>AI verification:</b> skipped — {_xml(qa['note'])}.", body_style))
    else:
        out.append(Paragraph(
            f"<b>AI verification:</b> FAILED — {_xml(qa['note'])}. "
            "This translation has had no automated check.", body_style))
    return out




# ---------------------------------------------------------------------------
# Image-based translation (Modes B and C)
# ---------------------------------------------------------------------------

MAX_IMAGE_BYTES = 1_500_000  # 1.5 MB per image — still inside OVHcloud body limits
# Max dimension before downscaling. This is the single most important knob for
# OCR quality: too small and dense forms become illegible, at which point the
# model stops transcribing and starts repeating itself until it hits the output
# cap. 2200px on the long edge is roughly 190dpi for A4, which keeps 8pt form
# print legible. Raise it if dense documents still loop; lower it if requests
# start hitting body-size limits.
MAX_IMAGE_PIXELS = 2200


def _compress_to_jpeg(img: Image.Image, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """
    Convert a Pillow image to JPEG, downscaling and reducing quality until
    the encoded size is under max_bytes. JPEG is ~10x smaller than PNG for
    scanned documents and more than adequate for OCR/translation quality.
    """
    img = img.convert("RGB")

    # Downscale if either dimension exceeds MAX_IMAGE_PIXELS
    w, h = img.size
    if max(w, h) > MAX_IMAGE_PIXELS:
        scale = MAX_IMAGE_PIXELS / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Try decreasing quality until under size limit
    for quality in (85, 70, 55, 40):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            log.debug("Image compressed to %d KB at quality %d", len(data) // 1024, quality)
            return data

    # Last resort: halve dimensions
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40, optimize=True)
    return buf.getvalue()


def render_page_b64(pdf_bytes: bytes, page_index: int, dpi: int = RENDER_DPI) -> str:
    """Render a PDF page to a size-capped JPEG and return as base64."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    # Convert pymupdf pixmap → Pillow for compression control
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    jpeg_bytes = _compress_to_jpeg(img)

    # Log the size the model actually receives, not the pre-downscale size —
    # the difference is what determines whether it can read the page.
    sent = Image.open(io.BytesIO(jpeg_bytes))
    log.info("Page %d: rendered %dx%d → sent %dx%d, %d KB JPEG",
             page_index + 1, pix.width, pix.height,
             sent.width, sent.height, len(jpeg_bytes) // 1024)
    return base64.b64encode(jpeg_bytes).decode()


def image_file_to_b64(image_bytes: bytes, ext: str) -> str:
    """Normalise and compress an uploaded image, return as base64 JPEG."""
    img = Image.open(io.BytesIO(image_bytes))
    jpeg_bytes = _compress_to_jpeg(img)
    log.info("Uploaded image compressed to %d KB JPEG", len(jpeg_bytes) // 1024)
    return base64.b64encode(jpeg_bytes).decode()


def _parse_assessment(raw: str) -> tuple[str, dict]:
    """
    Split Mistral's Stage 1 response into (extracted_text, assessment_dict).
    The assessment JSON block is the last ```json ... ``` in the response.
    Falls back to safe defaults if parsing fails.
    """
    default_assessment = {
        "language": "unknown", "script": "unknown",
        "confidence": 1.0, "escalate": False,
        "reason": "Assessment not parsed — using Mistral for translation"
    }
    try:
        # Find the last JSON block
        matches = re.findall(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if matches:
            assessment = json.loads(matches[-1])
            # Strip the JSON block from the text
            text = re.sub(r"```json\s*\{.*?\}\s*```", "", raw, flags=re.DOTALL).strip()
            return text, assessment
    except (json.JSONDecodeError, Exception) as e:
        log.warning("Could not parse assessment JSON: %s", e)
    return raw, default_assessment


def _should_escalate(assessment: dict) -> tuple[bool, str]:
    """
    Decide whether to escalate to Llama based on:
    1. Hardcoded language list (always escalate)
    2. Mistral's self-reported confidence below threshold
    3. Mistral's own escalate flag
    Returns (escalate: bool, reason: str)
    """
    language = assessment.get("language", "").lower()

    # Check hardcoded escalation list
    for lang in ALWAYS_ESCALATE_LANGUAGES:
        if lang in language:
            return True, f"Language '{assessment.get('language')}' is in the always-escalate list"

    # Check Mistral's own escalate flag
    if assessment.get("escalate", False):
        return True, assessment.get("reason", "Mistral recommended escalation")

    # Check confidence threshold
    confidence = assessment.get("confidence", 1.0)
    if confidence < ESCALATION_CONFIDENCE_THRESHOLD:
        return True, f"Mistral confidence {confidence:.0%} below threshold {ESCALATION_CONFIDENCE_THRESHOLD:.0%}"

    return False, ""


TESSERACT_LANGS = "ukr+rus+bul+srp+mkd+bel+kaz+eng"  # Cyrillic-script + English

def _tesseract_available() -> bool:
    """True if the tesseract binary and at least the Ukrainian pack are present."""
    try:
        langs = subprocess.run(["tesseract", "--list-langs"],
                               capture_output=True, text=True, timeout=10)
        return "ukr" in langs.stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


# --- Scan preprocessing (OpenCV) --------------------------------------------
# Poor scans — phone photos of creased paper, low contrast, slight skew — read
# badly whether by Tesseract or the vision model. Gentle preprocessing (deskew,
# CLAHE contrast equalisation, light denoise) materially improves legibility.
# Two hard-won rules from testing, both enforced below:
#   1. Preprocessing that rescues a BAD page DEGRADES a good one (adaptive
#      thresholding especially). So it is applied per page and kept ONLY when it
#      increases the confident-word yield. A clean page keeps its raw pixels.
#   2. The right metric is word YIELD (how many words read at conf>=40), not
#      mean confidence — raising the average by dropping hard words is worse,
#      not better, and mean confidence rewards exactly that.
# If OpenCV is unavailable, everything passes through unchanged.

try:
    import cv2 as _cv2
    import numpy as _np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _deskew_gray(gray):
    """Correct small skew estimated from text-pixel orientation."""
    inv = _cv2.bitwise_not(gray)
    thr = _cv2.threshold(inv, 0, 255, _cv2.THRESH_BINARY | _cv2.THRESH_OTSU)[1]
    coords = _np.column_stack(_np.where(thr > 0))
    if len(coords) < 100:
        return gray, 0.0
    angle = _cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3 or abs(angle) > 20:   # ignore noise / implausible skew
        return gray, 0.0
    h, w = gray.shape
    M = _cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return _cv2.warpAffine(gray, M, (w, h), flags=_cv2.INTER_CUBIC,
                           borderMode=_cv2.BORDER_REPLICATE), angle


def _enhance_gray(gray):
    """Gentle, non-destructive enhancement: deskew + CLAHE + light denoise."""
    g, ang = _deskew_gray(gray)
    g = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = _cv2.fastNlMeansDenoising(g, h=5)
    return g, ang


def _word_yield(gray) -> int:
    """Confident-word count (conf>=40) — the legibility metric that matters."""
    try:
        import pytesseract
        from pytesseract import Output
        d = pytesseract.image_to_data(gray, output_type=Output.DICT)
        return sum(1 for c, t in zip(d["conf"], d["text"])
                   if t.strip() and int(c) >= 40)
    except Exception:
        return -1   # can't measure (no pytesseract) — signal "unknown"


def preprocess_image_bytes(img_bytes: bytes, page_label: str = "") -> bytes:
    """
    Return possibly-enhanced JPEG bytes for one page image.

    Applies enhancement only when it measurably improves the confident-word
    yield; otherwise returns the original bytes unchanged. Falls back to the
    original on any error or when OpenCV/pytesseract are unavailable, so this is
    always safe to call.
    """
    if not _CV2_AVAILABLE:
        return img_bytes
    try:
        arr = _np.frombuffer(img_bytes, dtype=_np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if img is None:
            return img_bytes
        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)

        raw_y = _word_yield(gray)
        enh, ang = _enhance_gray(gray)
        enh_y = _word_yield(enh)

        # If we can't measure (no pytesseract), be conservative and keep raw.
        if raw_y < 0 or enh_y < 0:
            log.info("Preprocess%s: cannot measure yield — keeping raw",
                     f" ({page_label})" if page_label else "")
            return img_bytes

        # Keep enhanced only if it yields meaningfully more readable words.
        if enh_y > raw_y * 1.05:
            ok, buf = _cv2.imencode(".jpg", enh, [_cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                log.info("Preprocess%s: enhanced (%d→%d words, skew %.2f°) — using",
                         f" ({page_label})" if page_label else "",
                         raw_y, enh_y, ang)
                return buf.tobytes()
        log.info("Preprocess%s: raw better or equal (%d vs %d words) — keeping raw",
                 f" ({page_label})" if page_label else "", raw_y, enh_y)
        return img_bytes
    except Exception as e:
        log.warning("Preprocess%s failed (%s) — keeping raw",
                    f" ({page_label})" if page_label else "", e)
        return img_bytes


def ocr_with_tesseract(b64_images: list[str], page_label: str = "") -> str:
    """
    Deterministic local OCR via Tesseract, for Cyrillic-script scans that the
    vision model mangles. Returns extracted text (no assessment — the caller
    supplies language separately).

    Tesseract returns what is actually on the page rather than fabricating, so
    it both reads Cyrillic far better than Mistral-Small AND cannot produce the
    129k-character hallucination that a poor scan provokes from a vision LLM.
    Runs entirely on-box: no data leaves Ancilla, so no GDPR consideration.
    """
    import base64 as _b64
    texts = []
    for i, img in enumerate(b64_images):
        img_bytes = _b64.b64decode(img)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(img_bytes)
            tmp_path = tf.name
        try:
            proc = subprocess.run(
                ["tesseract", tmp_path, "stdout", "-l", TESSERACT_LANGS, "--psm", "3"],
                capture_output=True, text=True, timeout=OCR_TIMEOUT_SECONDS,
            )
            texts.append(proc.stdout)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    result = "\n\n---PAGE BREAK---\n\n".join(texts)
    log.info("Tesseract OCR%s: %d char(s) from %d page(s)",
             f" ({page_label})" if page_label else "", len(result), len(b64_images))
    return result


def ocr_images_with_assessment(b64_images: list[str], source_hint: str,
                               page_label: str = "") -> tuple[str, dict]:
    """
    Stage 1: Send page images to Mistral for OCR + language assessment.
    Returns (extracted_text, assessment_dict).

    Bounded three ways against poor scans:
      - a hard per-call timeout (OCR_TIMEOUT_SECONDS), so a struggling call
        fails in minutes rather than hanging on the client's 10-minute default;
      - a per-page character ceiling, so implausibly long (fabricated) output is
        rejected as an unreadable scan rather than passed downstream;
      - a Tesseract fallback for Cyrillic, retried locally before giving up.
    Raises TranslationTruncated on token-cap overflow, UnreadableScan on a scan
    too poor to OCR reliably.
    """
    hint = f"This is a {source_hint}. " if source_hint else ""

    # Preprocess each page: enhance poor scans, leave good ones untouched. Both
    # the Mistral call below and the Tesseract fallback use these versions.
    import base64 as _b64
    enhanced_b64 = []
    for i, img in enumerate(b64_images):
        pl = f"{page_label} img{i+1}" if page_label else f"img{i+1}"
        try:
            raw_bytes = _b64.b64decode(img)
            proc_bytes = preprocess_image_bytes(raw_bytes, pl)
            enhanced_b64.append(_b64.b64encode(proc_bytes).decode()
                                if proc_bytes is not raw_bytes else img)
        except Exception:
            enhanced_b64.append(img)
    b64_images = enhanced_b64

    content = []
    for img in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"},
        })
    content.append({
        "type": "text",
        "text": hint + OCR_ASSESSMENT_PROMPT,
    })
    log.info("Stage 1: OCR + assessment via %s (%d images%s)",
             MODEL_MISTRAL, len(b64_images), f", {page_label}" if page_label else "")
    response = ovh_client.chat.completions.create(
        model=MODEL_MISTRAL,
        messages=[{"role": "user", "content": content}],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.1,
        timeout=OCR_TIMEOUT_SECONDS,
    )
    _check_not_truncated(response, "OCR", page_label)

    raw = response.choices[0].message.content.strip()
    extracted_text, assessment = _parse_assessment(raw)

    # Poor-scan guard: implausibly long output means the model is fabricating,
    # not reading. Try Tesseract for Cyrillic before failing.
    ceiling = MAX_OCR_CHARS_PER_PAGE * max(1, len(b64_images))
    if len(extracted_text) > ceiling:
        log.warning("OCR produced %d chars for %d page(s) (ceiling %d) — likely a "
                    "poor scan provoking fabricated output%s",
                    len(extracted_text), len(b64_images), ceiling,
                    f" ({page_label})" if page_label else "")
        script = str(assessment.get("script", "")).lower()
        language = str(assessment.get("language", "")).lower()
        cyrillic = "cyrillic" in script or language in (
            "ukrainian", "russian", "bulgarian", "serbian", "macedonian",
            "belarusian", "kazakh")
        if cyrillic and _tesseract_available():
            log.info("Falling back to Tesseract for Cyrillic scan%s",
                     f" ({page_label})" if page_label else "")
            t_text = ocr_with_tesseract(b64_images, page_label)
            if t_text.strip() and len(t_text) <= ceiling:
                # Keep Mistral's language assessment; use Tesseract's text.
                assessment["ocr_engine"] = "tesseract"
                return t_text, assessment
            log.warning("Tesseract fallback did not yield usable text%s",
                        f" ({page_label})" if page_label else "")
        raise UnreadableScan(
            f"The document could not be read reliably by OCR"
            f"{' (' + page_label + ')' if page_label else ''}. This usually means "
            "the scan quality is too low — the text is blurred, skewed, low-"
            "contrast or creased. A clearer scan, or manual translation, is "
            "needed. (No translation has been produced, to avoid returning "
            "fabricated content.)"
        )

    log.info("Stage 1: extracted %d chars | language=%s confidence=%.2f escalate=%s",
             len(extracted_text), assessment.get("language"),
             assessment.get("confidence", 0), assessment.get("escalate"))
    return extracted_text, assessment


def _clean_ocr_text(text: str) -> str:
    """
    Remove headers and markers added by the OCR prompt before passing to translation.
    These confuse the translation model into thinking the document is already processed.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("AI Translation of original document"):
            continue
        if stripped.startswith("```json") or stripped.startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _split_on_words(text: str, max_chars: int) -> list[str]:
    """
    Last-resort splitter for a single line longer than max_chars: break on word
    boundaries. Only reached by pathological input (e.g. an OCR pass that
    returned a whole page with no line breaks).
    """
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for word in text.split(" "):
        if buf_len + len(word) + 1 > max_chars and buf:
            out.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(word)
        buf_len += len(word) + 1
    if buf:
        out.append(" ".join(buf))
    return out


def _split_for_translation(text: str, max_chars: int = MAX_TRANSLATE_CHARS) -> list[str]:
    """
    Split text into chunks no larger than max_chars, preferring the least
    disruptive boundary available: paragraphs first, then lines, then words.
    This guarantees every chunk is small enough to translate in one call,
    regardless of how the OCR pass happened to format the text.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0

    for para in text.split("\n\n"):
        para_len = len(para) + 2  # account for the separator

        if para_len > max_chars:
            # Oversized paragraph: flush, then break it down by line, and by
            # word for any line that is itself too long.
            flush()
            line_buf: list[str] = []
            line_len = 0
            for line in para.splitlines():
                if len(line) + 1 > max_chars:
                    if line_buf:
                        chunks.append("\n".join(line_buf))
                        line_buf, line_len = [], 0
                    chunks.extend(_split_on_words(line, max_chars))
                    continue
                if line_len + len(line) + 1 > max_chars and line_buf:
                    chunks.append("\n".join(line_buf))
                    line_buf, line_len = [], 0
                line_buf.append(line)
                line_len += len(line) + 1
            if line_buf:
                chunks.append("\n".join(line_buf))
            continue

        if current_len + para_len > max_chars and current:
            flush()
        current.append(para)
        current_len += para_len

    flush()
    return chunks


# ---------------------------------------------------------------------------
# DeepL translation engine (primary) — falls back to the model path below
# ---------------------------------------------------------------------------

class DeepLUnavailable(RuntimeError):
    """
    DeepL could not produce a usable translation. Never fatal: the caller logs
    the reason, records it for the provenance line, and translates with the
    OVHcloud models instead.
    """


_deepl_skip_until = 0.0            # cooldown after an outage / auth / quota failure
_deepl_source_names: list[str] | None = None
_MARKER_RE = re.compile(r"\[(?:ILLEGIBLE|UNCLEAR)\]", re.IGNORECASE)
_LIST_START_RE = re.compile(r"^\s*([\-•·\*–—]|\(?\w{1,3}[.)]\s|\d+[.):]?\s)")


def _deepl_target_code(target_lang: str) -> str | None:
    key = target_lang.strip().lower()
    if key in DEEPL_TARGET_CODES:
        return DEEPL_TARGET_CODES[key]
    if re.fullmatch(r"[a-z]{2}(-[a-z]{2,4})?", key):   # already a code, e.g. "de", "pt-br"
        return {"EN": DEEPL_ENGLISH_VARIANT, "PT": "PT-PT"}.get(key.upper(), key.upper())
    return None


def _deepl_supports_source(language: str) -> bool:
    """
    True unless the OCR assessment named a language DeepL does not list as a
    source. An unknown/empty language is allowed through — DeepL auto-detects,
    and the untranslated-output check catches a miss.
    """
    global _deepl_source_names
    name = (language or "").strip().lower()
    if not name or name == "unknown":
        return True
    if _deepl_source_names is None:
        _deepl_source_names = [l.name.lower().split()[0]
                               for l in deepl_client.get_source_languages()]
    for alias, canonical in DEEPL_SOURCE_ALIASES.items():
        if alias in name:
            name += " " + canonical
    # An empty list means the lookup told us nothing — let DeepL try.
    return not _deepl_source_names or any(n in name for n in _deepl_source_names)


def _unwrap_soft_breaks(text: str) -> str:
    """
    Join lines that are one sentence hard-wrapped by the PDF or the OCR pass.

    DeepL treats every newline as a sentence boundary (which is what keeps
    lists and table rows on their own lines), so a paragraph wrapped at the
    page margin would otherwise be translated as disconnected fragments. A
    line is joined to the next only when it is prose-length for its paragraph,
    does not end a sentence, and the next line does not start a list item —
    short lines (headings, list items, table cells, lab rows) are left alone.
    """
    out_paras = []
    for para in text.split("\n\n"):
        lines = para.split("\n")
        longest = max((len(l) for l in lines), default=0)
        min_len = max(40, int(longest * 0.7))
        joined: list[str] = []
        for line in lines:
            prev = joined[-1] if joined else ""
            if (prev and line.strip()
                    and len(prev) >= min_len
                    and not prev.rstrip().endswith((".", ":", ";", "!", "?", "。"))
                    and not _LIST_START_RE.match(line)):
                if prev.endswith("-") and prev[-2:-1].isalpha() and line[:1].islower():
                    joined[-1] = prev[:-1] + line.lstrip()      # de-hyphenate
                else:
                    joined[-1] = prev.rstrip() + " " + line.lstrip()
            else:
                joined.append(line)
        out_paras.append("\n".join(joined))
    return "\n\n".join(out_paras)


def deepl_attestation(translated_text: str) -> str:
    """Engine-accurate replacement for the attestation the models append."""
    return (
        "Translator's attestation: This translation was produced by DeepL machine "
        "translation directly from the text of the source document; nothing was "
        "added or summarised. Items marked [ILLEGIBLE] or [UNCLEAR] indicate text "
        "that could not be read with certainty. Number of such items: "
        f"{len(_MARKER_RE.findall(translated_text))}."
    )


def translate_with_deepl(text: str, target_lang: str, source_language: str = "") -> str:
    """
    Translate text with DeepL, preserving line and paragraph structure and the
    '---PAGE BREAK---' markers. Raises DeepLUnavailable for anything short of a
    complete, plausibly-translated result; the caller then uses the model path.
    Does not append an attestation — callers add deepl_attestation() once.
    """
    global _deepl_skip_until

    if deepl_client is None:
        raise DeepLUnavailable("DeepL is not configured")
    if time.time() < _deepl_skip_until:
        raise DeepLUnavailable("DeepL is paused after a recent failure")

    target_code = _deepl_target_code(target_lang)
    if not target_code:
        raise DeepLUnavailable(f"target language '{target_lang}' has no DeepL code")

    pages = _clean_ocr_text(text).split("---PAGE BREAK---")
    out_pages: list[str] = []
    n_requests = 0
    try:
        if not _deepl_supports_source(source_language):
            raise DeepLUnavailable(f"DeepL does not list '{source_language}' as a source language")

        for page in pages:
            out_chunks: list[str] = []
            for chunk in _split_for_translation(_unwrap_soft_breaks(page.strip("\n")),
                                                DEEPL_MAX_CHARS):
                if not chunk.strip():
                    out_chunks.append(chunk)
                    continue
                result = deepl_client.translate_text(
                    chunk, target_lang=target_code,
                    preserve_formatting=True,      # no silent punctuation/case "fixes"
                    split_sentences="1",           # newlines are boundaries and are kept
                    model_type=DEEPL_MODEL_TYPE or None,
                )
                n_requests += 1
                if not result.text.strip():
                    raise DeepLUnavailable("DeepL returned empty text for a non-empty chunk")
                out_chunks.append(result.text)
            out_pages.append("\n\n".join(out_chunks))

    except DeepLUnavailable:
        raise
    except Exception as e:                         # deepl.DeepLException, network, anything
        # Outage, bad key or exhausted quota: do not make every remaining
        # batch of this (and the next) document wait on the same failure.
        _deepl_skip_until = time.time() + DEEPL_COOLDOWN_SECONDS
        raise DeepLUnavailable(f"{type(e).__name__}: {e}"[:300]) from e

    translated = "\n\n---PAGE BREAK---\n\n".join(out_pages)
    log.info("DeepL: %d chars in -> %d chars out (%s, %d request%s)",
             len(text), len(translated), target_code, n_requests,
             "" if n_requests == 1 else "s")
    return translated


def try_deepl(text: str, target_lang: str, assessment: dict,
              model_override: str = "") -> str | None:
    """
    Attempt DeepL. Returns the translation, or None after recording why not in
    assessment["deepl_fallback_reason"] (shown on the PDF provenance line).
    An explicit model=mistral|llama request bypasses DeepL without comment.
    """
    if model_override or TRANSLATE_ENGINE != "deepl":
        return None
    source_language = str(assessment.get("language", ""))
    try:
        translated = translate_with_deepl(text, target_lang, source_language)
        if _looks_untranslated(translated, source_language, text, search_whole=True):
            raise DeepLUnavailable("DeepL output still contained the source text")
    except DeepLUnavailable as e:
        log.warning("DeepL not used — falling back to OVHcloud models: %s", e)
        assessment["deepl_fallback_reason"] = str(e)
        return None
    assessment["translation_model"] = DEEPL_ENGINE_NAME
    assessment.pop("escalation_reason", None)
    return translated


def _is_deepl_output(assessment: dict | None) -> bool:
    return str((assessment or {}).get("translation_model", "")).startswith(DEEPL_ENGINE_NAME)


def _provenance_notes(assessment: dict) -> str:
    """Suffix for the PDF provenance line: escalation and/or DeepL fallback."""
    notes = ""
    if assessment.get("escalation_reason"):
        notes += f" (escalated — {assessment['escalation_reason']})"
    if assessment.get("deepl_fallback_reason"):
        notes += f" | DeepL not used: {assessment['deepl_fallback_reason']}"
    return notes


def translate_text_with_model(text: str, target_lang: str, model: str) -> str:
    """
    Translate extracted text using the specified model.

    Long text is split into MAX_TRANSLATE_CHARS chunks and translated
    sequentially, so document length is bounded only by time, not by the
    per-call output cap. Every response is checked for truncation — if any
    chunk is cut off, the whole job fails rather than returning a partial
    translation with a clean-looking attestation on the end.

    Only the final chunk is asked for the attestation, so it appears once.
    """
    clean_text = _clean_ocr_text(text)
    chunks = _split_for_translation(clean_text)

    log.info("Translating with %s → %s (%d chars in %d chunk%s)",
             model, target_lang, len(clean_text), len(chunks),
             "" if len(chunks) == 1 else "s")

    translated_parts: list[str] = []

    for i, chunk in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        position = (
            "" if len(chunks) == 1
            else f"This is part {i + 1} of {len(chunks)} of a longer document. "
                 "Translate this part only; do not summarise or add a preamble. "
        )
        tail = (
            "Append the translator's attestation at the end." if is_last
            else "Do NOT append the translator's attestation — this is not the "
                 "final part of the document."
        )

        response = ovh_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(target_lang=target_lang)},
                {"role": "user", "content": (
                    f"The text below is raw OCR output from a scanned medical document. "
                    f"It has NOT been translated. Your task is to translate it into {target_lang}.\n\n"
                    f"{position}"
                    f"IMPORTANT: Your response must consist entirely of {target_lang} text. "
                    f"Do not reproduce any of the source language text in your response — "
                    f"not even partially. Every sentence must be a translation, not a copy.\n\n"
                    "Apply all fidelity rules from your system instructions. "
                    "Separate pages with '---PAGE BREAK---'. "
                    f"{tail}\n\n"
                    f"RAW OCR TEXT TO TRANSLATE:\n{chunk}"
                )},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.1,
        )
        _check_not_truncated(
            response, "Translation",
            f"chunk {i + 1} of {len(chunks)}, {len(chunk)} chars in",
        )

        part = response.choices[0].message.content.strip()
        log.info("Translation chunk %d/%d: %d chars in → %d chars out",
                 i + 1, len(chunks), len(chunk), len(part))
        translated_parts.append(part)

    result = "\n\n".join(translated_parts)
    log.info("translate_text_with_model: %d chars total", len(result))
    return result


# --- Untranslated-output detection -----------------------------------------
# Tuning for _looks_untranslated. The goal is to distinguish "the model echoed
# the source back at us" from "the model correctly preserved proper nouns".
_PROBE_COUNT           = 5     # How many prose lines to sample
_PROBE_LEN             = 60    # Chars of each line used as the probe
_MIN_PROBES            = 2     # Below this, skip the check rather than guess
_MIN_CANDIDATE_CHARS   = 40    # Prose lines are long; headings are short
_MIN_CANDIDATE_WORDS   = 6     # A name or heading is rarely 6+ words
_MIN_LOWERCASE_RATIO   = 0.5   # Excludes ALL-CAPS letterheads
_MAX_DIGIT_RATIO       = 0.30  # Excludes vitals/lab rows, which survive as-is


def _prose_candidates(text: str) -> list[str]:
    """
    Pick lines of running prose suitable for use as translation probes.

    Excluded deliberately, because a faithful translation legitimately leaves
    them unchanged and matching on them would produce false positives:
      - short lines: headings, labels, field names
      - ALL-CAPS lines: letterheads and institution names
      - digit-heavy lines: vital signs, lab results, dosages, dates
      - our own OCR markers
    What survives is narrative clinical text, which any genuine translation
    must rewrite entirely.
    """
    candidates: list[str] = []

    for line in text.splitlines():
        s = line.strip()

        if len(s) < _MIN_CANDIDATE_CHARS:
            continue
        if s.startswith("AI Translation") or s.startswith("---"):
            continue
        if len(s.split()) < _MIN_CANDIDATE_WORDS:
            continue

        alpha = [c for c in s if c.isalpha()]
        if not alpha:
            continue
        if sum(1 for c in alpha if c.islower()) / len(alpha) < _MIN_LOWERCASE_RATIO:
            continue   # ALL-CAPS: letterhead, section banner, institution name

        if sum(1 for c in s if c.isdigit()) / len(s) > _MAX_DIGIT_RATIO:
            continue   # numeric row: lab values, vitals, dosages

        candidates.append(s)

    return candidates


def _looks_untranslated(text: str, source_language: str, extracted_text: str = "",
                        search_whole: bool = False) -> bool:
    """
    Check whether the translation output still contains the source text verbatim.

    Naive approaches false-positive badly. Proper nouns are *correctly* left
    untranslated — "UNIDADE LOCAL DE SAÚDE ALGARVE" is a hospital's name, and a
    faithful English translation will still contain it. Drug names, place names
    and patient names behave the same way. So we deliberately do not probe on
    headings or name-like lines.

    Instead we sample several *prose* lines spread through the source — running
    clinical narrative, which any real translation must rewrite — and require a
    majority of them to survive verbatim before calling the output untranslated.
    One surviving line is normal; most of them surviving is not.
    """
    if not text or not extracted_text or len(extracted_text) < 30:
        return False

    candidates = _prose_candidates(extracted_text)
    if len(candidates) < _MIN_PROBES:
        # Not enough running prose to judge (e.g. a form or a results table).
        # Better to let it through than to fail a valid translation.
        log.info("_looks_untranslated: only %d prose candidate(s) — skipping check",
                 len(candidates))
        return False

    # Sample evenly across the document so we are not judging on one section
    step   = max(1, len(candidates) // _PROBE_COUNT)
    probes = [c[:_PROBE_LEN] for c in candidates[::step]][:_PROBE_COUNT]

    # The originals may be appended after the translation IN THE SAME TEXT, so
    # by default only search the first half — a verbatim copy down there is
    # expected. Mode A returns the translation alone (originals are a separate
    # appended PDF), so it passes search_whole=True to check everything.
    search_in = text if search_whole else text[:len(text) // 2]

    matched = [p for p in probes if p in search_in]
    threshold = max(2, (len(probes) + 1) // 2)   # majority, minimum of 2

    if len(matched) >= threshold:
        log.warning("_looks_untranslated: %d of %d prose probes found verbatim in the "
                    "output (threshold %d) — translation likely did not happen. "
                    "Source language: %s. First match: %r",
                    len(matched), len(probes), threshold, source_language, matched[0])
        return True

    log.info("_looks_untranslated: %d of %d prose probes matched (threshold %d) — "
             "translation looks genuine", len(matched), len(probes), threshold)
    return False


def _page_label(first_page: int, count: int) -> str:
    return (f"page {first_page}" if count == 1
            else f"pages {first_page}-{first_page + count - 1}")


def ocr_batch_adaptive(b64_images: list[str], source_hint: str,
                       first_page: int = 1) -> tuple[str, dict]:
    """
    OCR a batch of pages, halving the batch and retrying if the output overflows.

    Dense pages — closely typed forms, multi-column layouts — can produce far
    more text than sparse ones, so no fixed batch size is right for every
    document. Rather than failing a job because four pages happened to be
    dense, we split and retry, down to a single page.

    Only a single page that still overflows is genuinely fatal: that means one
    page alone exceeds the output limit, which in practice means the model is
    repeating itself rather than transcribing.
    """
    label = _page_label(first_page, len(b64_images))

    try:
        return ocr_images_with_assessment(b64_images, source_hint, label)

    except TranslationTruncated:
        if len(b64_images) == 1:
            raise TranslationTruncated(
                f"OCR of {label} overflowed the {MAX_OUTPUT_TOKENS}-token output "
                f"limit even as a single page. The page is either extraordinarily "
                f"dense or the model is looping. This job has been failed rather "
                f"than returning an incomplete transcription."
            )

        mid = len(b64_images) // 2
        log.warning("OCR of %s overflowed — splitting into %s and %s and retrying",
                    label,
                    _page_label(first_page, mid),
                    _page_label(first_page + mid, len(b64_images) - mid))

        text_a, assess_a = ocr_batch_adaptive(b64_images[:mid], source_hint, first_page)
        text_b, assess_b = ocr_batch_adaptive(b64_images[mid:], source_hint, first_page + mid)

        # Keep the first usable language assessment — the source language does
        # not change halfway through a document.
        assessment = assess_a if assess_a.get("language") else assess_b
        return f"{text_a}\n\n---PAGE BREAK---\n\n{text_b}", assessment


def translate_images(b64_images: list[str], target_lang: str, source_hint: str,
                     model_override: str = "", first_page: int = 1) -> tuple[str, dict, bool]:
    """
    Two-stage image translation pipeline for one batch of page images.
    Returns (translated_text, assessment, was_escalated, source_text).
    model_override: "mistral" or "llama" to bypass auto-selection.
    first_page: 1-based page number of the first image, for log and error text.
    """
    # Stage 1: OCR + assessment (always Mistral — only multimodal model),
    # splitting the batch automatically if the output overflows.
    extracted_text, assessment = ocr_batch_adaptive(
        b64_images, source_hint, first_page
    )

    # Determine translation model
    if model_override == "llama":
        escalate, reason = True, "Manual override: Llama requested"
    elif model_override == "mistral":
        escalate, reason = False, "Manual override: Mistral requested"
    else:
        escalate, reason = _should_escalate(assessment)

    source_language = assessment.get("language", "unknown")
    tx_model = MODEL_LLAMA if escalate else MODEL_MISTRAL

    if escalate:
        log.info("Stage 2: escalating to %s — %s", MODEL_LLAMA, reason)
    else:
        log.info("Stage 2: translating with %s (no escalation needed)", MODEL_MISTRAL)

    assessment["translation_model"] = tx_model
    if escalate:
        assessment["escalation_reason"] = reason

    # DeepL first; the model path below is the fallback.
    translation = try_deepl(extracted_text, target_lang, assessment, model_override)
    if translation is not None:
        return translation, assessment, False, extracted_text

    # Attempt translation — retry once with Llama if output looks untranslated
    for attempt in range(2):
        model_to_use = MODEL_LLAMA if (escalate or attempt > 0) else MODEL_MISTRAL
        if attempt > 0:
            log.warning("Stage 2: translation appeared untranslated — retrying with %s", MODEL_LLAMA)
            assessment["translation_model"] = MODEL_LLAMA
            assessment["escalation_reason"] = "Retry: initial translation appeared untranslated"

        translation = translate_text_with_model(extracted_text, target_lang, model_to_use)

        if not _looks_untranslated(translation, source_language, extracted_text):
            break
        log.warning("Stage 2: output still looks like %s after attempt %d",
                    source_language, attempt + 1)

    return translation, assessment, escalate, extracted_text


QA_SYSTEM_PROMPT = (
    "You are a bilingual medical translation verifier. You are given the source "
    "text of a medical document and its draft translation. Your task is to find "
    "clinically material discrepancies — errors that could change how a clinician "
    "reads the document. You are NOT a stylist: wording preferences, register, "
    "and phrasing differences are not findings.\n\n"
    "Check specifically for:\n"
    "- mistranslated diagnoses or section headings (a heading error misleads the "
    "reader about everything beneath it)\n"
    "- drug names, doses, routes, frequencies that differ from the source\n"
    "- laboratory values, units, reference ranges, or decimal separators that "
    "differ from the source\n"
    "- negations reversed (e.g. 'no known allergies' vs 'known allergies')\n"
    "- dates, times, or identifiers that differ\n"
    "- content present in the source but absent from the translation, or invented "
    "in the translation with no source basis\n"
    "- institutional or blood-product terms rendered into nonsense by literal "
    "word-by-word translation\n\n"
    "Return ONLY a JSON array (no prose, no markdown fences). Each finding:\n"
    '{"severity": "critical" or "minor", "source_excerpt": "...", '
    '"translation_excerpt": "...", "issue": "one sentence"}\n'
    "severity is critical when acting on the error could affect patient care.\n"
    "If the translation is faithful, return []"
)


def verify_translation(source_text: str, translation: str, source_language: str,
                       target_lang: str,
                       aligned_pairs: list[tuple[str, str]] | None = None) -> dict:
    """
    Second-model review of a draft translation against its source.

    Checking is an easier task than translating — a focused rubric at
    temperature 0 catches slips (phonetic heading guesses, dropped sections,
    unit substitutions) that the translating model itself made. Llama 70B is
    used regardless of which model translated; when Llama translated, this is
    a self-check, which is weaker but still catches inattention errors.

    Never raises: QA is advisory, and its own failure must not take down a
    completed translation. The status field is always honest about what
    happened — clean, findings, failed, or skipped — and that status is
    printed on the output so a reader never assumes a check that didn't run.

    Returns {"status", "findings", "note"}.
    """
    try:
        if aligned_pairs is not None:
            # Mode A: source and translation are block-aligned — present them
            # as pairs, which makes omissions and per-block errors obvious.
            lines = []
            for n, (s, t) in enumerate(aligned_pairs, 1):
                lines.append(f"{n}| SOURCE: {s}")
                lines.append(f"{n}| TRANSLATION: {t}")
            payload = (
                f"The document was translated block by block from "
                f"{source_language or 'the source language'} into {target_lang}. "
                "Each numbered pair is one block.\n\n" + "\n".join(lines)
            )
        else:
            payload = (
                f"SOURCE DOCUMENT ({source_language or 'source language'}):\n"
                f"{source_text}\n\n"
                f"DRAFT TRANSLATION ({target_lang}):\n{translation}"
            )

        if len(payload) > 180_000:
            payload = payload[:180_000]
            coverage_note = " (document truncated for verification — very long input)"
        else:
            coverage_note = ""

        log.info("QA: verifying translation (%d chars payload) via %s",
                 len(payload), MODEL_LLAMA)
        response = ovh_client.chat.completions.create(
            model=MODEL_LLAMA,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user",   "content": payload},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        try:
            finish = response.choices[0].finish_reason
        except (AttributeError, IndexError):
            finish = None
        if finish == "length":
            return {"status": "failed", "findings": [],
                    "note": "verification output overflowed — treat as not verified"}

        clean = re.sub(r"```json|```", "", raw).strip()
        if not clean.startswith("["):
            m = re.search(r"\[.*\]", clean, re.DOTALL)
            clean = m.group(0) if m else clean
        parsed = json.loads(clean)
        if not isinstance(parsed, list):
            raise ValueError("verifier did not return a JSON array")

        findings = []
        for item in parsed:
            if isinstance(item, dict) and item.get("issue"):
                findings.append({
                    "severity": str(item.get("severity", "minor")).lower(),
                    "source_excerpt": str(item.get("source_excerpt", ""))[:150],
                    "translation_excerpt": str(item.get("translation_excerpt", ""))[:150],
                    "issue": str(item.get("issue", ""))[:250],
                })

        critical = sum(1 for f in findings if f["severity"] == "critical")
        log.info("QA: %d finding(s), %d critical%s",
                 len(findings), critical, coverage_note)

        if findings:
            return {"status": "findings", "findings": findings,
                    "note": f"{len(findings)} finding(s), {critical} critical"
                            + coverage_note}
        return {"status": "clean", "findings": [],
                "note": "no clinically material discrepancies found" + coverage_note}

    except Exception as e:
        log.warning("QA: verification failed: %s", e)
        return {"status": "failed", "findings": [],
                "note": f"verification could not be completed ({type(e).__name__}) "
                        "— treat as not verified"}


def run_qa(source_text: str, translation: str, assessment: dict,
           target_lang: str, qa_mode: str,
           aligned_pairs: list[tuple[str, str]] | None = None) -> None:
    """
    Decide whether to verify, run it, and record the result in
    assessment["qa"] for the PDF builders to render.
    qa_mode: "auto" (skip exempt languages), "on", or "off".
    """
    language = str(assessment.get("language", "")).lower()

    if qa_mode == "off":
        assessment["qa"] = {"status": "skipped", "findings": [],
                            "note": "disabled by request"}
    elif qa_mode != "on" and language in QA_EXEMPT_LANGUAGES:
        assessment["qa"] = {"status": "skipped", "findings": [],
                            "note": f"{language.title()} is exempt (reliably translated)"}
    else:
        assessment["qa"] = verify_translation(
            source_text, translation, language, target_lang, aligned_pairs
        )


def get_all_page_images(pdf_bytes: bytes) -> list[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    doc.close()
    images = []
    for i in range(total):
        images.append(render_page_b64(pdf_bytes, i))
    return images


def translate_all_images(b64_images: list[str], target_lang: str, source_hint: str,
                         model_override: str = "") -> tuple[list[str], dict, bool]:
    """
    Translate all page images, batching MAX_IMAGES_PER_CALL pages at a time.

    Batching is what makes long documents work: a 30-page document becomes 8
    OCR calls of 4 pages each, every one comfortably inside the output cap,
    rather than one call that silently truncates. Any batch that does hit the
    cap raises TranslationTruncated and fails the whole job.

    Returns (text_parts, assessment, was_escalated, source_text). text_parts is a list of
    translated fragments in document order — the PDF builder joins and reflows
    them, so the list length carries no per-page meaning.
    """
    total_pages   = len(b64_images)
    total_batches = (total_pages + MAX_IMAGES_PER_CALL - 1) // MAX_IMAGES_PER_CALL

    all_text_parts: list[str] = []
    all_source_parts: list[str] = []
    final_assessment: dict = {}
    was_escalated = False
    engines: list[str] = []          # translation engine used for each batch
    fallback_reason = ""

    log.info("translate_all_images: %d page(s) in %d batch(es) of up to %d",
             total_pages, total_batches, MAX_IMAGES_PER_CALL)

    for batch_no, start in enumerate(range(0, total_pages, MAX_IMAGES_PER_CALL), 1):
        batch = b64_images[start:start + MAX_IMAGES_PER_CALL]
        first_page = start + 1
        page_label = _page_label(first_page, len(batch))

        log.info("Batch %d/%d: %s", batch_no, total_batches, page_label)

        text, assessment, escalated, batch_source = translate_images(
            batch, target_lang, source_hint, model_override, first_page
        )
        all_source_parts.append(batch_source)

        if not text.strip():
            raise TranslationTruncated(
                f"Translation returned no content for {page_label}. "
                "This job has been failed rather than producing a partial document."
            )

        if not final_assessment:
            final_assessment = assessment   # First batch's language assessment
        engines.append(assessment.get("translation_model", ""))
        if assessment.get("deepl_fallback_reason"):
            fallback_reason = assessment["deepl_fallback_reason"]
        was_escalated = was_escalated or escalated
        all_text_parts.extend(
            part for part in text.split("---PAGE BREAK---") if part.strip()
        )

    # Provenance must describe the whole document, not just the first batch:
    # DeepL may have failed part-way and handed later batches to the models.
    used = list(dict.fromkeys(e for e in engines if e))
    if len(used) > 1:
        final_assessment["translation_model"] = " + ".join(used)
    if fallback_reason:
        final_assessment["deepl_fallback_reason"] = fallback_reason
    if DEEPL_ENGINE_NAME in used:
        # The models append their own attestation; DeepL text gets one, once.
        all_text_parts.append(deepl_attestation("\n".join(all_text_parts)))

    total_chars = sum(len(p) for p in all_text_parts)
    log.info("translate_all_images: %d fragment(s), %d chars total from %d page(s)",
             len(all_text_parts), total_chars, total_pages)

    return all_text_parts, final_assessment, was_escalated, "\n\n".join(all_source_parts)



# ---------------------------------------------------------------------------
# Output PDF builders for Modes B and C
# ---------------------------------------------------------------------------
#
# Layout for both modes:
#   SECTION 1 — Full English translation, one page per source page, portrait A4.
#               Text flows naturally; no thumbnails. Header on each page.
#   SECTION 2 — Original document images appended at the end, one per page,
#               labelled "Original document — page N".
#   The two sections are joined into a single PDF via pymupdf.
# ---------------------------------------------------------------------------

def _b64_to_rl_image(b64: str, max_width_pt: float, max_height_pt: float) -> RLImage:
    """Scale a base64 image to fit within the given bounds, return ReportLab Image."""
    img_bytes = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(img_bytes))
    w_px, h_px = pil.size
    scale = min(max_width_pt / w_px, max_height_pt / h_px)
    return RLImage(io.BytesIO(img_bytes), width=w_px * scale, height=h_px * scale)


def _xml(text: str) -> str:
    """Escape text for ReportLab Paragraph."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_translation_section(page_texts: list[str], source_filename: str,
                                target_lang: str,
                                assessment: dict | None = None) -> bytes:
    """
    Build the translation section as a portrait A4 PDF.
    Joins all page_texts into a single flowing document — does not try to
    force a per-page split, as the model's page break markers are unreliable.
    The translation flows naturally across as many A4 pages as needed.
    """
    from reportlab.platypus import BaseDocTemplate, PageTemplate, PageBreak
    from reportlab.platypus import Frame as RLFrame

    buf = io.BytesIO()
    page_w, page_h = A4
    margin = 20 * mm

    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "TxHeader",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#555555"),
        spaceBefore=0,
        spaceAfter=4,
        borderPad=2,
        borderColor=colors.HexColor("#cccccc"),
        borderWidth=0,
        leading=10,
    )
    body_style = ParagraphStyle(
        "TxBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    attestation_style = ParagraphStyle(
        "Attestation",
        parent=styles["Normal"],
        fontSize=8,
        leading=12,
        textColor=colors.HexColor("#333333"),
        spaceBefore=10,
        borderPad=4,
        backColor=colors.HexColor("#f5f5f5"),
    )

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin
    frame = RLFrame(margin, margin, avail_w, avail_h, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    story = []

    # Build model provenance string from assessment
    if assessment:
        ocr_model  = MODEL_MISTRAL
        tx_model   = assessment.get("translation_model", MODEL_MISTRAL)
        language   = assessment.get("language", "unknown")
        confidence = assessment.get("confidence", 0)
        if tx_model == ocr_model:
            model_note = (f"OCR + Translation: {tx_model} | Language: {language} "
                          f"(confidence {confidence:.0%})"
                          + _xml(_provenance_notes(assessment)))
        else:
            model_note = (
                f"OCR: {ocr_model} | Translation: {tx_model} | "
                f"Language: {language}" + _xml(_provenance_notes(assessment))
            )
    else:
        model_note = f"Model: {MODEL_MISTRAL}"

    # Cover note
    story.append(Paragraph(
        f"AI TRANSLATION — {_xml(source_filename)} — {target_lang.upper()} | "
        f"{model_note} | "
        "THIS IS A MACHINE TRANSLATION. NOT VERIFIED BY A HUMAN TRANSLATOR. "
        "Original documents follow this translation.",
        header_style,
    ))
    story.append(Spacer(1, 6 * mm))

    # QA status on the cover note
    qa = (assessment or {}).get("qa")
    if qa and qa["status"] == "findings":
        n_crit = sum(1 for f in qa["findings"] if f["severity"] == "critical")
        story.append(Paragraph(
            f"<b>AI verification: {_xml(qa['note'])}"
            + (f" — {n_crit} CRITICAL" if n_crit else "")
            + ". See the findings list after the translation.</b>",
            header_style,
        ))
        story.append(Spacer(1, 3 * mm))
    elif qa and qa["status"] == "failed":
        story.append(Paragraph(
            "<b>AI verification FAILED — this translation has had no "
            "automated check.</b>", header_style))
        story.append(Spacer(1, 3 * mm))

    # Join all page text chunks into one flowing document.
    # Strip ---PAGE BREAK--- markers — we let ReportLab paginate naturally.
    full_text = "\n\n".join(
        chunk.strip() for chunk in page_texts if chunk.strip()
    ).replace("---PAGE BREAK---", "\n\n")

    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    keep_lines = _is_deepl_output(assessment)   # see _build_text_translation_pdf
    for para in paragraphs:
        style = attestation_style if para.startswith("Translator") else body_style
        markup = _xml(para)
        if keep_lines and style is body_style:
            markup = "<br/>".join(l.strip() for l in markup.split("\n") if l.strip())
        story.append(Paragraph(markup, style))

    # Verification outcome, rendered after the translation body
    story.extend(_qa_attestation_paragraphs(assessment, attestation_style))

    doc.build(story)
    return buf.getvalue()


def _build_originals_section(b64_images: list[str], source_filename: str) -> bytes:
    """
    Build the originals section: one source page image per A4 page,
    scaled to fit, with a caption. Portrait A4.
    """
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import Frame as RLFrame

    buf = io.BytesIO()
    page_w, page_h = A4
    margin = 10 * mm
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin - 8 * mm  # leave room for caption

    styles = getSampleStyleSheet()
    caption_style = ParagraphStyle(
        "OrigCaption",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#666666"),
        leading=10,
    )

    c = rl_canvas.Canvas(buf, pagesize=A4)

    # Section divider page
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#333333"))
    c.drawCentredString(page_w / 2, page_h / 2 + 10,
                        "ORIGINAL DOCUMENTS")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawCentredString(page_w / 2, page_h / 2 - 10,
                        f"Source: {source_filename}")
    c.showPage()

    for i, b64 in enumerate(b64_images):
        img_bytes = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(img_bytes))
        w_px, h_px = pil.size
        scale = min(avail_w / w_px, avail_h / h_px)
        draw_w = w_px * scale
        draw_h = h_px * scale

        # Centre image on page, above caption area
        x = margin + (avail_w - draw_w) / 2
        y = margin + 8 * mm + (avail_h - draw_h) / 2

        from reportlab.lib.utils import ImageReader
        c.saveState()
        # Light border around image
        c.setStrokeColor(colors.HexColor("#cccccc"))
        c.setLineWidth(0.5)
        c.rect(x - 1, y - 1, draw_w + 2, draw_h + 2)
        c.drawImage(ImageReader(io.BytesIO(img_bytes)), x, y, draw_w, draw_h)
        c.restoreState()

        # Caption at bottom
        caption_frame = RLFrame(
            margin, margin, avail_w, 8 * mm,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        caption_frame.addFromList([
            Paragraph(
                f"Original document — page {i + 1} of {len(b64_images)} | "
                f"{_xml(source_filename)}",
                caption_style,
            )
        ], c)

        c.showPage()

    c.save()
    return buf.getvalue()


def _merge_pdfs(pdf_a: bytes, pdf_b: bytes) -> bytes:
    """Concatenate two PDFs using pymupdf and return the merged bytes."""
    doc_a = fitz.open(stream=pdf_a, filetype="pdf")
    doc_b = fitz.open(stream=pdf_b, filetype="pdf")
    doc_a.insert_pdf(doc_b)
    buf = io.BytesIO()
    doc_a.save(buf)
    doc_a.close()
    doc_b.close()
    return buf.getvalue()


def build_mode_b_pdf(b64_images: list[str], page_texts: list[str],
                     source_filename: str, target_lang: str,
                     assessment: dict | None = None) -> bytes:
    log.info("Building Mode B output: translation section + originals appendix")
    translation_pdf = _build_translation_section(page_texts, source_filename, target_lang, assessment)
    originals_pdf   = _build_originals_section(b64_images, source_filename)
    return _merge_pdfs(translation_pdf, originals_pdf)


def build_mode_c_pdf(b64_images: list[str], page_texts: list[str],
                     source_filename: str, target_lang: str,
                     assessment: dict | None = None) -> bytes:
    log.info("Building Mode C output: translation section + originals appendix")
    translation_pdf = _build_translation_section(page_texts, source_filename, target_lang, assessment)
    originals_pdf   = _build_originals_section(b64_images, source_filename)
    return _merge_pdfs(translation_pdf, originals_pdf)


# ---------------------------------------------------------------------------
# Main pipeline dispatcher
# ---------------------------------------------------------------------------

def _page_text_with_paragraph_gaps(page: fitz.Page) -> str:
    """
    Page text in reading order, one source line per line, with a blank line
    wherever the page itself leaves a visible vertical gap.

    Plain get_text("text") emits no blank lines at all, so every paragraph,
    heading and field block on a well-laid-out page arrives as one unbroken
    run of lines. The models used to re-invent the paragraph breaks; DeepL
    translates exactly what it is given, so the breaks have to come from the
    page geometry instead.
    """
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return page.get_text("text")

    out: list[str] = []
    prev = None                                    # bbox of the previous text line
    for block in blocks:
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if not text.strip():
                continue
            x0, y0, x1, y1 = line["bbox"]
            if prev is not None:
                gap = y0 - prev[3]
                height = max(1.0, min(y1 - y0, prev[3] - prev[1]))
                # Only a clear downward gap is a paragraph break. A negative
                # gap is the same row (table cells) or the top of a new column.
                if gap > 0.4 * height and out and out[-1] != "":
                    out.append("")
            out.append(text.rstrip())
            prev = (x0, y0, x1, y1)
    return "\n".join(out)


def extract_text_layer(pdf_bytes: bytes) -> str:
    """
    Extract the text layer of a born-digital PDF, preserving the document's own
    line breaks and blank lines.

    This is the whole basis of the simplified Mode A: a clean text PDF already
    knows where its lines and paragraphs are, so we don't reconstruct layout
    from block geometry — we keep the newlines get_text gives us and carry them
    through translation. Lists stay lists; blank lines stay blank.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(_page_text_with_paragraph_gaps(page))
    doc.close()
    # A form feed between pages lets us drop page furniture later if needed;
    # for now double newline keeps paragraphs separated across page boundaries.
    return "\n\n".join(p.strip("\n") for p in pages if p.strip())


def translate_preserving_layout(text: str, target_lang: str, model: str) -> str:
    """
    Translate text while preserving its line and paragraph structure.

    The prompt puts TRANSLATION first and layout second, on purpose: an earlier
    version led with a "CRITICAL LAYOUT RULE" and several keep-every-line
    imperatives, with "only the language changes" buried mid-paragraph. A 24B
    model read the emphasis literally and returned the source unchanged but
    re-laid-out — a faithful copy, not a translation. Translation is now the
    unmistakable primary instruction; layout preservation is a formatting note
    under it. Each chunk is checked for an untranslated echo and retried once
    with an even blunter instruction before being accepted.
    """
    chunks = _split_for_translation(text, MAX_TRANSLATE_CHARS)
    log.info("Mode A: translating text layer via %s (%d chars in %d chunk%s)",
             model, len(text), len(chunks), "" if len(chunks) == 1 else "s")

    def call(chunk: str, i: int, blunt: bool) -> str:
        is_last = (i == len(chunks) - 1)
        position = ("" if len(chunks) == 1
                    else f"This is part {i + 1} of {len(chunks)} of one document. "
                         "Translate this part only. ")
        tail = ("Append the translator's attestation at the very end."
                if is_last else
                "Do NOT append any attestation — this is not the final part.")
        emphasis = (
            "The text is NOT yet in {tl}. Every sentence must be rewritten in "
            "{tl}. Returning the source text unchanged is a failure. "
            if blunt else ""
        ).format(tl=target_lang)

        response = ovh_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(target_lang=target_lang)},
                {"role": "user", "content": (
                    f"Translate the following medical document into {target_lang}. "
                    f"{emphasis}"
                    "Translate the meaning of every line; do not leave any source-"
                    "language text in your output except proper nouns, drug names "
                    "and institution names that a faithful translation would keep.\n\n"
                    "Formatting note: keep the same visual layout — preserve line "
                    "breaks and blank lines so lists stay one item per line. But "
                    "layout is secondary: never let it stop you translating. The "
                    "content must be in {tl}; only the line structure is copied.\n\n"
                    f"{position}"
                    "Apply every fidelity rule from your system instructions. "
                    f"{tail}\n\n"
                    f"TEXT TO TRANSLATE:\n{chunk}"
                ).format(tl=target_lang)},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.1,
        )
        _check_not_truncated(response, "Mode A translation",
                             f"chunk {i + 1} of {len(chunks)}")
        return response.choices[0].message.content.strip()

    out_parts = []
    for i, chunk in enumerate(chunks):
        result = call(chunk, i, blunt=False)
        # Guard against the model echoing the source instead of translating it.
        if _looks_untranslated(result, "", chunk, search_whole=True):
            log.warning("Mode A: chunk %d came back untranslated — retrying with "
                        "a blunter instruction", i + 1)
            retry = call(chunk, i, blunt=True)
            if not _looks_untranslated(retry, "", chunk, search_whole=True):
                result = retry
            else:
                log.warning("Mode A: chunk %d still appears untranslated after "
                            "retry — keeping best effort", i + 1)
        out_parts.append(result)

    return "\n\n".join(out_parts)


def _build_text_translation_pdf(translated_text: str, source_filename: str,
                                 target_lang: str, assessment: dict | None) -> bytes:
    """
    Render translated text as a clean A4 PDF, preserving line and list structure.

    Single newlines become line breaks (<br/>), blank lines separate paragraphs,
    and lines that look like list items are set tight with a slight indent so
    drug charts and code lists read as lists rather than running prose.
    """
    from reportlab.platypus import BaseDocTemplate, PageTemplate
    from reportlab.platypus import Frame as RLFrame

    buf = io.BytesIO()
    page_w, page_h = A4
    margin = 18 * mm

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle("TxHeader", parent=styles["Normal"], fontSize=7,
                                  textColor=colors.HexColor("#555555"),
                                  spaceAfter=4, leading=10)
    body_style = ParagraphStyle("TxBody", parent=styles["Normal"], fontSize=10,
                                leading=14, spaceAfter=6)
    list_style = ParagraphStyle("TxList", parent=styles["Normal"], fontSize=9.5,
                                leading=12, spaceAfter=1, leftIndent=8)
    attest_style = ParagraphStyle("Attestation", parent=styles["Normal"],
                                  fontSize=8, leading=12,
                                  textColor=colors.HexColor("#333333"),
                                  spaceBefore=10, borderPad=4,
                                  backColor=colors.HexColor("#f5f5f5"))

    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=margin, rightMargin=margin,
                          topMargin=margin, bottomMargin=margin)
    frame = RLFrame(margin, margin, page_w - 2 * margin, page_h - 2 * margin, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    # Provenance line
    if assessment:
        tx_model = assessment.get("translation_model", MODEL_LLAMA)
        language = assessment.get("language", "unknown")
        model_note = (f"Translation: {tx_model} | Source language: {language}"
                      + _xml(_provenance_notes(assessment)))
    else:
        model_note = f"Translation: {MODEL_LLAMA}"

    story = [
        Paragraph(
            f"AI TRANSLATION — {_xml(source_filename)} — {target_lang.upper()} | "
            f"{model_note} | THIS IS A MACHINE TRANSLATION. NOT VERIFIED BY A "
            "HUMAN TRANSLATOR. The original document follows this translation.",
            header_style),
        Spacer(1, 5 * mm),
    ]

    # QA note on the cover
    qa = (assessment or {}).get("qa")
    if qa and qa["status"] == "findings":
        n_crit = sum(1 for f in qa["findings"] if f["severity"] == "critical")
        story.append(Paragraph(
            f"<b>AI verification: {_xml(qa['note'])}"
            + (f" — {n_crit} CRITICAL" if n_crit else "")
            + ". See findings after the translation.</b>", header_style))
        story.append(Spacer(1, 3 * mm))
    elif qa and qa["status"] == "failed":
        story.append(Paragraph("<b>AI verification FAILED — treat as "
                               "unverified.</b>", header_style))
        story.append(Spacer(1, 3 * mm))

    def looks_like_list_item(line: str) -> bool:
        s = line.strip()
        if len(s) < 2:
            return False
        return bool(re.match(r"^([\-\u2022\*]|\(?[A-Z]?\d|\d+[.)])", s)) or " - " in s

    # DeepL returns exactly the line structure it was sent (wrapped prose was
    # already re-joined), so each line is a real line and must stay one. Model
    # output is re-flowed as before, since the models wrap lines arbitrarily.
    keep_lines = _is_deepl_output(assessment)

    # Split on blank lines into paragraph-ish groups, preserving inner newlines
    groups = re.split(r"\n\s*\n", translated_text)
    for group in groups:
        raw_lines = [ln for ln in group.split("\n")]
        stripped = [ln.strip() for ln in raw_lines if ln.strip()]
        if not stripped:
            story.append(Spacer(1, 3 * mm))
            continue

        if stripped and stripped[0].startswith("Translator"):
            story.append(Paragraph("<br/>".join(_xml(l) for l in stripped), attest_style))
            continue

        # A group where most lines look like list items is rendered line-by-line
        # so the list structure survives; otherwise join wrapped lines into a
        # flowing paragraph.
        list_lines = sum(1 for l in stripped if looks_like_list_item(l))
        if not keep_lines and len(stripped) >= 2 and list_lines >= len(stripped) * 0.5:
            for l in stripped:
                story.append(Paragraph(_xml(l), list_style))
            story.append(Spacer(1, 2 * mm))
        elif keep_lines:
            story.append(Paragraph("<br/>".join(_xml(l) for l in stripped), body_style))
        else:
            story.append(Paragraph(" ".join(_xml(l) for l in stripped), body_style))

    story.extend(_qa_attestation_paragraphs(assessment, attest_style))
    doc.build(story)
    return buf.getvalue()


def run_mode_a_textlayer(file_bytes: bytes, original_input: bytes,
                         target_lang: str, source_hint: str,
                         model_override: str, uploaded_filename: str,
                         qa_mode: str) -> bytes:
    """
    Simplified Mode A for born-digital text PDFs.

    Extract the text layer → translate it preserving layout → render a clean
    English PDF → append the untouched original pages behind it. No block
    geometry, no in-place rewriting, so watermarks and stamps are never a
    problem (they stay on the appended original, which is where a reader needs
    to see them anyway).
    """
    text = extract_text_layer(file_bytes)

    # Language assessment on a sample (same logic as before)
    sample = text[:1000]
    assessment: dict
    if model_override == "llama":
        assessment = {"language": "unknown"}
        tx_model = MODEL_LLAMA
        assessment["escalation_reason"] = "Manual override"
    else:
        resp = ovh_client.chat.completions.create(
            model=MODEL_MISTRAL,
            messages=[{"role": "user", "content": (
                "Identify the language of this text and assess your translation "
                "confidence. Output only JSON: "
                '```json\n{"language":"...","script":"...","confidence":0.0,'
                '"escalate":false,"reason":"..."}\n```\n\nText: ' + sample
            )}],
            max_tokens=200, temperature=0.0,
        )
        _, assessment = _parse_assessment(resp.choices[0].message.content)
        escalate, reason = _should_escalate(assessment)
        tx_model = MODEL_LLAMA if escalate else MODEL_MISTRAL
        if escalate:
            assessment["escalation_reason"] = reason
    assessment["translation_model"] = tx_model

    # DeepL first; the model path is the fallback.
    translated = try_deepl(text, target_lang, assessment, model_override)
    if translated is not None:
        translated += "\n\n" + deepl_attestation(translated)
    else:
        translated = translate_preserving_layout(text, target_lang, tx_model)

    run_qa(text, translated, assessment, target_lang, qa_mode)

    translation_pdf = _build_text_translation_pdf(
        translated, uploaded_filename, target_lang, assessment)

    # Append the untouched original pages
    try:
        orig_images = get_all_page_images(original_input)
        originals_pdf = _build_originals_section(orig_images, uploaded_filename)
        return _merge_pdfs(translation_pdf, originals_pdf)
    except Exception as e:
        log.warning("Mode A: could not append original pages: %s", e)
        return translation_pdf


def run_pipeline(file_bytes: bytes, ext: str, target_lang: str,
                 source_hint: str, forced_mode: str,
                 model_override: str = "",
                 uploaded_filename: str = "document.pdf",
                 qa_mode: str = "auto") -> tuple[bytes, str]:
    """
    Returns (output_pdf_bytes, mode_used).
    ext should be lowercase including the dot, e.g. ".pdf", ".jpg"
    model_override: "mistral", "llama", or "" for auto
    qa_mode: "auto" (verify unless language is exempt), "on", "off"
    """
    if ext in ACCEPTED_IMAGE_TYPES:
        log.info("Direct image upload (%s) → rotation correction → Mode C", ext)
        file_bytes = correct_image_rotation(file_bytes)
        b64 = image_file_to_b64(file_bytes, ".png")
        page_texts, assessment, _, source_text = translate_all_images(
            [b64], target_lang, source_hint, model_override)
        run_qa(source_text, "\n\n".join(page_texts), assessment, target_lang, qa_mode)
        pdf = build_mode_c_pdf([b64], page_texts, uploaded_filename, target_lang, assessment)
        return pdf, "C"

    # Keep a copy for Mode A's appended originals. Use the rotation-CORRECTED
    # bytes: get_all_page_images honours /Rotate, and any content-rotation fix
    # we applied should show in the appendix too, so the reader sees the pages
    # the same way up as the translation.
    log.info("Applying rotation correction to PDF")
    file_bytes = correct_pdf_rotation(file_bytes)
    original_input = file_bytes
    mode = detect_mode(file_bytes, forced_mode)

    if mode == "A":
        log.info("Pipeline: Mode A (text-based, clean translation + original appendix)")
        pdf = run_mode_a_textlayer(
            file_bytes, original_input, target_lang, source_hint,
            model_override, uploaded_filename, qa_mode)
        return pdf, "A"

    # Modes B and C: full two-stage image pipeline
    b64_images = get_all_page_images(file_bytes)
    page_texts, assessment, _, source_text = translate_all_images(
        b64_images, target_lang, source_hint, model_override
    )
    run_qa(source_text, "\n\n".join(page_texts), assessment, target_lang, qa_mode)

    if mode == "B":
        log.info("Pipeline: Mode B (clean scan)")
        pdf = build_mode_b_pdf(b64_images, page_texts, uploaded_filename, target_lang, assessment)
        return pdf, "B"
    else:
        log.info("Pipeline: Mode C (phone photo)")
        pdf = build_mode_c_pdf(b64_images, page_texts, uploaded_filename, target_lang, assessment)
        return pdf, "C"



# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

ACCEPTED_EXTENSIONS = {".pdf"} | ACCEPTED_IMAGE_TYPES


def require_api_key(f):
    """Decorator: checks X-API-Key header against SERVICE_API_KEY."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if SERVICE_API_KEY:
            incoming = request.headers.get("X-API-Key", "")
            if not incoming:
                log.warning("Request rejected: missing X-API-Key header")
                return jsonify({"error": "Missing X-API-Key header"}), 401
            if incoming != SERVICE_API_KEY:
                log.warning("Request rejected: invalid API key")
                return jsonify({"error": "Invalid API key"}), 403
        return f(*args, **kwargs)
    return decorated


def _validate_upload() -> tuple[object, str, str] | tuple[None, None, str]:
    """
    Validate the uploaded file from the current request.
    Returns (file_object, ext, error_message).
    error_message is empty string on success.
    """
    if "file" not in request.files:
        return None, None, "No file field in request"
    uploaded = request.files["file"]
    if not uploaded.filename:
        return None, None, "Empty filename"
    ext = Path(uploaded.filename).suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        return None, None, (
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS))}"
        )
    return uploaded, ext, ""


@app.route("/health", methods=["GET"])
def health():
    with _db() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as n FROM jobs GROUP BY status"
        ).fetchall()
    counts = {r["status"]: r["n"] for r in rows}
    return jsonify({
        "status":               "ok",
        "ocr_model":            MODEL_MISTRAL,
        "translation_engine":   DEEPL_ENGINE_NAME if deepl_client else "ovhcloud-models",
        "deepl": {
            "enabled":  deepl_client is not None,
            "plan":     ("free" if DEEPL_AUTH_KEY.endswith(":fx") else "pro") if deepl_client else None,
            "paused":   time.time() < _deepl_skip_until,
        },
        "fallback_models":      [MODEL_MISTRAL, MODEL_LLAMA],
        "translation_models":   [MODEL_MISTRAL, MODEL_LLAMA],
        "escalation_languages": sorted(ALWAYS_ESCALATE_LANGUAGES),
        "port":                 PORT,
        "queue": {
            "pending":    counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "total_jobs": sum(counts.values()),
            "max_workers": MAX_WORKERS,
        },
    })


@app.route("/translate/submit", methods=["POST"], strict_slashes=False)
@require_api_key
def translate_submit():
    """
    Async submit: accept file, enqueue translation job, return job_id immediately.
    """
    uploaded, ext, err = _validate_upload()
    if err:
        return jsonify({"error": err}), 400

    target_lang    = request.form.get("target_lang", "English").strip()
    source_hint    = request.form.get("source_hint", "").strip()
    forced_mode    = request.form.get("mode", "auto").upper()
    model_override = request.form.get("model", "").strip().lower()
    qa_mode        = request.form.get("qa", "auto").strip().lower()
    if qa_mode not in ("auto", "on", "off"):
        qa_mode = "auto"
    client_job_id  = (request.form.get("job_id") or request.form.get("jobID") or "").strip()

    if model_override and model_override not in ("mistral", "llama"):
        return jsonify({"error": "model must be 'mistral', 'llama', or omitted for auto"}), 400

    # Validate client-supplied job_id if provided — must be a valid UUID
    if client_job_id:
        try:
            uuid.UUID(client_job_id)
        except ValueError:
            return jsonify({"error": "job_id must be a valid UUID if supplied"}), 400

    file_bytes = uploaded.read()
    job_id     = _create_job(uploaded.filename, client_job_id or None)

    log.info("Job %s submitted: file=%s ext=%s target=%s mode=%s model=%s",
             job_id, uploaded.filename, ext, target_lang, forced_mode,
             model_override or "auto")

    _executor.submit(
        _run_job,
        job_id, file_bytes, ext, target_lang,
        source_hint, forced_mode, model_override, uploaded.filename, qa_mode,
    )

    base_url = request.host_url.rstrip("/")
    return jsonify({
        "job_id":     job_id,
        "status":     "pending",
        "poll_url":   f"{base_url}/translate/status/{job_id}",
        "result_url": f"{base_url}/translate/result/{job_id}",
    }), 202


@app.route("/translate/status/<job_id>", methods=["GET"])
@require_api_key
def translate_status(job_id: str):
    """Poll job status."""
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    resp = {
        "job_id":   job_id,
        "status":   job["status"],
        "filename": job["filename"],
        "elapsed_seconds": round(
            (job["completed_at"] or time.time()) - job["submitted_at"], 1
        ),
    }
    if job["status"] == "failed":
        resp["error"] = job.get("error", "Unknown error")
    if job["status"] == "complete":
        resp["mode_used"] = job.get("mode_used", "")

    return jsonify(resp)


@app.route("/translate/result/<job_id>", methods=["GET"])
@require_api_key
def translate_result(job_id: str):
    """
    Fetch completed PDF. Cleans up the job and result file after successful delivery.
    """
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found — may have expired or already been retrieved"}), 404

    if job["status"] == "failed":
        # If an error PDF was written, serve it so FileMaker's result field
        # holds a readable explanation. The job status remains "failed" on the
        # status endpoint, so status-based branching is unaffected; this only
        # changes what the RESULT endpoint returns for a failed job.
        err_path = Path(job["result_path"])
        if err_path.exists():
            pdf_bytes = err_path.read_bytes()
            err_path.unlink(missing_ok=True)
            with _db() as conn:
                conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                conn.commit()
            stem = Path(job["filename"]).stem
            log.info("Job %s failed — error PDF delivered (%d bytes), cleaned up",
                     job_id, len(pdf_bytes))
            return send_file(
                io.BytesIO(pdf_bytes),
                mimetype="application/pdf",
                as_attachment=True,
                download_name=f"{stem}_TRANSLATION_FAILED.pdf",
            )
        return jsonify({"error": job.get("error", "Translation failed")}), 500

    if job["status"] != "complete":
        return jsonify({
            "error": "Job not yet complete",
            "status": job["status"],
            "elapsed_seconds": round(time.time() - job["submitted_at"], 1),
        }), 409

    result_path = Path(job["result_path"])
    if not result_path.exists():
        return jsonify({"error": "Result file missing — job may have expired"}), 404

    stem           = Path(job["filename"]).stem
    target_lang    = "english"   # stored in job in future; default for now
    mode_used      = job.get("mode_used", "")
    download_name  = f"{stem}_translated_{target_lang}_mode{mode_used}.pdf"

    pdf_bytes = result_path.read_bytes()

    # Clean up job and result file after delivery
    result_path.unlink(missing_ok=True)
    with _db() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()

    log.info("Job %s result delivered (%d bytes) — cleaned up", job_id, len(pdf_bytes))

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=download_name,
    )


@app.route("/translate", methods=["POST"], strict_slashes=False)
@require_api_key
def translate_sync():
    """
    Synchronous endpoint — retained for direct testing and non-FileMaker callers.
    Waits for completion and returns PDF directly. Not recommended for FileMaker.
    """
    uploaded, ext, err = _validate_upload()
    if err:
        return jsonify({"error": err}), 400

    target_lang    = request.form.get("target_lang", "English").strip()
    source_hint    = request.form.get("source_hint", "").strip()
    forced_mode    = request.form.get("mode", "auto").upper()
    model_override = request.form.get("model", "").strip().lower()
    qa_mode        = request.form.get("qa", "auto").strip().lower()
    if qa_mode not in ("auto", "on", "off"):
        qa_mode = "auto"
    stem           = Path(uploaded.filename).stem

    client_job_id  = (request.form.get("job_id") or request.form.get("jobID") or "").strip()
    if client_job_id:
        try:
            uuid.UUID(client_job_id)
        except ValueError:
            return jsonify({"error": "job_id must be a valid UUID if supplied"}), 400

    try:
        file_bytes = uploaded.read()
        output_pdf, mode_used = run_pipeline(
            file_bytes, ext, target_lang, source_hint,
            forced_mode, model_override, uploaded.filename, qa_mode,
        )
        output_name = f"{stem}_translated_{target_lang.lower()}_mode{mode_used}.pdf"
        return send_file(
            io.BytesIO(output_pdf),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=output_name,
        )
    except Exception as e:
        log.exception("Sync translation failed")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("SkyCare PDF Translation Service v3 | port=%d", PORT)
    log.info("OCR model: %s", MODEL_MISTRAL)
    log.info("Translation models: %s / %s", MODEL_MISTRAL, MODEL_LLAMA)
    log.info("Job DB: %s | Job TTL: %ds | Workers: %d", DB_PATH, JOB_TTL_SECS, MAX_WORKERS)
    app.run(host="0.0.0.0", port=PORT, debug=False)
