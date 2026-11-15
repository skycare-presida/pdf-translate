# FileMaker integration

Notes on calling the translation service from Presida (the FileMaker-based
case management system). The service is async by design — see
[history.md](history.md#v2-%E2%86%92-v3-async-job-queue) for why — so
integration is a submit/poll/fetch loop rather than a single request.

## The three endpoints

| Call | What it does |
|------|--------------|
| `POST /translate/submit` | Accepts the PDF, returns `{ job_id, status: "queued" }` immediately |
| `GET /translate/status/<job_id>` | Returns `{ status: queued \| processing \| complete \| failed, error?, elapsed_seconds, ... }` |
| `GET /translate/result/<job_id>` | Returns the translated PDF (on `complete`) or the error PDF (on `failed`); deletes the job after delivery |

All three require `Authorization: Bearer <TRANSLATE_API_KEY>`. The key is
set in `/opt/skycare/pdf-translate-v3/.env` by `install.sh` on first run
(either inherited from v1/v2 or newly generated; printed on completion).

A synchronous `POST /translate` endpoint also exists for direct testing
but should not be called from FileMaker — it will time out on real
documents.

## FileMaker script pattern

```
# ────────────────────────────────────────────────────────────────────────
# Script: Translate Medical Document
# Called with: $$DocContainer (container field holding the source PDF)
#              $$TargetLang  (optional; defaults to English)
# Sets:        $$TranslatedPDF (translated PDF, on success)
#              $$TranslationError (error message, on failure)
# ────────────────────────────────────────────────────────────────────────

# Generate our own job ID so we can log it before the HTTP call returns.
# The service accepts jobID (camelCase, what Presida sends) OR job_id.
Set Variable [ $jobID ; Get(UUID) ]

# --- Submit ---
Insert from URL [ Select ; With dialog: Off ;
    Target: $submitResponse ;
    URL: "https://ancilla.skycare.uk:5107/translate/submit" ;
    Verify SSL: On ;
    cURL options:
        "--request POST" &
        " --header " & Quote("Authorization: Bearer " & $$TRANSLATE_API_KEY) &
        " --form "   & Quote("jobID=" & $jobID) &
        " --form "   & Quote("target_lang=" & If(IsEmpty($$TargetLang); "English"; $$TargetLang)) &
        " --form "   & Quote("file=@$doc.pdf;type=application/pdf") &
        " --upload-file-with-path " & Quote("$$DocContainer")
]

# The container-variable-with-.pdf-extension gotcha — see below.

# --- Poll loop ---
Loop
    Pause/Resume Script [ Duration (seconds): 15 ]

    Insert from URL [ Select ; With dialog: Off ;
        Target: $statusResponse ;
        URL: "https://ancilla.skycare.uk:5107/translate/status/" & $jobID ;
        Verify SSL: On ;
        cURL options: " --header " & Quote("Authorization: Bearer " & $$TRANSLATE_API_KEY)
    ]

    Set Variable [ $status ; JSONGetElement($statusResponse; "status") ]
    Exit Loop If [ $status = "complete" or $status = "failed" ]
End Loop

# --- Branch on outcome ---
If [ $status = "failed" ]
    Set Variable [ $$TranslationError ; JSONGetElement($statusResponse; "error") ]
    # The error PDF is still worth fetching — it's the readable explanation.
End If

# --- Fetch result (succeeds for both "complete" and "failed" — the error PDF
#     lives at the same endpoint on failure) ---
Insert from URL [ Select ; With dialog: Off ;
    Target: $$TranslatedPDF ;
    URL: "https://ancilla.skycare.uk:5107/translate/result/" & $jobID ;
    Verify SSL: On ;
    cURL options: " --header " & Quote("Authorization: Bearer " & $$TRANSLATE_API_KEY)
]
```

The result endpoint deletes the job and result file after delivery — a
second fetch will 404. If you need the PDF in more than one place in the
calling script, assign it to a variable once and reuse the variable.

## Submit-form fields

Everything except `file` is optional:

| Field | Values | Notes |
|-------|--------|-------|
| `file` | the PDF or image | Required. See cURL gotcha below. |
| `target_lang` | any language name, e.g. `English`, `French` | Default `English` |
| `mode` | `A` / `B` / `C` / `AUTO` | Default `AUTO` (detect text layer, scan, photo) |
| `model` | `mistral` / `llama` / empty | Empty = auto-select by language and confidence |
| `qa` | `auto` / `on` / `off` | Default `auto` (skip for high-resource European languages) |
| `source_hint` | free text | Optional context, e.g. "Turkish hospital discharge summary" |
| `jobID` / `job_id` | client-generated UUID | Either name accepted; use client-side ID so Presida can log before the submit returns |

## Gotchas

### The container-variable-with-.pdf-extension cURL quirk

When passing a container field held in a FileMaker variable to a cURL
multipart form, the variable must be **literally named with a `.pdf`
extension** or cURL silently sends the wrong content type and the service
rejects the upload with no useful error.

**Works:**
```
--form "file=@$doc.pdf;type=application/pdf"
```
with the container assigned to `$doc.pdf` (the dot and `pdf` are part of
the variable name, not a file path).

**Fails silently:**
```
--form "file=@$doc;type=application/pdf"
```
with the container assigned to `$doc`.

Discovered through trial and error; the behaviour is a FileMaker/cURL
interaction, not a service issue.

### `jobID` vs `job_id`

Presida sends `Get(UUID)` as `jobID` (camelCase, FileMaker idiom). The
service accepts both spellings, so new scripts can use either; existing
scripts don't need updating.

### Branching on failure

`JSONGetElement($statusResponse; "status")` gives `complete` or `failed`;
`JSONGetElement($statusResponse; "error")` gives the error message on
failure. Always branch on both:

```
If [ $status = "complete" ]
    # happy path
Else If [ $status = "failed" ]
    # fetch the error PDF, surface $error in the UI
End If
```

The result endpoint returns the error PDF for failed jobs, so the
calling script can treat success and failure identically for the fetch
step — only the display differs.

### Timeouts

The service designs around FileMaker's HTTP timeout (~5 minutes via
cURL `Insert from URL`) by keeping every call short: submit returns in
~1 second, status polls return in ~100ms, result returns as fast as the
PDF can be streamed. The only call that can genuinely take time is the
poll loop itself, which the `Pause/Resume` handles correctly.

A full translation of a typical discharge summary takes 1-5 minutes
end-to-end; a long scan with QA on and Llama translation can reach 10-15
minutes. Set an upper bound on the poll loop (say, 30 iterations ×
15 seconds = 7.5 minutes before showing the user "still working…") so
the UI doesn't spin indefinitely if something goes wrong.

## Testing from the command line

Useful for verifying the service is reachable before touching FileMaker:

```bash
# Submit
curl -X POST https://ancilla.skycare.uk:5107/translate/submit \
    -H "Authorization: Bearer $TRANSLATE_API_KEY" \
    -F "file=@/path/to/test.pdf" \
    -F "target_lang=English"
# → { "job_id": "...", "status": "queued", "status_url": "...", "result_url": "..." }

# Poll
curl -H "Authorization: Bearer $TRANSLATE_API_KEY" \
    https://ancilla.skycare.uk:5107/translate/status/JOB_ID

# Fetch
curl -H "Authorization: Bearer $TRANSLATE_API_KEY" \
    https://ancilla.skycare.uk:5107/translate/result/JOB_ID \
    --output translated.pdf
```

`install.sh` prints these three commands with the real API key filled in
at the end of a successful install.
