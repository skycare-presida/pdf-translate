"""
Tests for the DeepL engine and its fallback to the OVHcloud models.
No network: the DeepL client is replaced by a fake.

    pip install pytest && pytest tests/
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "service"))
import pdf_translate_v3_service as svc  # noqa: E402

SOURCE = (
    "Entlassungsbericht\n\n"
    "Der Patient wurde am 3. Mai mit akuten Brustschmerzen in unsere Klinik\n"
    "aufgenommen und nach einer umfassenden kardiologischen Abklärung auf die\n"
    "Normalstation verlegt.\n\n"
    "Medikation:\n- Ramipril 5 mg 1-0-0\n- Bisoprolol 2,5 mg 1-0-1\n\n"
    "Die Laborwerte zeigten im weiteren Verlauf eine deutliche Besserung der Entzündungsparameter.\n"
    "Der Patient konnte in gutem Allgemeinzustand nach Hause entlassen werden und wird ambulant weiterbetreut."
)


class FakeDeepL:
    def __init__(self, fn=None, languages=("German", "French", "Norwegian (bokmål)")):
        self.fn = fn or (lambda t: "EN<" + t.upper() + ">")
        self.languages = languages
        self.calls = []
        self.doc_fn = None          # whole-document translation off unless a test sets it

    def get_source_languages(self):
        return [SimpleNamespace(name=n, code=n[:2].upper()) for n in self.languages]

    def translate_document(self, src, out, **kw):
        self.doc_calls = getattr(self, "doc_calls", []) + [kw]
        if self.doc_fn is None:
            raise svc.deepl.DocumentTranslationException("document translation unavailable", None)
        assert isinstance(src, bytes)        # the SDK ignores `filename` for file objects
        out.write(self.doc_fn(src))
        return SimpleNamespace(billed_characters=50000)

    def translate_text(self, text, **kw):
        self.calls.append((text, kw))
        return SimpleNamespace(text=self.fn(text), detected_source_lang="DE")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(svc, "TRANSLATE_ENGINE", "deepl")
    monkeypatch.setattr(svc, "_deepl_skip_until", 0.0)
    monkeypatch.setattr(svc, "_deepl_source_names", None)
    monkeypatch.setattr(svc, "deepl_client", FakeDeepL())


def test_target_codes():
    assert svc._deepl_target_code("English") == "EN-GB"
    assert svc._deepl_target_code(" german ") == "DE"
    assert svc._deepl_target_code("pt-br") == "PT-BR"
    assert svc._deepl_target_code("en") == "EN-GB"
    assert svc._deepl_target_code("Klingon") is None


def test_unwrap_joins_wrapped_prose_but_not_lists_or_headings():
    out = svc._unwrap_soft_breaks(SOURCE)
    assert "Klinik aufgenommen und" in out and "auf die Normalstation verlegt." in out
    assert "Medikation:\n- Ramipril 5 mg 1-0-0\n- Bisoprolol 2,5 mg 1-0-1" in out
    assert out.startswith("Entlassungsbericht\n\n")
    # two complete sentences on consecutive lines stay on separate lines
    assert "Entzündungsparameter.\nDer Patient konnte" in out


def test_deepl_used_and_page_breaks_survive():
    a = {"language": "German", "escalation_reason": "x"}
    out = svc.try_deepl(SOURCE + "\n---PAGE BREAK---\nSeite zwei des Berichts.", "English", a)
    assert out is not None and out.count("---PAGE BREAK---") == 1
    assert a["translation_model"] == "DeepL" and "escalation_reason" not in a
    assert all("PAGE BREAK" not in text for text, _ in svc.deepl_client.calls)
    assert svc.deepl_client.calls[0][1]["target_lang"] == "EN-GB"


def test_large_text_is_split_under_request_limit():
    big = "\n\n".join(f"Absatz {i}: " + "wort " * 400 for i in range(40))
    svc.translate_with_deepl(big, "English", "German")
    assert len(svc.deepl_client.calls) > 1
    assert all(len(t) <= svc.DEEPL_MAX_CHARS for t, _ in svc.deepl_client.calls)


@pytest.mark.parametrize("setup,expect", [
    (lambda m: m.setattr(svc, "deepl_client", None), "not configured"),
    (lambda m: m.setattr(svc, "deepl_client", FakeDeepL(fn=lambda t: t)), "source text"),
    (lambda m: m.setattr(svc, "deepl_client", FakeDeepL(fn=lambda t: " ")), "empty"),
])
def test_falls_back_with_reason(monkeypatch, setup, expect):
    setup(monkeypatch)
    a = {"language": "German"}
    assert svc.try_deepl(SOURCE, "English", a) is None
    assert expect in a["deepl_fallback_reason"]
    assert "translation_model" not in a


def test_unsupported_languages_fall_back_without_translating():
    a = {"language": "Pashto"}
    assert svc.try_deepl(SOURCE, "English", a) is None
    assert "Pashto" in a["deepl_fallback_reason"]
    a = {"language": "German"}
    assert svc.try_deepl(SOURCE, "Klingon", a) is None
    assert svc.deepl_client.calls == []
    assert svc._deepl_supports_source("Norwegian") and svc._deepl_supports_source("unknown")


def test_api_error_falls_back_and_pauses_deepl(monkeypatch):
    def boom(_):
        raise RuntimeError("Quota for this billing period has been exceeded")
    client = FakeDeepL(fn=boom)
    monkeypatch.setattr(svc, "deepl_client", client)
    a = {"language": "German"}
    assert svc.try_deepl(SOURCE, "English", a) is None
    assert "Quota" in a["deepl_fallback_reason"]
    assert svc.try_deepl(SOURCE, "English", {"language": "German"}) is None
    assert len(client.calls) == 1          # second attempt skipped by the cooldown


def test_explicit_model_or_engine_switch_bypasses_deepl(monkeypatch):
    a = {"language": "German"}
    assert svc.try_deepl(SOURCE, "English", a, model_override="llama") is None
    monkeypatch.setattr(svc, "TRANSLATE_ENGINE", "llm")
    assert svc.try_deepl(SOURCE, "English", a) is None
    assert "deepl_fallback_reason" not in a and svc.deepl_client.calls == []


def _fitz_pdf(text):
    doc = svc.fitz.open()
    page = doc.new_page()
    page.insert_textbox(svc.fitz.Rect(50, 50, 550, 800), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_text(pdf):
    doc = svc.fitz.open(stream=pdf, filetype="pdf")
    return " ".join(" ".join(p.get_text().split()) for p in doc)


def _stub_assessment(monkeypatch):
    resp = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content='```json\n{"language":"German","confidence":0.95,"escalate":false}\n```'))])
    monkeypatch.setattr(svc.ovh_client.chat.completions, "create", lambda **kw: resp)


def test_mode_a_end_to_end_with_deepl(monkeypatch):
    _stub_assessment(monkeypatch)
    pdf, mode = svc.run_pipeline(_fitz_pdf(SOURCE), ".pdf", "English", "", "A", qa_mode="off")
    text = _pdf_text(pdf)
    assert mode == "A"
    assert "Translation: DeepL" in text and "EN<" in text
    # short DeepL statement; no OCR ran on a text PDF, so no [ILLEGIBLE] count
    assert "Translated by DeepL, a machine translation service" in text
    assert "attestation" not in text and "ILLEGIBLE" not in text


def test_mode_a_end_to_end_falls_back_to_models(monkeypatch):
    _stub_assessment(monkeypatch)
    monkeypatch.setattr(svc, "deepl_client", FakeDeepL(fn=lambda t: ""))
    monkeypatch.setattr(svc, "translate_preserving_layout",
                        lambda text, tl, model: "Model translation of the discharge report.")
    pdf, _ = svc.run_pipeline(_fitz_pdf(SOURCE), ".pdf", "English", "", "A", qa_mode="off")
    text = _pdf_text(pdf)
    assert "Model translation of the discharge report." in text
    assert "DeepL not used" in text and svc.MODEL_MISTRAL in text


def test_scan_batches_mixed_engines_reported(monkeypatch):
    monkeypatch.setattr(svc, "MAX_IMAGES_PER_CALL", 1)
    monkeypatch.setattr(svc, "ocr_batch_adaptive",
                        lambda imgs, hint, first_page=1: (SOURCE, {"language": "German", "confidence": 0.9}))
    monkeypatch.setattr(svc, "translate_text_with_model",
                        lambda text, tl, model: "Model translation. Translator's attestation: n/a")
    n = {"calls": 0}

    def flaky(t):
        n["calls"] += 1
        if n["calls"] > 1:
            raise ConnectionError("DeepL unreachable")
        return "EN<" + t.upper() + ">"
    monkeypatch.setattr(svc, "deepl_client", FakeDeepL(fn=flaky))

    parts, assessment, _, _ = svc.translate_all_images(["img1", "img2"], "English", "")
    assert assessment["translation_model"] == f"DeepL + {svc.MODEL_MISTRAL}"
    assert "ConnectionError" in assessment["deepl_fallback_reason"]
    notes = [p for p in parts if p.startswith("Translated by DeepL")]
    assert len(notes) == 1 and "Number of such items: 0." in notes[0]


def test_formatted_text_pdf_keeps_lines_and_paragraph_gaps(monkeypatch):
    """A well-laid-out text PDF must not come back as one squashed paragraph."""
    _stub_assessment(monkeypatch)
    doc = svc.fitz.open()
    page = doc.new_page()
    y = 60
    for text, step in [("Entlassungsbericht der Klinik", 30),
                       ("Patient: Hartley, James", 14), ("Geburtsdatum: 03.05.1961", 14),
                       ("Aufnahme: 01.11.2026", 30),
                       ("Der Patient wurde mit akuten Brustschmerzen in unsere Klinik aufgenommen", 14),
                       ("und nach einer kardiologischen Abklärung auf die Normalstation verlegt.", 30),
                       ("Ramipril 5 mg 1-0-0", 14), ("Bisoprolol 2,5 mg 1-0-1", 14)]:
        page.insert_text((50, y), text, fontsize=10)
        y += step
    src = doc.tobytes()
    doc.close()

    text = svc.extract_text_layer(src)
    assert text.count("\n\n") == 3                       # the three visible gaps
    assert "Patient: Hartley, James\nGeburtsdatum: 03.05.1961\nAufnahme" in text

    monkeypatch.setattr(svc, "deepl_client", FakeDeepL(fn=lambda t: t.swapcase()))
    pdf, _ = svc.run_pipeline(src, ".pdf", "English", "", "A", qa_mode="off")
    out = svc.fitz.open(stream=pdf, filetype="pdf")[0].get_text()
    # field lines stay on separate lines; the wrapped sentence is one paragraph
    assert "pATIENT: hARTLEY, jAMES\ngEBURTSDATUM: 03.05.1961\naUFNAHME: 01.11.2026" in out
    assert "rAMIPRIL 5 MG 1-0-0\nbISOPROLOL 2,5 MG 1-0-1" in out
    assert "AUFGENOMMEN UND NACH" in " ".join(out.split())


def _translated_pdf(_src_bytes):
    return _fitz_pdf(SOURCE.swapcase())


def test_mode_a_whole_document_keeps_deepl_pdf(monkeypatch):
    _stub_assessment(monkeypatch)
    svc.deepl_client.doc_fn = _translated_pdf
    pdf, _ = svc.run_pipeline(_fitz_pdf(SOURCE), ".pdf", "English", "", "A",
                              uploaded_filename="Hartley_James.pdf", qa_mode="off")
    doc = svc.fitz.open(stream=pdf, filetype="pdf")
    assert len(doc) >= 3                                   # cover, translated page, originals
    assert "whole document" in " ".join(doc[0].get_text().split())
    cover = " ".join(doc[0].get_text().split())
    assert "attestation" not in cover and "follows on the next page" not in cover
    assert "Translated by DeepL, a machine translation service" in cover and "ILLEGIBLE" not in cover
    assert "eNTLASSUNGSBERICHT" in doc[1].get_text()       # DeepL's own page, untouched
    assert svc.deepl_client.calls == []                    # text API not used
    assert svc.deepl_client.doc_calls[0]["filename"] == "document.pdf"   # patient name not sent


@pytest.mark.parametrize("doc_fn", [
    None,                                        # API error
    lambda b: b,                                 # came back untranslated
    lambda b: _fitz_pdf("x"),                    # most of the content lost
    lambda b: b"not a pdf",
])
def test_whole_document_failure_falls_back_to_text_translation(monkeypatch, doc_fn):
    _stub_assessment(monkeypatch)
    svc.deepl_client.doc_fn = doc_fn
    pdf, _ = svc.run_pipeline(_fitz_pdf(SOURCE), ".pdf", "English", "", "A", qa_mode="off")
    text = _pdf_text(pdf)
    assert "Translation: DeepL |" in text and "EN<" in text
    assert len(svc.deepl_client.calls) >= 1


def test_whole_document_respects_switch_and_page_limit(monkeypatch):
    _stub_assessment(monkeypatch)
    svc.deepl_client.doc_fn = _translated_pdf
    monkeypatch.setattr(svc, "DEEPL_DOCUMENT_MAX_PAGES", 1)
    two_pages = svc.fitz.open(stream=_fitz_pdf(SOURCE), filetype="pdf")
    two_pages.insert_pdf(svc.fitz.open(stream=_fitz_pdf(SOURCE), filetype="pdf"))
    svc.run_pipeline(two_pages.tobytes(), ".pdf", "English", "", "A", qa_mode="off")
    assert not getattr(svc.deepl_client, "doc_calls", [])
    monkeypatch.setattr(svc, "DEEPL_DOCUMENT_MAX_PAGES", 0)
    monkeypatch.setattr(svc, "DEEPL_DOCUMENT_TRANSLATION", False)
    svc.run_pipeline(_fitz_pdf(SOURCE), ".pdf", "English", "", "A", qa_mode="off")
    assert not getattr(svc.deepl_client, "doc_calls", [])
