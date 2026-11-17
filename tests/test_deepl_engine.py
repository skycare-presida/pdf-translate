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

    def get_source_languages(self):
        return [SimpleNamespace(name=n, code=n[:2].upper()) for n in self.languages]

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
    assert "produced by DeepL machine translation" in text


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
    assert sum("produced by DeepL" in p for p in parts) == 1
