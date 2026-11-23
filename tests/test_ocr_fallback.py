"""
Tests for the local OCR fallback: a page the vision model cannot read must not
fail the whole document. No network and no Tesseract binary needed.

    pytest tests/
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "service"))
import pdf_translate_v3_service as svc  # noqa: E402

PAGE = ("Informe de alta hospitalaria del servicio de cardiologia.\n"
        "El paciente ingreso el dia tres de mayo por dolor toracico agudo y fue trasladado a planta.\n"
        "Tratamiento: Ramipril 5 mg cada 24 horas.")


def _response(text, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=text))])


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setattr(svc, "preprocess_image_bytes", lambda b, label="": b)
    monkeypatch.setattr(svc, "_tesseract_installed", {"eng", "spa", "ukr", "osd"})
    monkeypatch.setattr(svc, "TRANSLATE_ENGINE", "llm")          # translation is stubbed
    monkeypatch.setattr(svc, "translate_text_with_model",
                        lambda text, tl, model: "EN: " + text[:40])
    monkeypatch.setattr(svc, "_looks_untranslated", lambda *a, **k: False)


def _vision(monkeypatch, fn):
    calls = []

    def create(**kw):
        n_images = sum(1 for c in kw["messages"][0]["content"] if c["type"] == "image_url")
        calls.append((n_images, kw["max_tokens"]))
        return fn(n_images)
    monkeypatch.setattr(svc.ovh_client.chat.completions, "create", create)
    return calls


def test_ocr_budget_scales_with_pages(monkeypatch):
    ok = _response(PAGE + '\n```json\n{"language":"Spanish","confidence":0.9}\n```')
    calls = _vision(monkeypatch, lambda n: ok)
    svc.ocr_images_with_assessment(["aGk="], "")
    svc.ocr_images_with_assessment(["aGk=", "aGk="], "")
    assert calls == [(1, 6600), (2, 12600)]
    assert all(budget < svc.MAX_OUTPUT_TOKENS for _, budget in calls)


def test_runaway_single_page_is_read_locally(monkeypatch):
    """The 21 Nov failure: pages 1-2 overflow, page 1 alone overflows again."""
    good = _response("Segunda pagina.\n```json\n{\"language\":\"Spanish\",\"confidence\":0.9}\n```")
    state = {"single": 0}

    def vision(n):
        if n == 2:
            return _response("x" * 500, finish="length")
        state["single"] += 1
        return _response("x" * 500, finish="length") if state["single"] == 1 else good
    _vision(monkeypatch, vision)
    seen = []
    monkeypatch.setattr(svc, "ocr_with_tesseract",
                        lambda imgs, label="", script="": seen.append(script) or PAGE)

    parts, assessment, _, source = svc.translate_all_images(["aGk=", "aGk="], "English", "")
    assert "Informe de alta" in source and "Segunda pagina" in source
    assert assessment["local_ocr_pages"] == ["pages 1-2"]
    assert assessment["language"] == "Spanish"          # taken from the page that was assessed
    assert assessment["no_text_pages"] == 0
    assert "Local OCR (Tesseract)" in svc._provenance_notes(assessment)
    assert seen == [""]                                  # script left to Tesseract to detect


def test_timeout_on_single_page_also_falls_back(monkeypatch):
    def boom(n):
        raise svc.APITimeoutError(request=None)
    _vision(monkeypatch, boom)
    monkeypatch.setattr(svc, "ocr_with_tesseract", lambda imgs, label="", script="": PAGE)
    text, assessment = svc.ocr_images_with_assessment(["aGk="], "", "page 1")
    assert text == PAGE and assessment["ocr_engine"] == "tesseract"
    assert assessment["language"] == "unknown" and assessment["confidence"] == 0.0


def test_page_with_no_text_gets_marker_not_failure(monkeypatch):
    ok = _response(PAGE + '\n```json\n{"language":"Spanish","confidence":0.9}\n```')
    monkeypatch.setattr(svc, "MAX_IMAGES_PER_CALL", 1)
    state = {"n": 0}

    def vision(n):
        state["n"] += 1
        return ok if state["n"] == 1 else _response("y" * 500, finish="length")
    _vision(monkeypatch, vision)
    monkeypatch.setattr(svc, "ocr_with_tesseract", lambda imgs, label="", script="": " .. \n")

    parts, assessment, _, _ = svc.translate_all_images(["aGk=", "aGk="], "English", "")
    assert svc.NO_TEXT_MARKER in parts
    assert assessment["no_text_pages"] == 1
    assert "NOT TRANSLATED" in svc._provenance_notes(assessment)


def test_document_with_no_readable_page_still_fails(monkeypatch):
    _vision(monkeypatch, lambda n: _response("z" * 500, finish="length"))
    monkeypatch.setattr(svc, "ocr_with_tesseract", lambda imgs, label="", script="": "")
    with pytest.raises(svc.UnreadableScan):
        svc.translate_all_images(["aGk="], "English", "")


def test_local_ocr_overproduction_and_missing_tesseract_fail(monkeypatch):
    _vision(monkeypatch, lambda n: _response("z" * 500, finish="length"))
    monkeypatch.setattr(svc, "ocr_with_tesseract",
                        lambda imgs, label="", script="": "word " * 5000)
    with pytest.raises(svc.UnreadableScan):
        svc.ocr_images_with_assessment(["aGk="], "")
    monkeypatch.setattr(svc, "_tesseract_installed", set())
    with pytest.raises(svc.UnreadableScan):
        svc.ocr_images_with_assessment(["aGk="], "")


def test_cyrillic_over_ceiling_still_uses_cyrillic_packs(monkeypatch):
    long = "слово " * 3000 + '\n```json\n{"language":"Ukrainian","script":"Cyrillic","confidence":0.8}\n```'
    _vision(monkeypatch, lambda n: _response(long))
    seen = []
    monkeypatch.setattr(svc, "ocr_with_tesseract",
                        lambda imgs, label="", script="": seen.append(script) or "Виписка з історії хвороби пацієнта лікарні")
    text, assessment = svc.ocr_images_with_assessment(["aGk="], "")
    assert seen == ["cyrillic"] and assessment["language"] == "Ukrainian"
