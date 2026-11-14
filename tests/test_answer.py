from nts_rag.config import Settings
from nts_rag.rag import answer as answer_mod
from nts_rag.rag.answer import (
    AnswerResult,
    _citation,
    _extractive,
    _verify_grounded,
    generate_answer,
    is_answerable,
    rerank_with_llm,
)
from nts_rag.rag.retrieve import RetrievedChunk


def _chunk(text: str, page: int = 1, page_fin: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id="x",
        text=text,
        metadata={
            "pagina": page,
            "pagina_fin": page_fin or page,
            "seccion_corta": "I. FINALIDAD",
            "tipo": "texto",
        },
    )


def test_answerable_when_term_is_present():
    chunks = [_chunk("FINALIDAD Contribuir a disminuir la morbimortalidad.")]
    assert is_answerable("¿Cuál es la finalidad de la norma?", chunks)


def test_abstains_on_price():
    chunks = [_chunk("Vacuna contra el VPH. Esquema de dos dosis.")]
    assert not is_answerable("¿Cuál es el precio de la vacuna contra el VPH?", chunks)


def test_abstains_on_unknown_disease():
    chunks = [_chunk("Vacuna contra la hepatitis A. Composición y dosis.")]
    assert not is_answerable("¿Qué dice la norma sobre la vacuna contra el ébola?", chunks)


def test_abstains_when_topic_is_only_a_passing_mention():
    chunks = [_chunk("Población con diabetes, hipertensión o enfermedad renal crónica.")]
    assert not is_answerable("¿Cuál es el tratamiento recomendado para la diabetes tipo 2?", chunks)



def test_citation_with_page_range():
    chunk = _chunk("texto", page=24, page_fin=25)
    assert "pp. 24–25" in _citation(chunk)
    assert "I. FINALIDAD" in _citation(chunk)


def test_extractive_answer_contains_citation():
    result = _extractive([_chunk("FINALIDAD Contribuir a disminuir la morbimortalidad.")])
    assert result.mode == "extractivo"
    assert "Según la normativa cargada" in result.text
    assert "Fuente:" in result.text


def test_no_chunks_abstains():
    result = generate_answer(Settings(), "¿Cuál es la finalidad?", [])
    assert result.mode == "extractivo"
    assert "No se encontró" in result.text


def test_rerank_reorders_candidates(monkeypatch):
    settings = Settings(llm_base_url="http://localhost:11434/v1")
    chunks = [_chunk("fragmento A", 1), _chunk("fragmento B", 2)]
    monkeypatch.setattr(answer_mod, "_chat", lambda *a, **k: "[2, 1]")

    ranked = rerank_with_llm(settings, "pregunta", chunks, top_k=2)

    assert [c.pagina for c in ranked] == [2, 1]


def test_rerank_keeps_order_on_bad_output(monkeypatch):
    settings = Settings(llm_base_url="http://localhost:11434/v1")
    chunks = [_chunk("fragmento A", 1), _chunk("fragmento B", 2)]
    monkeypatch.setattr(answer_mod, "_chat", lambda *a, **k: "sin json")

    ranked = rerank_with_llm(settings, "pregunta", chunks, top_k=2)

    assert [c.pagina for c in ranked] == [1, 2]


def test_verify_grounded(monkeypatch):
    settings = Settings(llm_base_url="http://localhost:11434/v1", verify=True)
    chunks = [_chunk("FINALIDAD Contribuir a disminuir la morbimortalidad.")]
    monkeypatch.setattr(answer_mod, "_chat", lambda *a, **k: "OK")
    assert _verify_grounded(settings, "pregunta", "respuesta", chunks)

    monkeypatch.setattr(answer_mod, "_chat", lambda *a, **k: "NO")
    assert not _verify_grounded(settings, "pregunta", "respuesta", chunks)


def test_generate_falls_back_to_extractive_when_not_verified(monkeypatch):
    chunks = [_chunk("FINALIDAD Contribuir a disminuir la morbimortalidad.")]
    monkeypatch.setattr(
        answer_mod,
        "_llm_answer",
        lambda *a, **k: AnswerResult(text="una afirmación inventada", mode="llm"),
    )
    monkeypatch.setattr(answer_mod, "_verify_grounded", lambda *a, **k: False)

    result = generate_answer(Settings(), "¿Cuál es la finalidad?", chunks)

    assert result.mode == "extractivo"
    assert "Según la normativa cargada" in result.text
