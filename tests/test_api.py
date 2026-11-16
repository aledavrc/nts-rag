from fastapi.testclient import TestClient

from nts_rag.api import app
from nts_rag.config import Settings
from nts_rag.rag import answer as answer_mod
from nts_rag.rag.answer import AnswerResult
from nts_rag.rag.retrieve import RetrievedChunk


def test_v1_models_is_openai_compatible():
    from nts_rag.api import v1_models

    payload = v1_models()

    assert payload["object"] == "list"
    assert payload["data"]
    for model in payload["data"]:
        assert model["object"] == "model"
        assert model["id"]


class _FakeRetriever:
    def search(self, query: str, *, k: int = 6, filtros=None):
        return [
            RetrievedChunk(
                id="x",
                text="Vacuna contra el VPH. Esquema de dos dosis.",
                metadata={"pagina": 1, "seccion_corta": "6.1", "tipo": "texto"},
            )
        ]


def test_run_query_skips_rerank_when_disabled(monkeypatch):
    def fail_rerank(*args, **kwargs):
        raise AssertionError("rerank_with_llm no debe llamarse con rerank=False")

    monkeypatch.setattr(answer_mod, "rerank_with_llm", fail_rerank)
    monkeypatch.setattr(
        answer_mod, "generate_answer", lambda *args, **kwargs: AnswerResult("ok", "llm")
    )

    settings = Settings(llm_base_url="http://localhost/v1", rerank=False)
    chunks, result = answer_mod.run_query(settings, _FakeRetriever(), "¿vacuna VPH?")

    assert len(chunks) == 1
    assert result.text == "ok"


def test_salud_degrades_when_the_index_is_incomplete():
    from types import SimpleNamespace

    from nts_rag.api import _estado

    state = SimpleNamespace(
        retriever=SimpleNamespace(chunk_count=900, chunks=[{}] * 994),
        settings=Settings(),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    estado = _estado(request)
    assert estado.estado == "degradado"
    assert estado.indice_listo is False
    assert estado.chunks == 900

    state.retriever = SimpleNamespace(chunk_count=994, chunks=[{}] * 994)
    assert _estado(request).estado == "ok"


def test_buscar_route_returns_sources(monkeypatch):
    monkeypatch.setattr(app.state, "retriever", _FakeRetriever(), raising=False)

    response = TestClient(app).post("/buscar", json={"consulta": "vph", "k": 2})

    assert response.status_code == 200
    fuente = response.json()["resultados"][0]
    assert fuente["pagina"] == 1
    assert fuente["seccion"] == "6.1"
    assert fuente["tipo"] == "texto"


def test_buscar_route_returns_503_without_index(monkeypatch):
    monkeypatch.setattr(app.state, "retriever", None, raising=False)
    monkeypatch.setattr(app.state, "error", "El índice no está construido.", raising=False)

    response = TestClient(app).post("/buscar", json={"consulta": "vph"})

    assert response.status_code == 503
    assert "no disponible" in response.json()["detail"]
