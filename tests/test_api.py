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
