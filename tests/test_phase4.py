"""Fase 4: presupuesto de llamadas y tokens, sustitución silenciosa, auth y registro."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nts_rag.api import app
from nts_rag.config import Settings
from nts_rag.evaluation import silent_substitution
from nts_rag.index import load_chunks
from nts_rag.querylog import log_query
from nts_rag.rag import answer as answer_mod
from nts_rag.rag.abstain import judge
from nts_rag.rag.answer import AnswerResult, context_chars, run_query
from nts_rag.rag.interpret import PLAN_SYSTEM, deterministic_plan
from nts_rag.rag.retrieve import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return load_chunks(ROOT / "data/chunks/nts-246.chunks.jsonl")


class _Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, *, k=6, filtros=None):
        return []


def _counting_chat(calls, *, plan="no es json"):
    def fake_chat(settings, system, user, **kwargs):
        usage = answer_mod._USAGE.get()
        budget = answer_mod._BUDGET.get()
        if usage is not None and budget is not None and len(usage) >= budget:
            return None  # mismo tope que _chat
        usage.append((100, 10))
        calls.append(system)
        if system == PLAN_SYSTEM:
            return plan
        if system == answer_mod.VERIFY_SYSTEM:
            return "OK"
        if system == answer_mod.RERANK_SYSTEM:
            return "[1]"
        return "Contiene bacilos atenuados [c1]."

    return fake_chat


# --- Presupuesto de llamadas -----------------------------------------------------


def test_worst_case_stays_within_three_calls(monkeypatch, corpus):
    """Plan del LLM inválido: sin reintento ni re-rankeo, redacción y verificación."""

    calls = []
    monkeypatch.setattr(answer_mod, "_chat", _counting_chat(calls))
    settings = Settings(llm_base_url="http://localhost/v1", rerank=True, _env_file=None)

    _, result = run_query(settings, _Retriever(corpus), "¿Cuál es la composición de la BCG?")

    assert result.llamadas <= 3
    assert calls == [PLAN_SYSTEM, answer_mod.SYSTEM_PROMPT, answer_mod.VERIFY_SYSTEM]
    assert result.plan["origen"] == "determinista"


def test_without_verification_the_plan_may_retry(monkeypatch, corpus):
    calls = []
    monkeypatch.setattr(answer_mod, "_chat", _counting_chat(calls))
    settings = Settings(llm_base_url="http://localhost/v1", verify=False, _env_file=None)

    _, result = run_query(settings, _Retriever(corpus), "¿Cuál es la composición de la BCG?")

    assert calls.count(PLAN_SYSTEM) == 2  # un reintento cabe en el presupuesto
    assert result.llamadas <= 3


def test_chat_refuses_calls_beyond_the_budget(monkeypatch):
    def fail_post(*args, **kwargs):
        raise AssertionError("no debe llamar al LLM con el presupuesto agotado")

    monkeypatch.setattr(answer_mod._client, "post", fail_post)
    usage = [(1, 1), (1, 1), (1, 1)]
    usage_token = answer_mod._USAGE.set(usage)
    budget_token = answer_mod._BUDGET.set(3)
    try:
        settings = Settings(llm_base_url="http://localhost/v1", _env_file=None)
        assert answer_mod._chat(settings, "s", "u") is None
    finally:
        answer_mod._USAGE.reset(usage_token)
        answer_mod._BUDGET.reset(budget_token)


def test_a_failed_call_still_counts(monkeypatch):
    def broken_post(*args, **kwargs):
        raise ConnectionError("sin servidor")

    monkeypatch.setattr(answer_mod._client, "post", broken_post)
    usage = []
    token = answer_mod._USAGE.set(usage)
    try:
        settings = Settings(llm_base_url="http://localhost/v1", _env_file=None)
        assert answer_mod._chat(settings, "s", "u") is None
    finally:
        answer_mod._USAGE.reset(token)
    assert usage == [(0, 0)]


def test_token_budget_depends_on_intent():
    lookup = context_chars(deterministic_plan("¿Cuál es la dosis de la vacuna Tdap?"))
    edad = context_chars(deterministic_plan("¿Qué vacunas le tocan a un niño de 5 años?"))
    assert lookup == (2000 - 600) * 4
    assert edad == (6000 - 600) * 4
    assert context_chars(None) == (12_000 - 600) * 4


# --- Sustitución silenciosa ------------------------------------------------------


def _chunk(slug, numero, doc="nts-246-2026"):
    return RetrievedChunk(
        f"{slug}-x",
        "texto",
        {"vacuna_slug": slug, "seccion_numero": numero, "documento_id": doc, "pagina": 1},
    )


def test_answering_with_another_vaccine_is_partial():
    plan = deterministic_plan("¿Cuál es la composición de la vacuna contra la hepatitis A?")
    verdict = judge(plan, [_chunk("hvb", "6.1.1.2")], answerable=True)
    assert (verdict.estado, verdict.motivo) == ("parcial", "evidencia_insuficiente")
    assert "hepatitis A" in verdict.faltan_fuentes[0]
    assert judge(plan, [_chunk("hva", "6.1.1.18")], answerable=True).estado == "respondida"


def test_silent_substitution_metric():
    plan = deterministic_plan("¿Cuál es la composición de la vacuna contra la hepatitis A?")
    answered = AnswerResult(text="…", mode="extractivo", estado="respondida")
    assert silent_substitution(plan, [_chunk("hvb", "6.1.1.2")], answered)
    assert not silent_substitution(plan, [_chunk("hva", "6.1.1.18")], answered)
    hidden = AnswerResult(text="…", mode="llm", reformulacion="Composición de la HVA")
    assert silent_substitution(plan, [_chunk("hva", "6.1.1.18")], hidden)
    shown = AnswerResult(
        text="> Pregunta interpretada como: *Composición de la HVA*\n…",
        mode="llm",
        reformulacion="Composición de la HVA",
    )
    assert not silent_substitution(plan, [_chunk("hva", "6.1.1.18")], shown)


def test_abstention_shows_the_interpreted_question():
    """Una abstención también declara cómo se leyó la pregunta (sustitución = 0)."""

    settings = Settings(_env_file=None)
    plan = deterministic_plan("¿Qué dice la norma del ébola?")
    plan.pregunta_normalizada = "Esquema de la vacuna contra el ébola"
    result = answer_mod.generate_answer(settings, plan.pregunta, [], plan=plan)
    assert result.estado == "abstenida"
    assert "Pregunta interpretada como" in result.text


# --- Clave de API y registro ------------------------------------------------------


class _SearchRetriever:
    chunk_count = 1
    chunks = [{}]

    def search(self, query, *, k=6, filtros=None):
        return [RetrievedChunk("x", "texto", {"pagina": 1, "seccion_corta": "6.1"})]


def test_api_key_protects_query_routes(monkeypatch):
    monkeypatch.setattr(app.state, "retriever", _SearchRetriever(), raising=False)
    monkeypatch.setattr(
        app.state, "settings", Settings(api_key="secreto", _env_file=None), raising=False
    )
    client = TestClient(app)
    body = {"consulta": "dosis de BCG"}

    assert client.post("/buscar", json=body).status_code == 401
    assert client.post("/buscar", json=body, headers={"X-API-Key": "otra"}).status_code == 401
    assert client.post("/buscar", json=body, headers={"X-API-Key": "secreto"}).status_code == 200
    bearer = {"Authorization": "Bearer secreto"}
    assert client.post("/buscar", json=body, headers=bearer).status_code == 200
    # La salud sigue abierta para los healthchecks.
    assert client.get("/health").status_code == 200


def test_without_api_key_routes_stay_open(monkeypatch):
    monkeypatch.setattr(app.state, "retriever", _SearchRetriever(), raising=False)
    monkeypatch.setattr(app.state, "settings", Settings(_env_file=None), raising=False)
    client = TestClient(app)
    assert client.post("/buscar", json={"consulta": "dosis de BCG"}).status_code == 200


def test_query_log_writes_one_json_line(tmp_path):
    settings = Settings(query_log=tmp_path / "logs" / "consultas.jsonl", _env_file=None)
    answer = AnswerResult(
        text="…", mode="estructurado", estado="parcial", motivo="conflicto", llamadas=0,
        plan={"intencion": "comparacion", "origen": "determinista"},
    )
    chunk = RetrievedChunk("c1", "t", {"documento_id": "nts-246-2026"})

    log_query(settings, "¿pregunta?", [chunk], answer, 12)
    log_query(settings, "¿otra?", [], answer, 3)

    lines = settings.query_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["estado"] == "parcial" and record["fuentes"] == ["c1"]
    assert record["intencion"] == "comparacion" and record["tiempo_ms"] == 12


def test_query_log_is_off_by_default(tmp_path):
    answer = AnswerResult(text="…", mode="extractivo")
    log_query(Settings(_env_file=None), "¿pregunta?", [], answer, 1)
    assert not any(tmp_path.iterdir())
