"""Fase 2: plan, validador, evidencia guiada, veredicto y contrato de respuesta."""

import json
from pathlib import Path

import pytest

from nts_rag.catalogo import CANONICAL_FIELDS, FICHAS, NO_APLICA
from nts_rag.config import Settings
from nts_rag.index import load_chunks
from nts_rag.rag import answer as answer_mod
from nts_rag.rag.abstain import check_shared_allergen, find_conflicts, judge
from nts_rag.rag.answer import dedupe_citations, plan_and_gather, render_citations, run_query
from nts_rag.rag.evidence import _foreign_ficha, expand_context, gather_evidence
from nts_rag.rag.interpret import (
    PLAN_SYSTEM,
    Plan,
    deterministic_plan,
    plan_query,
    question_parts,
    validate_plan,
)
from nts_rag.rag.retrieve import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return load_chunks(ROOT / "data/chunks/nts-246.chunks.jsonl")


class _Retriever:
    """Retriever sin embeddings: los slots salen de `chunks`, la búsqueda no trae nada."""

    def __init__(self, chunks, results=()):
        self.chunks = chunks
        self._results = list(results)

    def search(self, query, *, k=6, filtros=None):
        return [RetrievedChunk(c.id, c.text, dict(c.metadata), rrf=c.rrf) for c in self._results]


# --- Plan -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "intencion", "entidades", "edad"),
    [
        ("¿Cuál es la dosis de la vacuna Tdap?", "lookup", ["tdap"], None),
        ("¿Qué vacunas le tocan a un niño de 5 años?", "edad", [], 60),
        ("¿Qué alergia comparten la SPR, la antiamarílica y la influenza?", "comun",
         ["spr", "ama", "influenza"], None),
        ("¿Qué vacuna previene contra más cepas?", "max_atributo", [], None),
        ("¿Cuál es la mejor vacuna para un niño?", "evaluacion", [], None),
        ("¿Qué cambió en la BCG entre 2022 y 2026?", "comparacion_normas", ["bcg"], None),
        ("Diferencia entre pentavalente y hexavalente", "comparacion",
         ["pentavalente", "hexavalente"], None),
    ],
)
def test_deterministic_plan_intents(question, intencion, entidades, edad):
    plan = deterministic_plan(question)
    assert plan.intencion == intencion
    assert plan.entidades == entidades
    assert plan.edad_meses == edad


def test_plan_flags_rescate_ambiguity_and_therapy():
    assert deterministic_plan("niño de 5 años que no ha recibido vacunas").rescate
    assert deterministic_plan("¿Cuántas dosis se aplican?").requiere_aclaracion
    assert not deterministic_plan("¿Cuántas dosis de Tdap recibe la gestante?").requiere_aclaracion
    assert deterministic_plan("¿Cuánto paracetamol se le da a un niño?").terapia


def test_age_question_keeps_the_named_vaccine_ficha(corpus):
    """La ruta de edad no debe descartar la ficha clínica (6.1.1.x) de la vacuna."""

    question = "¿Se le puede poner la primera dosis de rotavirus a un niño de 12 meses?"
    evidence = gather_evidence(_Retriever(corpus), question, deterministic_plan(question))
    assert any(c.seccion_numero == "6.1.1.9" for c in evidence.fuentes)


def test_age_rescate_context_includes_the_rescue_limits(corpus):
    """El redactor necesita los 13 límites del Anexo 2, no solo la fila buscada."""

    question = "¿Qué vacunas debe tener un menor de 5 años que no ha recibido vacunas previas?"
    plan, evidence = plan_and_gather(_Retriever(corpus), question)
    assert plan.edad_meses is not None and plan.rescate
    limits = [
        chunk
        for chunk in evidence.contexto
        if chunk.metadata.get("vacuna_slug") and "rescate" in chunk.seccion.lower()
    ]
    assert len(limits) >= 10


def test_subqueries_split_only_between_questions():
    plan = deterministic_plan(
        "¿Qué vacunas recibe una gestante y cuáles están contraindicadas en el embarazo?"
    )
    assert plan.subconsultas == [
        "Qué vacunas recibe una gestante",
        "cuáles están contraindicadas en el embarazo",
    ]
    # "difteria y tétanos" es una sola vacuna: no se parte.
    assert deterministic_plan("¿Cuál es la dosis de difteria y tétanos?").subconsultas == []


def test_question_parts_are_the_literal_clauses():
    second = (
        "¿Qué vacunas debe tener un menor de 5 años que no ha recibido vacunas previas? "
        "¿Qué vacunas no son recuperables?"
    )
    assert question_parts(second) == [
        "Qué vacunas debe tener un menor de 5 años que no ha recibido vacunas previas",
        "Qué vacunas no son recuperables",
    ]
    both = (
        "¿Qué vacunas debe tener un menor de 5 años que no ha recibido vacunas previas? "
        "¿Qué vacunas se deben priorizar, cuáles son recuperables y cuáles no son recuperables?"
    )
    assert len(question_parts(both)) == 3


def test_llm_prompt_uses_the_literal_parts(monkeypatch):
    captured = {}

    def fake_chat(settings, system, user, **kwargs):
        captured["user"] = user
        return "respuesta [c1]"

    monkeypatch.setattr(answer_mod, "_chat", fake_chat)
    settings = Settings(llm_base_url="http://localhost/v1", _env_file=None)
    question = (
        "¿Qué vacunas debe tener un menor de 5 años que no ha recibido vacunas previas? "
        "¿Qué vacunas no son recuperables?"
    )
    answer_mod._llm_answer(
        settings, question, [_chunk(13)], None, deterministic_plan(question)
    )
    prompt = captured["user"]
    assert "Partes de la pregunta" in prompt
    assert "no son recuperables" in prompt
    assert "esquema de rescate" not in prompt.lower()


def test_dedupe_citations_collapses_repeated_ones():
    assert dedupe_citations("[A][A] texto [B] [B]") == "[A] texto [B]"


def test_validator_rejects_values_outside_the_catalogs():
    errors = validate_plan(
        {
            "intencion": "lookup",
            "entidades": [{"slug": "ebola"}],
            "campos": ["PRECIO"],
            "version": "2030",
            "edad_meses": -3,
            "subconsultas": ["a"] * 5,
        }
    )
    assert len(errors) == 5


def _llm_plan(**overrides):
    data = {
        "pregunta_normalizada": "Composición de la vacuna BCG",
        "intencion": "lookup",
        "entidades": [{"slug": "bcg"}],
        "campos": ["COMPOSICION"],
        "edad_meses": None,
        "version": "vigente",
        "subconsultas": [],
        "supuestos": [],
        "premisas_no_soportadas": [],
        "requiere_aclaracion": False,
    }
    data.update(overrides)
    return json.dumps(data)


def test_plan_query_uses_a_valid_llm_plan():
    plan = plan_query("¿de qué está hecha la BCG?", lambda system, user: _llm_plan())
    assert plan.origen == "llm"
    assert plan.campos == ["COMPOSICION"]
    assert plan.pregunta_normalizada == "Composición de la vacuna BCG"


def test_llm_plan_cannot_invent_entities_or_premises():
    """Un modelo real lista las 24 fichas y duda de la pregunta; el catálogo manda."""

    plan = plan_query(
        "¿Qué vacunas debe tener un niño de 5 años?",
        lambda system, user: _llm_plan(
            intencion="conjunto",
            entidades=[
                {"slug": slug}
                for slug in ("bcg", "hvb", "hexavalente", "pentavalente", "ipv", "apo")
            ],
            premisas_no_soportadas=["La norma clasifica las vacunas en prioritarias"],
            requiere_aclaracion=True,
        ),
    )
    assert plan.origen == "llm"
    assert plan.entidades == []
    assert plan.premisas_no_soportadas == []
    assert not plan.requiere_aclaracion


def test_plan_query_retries_once_then_falls_back():
    calls = []

    def chat(system, user):
        calls.append(user)
        return _llm_plan(entidades=[{"slug": "ebola"}])

    plan = plan_query("¿Cuál es la dosis de la vacuna Tdap?", chat)
    assert len(calls) == 2 and "Corrige" in calls[1]
    assert plan.origen == "determinista"
    assert plan.entidades == ["tdap"]
    assert plan.avisos


# --- Evidencia ----------------------------------------------------------------


def test_slots_bring_the_exact_field_of_each_named_vaccine(corpus):
    plan = deterministic_plan("Diferencia de conservación entre la SPR y la AMA")
    evidence = gather_evidence(_Retriever(corpus), plan.pregunta, plan, k=6)
    found = {(c.metadata["vacuna_slug"], c.campo) for c in evidence.fuentes if c.tipo == "campo"}
    assert {("spr", "CONSERVACION"), ("ama", "CONSERVACION")} <= found


def test_age_route_brings_calendar_and_rescue(corpus):
    plan = deterministic_plan("¿Se puede vacunar con BCG a un niño de 6 años?")
    evidence = gather_evidence(_Retriever(corpus), plan.pregunta, plan, k=6)
    rescue = [c for c in evidence.fuentes if c.tipo == "fila" and "RESCATE" in c.seccion_ruta]
    assert rescue and rescue[0].metadata["vacuna_slug"] == "bcg"


def test_comparison_of_norms_marks_new_vaccines_as_absent_in_2022(corpus):
    plan = deterministic_plan("¿Qué cambió en la composición de la hexavalente entre 2022 y 2026?")
    evidence = gather_evidence(_Retriever(corpus), plan.pregunta, plan, k=6)
    assert ("hexavalente", "FICHA", "nts-196-2022") in evidence.faltantes
    assert any(c.documento_id == "nts-246-2026" for c in evidence.fuentes)


def test_foreign_ficha_is_contamination(corpus):
    plan = deterministic_plan("¿Cuál es la composición de la vacuna contra la hepatitis A?")
    hvb = next(c for c in corpus if c["tipo"] == "campo" and c["vacuna_slug"] == "hvb")
    hva = next(c for c in corpus if c["tipo"] == "campo" and c["vacuna_slug"] == "hva")
    assert _foreign_ficha(RetrievedChunk(hvb["id"], hvb["texto"], hvb), plan)
    assert not _foreign_ficha(RetrievedChunk(hva["id"], hva["texto"], hva), plan)


def test_section_expansion_brings_every_field_of_the_ficha(corpus):
    """Métrica de Fase 2: expansión de sección 13/13 (12/13 donde no aplica)."""

    retriever = _Retriever(corpus)
    plan = Plan(pregunta="x")
    complete = 0
    for doc_id, fichas in FICHAS.items():
        for number, slug in fichas.items():
            seed = next(
                c
                for c in corpus
                if c["tipo"] == "campo"
                and c["documento_id"] == doc_id
                and c["seccion_numero"] == number
            )
            seed_chunk = RetrievedChunk(seed["id"], seed["texto"], seed)
            context = expand_context(retriever, [seed_chunk], plan)
            campos = {c.campo for c in context if c.tipo == "campo"}
            expected = {f for f in CANONICAL_FIELDS if slug not in NO_APLICA.get(f, set())}
            complete += campos == expected
    assert complete == 42


# --- Veredicto ---------------------------------------------------------------


def _source():
    return [RetrievedChunk("x", "texto", {"pagina": 1, "seccion_corta": "6.1", "tipo": "texto"})]


def _verdict(question, sources=None, answerable=True):
    sources = _source() if sources is None else sources
    return judge(deterministic_plan(question), sources, answerable=answerable)


def test_judge_order_of_states():
    assert _verdict("¿Cuál es la mejor vacuna?").motivo == "requiere_conocimiento_externo"
    assert _verdict("¿Qué dice la norma del ébola?", [], False).motivo == "fuera_de_alcance"
    assert _verdict("¿Cuántas dosis se aplican?").estado == "aclaracion"
    assert _verdict("¿Qué vacuna tiene más cepas?").motivo == "metrica_no_comparable"
    verdict = judge(
        deterministic_plan("¿Cuál es la dosis de la vacuna Tdap?"),
        _source(),
        answerable=True,
        faltantes=[("tdap", "DOSIS", "nts-246-2026")],
    )
    assert (verdict.estado, verdict.motivo) == ("parcial", "evidencia_insuficiente")


def test_conflict_between_ficha_and_calendar(corpus):
    plan = deterministic_plan("¿Cuándo se coloca el anticuerpo monoclonal?")
    conflicts = find_conflicts(plan, corpus)
    assert conflicts and conflicts[0]["vacuna"] == "mab_vrs"
    assert "ficha 6.1.1.24: recién nacido" in conflicts[0]["valores"]
    # Sin discrepancia no hay conflicto: BCG al nacer en ficha y calendario.
    assert find_conflicts(deterministic_plan("¿Cuándo se aplica la BCG?"), corpus) == []


def test_shared_allergen_premise(corpus):
    plan = deterministic_plan("¿Qué alergia comparten la SPR, la antiamarílica y la influenza?")
    assert check_shared_allergen(plan, corpus)
    plan = deterministic_plan("¿Qué alergia comparten la SPR y la SR?")
    assert check_shared_allergen(plan, corpus) == []


# --- Respuesta ---------------------------------------------------------------


def _chunk(page, doc="NTS N° 246-MINSA/DGIESP-2026", vigencia="vigente"):
    return RetrievedChunk(
        f"c{page}",
        f"texto {page}",
        {"pagina": page, "seccion_corta": "6.1.1.1", "documento": doc, "vigencia": vigencia},
    )


def test_placeholders_become_metadata_citations_and_memory_ones_are_dropped():
    chunks = [_chunk(13), _chunk(50, "NTS N° 196-MINSA/DGIESP-2022", "derogada")]
    text, valid = render_citations(
        "BCG al nacer [c1]. Antes, igual [c2]. Inventada [NTS 246, p. 99, sección X]. [c9]",
        chunks,
    )
    assert valid == 2
    assert "[NTS 246-2026, p. 13, 6.1.1.1]" in text
    assert "[NTS 196-2022, p. 50, 6.1.1.1]" in text
    assert "p. 99" not in text and "[c9]" not in text


def test_llm_answer_without_citations_is_rejected(monkeypatch):
    monkeypatch.setattr(answer_mod, "_chat", lambda *a, **k: "Se aplica al nacer.")
    settings = Settings(llm_base_url="http://localhost/v1")
    assert answer_mod._llm_answer(settings, "pregunta", [_chunk(13)]) is None


def test_with_llm_the_plan_replaces_the_rerank(monkeypatch, corpus):
    calls = []

    def fake_chat(settings, system, user, **kwargs):
        usage = answer_mod._USAGE.get()
        usage.append((100, 10))
        calls.append(system)
        if system == PLAN_SYSTEM:
            return _llm_plan()
        if system == answer_mod.VERIFY_SYSTEM:
            return "OK"
        return "Contiene bacilos atenuados [c1]."

    def fail_rerank(*args, **kwargs):
        raise AssertionError("con plan del LLM no se re-rankea")

    monkeypatch.setattr(answer_mod, "_chat", fake_chat)
    monkeypatch.setattr(answer_mod, "rerank_with_llm", fail_rerank)
    settings = Settings(llm_base_url="http://localhost/v1", _env_file=None)

    chunks, result = run_query(settings, _Retriever(corpus), "¿de qué está hecha la BCG?")

    assert calls[0] == PLAN_SYSTEM
    assert result.llamadas == 3 and result.tokens_entrada == 300
    assert result.mode == "llm" and result.estado == "respondida"
    assert result.reformulacion == "Composición de la vacuna BCG"
    assert "[NTS 246-2026" in result.text
    assert chunks[0].campo == "COMPOSICION" and chunks[0].metadata["vacuna_slug"] == "bcg"


def test_extractive_query_makes_no_llm_call(corpus):
    settings = Settings(_env_file=None)
    _, result = run_query(settings, _Retriever(corpus), "¿Cuál es la composición de la vacuna BCG?")
    assert result.llamadas == 0 and result.mode == "extractivo"
    assert result.plan["intencion"] == "lookup"
