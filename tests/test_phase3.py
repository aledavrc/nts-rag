"""Fase 3: matriz vacuna × campo × versión, diff entre normas y rutas estructuradas."""

import json
from pathlib import Path

import pytest

from nts_rag.config import Settings
from nts_rag.index import load_chunks
from nts_rag.rag.answer import plan_and_gather, run_query
from nts_rag.rag.fichas import (
    CURRENT,
    PREVIOUS,
    RESCATE,
    TIPO_LABELS,
    Matrix,
    classify_change,
    labels,
    remissions,
    valence,
)
from nts_rag.rag.interpret import deterministic_plan
from nts_rag.rag.structured import structured_answer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return load_chunks(ROOT / "data/chunks/nts-246.chunks.jsonl")


@pytest.fixture(scope="module")
def matrix(corpus):
    return Matrix(corpus)


class _Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, *, k=6, filtros=None):
        return []


# --- Valores normalizados ---------------------------------------------------------


def test_tipo_labels_are_multi_label():
    assert labels("Inactivados Polisacárido conjugado", TIPO_LABELS) == {
        "inactivada",
        "polisacarido",
        "conjugada",
    }
    assert "viva_atenuada" in labels("Virus Vivos Atenuados", TIPO_LABELS)
    assert "inactivada" in labels("Inactivadas o muertas", TIPO_LABELS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Vacuna PCV 20, cada dosis…", 20),
        ("Vacuna nonavalente. Cada dosis…", 9),
        ("Vacuna meningocócica tetravalente", 4),
        ("Vacuna pentavalente (DPT-HvB-Hib)", None),
    ],
)
def test_valence(text, expected):
    assert valence(text) == expected


def test_remissions():
    text = "Dos dosis. Ver: Dosis de rescate en Anexo 2 y casos especiales en ítem 6.3."
    assert remissions(text) == ["Anexo 2", "ítem 6.3"]


@pytest.mark.parametrize(
    ("old", "new", "campo", "expected"),
    [
        ("+ 2°C a +8°C", "+ 2°C a +8°C", "CONSERVACION", "sin_cambio"),
        ("0.5 ml. Ver Anexo 5", "0.5 ml", "DOSIS", "sin_cambio"),
        ("Inactivadas o muertas", "Inactivadas", "TIPO_DE_INMUNIZACION", "redaccional"),
        ("Vacuna inactivada", "Virus vivo atenuado", "TIPO_DE_INMUNIZACION", "cambio_real"),
        ("1cc y aguja 25 G x 1", "Jeringa de 1ml con aguja 25G x 1", "JERINGAS", "redaccional"),
        ("Dosis a los 6 meses", "Dosis a los 9 meses", "ESQUEMA_DE_VACUNACION", "cambio_real"),
        ("Fiebre. Alergia al huevo. Menores de 6 meses.", "Fiebre. Menores de 6 meses.",
         "CONTRAINDICACIONES", "cambio_real"),
        ("Hasta los 7 años", "Ver Anexo 2", "ESQUEMA_DE_VACUNACION", "movido"),
        # El límite pasa a la remisión: la nueva versión es parte de la anterior.
        ("Refuerzo a los 18 meses y 4 años, hasta menores de 7 años",
         "Refuerzo a los 18 meses y 4 años. Ver: Dosis de rescate en anexo 2",
         "ESQUEMA_DE_VACUNACION", "movido"),
        # Límites de edad equivalentes; "5 años 0 días" no lo es.
        ("Se aplica hasta menores de 5 años", "Se aplica hasta los 4 años 11 meses 29 días",
         RESCATE, "sin_cambio"),
        ("Se aplica hasta los 5 años 0 días", "Se aplica hasta los 4 años 11 meses 29 días",
         RESCATE, "cambio_real"),
        # Composición: añadir detalle no cambia la vacuna; cambiar una cifra, sí.
        ("Cepa RIX 4414 no menos de 10^6 DICC 50", "1 dosis (1,5 ml): cepa RIX 4414 no menos "
         "de 10^6 DICC 50, producido en células Vero", "COMPOSICION", "redaccional"),
        ("Toxoide tetánico 5 Lf", "Toxoide tetánico 2.5 Lf", "COMPOSICION", "cambio_real"),
    ],
)
def test_classify_change(old, new, campo, expected):
    assert classify_change(old, new, campo) == expected


# --- Matriz -----------------------------------------------------------------------


def test_matrix_has_every_ficha_cell(matrix):
    assert len(matrix.slugs(CURRENT)) == 24 and len(matrix.slugs(PREVIOUS)) == 18
    assert matrix.status("apo", "JERINGAS") == "no_aplica"
    assert matrix.status("hexavalente", "COMPOSICION", PREVIOUS) == "sin_ficha"
    assert matrix.get("bcg", "RESCATE") is not None  # fila del Anexo 2


def test_filter_lists_are_complete(matrix):
    vivas = matrix.filter(
        "TIPO_DE_INMUNIZACION", CURRENT, lambda c: "viva_atenuada" in labels(c.valor, TIPO_LABELS)
    )
    assert [c.slug for c in vivas] == ["bcg", "apo", "rotavirus", "spr", "sr", "varicela", "ama"]


def test_catalog_diff_finds_the_six_new_fichas(matrix):
    nuevas = [row.slug for row in matrix.catalog_diff() if row.clase == "nuevo"]
    assert nuevas == ["hexavalente", "sr", "vrs", "hexavalente_vih", "meningococo", "mab_vrs"]


@pytest.mark.parametrize(
    ("name", "minimum"),
    [("diff.jsonl", 0.9), ("diff_control.jsonl", 0.9), ("diff_control2.jsonl", 0.9)],
)
def test_diff_precision_does_not_regress(matrix, name, minimum):
    """Precisión del diff frente a los sets etiquetados a mano (§9.3 registra la cifra)."""

    nuevas = {row.slug: row.clase for row in matrix.catalog_diff()}
    rows = [json.loads(line) for line in (ROOT / "eval" / name).read_text("utf-8").splitlines()]

    def got(row):
        if row["campo"] == "FICHA":
            return nuevas.get(row["vacuna"])
        return matrix.diff_cell(row["vacuna"], row["campo"]).clase

    hits = sum(got(row) == row["esperado"] for row in rows)
    assert hits / len(rows) >= minimum


# --- Rutas estructuradas -------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "kind", "cells"),
    [
        ("Compara la conservación de la vacuna SPR y la antiamarílica", "comparacion", 2),
        ("¿Qué vacunas son de virus vivos atenuados?", "filtro", 7),
        ("¿Qué vacunas están contraindicadas en el embarazo?", "filtro", 3),
        ("¿Qué vacunas se aplican en el muslo y cuáles en el deltoides?", "sitios", 24),
        ("¿Qué vacuna previene contra más cepas?", "valencias", 6),
        ("¿Qué vacunas nuevas incorpora la NTS 246-2026 respecto a la NTS 196-2022?",
         "diff_catalogo", 6),
        ("¿Qué cambió en la dosis de rescate de la BCG entre 2022 y 2026?", "diff", 2),
        ("¿Cuál es la conservación de las vacunas?", "lista_campo", 24),
        ("¿Cuánto es el margen para la rotura de cadena de frío de cada vacuna?",
         "lista_campo", 24),
        ("¿Qué vacunas no usan jeringa?", "sin_jeringa", 2),
    ],
)
def test_structured_routes(matrix, question, kind, cells):
    result = structured_answer(deterministic_plan(question), matrix)
    assert result is not None and result.kind == kind
    assert len(result.cells) == cells
    # D11: cada celda lleva su cita.
    assert all(cell.cita in result.text for cell in result.cells)


def test_catalog_diff_answers_when_nothing_was_removed(matrix):
    """"¿Qué vacunas ya no están…?" no debe responder con las fichas nuevas."""

    result = structured_answer(
        deterministic_plan("¿Qué vacunas ya no están en la normativa actual?"), matrix
    )
    assert result is not None and result.kind == "diff_catalogo"
    assert "No existen" in result.text and not result.cells


def test_lookup_is_not_routed_to_the_matrix(matrix):
    plan = deterministic_plan("¿Cuál es la dosis de la vacuna Tdap?")
    assert structured_answer(plan, matrix) is None


def test_structured_cells_become_the_sources(corpus):
    question = "¿Qué vacunas son de virus vivos atenuados?"
    plan, evidence = plan_and_gather(_Retriever(corpus), question)
    slugs = [c.metadata["vacuna_slug"] for c in evidence.fuentes]
    assert slugs[:7] == ["bcg", "apo", "rotavirus", "spr", "sr", "varicela", "ama"]


def test_structured_answer_costs_no_llm_call(corpus, monkeypatch):
    from nts_rag.rag import answer as answer_mod

    def no_chat(settings, system, user, **kwargs):
        if system == answer_mod.SYSTEM_PROMPT:
            raise AssertionError("una ruta estructurada no redacta con el LLM")
        return None

    monkeypatch.setattr(answer_mod, "_chat", no_chat)
    settings = Settings(llm_base_url="http://localhost/v1", _env_file=None)
    _, result = run_query(
        settings, _Retriever(corpus), "Compara la conservación de la vacuna SPR y la antiamarílica"
    )
    assert result.mode == "estructurado" and result.estado == "respondida"
    assert "| Conservación |" in result.text
