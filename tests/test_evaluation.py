import json
from pathlib import Path

from nts_rag.evaluation import (
    ESTADOS,
    RowResult,
    expected_sources,
    expected_state,
    row_type,
    score_retrieval,
    source_matches,
    summarize,
    summarize_by_type,
)
from nts_rag.rag.retrieve import RetrievedChunk

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def _chunk(
    seccion: str, page: int, page_fin: int | None = None, documento_id: str | None = None
) -> RetrievedChunk:
    metadata = {
        "pagina": page,
        "pagina_fin": page_fin or page,
        "seccion_corta": seccion,
        "seccion": f"NTS > {seccion}",
        "tipo": "tabla",
    }
    if documento_id:
        metadata["documento_id"] = documento_id
    return RetrievedChunk(id=seccion, text="", metadata=metadata)


def test_legacy_row_is_lookup_with_one_source():
    row = {"pregunta": "x", "pagina": 22, "seccion": "6.1.1.8"}
    assert row_type(row) == "lookup"
    assert expected_sources(row) == [{"pagina": 22, "seccion": "6.1.1.8"}]
    assert expected_state(row) == "respondida"


def test_legacy_null_page_is_negative_expected_to_abstain():
    row = {"pregunta": "x", "pagina": None, "seccion": None}
    assert row_type(row) == "negativa"
    assert expected_sources(row) == []
    assert expected_state(row) == "abstenida"


def test_section_number_needs_boundary():
    bcg = {"pagina": 13, "pagina_fin": 14, "seccion": "6.1.1.1"}
    assert source_matches(_chunk("6.1.1.1. VACUNA BCG", 13, 14), bcg)
    assert not source_matches(_chunk("6.1.1.10. VACUNA ANTINEUMOCÓCICA", 13, 14), bcg)


def test_annex_number_needs_boundary():
    anexo1 = {"pagina": 79, "seccion": "ANEXO 1"}
    assert source_matches(_chunk("ANEXO 1 > ADMINISTRACIÓN SIMULTÁNEA", 79), anexo1)
    assert not source_matches(_chunk("ANEXO 10 > REGISTRO", 79), anexo1)


def test_page_ranges_overlap():
    source = {"pagina": 45, "pagina_fin": 47, "seccion": "6.2.2"}
    assert source_matches(_chunk("6.2.2. ESQUEMA", 46, 47), source)
    assert not source_matches(_chunk("6.2.2. ESQUEMA", 48), source)


def test_document_defaults_to_current_norm():
    source_2022 = {"documento": "nts-196-2022", "pagina": 50, "seccion": "ANEXO 2"}
    assert not source_matches(_chunk("ANEXO 2", 50), source_2022)
    assert source_matches(_chunk("ANEXO 2", 50, documento_id="nts-196-2022"), source_2022)


def test_score_retrieval_recall_completeness_precision():
    sources = [
        {"pagina": 45, "pagina_fin": 47, "seccion": "6.2.2"},
        {"pagina": 80, "seccion": "ANEXO 2"},
    ]
    chunks = [_chunk("6.2.2. ESQUEMA", 45), _chunk("ANEXO 20", 121), _chunk("ANEXO 2", 80)]

    top2 = score_retrieval(chunks, sources, 2)
    assert top2.hit and not top2.completa
    assert top2.recall == 0.5
    assert top2.precision == 0.5

    top3 = score_retrieval(chunks, sources, 3)
    assert top3.completa
    assert top3.recall == 1.0
    assert round(top3.precision, 2) == 0.67


def test_summary_counts_abstention_and_type():
    found = RowResult(row={"pregunta": "a", "pagina": 22, "seccion": "6.1.1.8"}, tipo="lookup")
    found.por_k[5] = score_retrieval([_chunk("6.1.1.8", 22)], expected_sources(found.row), 5)
    found.estado = "respondida"

    negative = RowResult(row={"pregunta": "b", "pagina": None}, tipo="negativa")
    negative.por_k[5] = score_retrieval([], [], 5)
    negative.estado = "respondida"  # no se abstuvo: error

    edge = RowResult(
        row={
            "pregunta": "c",
            "tipo": "borde",
            "fuentes_esperadas": [{"pagina": 90, "seccion": "ANEXO 6"}],
            "estado_esperado": "parcial",
            "motivo_esperado": "conflicto",
        },
        tipo="borde",
    )
    edge.por_k[5] = score_retrieval([], expected_sources(edge.row), 5)
    edge.estado = "respondida"  # decisión correcta, estado inexacto

    summary = summarize([found, negative, edge], [5])
    assert summary["con_fuentes"] == 2
    assert summary["@5"]["hit"] == 1
    assert summary["@5"]["recall"] == 0.5
    assert summary["estado"]["decision_correcta"] == 2
    assert summary["estado"]["estado_exacto"] == 1
    assert summary["estado"]["abstiene"] == [0, 1]
    assert summary["estado"]["motivo_exacto"] == 0
    assert summary["reformulacion_util"] is None

    by_type = summarize_by_type([found, negative, edge], [5])
    assert set(by_type) == {"borde", "lookup", "negativa"}


def test_eval_files_are_well_formed():
    for path in sorted(EVAL_DIR.glob("*.jsonl")):
        ids = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["pregunta"].strip(), path
            assert expected_state(row) in ESTADOS, (path, row)
            if "id" in row:
                assert row["id"] not in ids, (path, row["id"])
                ids.add(row["id"])
            for source in expected_sources(row):
                assert source.get("seccion"), (path, row)
                start = source.get("pagina")
                assert start is None or start <= (source.get("pagina_fin") or start), (path, row)
            if expected_state(row) == "abstenida" and "fuentes_esperadas" in row:
                assert not row["fuentes_esperadas"], (path, row)
