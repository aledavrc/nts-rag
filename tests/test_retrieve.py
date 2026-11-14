from nts_rag.index import normalize_record
from nts_rag.rag.retrieve import (
    Retriever,
    _where,
    section_number_in_query,
)
from nts_rag.schemas import Filtros
from nts_rag.text import expand_tokens, tokenize


def _chunk(**overrides):
    record = {
        "id": "abc",
        "section": "NTS 246 > VI > 6.1.1.10. VACUNA ANTINEUMOCÓCICA",
        "section_number": "6.1.1.10",
        "pages": [24, 25],
        "page_start": 24,
        "page_end": 25,
        "type": "table",
        "text": "| COMPOSICIÓN | PCV 20 |",
    }
    record.update(overrides)
    return normalize_record(record)


def test_where_without_filters_is_none():
    assert _where(None) is None
    assert _where(Filtros()) is None


def test_where_combines_type_and_page_range():
    where = _where(Filtros(tipo="tabla", pagina=24))
    assert where == {
        "$and": [
            {"tipo": {"$eq": "tabla"}},
            {"pagina": {"$lte": 24}},
            {"pagina_fin": {"$gte": 24}},
        ]
    }


def test_matches_page_range_and_type():
    chunk = _chunk()
    assert Retriever._matches(chunk, Filtros(pagina=24))
    assert Retriever._matches(chunk, Filtros(pagina=25))
    assert not Retriever._matches(chunk, Filtros(pagina=26))
    assert Retriever._matches(chunk, Filtros(tipo="tabla"))
    assert not Retriever._matches(chunk, Filtros(tipo="texto"))


def test_matches_section_by_substring():
    chunk = _chunk()
    assert Retriever._matches(chunk, Filtros(seccion="antineumocócica"))
    assert Retriever._matches(chunk, Filtros(seccion="6.1.1.10"))
    assert Retriever._matches(chunk, Filtros(seccion="6.1.1"))
    assert not Retriever._matches(chunk, Filtros(seccion="rotavirus"))


def test_section_number_is_not_a_suffix_match():
    chunk = _chunk(
        section="NTS 246 > VI > 6.6.2 COMPONENTES > 6.6.2.1 MICROPLANIFICACION",
        section_number="6.6.2.1",
    )
    assert not Retriever._matches(chunk, Filtros(seccion="6.2.1"))
    assert Retriever._matches(chunk, Filtros(seccion="6.6.2"))


def test_section_number_in_query_uses_boundaries():
    assert section_number_in_query("6.1.1.17", "¿Qué dice el 6.1.1.17?")
    assert not section_number_in_query("6.1.1.1", "¿Qué dice el 6.1.1.17?")
    assert not section_number_in_query("6.1.1.1", "¿Qué dice el 6.1.1.10?")


def test_query_expansion_for_gestante():
    tokens = expand_tokens(set(tokenize("¿Qué vacunas se aplican a gestantes?")))
    assert "embarazo" in tokens
    assert "gestacion" in tokens
    assert "prenatal" in tokens



