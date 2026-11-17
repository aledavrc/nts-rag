"""Esquema v2: catálogos, edades, versión y filtros de vigencia."""

import pytest

from nts_rag.catalogo import (
    CANONICAL_FIELDS,
    FICHAS,
    FIELD_VARIANTS,
    canonical_fields,
    detect_vaccines,
    is_domain_section,
)
from nts_rag.rag.retrieve import Retriever, _where, resolve_vigencias
from nts_rag.schemas import Filtros
from nts_rag.text import (
    age_limit_months,
    age_range_months,
    detect_vigencias,
    tokenize,
)


def test_catalog_maps_only_to_the_13_canonical_fields():
    assert len(CANONICAL_FIELDS) == 13
    assert {field for fields in FIELD_VARIANTS.values() for field in fields} == set(
        CANONICAL_FIELDS
    )
    assert canonical_fields("POSIBLES EFECTOS POST-VACUNALES") == ("EFECTOS_POSTVACUNALES",)
    assert canonical_fields("VÍA Y SITIO DE ADMINISTRACIÓN") == ("VIA_Y_SITIO",)
    assert canonical_fields("PRECIO") == ()


def test_fichas_match_annex_a():
    assert len(FICHAS["nts-246-2026"]) == 24
    assert len(FICHAS["nts-196-2022"]) == 18
    assert FICHAS["nts-196-2022"]["6.1.1.3"] == FICHAS["nts-246-2026"]["6.1.1.4"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Vacuna Pentavalente (DPT-HvB-Hib) o Hexavalente (DPT-HvB-Hib-IPV)",
         ["pentavalente", "hexavalente"]),
        ("VACUNA COMBINADA HEXAVALENTE (DTaP-HvB-Hib-IPV) (NIÑAS Y NIÑOS CON VIH)",
         ["hexavalente_vih"]),
        ("Vacuna dT adulto", ["dt_adulto"]),
        ("VACUNA CONTRA DIFTERIA Y TÉTANOS (DT PEDIÁTRICO)", ["dt_pediatrico"]),
        ("Vacuna SR/SPR", ["sr", "spr"]),
        ("Haemophilus influenzae tipo b", ["hib"]),
        ("Anticuerpo Monoclonal contra el Virus Respiratorio Sincitial", ["mab_vrs"]),
        ("Vacuna contra la Influenza adulto", ["influenza"]),
    ],
)
def test_detect_vaccines_avoids_known_traps(text, expected):
    assert detect_vaccines(text) == expected


def test_domain_sections_are_recognised_by_title_not_annex_number():
    assert is_domain_section("NTS > VI > 6.2.2. ESQUEMA", "6.2.2")
    assert is_domain_section("NTS > ANEXO 5 > VÍAS Y TÉCNICAS DE ADMINISTRACIÓN", "")
    assert not is_domain_section("NTS > ANEXO 20 > PROGRAMACIÓN PRESUPUESTAL", "")
    assert not is_domain_section("NTS > VI > 6.20 OTRA", "6.20")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Recién Nacido", (0, 0)),
        ("2 meses", (2, 2)),
        ("4 años", (48, 59)),
        ("5 a 11 años", (60, 143)),
        ("De 12 a 17 años", (144, 215)),
        ("Menor de 2 años", (0, 23)),
        ("Mayores de 60 años", (720, None)),
        ("Embarazo de 20 a 36 semanas", None),
    ],
)
def test_age_range_months(text, expected):
    assert age_range_months(text) == expected


def test_age_limit_takes_the_largest_hasta():
    text = "hasta los 11meses 29 días. Después de un 1 año hasta los 4 años 11 meses 29 días"
    assert age_limit_months(text) == 59
    assert age_limit_months("sin límite") is None


def test_age_is_no_longer_lost_in_tokenize():
    question = tokenize("¿Qué vacunas debe tener un niño de 5 años?")
    row = tokenize("| 5 a 11 años | Vacuna SR/SPR |")
    assert "edad5a" in question
    assert "edad5a" in row and "edad11a" in row
    assert "ano" in question


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("¿Cuál es la dosis de la vacuna Tdap?", ("vigente",)),
        ("¿Qué vacunas debe tener un niño sin vacunas previas?", ("vigente",)),
        ("¿Qué hacer antes de vacunar?", ("vigente",)),
        ("¿Qué decía la NTS 196 sobre la BCG?", ("derogada",)),
        ("¿Qué decía la norma anterior sobre la BCG?", ("derogada",)),
        ("¿Qué cambió en la BCG entre 2022 y 2026?", ("vigente", "derogada")),
        ("¿Qué vacunas nuevas incorpora la NTS 246-2026?", ("vigente", "derogada")),
    ],
)
def test_detect_vigencias(question, expected):
    assert detect_vigencias(question) == expected


def test_explicit_filter_overrides_question_signals():
    assert resolve_vigencias("¿Qué cambió en 2022?", Filtros(vigencia="vigente")) == ("vigente",)
    assert resolve_vigencias("dosis", Filtros(vigencia="todas")) == ("vigente", "derogada")
    assert resolve_vigencias("dosis", Filtros(documento="nts-196-2022")) == (
        "vigente",
        "derogada",
    )


def test_where_filters_current_norm_by_default():
    assert _where(None, ("vigente",)) == {"vigencia": {"$eq": "vigente"}}
    assert _where(None, ("vigente", "derogada")) is None
    assert _where(Filtros(documento="nts-196-2022"), ("vigente", "derogada")) == {
        "documento_id": {"$eq": "nts-196-2022"}
    }


def test_matches_excludes_other_vigencia():
    chunk = {"vigencia": "derogada", "documento_id": "nts-196-2022"}
    assert not Retriever._matches(chunk, None, ("vigente",))
    assert Retriever._matches(chunk, None, ("vigente", "derogada"))
    assert not Retriever._matches(chunk, Filtros(documento="nts-246-2026"), None)
