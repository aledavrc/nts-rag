from nts_rag.cli import _print_sources, _source_groups
from nts_rag.rag.retrieve import RetrievedChunk


def _chunk(page, section, tipo, slug=None):
    return RetrievedChunk(
        f"{page}-{section}-{slug}",
        "texto",
        {
            "pagina": page,
            "pagina_fin": page,
            "seccion_corta": section,
            "tipo": tipo,
            "vacuna_slug": slug,
        },
    )


def test_sources_group_by_section():
    chunks = [
        _chunk(80, "ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE", "fila", "bcg"),
        _chunk(80, "ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE", "fila", "hvb"),
        _chunk(45, "6.2.2. ESQUEMA DE VACUNACIÓN", "tabla"),
    ]
    groups = _source_groups(chunks)
    assert len(groups) == 2
    assert groups[0]["vacunas"] == ["bcg", "hvb"]
    assert groups[1]["tipos"] == ["tabla"]


def test_print_sources_is_compact_and_names_a_single_vaccine(capsys):
    chunks = [_chunk(13, "6.1.1.1. VACUNA BCG", "campo", "bcg")]
    _print_sources(chunks)
    out = capsys.readouterr().out
    assert "Fuentes (1 sección):" in out
    assert "vacuna BCG" in out


def test_print_sources_groups_several_vaccines_without_repeating_the_section(capsys):
    chunks = [
        _chunk(80, "ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE", "fila", "bcg"),
        _chunk(80, "ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE", "fila", "hvb"),
    ]
    _print_sources(chunks)
    out = capsys.readouterr().out
    assert "Fuentes (1 sección):" in out
    assert "varias vacunas" in out
    assert out.count("ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE") == 1


def test_print_sources_only_shows_the_cited_sections(capsys):
    chunks = [
        _chunk(80, "ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE", "fila", "bcg"),
        _chunk(45, "6.2.2. ESQUEMA DE VACUNACIÓN", "tabla"),
    ]
    _print_sources(
        chunks, text="Ver [NTS 246-2026, p. 80, ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE]."
    )
    out = capsys.readouterr().out
    assert "ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE" in out
    assert "6.2.2. ESQUEMA DE VACUNACIÓN" not in out


def test_print_sources_does_not_name_one_row_of_a_multi_vaccine_annex(capsys):
    chunks = [_chunk(80, "ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE", "fila", "bcg")]
    _print_sources(chunks)
    out = capsys.readouterr().out
    assert "vacuna BCG" not in out
    assert "ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE" in out
