import pytest

import nts_rag.index as index_mod
from nts_rag.config import Settings
from nts_rag.index import (
    build_index,
    corpus_lint,
    expand_field_chunks,
    expand_row_chunks,
    load_chunks,
    normalize_record,
)


def test_normalize_table_record_from_jsonl():
    record = {
        "id": "abc",
        "source": "parte-1.md",
        "section": "NTS 246 > VI > 6.1.1.10. VACUNA ANTINEUMOCÓCICA",
        "section_number": "6.1.1.10",
        "pages": [24, 25],
        "page_start": 24,
        "page_end": 25,
        "type": "table",
        "part": 1,
        "parts": 1,
        "text": "| COMPOSICIÓN | PCV 20 |",
        "embed_text": "6.1.1.10 VACUNA ANTINEUMOCÓCICA (págs. 24–25)\n| COMPOSICIÓN | PCV 20 |",
        "chars": 30,
    }

    chunk = normalize_record(record)

    assert chunk["id"] == "abc"
    assert chunk["pagina"] == 24
    assert chunk["pagina_fin"] == 25
    assert chunk["seccion_corta"] == "6.1.1.10. VACUNA ANTINEUMOCÓCICA"
    assert chunk["seccion_numero"] == "6.1.1.10"
    assert chunk["tipo"] == "tabla"
    assert chunk["fuente"] == "parte-1.md"
    assert chunk["texto"].startswith("| COMPOSICIÓN")
    assert chunk["embed_text"].startswith("6.1.1.10")


def test_normalize_text_record_from_chroma():
    record = {
        "id": "x",
        "texto": "Contribuir a disminuir la morbimortalidad.",
        "pagina": 1,
        "pagina_fin": 1,
        "seccion": "NTS 246 > I. FINALIDAD",
        "seccion_corta": "I. FINALIDAD",
        "tipo": "texto",
    }

    chunk = normalize_record(record)

    assert chunk["pagina"] == 1
    assert chunk["seccion_corta"] == "I. FINALIDAD"
    assert chunk["tipo"] == "texto"
    assert chunk["embed_text"] == (
        "NTS 246 > I. FINALIDAD\nContribuir a disminuir la morbimortalidad."
    )


def test_load_chunks_makes_duplicate_ids_unique(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "abc", "section": "X > Y", "type": "table",'
                ' "part": 1, "parts": 2, "text": "a"}',
                '{"id": "abc", "section": "X > Y", "type": "table",'
                ' "part": 2, "parts": 2, "text": "b"}',
            ]
        ),
        encoding="utf-8",
    )

    chunks = load_chunks(path)

    assert [chunk["id"] for chunk in chunks] == ["abc", "abc-2"]


def _ficha(**overrides):
    record = {
        "id": "f1",
        "section": "NTS 246 > VI > 6.1.1.10. VACUNA ANTINEUMOCÓCICA",
        "section_number": "6.1.1.10",
        "pages": [24, 25],
        "type": "table",
        "text": (
            "| Característica | Descripción |\n"
            "| --- | --- |\n"
            "| TIPO DE VACUNA | Inactivado Conjugada |\n"
            "| COMPOSICIÓN | PCV 20, cada dosis de 0.5 ml |\n"
            "| DOSIS (CANTIDAD) | 0.5 ml |"
        ),
    }
    record.update(overrides)
    return normalize_record(record)


def test_expand_ficha_into_canonical_field_chunks():
    chunks = expand_field_chunks([_ficha()])

    assert [c["tipo"] for c in chunks] == ["tabla", "campo", "campo", "campo"]
    campos = [c for c in chunks if c["tipo"] == "campo"]
    assert [c["campo"] for c in campos] == ["TIPO_DE_INMUNIZACION", "COMPOSICION", "DOSIS"]
    assert [c["campo_original"] for c in campos] == [
        "TIPO DE VACUNA",
        "COMPOSICIÓN",
        "DOSIS (CANTIDAD)",
    ]
    comp = campos[1]
    assert comp["embed_text"].startswith("6.1.1.10. VACUNA ANTINEUMOCÓCICA — COMPOSICIÓN")
    assert comp["texto"].startswith("**COMPOSICIÓN**")
    assert comp["seccion_numero"] == "6.1.1.10"
    assert comp["vacuna_slug"] == "neumococo"
    assert comp["id"] == "f1-composicion"


def test_record_gets_document_metadata_from_its_source():
    current = _ficha(source="8265031-nts-n-246-minsa-digiesp-2026-parte-1.md")
    old = _ficha(source="nts-n-196-minsa-digiesp-2022.md", section_number="6.1.1.9")

    assert (current["documento_id"], current["vigencia"], current["version"]) == (
        "nts-246-2026",
        "vigente",
        2026,
    )
    assert (old["documento_id"], old["vigencia"], old["version"]) == (
        "nts-196-2022",
        "derogada",
        2022,
    )
    # Mismo numeral, otra vacuna: la ficha se alinea por slug, no por numeral.
    assert old["vacuna_slug"] == "neumococo"
    assert _ficha(source="nts-n-196-minsa-digiesp-2022.md")["vacuna_slug"] == "spr"


def test_combined_label_maps_to_two_canonical_fields():
    table = _ficha(
        text="| Característica | Descripción |\n| --- | --- |\n"
        "| ESQUEMA DE VACUNACIÓN Y DOSIS | 2 dosis de 0.5 ml |"
    )
    campos = [c["campo"] for c in expand_field_chunks([table]) if c["tipo"] == "campo"]
    assert campos == ["ESQUEMA_DE_VACUNACION", "DOSIS"]


def test_one_row_continuation_of_a_ficha_is_a_field():
    # La continuación en la página siguiente trae una sola fila y sin encabezado.
    table = _ficha(
        text="| USO SIMULTÁNEO CON OTRAS VACUNAS | Puede administrarse |\n| --- | --- |"
    )
    campos = [c["campo"] for c in expand_field_chunks([table]) if c["tipo"] == "campo"]
    assert campos == ["USO_SIMULTANEO"]


def test_unknown_ficha_label_is_reported_by_lint():
    table = _ficha(
        source="8265031-nts-n-246-minsa-digiesp-2026-parte-1.md",
        text="| Característica | Descripción |\n| --- | --- |\n| PRECIO | 10 soles |",
    )
    errors = corpus_lint(expand_field_chunks([table]))
    assert any("etiqueta fuera del catálogo 'PRECIO'" in error for error in errors)
    assert any("6.1.1.1 (bcg): ficha ausente" in error for error in errors)


def test_does_not_expand_matrix_table():
    table = _ficha(
        text="| N° | DENOMINACIÓN | TIPO |\n|---|---|---|\n| 1 | Vacuna BCG | Vivo |"
    )
    assert expand_field_chunks([table]) == [table]


def test_does_not_expand_mixed_case_two_columns():
    table = _ficha(
        text="| BCG | Si no recibe la vacuna, administrar lo antes posible |\n"
        "|---|---|\n"
        "| HvB | Si no se administró en las primeras 12 horas |"
    )
    assert expand_field_chunks([table]) == [table]


def _anexo2(**overrides):
    record = {
        "id": "a2",
        "source": "8265031-nts-n-246-minsa-digiesp-2026-parte-1.md",
        "section": "NTS > ANEXO 2 > ESQUEMAS DE VACUNACIÓN CON DOSIS DE RESCATE",
        "section_number": "",
        "pages": [80],
        "type": "table",
        "text": (
            "| VACUNA | PRECISIONES |\n"
            "|---|---|\n"
            "| BCG | Después de 1 año hasta los 4 años 11 meses 29 días |\n"
            "| ROTAVIRUS | Inicia desde la captación hasta los 8 meses 0 días. |\n"
            "| DPT | Refuerzo; si no recibió Hexavalente |"
        ),
    }
    record.update(overrides)
    return normalize_record(record)


def test_domain_table_is_split_into_rows_not_fields():
    chunks = expand_row_chunks(expand_field_chunks([_anexo2()]))

    assert [c["tipo"] for c in chunks] == ["tabla", "fila", "fila", "fila"]
    bcg, rotavirus, dpt = chunks[1:]
    assert bcg["id"] == "a2-fila-1"
    assert bcg["texto"].splitlines()[0] == "| VACUNA | PRECISIONES |"
    assert bcg["embed_text"].endswith("VACUNA: BCG; PRECISIONES: Después de 1 año hasta los "
                                      "4 años 11 meses 29 días")
    assert (bcg["vacuna_slug"], bcg["edad_max_meses"]) == ("bcg", 59)
    assert (rotavirus["vacuna_slug"], rotavirus["edad_max_meses"]) == ("rotavirus", 8)
    # La primera celda decide la vacuna aunque la fila nombre otra.
    assert dpt["vacuna_slug"] == "dpt"
    assert dpt["vacunas"] == "dpt,hexavalente"


def test_operational_table_is_not_split():
    table = _anexo2(section="NTS > ANEXO 20 > PROGRAMACIÓN PRESUPUESTAL")
    assert expand_row_chunks([table]) == [table]


def test_calendar_rows_carry_age_range():
    table = _anexo2(
        section="NTS > VI > 6.2.2. ESQUEMA DE VACUNACIÓN DEL NIÑO",
        section_number="6.2.2",
        text="| EDAD | TIPO DE VACUNA | DOSIS |\n|---|---|---|\n"
        "| 5 a 11 años | Vacuna SR/SPR | Dosis única |",
    )
    row = expand_row_chunks([table])[1]
    assert (row["edad_min_meses"], row["edad_max_meses"]) == (60, 143)
    assert row["vacunas"] == "sr,spr"
    assert row["vacuna_slug"] == ""


def test_failed_build_keeps_the_previous_index(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        chunks_path=tmp_path / "chunks.jsonl",
    )
    settings.chunks_path.write_text(
        '{"id": "a", "section": "X > Y", "type": "text", "text": "hola"}',
        encoding="utf-8",
    )
    settings.chroma_dir.mkdir(parents=True)
    (settings.chroma_dir / "keep").write_text("índice anterior", encoding="utf-8")

    class BrokenEmbedder:
        def __init__(self, _settings):
            raise RuntimeError("sin modelo")

    monkeypatch.setattr(index_mod, "Embedder", BrokenEmbedder)
    with pytest.raises(RuntimeError):
        build_index(settings)

    assert (settings.chroma_dir / "keep").exists()


