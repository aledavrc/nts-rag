from nts_rag.index import (
    expand_field_chunks,
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


def test_expand_ficha_into_field_chunks():
    chunks = expand_field_chunks([_ficha()])

    assert [c["tipo"] for c in chunks] == ["tabla", "campo", "campo", "campo"]
    campos = [c for c in chunks if c["tipo"] == "campo"]
    assert [c["campo"] for c in campos] == [
        "TIPO DE VACUNA",
        "COMPOSICIÓN",
        "DOSIS (CANTIDAD)",
    ]
    comp = campos[1]
    assert comp["embed_text"].startswith("6.1.1.10. VACUNA ANTINEUMOCÓCICA — COMPOSICIÓN")
    assert comp["texto"].startswith("**COMPOSICIÓN**")
    assert comp["seccion_numero"] == "6.1.1.10"


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


