from __future__ import annotations

import gc
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from nts_rag import catalogo
from nts_rag.catalogo import (
    CANONICAL_FIELDS,
    FICHA_HEADERS,
    FICHAS,
    NO_APLICA,
    canonical_fields,
    detect_vaccines,
    ficha_slug,
    is_domain_section,
)
from nts_rag.config import (
    DOCUMENTS,
    INDEX_SCHEMA_VERSION,
    Settings,
    document_for_source,
)
from nts_rag.rag.embeddings import Embedder
from nts_rag.text import age_limit_months, age_range_months, normalize_for_match, short_section

# Campos que se guardan como metadatos en ChromaDB (esquema v2, §6.2).
META_FIELDS = (
    "documento_id",
    "documento",
    "version",
    "vigencia",
    "pagina",
    "pagina_fin",
    "seccion",
    "seccion_corta",
    "seccion_numero",
    "tipo",
    "campo",
    "campo_original",
    "vacuna_slug",
    "vacunas",
    "edad_min_meses",
    "edad_max_meses",
    "fuente",
    "parte",
    "parte_total",
)

TIPOS = ("texto", "tabla", "campo", "fila")
CATALOG_KEY = "src/nts_rag/catalogo.py"

_SEPARATOR_RE = re.compile(r"^\|[\s:\-|]+\|$")
_LABEL_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\/().,:°'0-9\-]+$")
_FICHA_NUMBER_RE = re.compile(r"^6\.1\.1\.\d+$")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Unifica un registro del JSONL o de ChromaDB en un chunk interno (v2)."""

    section = str(record.get("section") or record.get("seccion") or "")
    pages = record.get("pages") or []
    start = record.get("page_start") or record.get("pagina")
    end = record.get("page_end") or record.get("pagina_fin")
    if start is None and pages:
        start = pages[0]
    if end is None and pages:
        end = pages[-1]
    start = int(start or 0)
    end = int(end or start)

    raw_type = str(record.get("type") or record.get("tipo") or "text")
    if raw_type in {"table", "tabla"}:
        tipo = "tabla"
    elif raw_type in {"campo", "field"}:
        tipo = "campo"
    elif raw_type in {"fila", "row"}:
        tipo = "fila"
    else:
        tipo = "texto"

    text = str(record.get("text") or record.get("texto") or "")
    embed_text = str(record.get("embed_text") or "")
    if not embed_text:
        embed_text = f"{section}\n{text}".strip()

    fuente = str(record.get("source") or record.get("fuente") or "")
    doc = DOCUMENTS.get(str(record.get("documento_id") or "")) or document_for_source(fuente)
    seccion_numero = str(record.get("section_number") or record.get("seccion_numero") or "")

    return {
        "id": str(record.get("id") or ""),
        "texto": text,
        "embed_text": embed_text,
        "documento_id": doc.id,
        "documento": doc.titulo,
        "version": doc.version,
        "vigencia": str(record.get("vigencia") or doc.vigencia),
        "pagina": start,
        "pagina_fin": end,
        "seccion": section,
        "seccion_corta": str(record.get("seccion_corta") or short_section(section)),
        "seccion_numero": seccion_numero,
        "tipo": tipo,
        "campo": str(record.get("campo") or ""),
        "campo_original": str(record.get("campo_original") or ""),
        "vacuna_slug": str(
            record.get("vacuna_slug") or ficha_slug(doc.id, seccion_numero) or ""
        ),
        "vacunas": str(record.get("vacunas") or ""),
        "edad_min_meses": _optional_int(record.get("edad_min_meses")),
        "edad_max_meses": _optional_int(record.get("edad_max_meses")),
        "fuente": fuente,
        "parte": int(record.get("part") or record.get("parte") or 0),
        "parte_total": int(record.get("parts") or record.get("parte_total") or 1),
    }


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(normalize_record(json.loads(line)))
    return _unique_ids(expand_row_chunks(expand_field_chunks(_unique_ids(chunks))))


def _unique_ids(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sufijo solo para duplicados reales, en orden de aparición (D14)."""

    seen: dict[str, int] = {}
    for chunk in chunks:
        base = chunk["id"] or "chunk"
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            chunk["id"] = f"{base}-{count + 1}"
    return chunks


def _split_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _table_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("|") and not _SEPARATOR_RE.match(line.strip())
    ]


def _is_label(value: str) -> bool:
    return bool(_LABEL_RE.match(value)) and len(value) >= 3


def is_ficha(chunk: dict[str, Any]) -> bool:
    number = chunk.get("seccion_numero") or ""
    return bool(_FICHA_NUMBER_RE.match(number)) and number in FICHAS.get(
        chunk.get("documento_id") or "", {}
    )


def ficha_rows(text: str) -> list[tuple[str, str]]:
    """Filas `etiqueta → valor` de una tabla de ficha 6.1.1.x.

    Acepta N columnas (el valor une el resto de celdas) y cualquier número de
    filas, incluida la continuación de una sola fila en la página siguiente
    (causa raíz 9). Se descarta el encabezado `Característica`; una fila sin
    etiqueta continúa el valor de la anterior.
    """

    rows: list[tuple[str, str]] = []
    for line in _table_lines(text):
        cells = _split_cells(line)
        if not cells:
            continue
        label, rest = cells[0], [cell for cell in cells[1:] if cell]
        if normalize_for_match(label) in FICHA_HEADERS:
            continue
        value = " ".join(rest)
        if not label and rows and value:
            previous_label, previous_value = rows[-1]
            rows[-1] = (previous_label, f"{previous_value} {value}".strip())
        elif _is_label(label) and value:
            rows.append((label, value))
    return rows


def _field_chunk(
    base: dict[str, Any], canonical: str, label: str, value: str
) -> dict[str, Any]:
    return {
        **base,
        "id": f"{base['id']}-{canonical.lower()}",
        "texto": f"**{label}**\n\n{value}".strip(),
        "embed_text": f"{base['seccion_corta']} — {label}\n{value}",
        "tipo": "campo",
        "campo": canonical,
        "campo_original": label,
    }


def expand_field_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Añade un chunk por campo canónico de las fichas, sin quitar la tabla.

    Solo se parten las fichas 6.1.1.x del catálogo: el resto de tablas de dos
    columnas (p. ej. el Anexo 2) se parte en filas (`expand_row_chunks`). Una
    etiqueta fuera del catálogo no genera campo; la detecta `corpus_lint`.
    """

    out: list[dict[str, Any]] = []
    for chunk in chunks:
        out.append(chunk)
        if chunk["tipo"] != "tabla" or not is_ficha(chunk):
            continue
        for label, value in ficha_rows(chunk["texto"]):
            for canonical in canonical_fields(label):
                out.append(_field_chunk(chunk, canonical, label, value))
    return out


def _row_chunk(
    base: dict[str, Any],
    header: list[str],
    cells: list[str],
    header_line: str,
    line: str,
    index: int,
) -> dict[str, Any]:
    pairs = [
        f"{name}: {value}" if name else value
        for name, value in zip(header, cells, strict=False)
        if value
    ]
    row_text = " ".join(cell for cell in cells if cell)
    vacunas = detect_vaccines(row_text)
    # Si la primera celda nombra una sola vacuna (Anexo 2: "| DPT | … |"), es la
    # vacuna de la fila aunque el resto mencione otras.
    first = detect_vaccines(cells[0]) if cells else []
    if len(first) == 1:
        vacunas = [first[0], *[slug for slug in vacunas if slug != first[0]]]
    edades = age_range_months(cells[0]) if cells else None
    edad_min, edad_max = edades if edades else (None, age_limit_months(row_text))
    separator = "|" + "|".join(" --- " for _ in header) + "|"
    return {
        **base,
        "id": f"{base['id']}-fila-{index}",
        "texto": "\n".join([header_line, separator, line]),
        "embed_text": f"{base['seccion_corta']}\n" + "; ".join(pairs),
        "tipo": "fila",
        "campo": "",
        "campo_original": "",
        "vacuna_slug": vacunas[0] if len(vacunas) == 1 or len(first) == 1 else "",
        "vacunas": ",".join(vacunas),
        "edad_min_meses": edad_min,
        "edad_max_meses": edad_max,
    }


def expand_row_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Añade un chunk por fila de las tablas de dominio (D2), sin quitar la tabla.

    Cada fila lleva el encabezado para leerse sola, y sus vacunas y edades
    normalizadas. Las tablas operativas (presupuesto, registros) no se parten.
    """

    out: list[dict[str, Any]] = []
    for chunk in chunks:
        out.append(chunk)
        if chunk["tipo"] != "tabla" or is_ficha(chunk):
            continue
        if not is_domain_section(chunk["seccion"], chunk["seccion_numero"]):
            continue
        lines = _table_lines(chunk["texto"])
        if len(lines) < 2:
            continue
        header = _split_cells(lines[0])
        for index, line in enumerate(lines[1:], start=1):
            cells = _split_cells(line)
            if any(cells):
                out.append(_row_chunk(chunk, header, cells, lines[0], line, index))
    return out


def load_from_collection(collection: Any) -> list[dict[str, Any]]:
    data = collection.get(include=["documents", "metadatas"])
    chunks: list[dict[str, Any]] = []
    for chunk_id, text, metadata in zip(
        data["ids"], data["documents"], data["metadatas"], strict=False
    ):
        chunks.append(normalize_record({"id": chunk_id, "texto": text, **(metadata or {})}))
    return chunks


def _metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key in META_FIELDS:
        value = chunk.get(key)
        if value is None or value == "":
            continue
        metadata[key] = value
    return metadata


def corpus_lint(chunks: list[dict[str, Any]]) -> list[str]:
    """Errores estructurales del corpus indexado (D13). Lista vacía = 100%.

    Comprueba metadatos de documento, que estén todas las fichas del Anexo A con
    sus 13 campos (salvo los `NO_APLICA`), que ninguna etiqueta de ficha quede
    fuera del catálogo y que todas las páginas de cada norma estén cubiertas.
    """

    errors: list[str] = []
    for chunk in chunks:
        if chunk.get("documento_id") not in DOCUMENTS or not chunk.get("vigencia"):
            errors.append(f"{chunk['id']}: sin documento_id o vigencia")
        if chunk.get("tipo") not in TIPOS:
            errors.append(f"{chunk['id']}: tipo desconocido {chunk.get('tipo')!r}")

    for doc in DOCUMENTS.values():
        own = [chunk for chunk in chunks if chunk.get("documento_id") == doc.id]
        if not own:
            continue
        pages: set[int] = set()
        for chunk in own:
            pages.update(range(chunk["pagina"], chunk["pagina_fin"] + 1))
        missing = sorted(set(range(1, doc.paginas + 1)) - pages)
        if missing:
            errors.append(f"{doc.id}: páginas sin chunks {missing}")
        outside = sorted(page for page in pages if page < 1 or page > doc.paginas)
        if outside:
            errors.append(f"{doc.id}: páginas fuera de rango {outside}")

        for number, slug in FICHAS.get(doc.id, {}).items():
            tables = [
                c for c in own if c["tipo"] == "tabla" and c["seccion_numero"] == number
            ]
            if not tables:
                errors.append(f"{doc.id} {number} ({slug}): ficha ausente")
                continue
            for table in tables:
                for label, _ in ficha_rows(table["texto"]):
                    if not canonical_fields(label):
                        errors.append(f"{doc.id} {number}: etiqueta fuera del catálogo {label!r}")
            present = {
                c["campo"] for c in own if c["tipo"] == "campo" and c["seccion_numero"] == number
            }
            expected = {
                field for field in CANONICAL_FIELDS if slug not in NO_APLICA.get(field, set())
            }
            if expected - present:
                errors.append(
                    f"{doc.id} {number} ({slug}): faltan campos {sorted(expected - present)}"
                )
    return errors


def _sha256(path: Path) -> str:
    """Hash del contenido con saltos de línea normalizados a LF.

    Git puede guardar LF y extraer CRLF (core.autocrlf en Windows): sin
    normalizar, el mismo corpus daría hashes distintos según la máquina.
    """

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_paths(settings: Settings) -> list[Path]:
    """Archivos de los que depende el índice: Markdown del registro + chunks."""

    paths = [settings.corpus_dir / name for doc in DOCUMENTS.values() for name in doc.fuentes]
    return [*paths, settings.chunks_path]


def source_hashes(settings: Settings) -> dict[str, str]:
    hashes = {path.as_posix(): _sha256(path) for path in source_paths(settings) if path.exists()}
    # El catálogo genera metadatos (vacunas, campos, tablas de dominio): si cambia,
    # el índice queda desfasado aunque el corpus y el JSONL sigan iguales.
    hashes[CATALOG_KEY] = _sha256(Path(catalogo.__file__))
    return hashes


def read_manifest(settings: Settings) -> dict[str, Any] | None:
    if not settings.manifest_path.exists():
        return None
    return json.loads(settings.manifest_path.read_text(encoding="utf-8"))


def freshness_problems(settings: Settings) -> list[str]:
    """Desfases entre el manifiesto del índice y los archivos actuales (D13)."""

    manifest = read_manifest(settings)
    if manifest is None:
        return ["no hay manifiesto: ejecuta `make index`"]
    problems: list[str] = []
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        problems.append(
            f"schema_version {manifest.get('schema_version')} ≠ {INDEX_SCHEMA_VERSION}: reindexa"
        )
    recorded = manifest.get("fuentes") or {}
    current = source_hashes(settings)
    for path in sorted(set(recorded) | set(current)):
        if path not in current:
            problems.append(f"{path}: falta el archivo")
        elif path not in recorded:
            problems.append(f"{path}: no está en el manifiesto")
        elif recorded[path] != current[path]:
            problems.append(f"{path}: cambió desde el último índice")
    return problems


def get_collection(settings: Settings, *, reset: bool = False, path: Path | None = None):
    path = path or settings.chroma_dir
    if reset:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def _release_chroma() -> None:
    """Cierra los clientes de Chroma en caché: en Windows retienen chroma.sqlite3
    abierto y el directorio no se puede renombrar."""

    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001 - API interna; sin ella basta con gc
        pass
    gc.collect()


def _swap_index(new: Path, current: Path) -> None:
    """Sustituye el índice sin dejar nunca el directorio vacío.

    El anterior se aparta a `<nombre>.old` antes de mover el nuevo; si el
    movimiento falla, se restaura.
    """

    backup = current.with_name(f"{current.name}.old")
    shutil.rmtree(backup, ignore_errors=True)
    if current.exists():
        current.rename(backup)
    try:
        new.rename(current)
    except OSError:
        if backup.exists() and not current.exists():
            backup.rename(current)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def build_manifest(settings: Settings, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    por_documento = Counter(chunk["documento_id"] for chunk in chunks)
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generado_en": datetime.now(UTC).isoformat(),
        "chunks": len(chunks),
        "tipos": dict(sorted(Counter(chunk["tipo"] for chunk in chunks).items())),
        "documentos": [
            {
                "id": doc.id,
                "titulo": doc.titulo,
                "version": doc.version,
                "vigencia": doc.vigencia,
                "chunks": por_documento.get(doc.id, 0),
            }
            for doc in DOCUMENTS.values()
        ],
        "embedding_model": settings.embedding_model,
        "coleccion": settings.chroma_collection,
        "origen": settings.chunks_path.as_posix(),
        "fuentes": source_hashes(settings),
    }


def build_index(settings: Settings, *, reset: bool = True) -> int:
    settings.ensure_dirs()
    chunks = load_chunks(settings.chunks_path)
    if not chunks:
        raise ValueError(f"No hay chunks en {settings.chunks_path}")

    errors = corpus_lint(chunks)
    for error in errors:
        print(f"[lint] {error}")
    print(f"[lint] {'OK' if not errors else f'{len(errors)} problemas'}")

    # Se construye aparte y se cambia al final: si falla a mitad, el índice
    # anterior sigue sirviendo.
    target = settings.chroma_dir.with_name(f"{settings.chroma_dir.name}.tmp") if reset else None
    collection = get_collection(settings, reset=reset, path=target)
    embedder = Embedder(settings)

    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embedder.embed_documents(chunk["embed_text"] for chunk in batch)
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["texto"] for chunk in batch],
            embeddings=embeddings,
            metadatas=[_metadata(chunk) for chunk in batch],
        )
        print(f"[index] {min(start + batch_size, len(chunks))}/{len(chunks)}")

    if target is not None:
        del collection
        _release_chroma()
        _swap_index(target, settings.chroma_dir)

    settings.manifest_path.write_text(
        json.dumps(build_manifest(settings, chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[index] listo: {len(chunks)} chunks en {settings.chroma_dir}")
    return len(chunks)
