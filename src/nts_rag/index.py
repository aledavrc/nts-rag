from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from nts_rag.config import DOC_TITLE, Settings
from nts_rag.rag.embeddings import Embedder
from nts_rag.text import short_section

# Campos que se guardan como metadatos en ChromaDB.
META_FIELDS = (
    "documento",
    "pagina",
    "pagina_fin",
    "seccion",
    "seccion_corta",
    "seccion_numero",
    "tipo",
    "campo",
    "fuente",
    "parte",
    "parte_total",
)

_SEPARATOR_RE = re.compile(r"^\|[\s:\-|]+\|$")
_LABEL_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\/().,:°'0-9\-]+$")


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Unifica un registro del JSONL o de ChromaDB en un chunk interno."""

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
    else:
        tipo = "texto"

    text = str(record.get("text") or record.get("texto") or "")
    embed_text = str(record.get("embed_text") or "")
    if not embed_text:
        embed_text = f"{section}\n{text}".strip()

    return {
        "id": str(record.get("id") or ""),
        "texto": text,
        "embed_text": embed_text,
        "documento": str(record.get("documento") or DOC_TITLE),
        "pagina": start,
        "pagina_fin": end,
        "seccion": section,
        "seccion_corta": str(record.get("seccion_corta") or short_section(section)),
        "seccion_numero": str(record.get("section_number") or record.get("seccion_numero") or ""),
        "tipo": tipo,
        "campo": str(record.get("campo") or ""),
        "fuente": str(record.get("source") or record.get("fuente") or ""),
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
    return expand_field_chunks(_unique_ids(chunks))


def _unique_ids(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evita IDs repetidos: md_to_vector corta el hash a los primeros 80 chars.

    Dos partes de una misma tabla comparten encabezado y, por tanto, hash. El
    sufijo mantiene el ID estable para un mismo archivo de chunks.
    """

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


def _is_label(value: str) -> bool:
    return bool(_LABEL_RE.match(value)) and len(value) >= 3


def _kv_rows(text: str) -> list[tuple[str, str]] | None:
    """Detecta una ficha key-value (tabla de 2 columnas con etiquetas en MAYÚSCULA).

    Las fichas 6.1.1.x tienen forma `| TIPO DE VACUNA | ... |`. Se descarta el
    encabezado y se parten las filas que parecen un campo; así una fila suelta
    (p. ej. `HvB` en el Anexo 2) no anula el resto. Las matrices de varias
    columnas siguen sin partirse.
    """

    lines = [line for line in text.splitlines() if line.lstrip().startswith("|")]
    if len(lines) < 3:
        return None
    non_sep = [line for line in lines if not _SEPARATOR_RE.match(line)]
    if not non_sep:
        return None

    rows: list[tuple[str, str]] = []
    for line in non_sep[1:]:
        cells = _split_cells(line)
        if len(cells) == 2 and _is_label(cells[0]) and cells[1]:
            rows.append((cells[0], cells[1]))
    if len(rows) < 3:
        return None
    return rows


def _field_chunk(base: dict[str, Any], label: str, value: str, index: int) -> dict[str, Any]:
    texto = f"**{label}**\n\n{value}".strip()
    return {
        "id": f"{base['id']}-campo-{index}",
        "texto": texto,
        "embed_text": f"{base['seccion_corta']} — {label}\n{value}",
        "documento": base["documento"],
        "pagina": base["pagina"],
        "pagina_fin": base["pagina_fin"],
        "seccion": base["seccion"],
        "seccion_corta": base["seccion_corta"],
        "seccion_numero": base["seccion_numero"],
        "tipo": "campo",
        "campo": label,
        "fuente": base["fuente"],
        "parte": base["parte"],
        "parte_total": base["parte_total"],
    }


def expand_field_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Añade un chunk por campo de las fichas key-value, sin quitar la tabla.

    El chunk de campo es corto y su `embed_text` lleva la sección y la etiqueta,
    lo que hace que una pregunta como "composición de la antineumocócica" acierte
    con el campo exacto en lugar de recuperar la ficha entera.
    """

    out: list[dict[str, Any]] = []
    for chunk in chunks:
        out.append(chunk)
        if chunk["tipo"] == "tabla":
            rows = _kv_rows(chunk["texto"])
            if rows:
                for index, (label, value) in enumerate(rows, start=1):
                    out.append(_field_chunk(chunk, label, value, index))
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


def build_index(settings: Settings, *, reset: bool = True) -> int:
    settings.ensure_dirs()
    chunks = load_chunks(settings.chunks_path)
    if not chunks:
        raise ValueError(f"No hay chunks en {settings.chunks_path}")

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
        shutil.rmtree(settings.chroma_dir, ignore_errors=True)
        target.rename(settings.chroma_dir)

    manifest = {
        "documento": DOC_TITLE,
        "generado_en": datetime.now(UTC).isoformat(),
        "chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "coleccion": settings.chroma_collection,
        "origen": str(settings.chunks_path),
    }
    settings.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[index] listo: {len(chunks)} chunks en {settings.chroma_dir}")
    return len(chunks)
