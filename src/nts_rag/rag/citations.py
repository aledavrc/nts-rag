"""Citas desde metadatos: documento, páginas y sección. Nunca las escribe el LLM."""

from __future__ import annotations

from nts_rag.config import DOC_TITLE
from nts_rag.rag.retrieve import RetrievedChunk


def pages(chunk: RetrievedChunk) -> str:
    if chunk.pagina_fin > chunk.pagina:
        return f"pp. {chunk.pagina}–{chunk.pagina_fin}"
    return f"p. {chunk.pagina}"


def document_label(chunk: RetrievedChunk) -> str:
    title = chunk.documento or DOC_TITLE
    return f"{title} (derogada)" if chunk.vigencia == "derogada" else title


def citation(chunk: RetrievedChunk) -> str:
    section = chunk.seccion or "sin sección"
    return f"[{document_label(chunk)}, {pages(chunk)}, sección {section}]"


def short_citation(chunk: RetrievedChunk) -> str:
    """Cita compacta para celdas de tabla: norma, página y numeral."""

    norma = "NTS 196-2022" if chunk.vigencia == "derogada" else "NTS 246-2026"
    numero = chunk.seccion_numero or chunk.seccion
    return f"[{norma}, {pages(chunk)}, {numero}]"
