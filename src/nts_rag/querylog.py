"""Registro de consultas en JSONL (Fase 4). Desactivado salvo `NTS_RAG_QUERY_LOG`.

Una línea por consulta con lo necesario para auditar y medir coste: estado,
motivo, intención, fuentes citadas, llamadas y tokens. Las preguntas pueden
contener datos de salud: el registro es opt-in y el archivo no se versiona.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any

from nts_rag.config import Settings

logger = logging.getLogger(__name__)
_lock = threading.Lock()


def log_query(
    settings: Settings,
    pregunta: str,
    chunks: list[Any],
    answer: Any,
    tiempo_ms: int,
    *,
    ruta: str = "/consulta",
) -> None:
    if settings.query_log is None:
        return
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "ruta": ruta,
        "pregunta": pregunta,
        "estado": answer.estado,
        "motivo": answer.motivo,
        "modo": answer.mode,
        "intencion": (answer.plan or {}).get("intencion"),
        "origen_plan": (answer.plan or {}).get("origen"),
        "fuentes": [chunk.id for chunk in chunks],
        "documentos": sorted({chunk.documento_id for chunk in chunks if chunk.documento_id}),
        "llamadas_llm": answer.llamadas,
        "tokens_entrada": answer.tokens_entrada,
        "tokens_salida": answer.tokens_salida,
        "tiempo_ms": tiempo_ms,
    }
    try:
        settings.query_log.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock, open(settings.query_log, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:  # el registro nunca debe tumbar una consulta
        logger.warning("No se pudo escribir el registro de consultas: %s", exc)
