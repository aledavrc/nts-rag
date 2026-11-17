from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from nts_rag import __version__
from nts_rag.config import DOC_TITLE, get_settings
from nts_rag.rag.answer import run_query
from nts_rag.rag.formatting import clip_markdown
from nts_rag.rag.retrieve import Retriever
from nts_rag.schemas import (
    BusquedaRequest,
    BusquedaResponse,
    ConsultaRequest,
    ConsultaResponse,
    Fuente,
    SaludResponse,
)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.retriever = None
    app.state.error = None
    try:
        app.state.retriever = Retriever(settings)
    except Exception as exc:  # noqa: BLE001
        app.state.error = str(exc)
        logging.getLogger(__name__).warning("Índice no disponible: %s", exc)
    yield


app = FastAPI(
    title="NTS 246 RAG",
    description=f"Consulta con RAG de la {DOC_TITLE}",
    version=__version__,
    lifespan=lifespan,
)


def _retriever(request: Request) -> Retriever:
    retriever = request.app.state.retriever
    if retriever is None:
        detail = request.app.state.error or "El índice no está construido."
        raise HTTPException(status_code=503, detail=f"Servicio no disponible: {detail}")
    return retriever


def _fuente(chunk) -> Fuente:
    return Fuente(
        chunk_id=chunk.id,
        documento=chunk.metadata.get("documento", DOC_TITLE),
        documento_id=chunk.metadata.get("documento_id"),
        version=chunk.metadata.get("version"),
        vigencia=chunk.metadata.get("vigencia"),
        pagina=chunk.pagina,
        pagina_fin=chunk.pagina_fin,
        seccion=chunk.seccion,
        seccion_numero=chunk.seccion_numero or None,
        tipo=chunk.tipo,
        campo=chunk.campo,
        campo_original=chunk.campo_original,
        vacuna_slug=chunk.metadata.get("vacuna_slug") or None,
        fragmento=clip_markdown(chunk.text, 1200),
        score=chunk.score,
    )


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "servicio": "NTS 246 RAG",
        "documento": DOC_TITLE,
        "docs": "/docs",
        "salud": "/salud",
        "health": "/health",
    }


def _estado(request: Request) -> SaludResponse:
    retriever = request.app.state.retriever
    settings = request.app.state.settings
    indexed = retriever.chunk_count if retriever is not None else 0
    expected = len(retriever.chunks) if retriever is not None else 0
    # Listo solo si el índice tiene exactamente los chunks esperados: uno
    # parcial o desfasado responde, pero no es fiable.
    listo = 0 < indexed == expected
    return SaludResponse(
        estado="ok" if listo else "degradado",
        indice_listo=listo,
        chunks=indexed,
        coleccion=settings.chroma_collection,
        embedding_model=settings.embedding_model,
        llm_configurado=bool(settings.llm_base_url),
        version=__version__,
    )


@app.get("/salud", response_model=SaludResponse, tags=["salud"])
def salud(request: Request) -> SaludResponse:
    return _estado(request)


@app.get("/health", response_model=SaludResponse, tags=["salud"])
def health(request: Request) -> SaludResponse:
    """Alias en inglés de /salud, para healthchecks y orquestadores."""
    return _estado(request)


@app.get("/v1/models", include_in_schema=False)
def v1_models() -> dict:
    """Stub compatible con OpenAI para clientes que sondean el endpoint.

    El RAG no expone /v1/chat/completions; esto solo evita el 404 en la consola.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "nts-rag",
                "object": "model",
                "created": 0,
                "owned_by": "nts-rag",
            }
        ],
    }


@app.post("/consulta", response_model=ConsultaResponse)
def consulta(request: Request, body: ConsultaRequest) -> ConsultaResponse:
    retriever = _retriever(request)
    settings = request.app.state.settings
    started = time.perf_counter()
    chunks, answer = run_query(settings, retriever, body.pregunta, k=body.k, filtros=body.filtros)
    elapsed = int((time.perf_counter() - started) * 1000)
    return ConsultaResponse(
        respuesta=answer.text,
        fuentes=[_fuente(chunk) for chunk in chunks],
        modo=answer.mode,  # type: ignore[arg-type]
        estado=answer.estado,  # type: ignore[arg-type]
        motivo=answer.motivo,  # type: ignore[arg-type]
        reformulacion=answer.reformulacion,
        supuestos=answer.supuestos,
        faltan_fuentes=answer.faltan_fuentes,
        conflictos=answer.conflictos,  # type: ignore[arg-type]
        llamadas_llm=answer.llamadas,
        tokens_entrada=answer.tokens_entrada,
        tokens_salida=answer.tokens_salida,
        tiempo_ms=elapsed,
    )


@app.post("/buscar", response_model=BusquedaResponse)
def buscar(request: Request, body: BusquedaRequest) -> BusquedaResponse:
    retriever = _retriever(request)
    started = time.perf_counter()
    chunks = retriever.search(body.consulta, k=body.k, filtros=body.filtros)
    elapsed = int((time.perf_counter() - started) * 1000)
    return BusquedaResponse(
        consulta=body.consulta,
        resultados=[_fuente(chunk) for chunk in chunks],
        tiempo_ms=elapsed,
    )
