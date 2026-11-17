from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Filtros(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seccion: str | None = None
    pagina: int | None = Field(default=None, ge=1)
    tipo: Literal["texto", "tabla", "campo", "fila"] | None = None
    # Documento exacto (`nts-246-2026`, `nts-196-2022`). Sin él, la vigencia decide.
    documento: Literal["nts-246-2026", "nts-196-2022"] | None = None
    # Sin valor: solo la norma vigente, salvo que la pregunta nombre otra versión.
    vigencia: Literal["vigente", "derogada", "todas"] | None = None


class ConsultaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pregunta: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=6, ge=1, le=12)
    filtros: Filtros | None = None


class BusquedaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consulta: str = Field(min_length=2, max_length=1000)
    k: int = Field(default=6, ge=1, le=12)
    filtros: Filtros | None = None


class Fuente(BaseModel):
    chunk_id: str
    documento: str
    documento_id: str | None = None
    version: int | None = None
    vigencia: str | None = None
    pagina: int
    pagina_fin: int
    seccion: str
    seccion_numero: str | None = None
    tipo: str
    campo: str | None = None
    campo_original: str | None = None
    vacuna_slug: str | None = None
    fragmento: str
    score: float


class ConsultaResponse(BaseModel):
    respuesta: str
    fuentes: list[Fuente]
    modo: Literal["llm", "extractivo"]
    formato: Literal["markdown"] = "markdown"
    tiempo_ms: int


class BusquedaResponse(BaseModel):
    consulta: str
    resultados: list[Fuente]
    formato: Literal["markdown"] = "markdown"
    tiempo_ms: int


class SaludResponse(BaseModel):
    estado: Literal["ok", "degradado"]
    indice_listo: bool
    chunks: int
    coleccion: str
    embedding_model: str
    llm_configurado: bool
    version: str
