from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DOC_ID = "nts-246-2026"
DOC_TITLE = "NTS N° 246-MINSA/DGIESP-2026"

# Versión del esquema de chunks/metadatos del índice. Cambiarla obliga a reindexar.
INDEX_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DocumentInfo:
    """Una norma del corpus. `fuentes` son los Markdown auditados que la componen."""

    id: str
    titulo: str
    version: int
    vigencia: str  # "vigente" | "derogada"
    paginas: int
    fuentes: tuple[str, ...]


DOCUMENTS: dict[str, DocumentInfo] = {
    doc.id: doc
    for doc in (
        DocumentInfo(
            id="nts-246-2026",
            titulo=DOC_TITLE,
            version=2026,
            vigencia="vigente",
            paginas=140,
            fuentes=(
                "8265031-nts-n-246-minsa-digiesp-2026-parte-1.md",
                "8265031-nts-n-246-minsa-digiesp-2026-parte-2.md",
            ),
        ),
        DocumentInfo(
            id="nts-196-2022",
            titulo="NTS N° 196-MINSA/DGIESP-2022",
            version=2022,
            vigencia="derogada",
            paginas=103,
            fuentes=("nts-n-196-minsa-digiesp-2022.md",),
        ),
    )
}

VIGENCIAS = ("vigente", "derogada")


def document_for_source(source: str) -> DocumentInfo:
    """Documento al que pertenece un Markdown; sin coincidencia, la norma vigente."""

    name = Path(source).name if source else ""
    for doc in DOCUMENTS.values():
        if name in doc.fuentes:
            return doc
    return DOCUMENTS[DOC_ID]


class Settings(BaseSettings):
    """Configuración tomada de variables NTS_RAG_* o de un archivo .env."""

    model_config = SettingsConfigDict(
        env_prefix="NTS_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Chunks producidos por pipelines/ocr/md_to_vector.py.
    chunks_path: Path = Path("data/chunks/nts-246.chunks.jsonl")

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    # Markdown auditados: se usan para los hashes de frescura del manifiesto.
    corpus_dir: Path = Path("corpus/markdown")

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chroma_collection: str = "nts_246"

    top_k: int = 6
    # Pool de candidatos de la búsqueda híbrida (Fase 2: 40–50 en total).
    vector_k: int = 24
    bm25_k: int = 24

    llm_base_url: str | None = None
    llm_model: str = "qwen2.5:7b"
    llm_api_key: str = "ollama"
    llm_timeout: float = 60.0
    # Re-ordenar los candidatos con el LLM antes de responder.
    rerank: bool = True
    # Verificación de fidelidad de la respuesta del LLM frente a las fuentes.
    verify: bool = True
    # Máximo de llamadas al LLM por consulta (plan + redacción + verificación).
    llm_max_calls: int = 3

    # Si se define, /consulta y /buscar exigen `X-API-Key` (o `Authorization:
    # Bearer`) con este valor. Pensado para exponer la API fuera de localhost.
    api_key: str | None = None
    # Registro de consultas en JSONL (una línea por consulta). Desactivado por
    # defecto: las preguntas pueden contener datos de salud.
    query_log: Path | None = None

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "index" / "chroma"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def fastembed_cache(self) -> Path:
        return self.models_dir / "fastembed"

    def ensure_dirs(self) -> None:
        for path in (self.chroma_dir, self.fastembed_cache):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
