from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DOC_ID = "nts-246-2026"
DOC_TITLE = "NTS N° 246-MINSA/DGIESP-2026"


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

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chroma_collection: str = "nts_246"

    top_k: int = 6
    vector_k: int = 12
    bm25_k: int = 12

    llm_base_url: str | None = None
    llm_model: str = "qwen2.5:7b"
    llm_api_key: str = "ollama"
    llm_timeout: float = 60.0
    # Re-ordenar los candidatos con el LLM antes de responder.
    rerank: bool = True
    # Verificación de fidelidad de la respuesta del LLM frente a las fuentes.
    verify: bool = True

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
