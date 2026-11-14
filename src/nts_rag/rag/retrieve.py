from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from nts_rag.config import Settings
from nts_rag.index import load_chunks, load_from_collection
from nts_rag.rag.embeddings import Embedder
from nts_rag.text import expand_tokens, normalize_for_match, stem, tokenize

# Palabras que aparecen en casi todos los títulos y no distinguen secciones.
SECTION_GENERIC = {
    stem(word)
    for word in {
        "contra",
        "curso",
        "dosis",
        "esquema",
        "norma",
        "nts",
        "periodo",
        "vacuna",
        "vacunas",
        "vacunacion",
        "vida",
    }
}

# Expansión de términos coloquiales definida en nts_rag.text.


# Términos que, en la pregunta, apuntan a un campo concreto de una ficha.
# "esquema" y "tipo" se excluyen porque también aparecen en preguntas generales.
FIELD_INDICATORS = {
    "administracion",
    "cantidad",
    "composicion",
    "conservacion",
    "contraindicacion",
    "descripcion",
    "dosis",
    "efectos",
    "indicacion",
    "jeringa",
    "jeringas",
    "presentacion",
    "simultaneo",
    "sitio",
    "via",
}


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0
    rrf: float = 0.0
    origenes: set[str] = field(default_factory=set)

    @property
    def pagina(self) -> int:
        return int(self.metadata.get("pagina", 0))

    @property
    def pagina_fin(self) -> int:
        return int(self.metadata.get("pagina_fin") or self.pagina)

    @property
    def seccion(self) -> str:
        return str(self.metadata.get("seccion_corta") or self.metadata.get("seccion", ""))

    @property
    def seccion_ruta(self) -> str:
        return str(self.metadata.get("seccion", ""))

    @property
    def seccion_numero(self) -> str:
        return str(self.metadata.get("seccion_numero", ""))

    @property
    def tipo(self) -> str:
        return str(self.metadata.get("tipo", "texto"))

    @property
    def campo(self) -> str | None:
        value = self.metadata.get("campo")
        return str(value) if value else None


def section_number_in_query(number: str, query: str) -> bool:
    """Comprueba que el numeral citado no sea prefijo de otro (6.1.1.1 vs 6.1.1.17)."""

    if not number:
        return False
    return bool(re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])", query))


def _where(filtros: Any) -> dict | None:
    if filtros is None:
        return None
    conditions: list[dict] = []
    if getattr(filtros, "tipo", None):
        conditions.append({"tipo": {"$eq": filtros.tipo}})
    if getattr(filtros, "pagina", None):
        conditions.append({"pagina": {"$lte": filtros.pagina}})
        conditions.append({"pagina_fin": {"$gte": filtros.pagina}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = client.get_collection(settings.chroma_collection)
        self.embedder = Embedder(settings)

        if settings.chunks_path.exists():
            self.chunks = load_chunks(settings.chunks_path)
        else:
            self.chunks = load_from_collection(self.collection)

        self.by_id = {chunk["id"]: chunk for chunk in self.chunks}
        self.bm25 = None
        if self.chunks:
            corpus = [
                tokenize(f"{chunk['seccion_corta']} {chunk['texto']}") for chunk in self.chunks
            ]
            self.bm25 = BM25Okapi(corpus, b=0.3)

    def search(
        self, query: str, *, k: int | None = None, filtros: Any = None
    ) -> list[RetrievedChunk]:
        k = k or self.settings.top_k
        where = _where(filtros)
        query_embedding = self.embedder.embed_query(query)
        query_tokens = expand_tokens(set(tokenize(query)))
        query_section_tokens = query_tokens - SECTION_GENERIC
        field_named = bool(query_tokens & FIELD_INDICATORS)

        candidates: dict[str, RetrievedChunk] = {}
        vector_ranked: list[str] = []
        bm25_ranked: list[str] = []

        # 1. Búsqueda vectorial. Con filtro de sección se piden más candidatos
        # porque ese filtro se aplica en memoria.
        n_results = self.settings.vector_k
        if filtros is not None and getattr(filtros, "seccion", None):
            n_results = max(n_results, 40)
        vector = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas"],
        )
        if vector["ids"]:
            for chunk_id, text, metadata in zip(
                vector["ids"][0],
                vector["documents"][0],
                vector["metadatas"][0],
                strict=False,
            ):
                if not self._matches(metadata or {}, filtros):
                    continue
                vector_ranked.append(chunk_id)
                candidates[chunk_id] = RetrievedChunk(
                    id=chunk_id,
                    text=text or "",
                    metadata=metadata or {},
                    origenes={"vector"},
                )

        # 2. Búsqueda léxica (BM25): útil para siglas, números y dosis.
        if self.bm25 is not None:
            scores = self.bm25.get_scores(list(query_tokens))
            order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            for index in order:
                score = float(scores[index])
                if score <= 0:
                    break
                chunk = self.chunks[index]
                if not self._matches(chunk, filtros):
                    continue
                chunk_id = chunk["id"]
                bm25_ranked.append(chunk_id)
                if chunk_id not in candidates:
                    candidates[chunk_id] = RetrievedChunk(
                        id=chunk_id,
                        text=chunk["texto"],
                        metadata=chunk,
                        origenes={"bm25"},
                    )
                else:
                    candidates[chunk_id].origenes.add("bm25")
                if len(bm25_ranked) >= self.settings.bm25_k:
                    break

        # 3. Reciprocal Rank Fusion con refuerzos pequeños y deterministas.
        for rank, chunk_id in enumerate(vector_ranked, start=1):
            if chunk_id in candidates:
                candidates[chunk_id].rrf += 1.0 / (60 + rank)
        for rank, chunk_id in enumerate(bm25_ranked, start=1):
            if chunk_id in candidates:
                candidates[chunk_id].rrf += 1.0 / (60 + rank)

        for item in candidates.values():
            # Los chunks de campo son la respuesta precisa solo cuando la
            # pregunta nombra ese campo; en preguntas generales la tabla o el
            # texto completos son mejores, así que se atenúan.
            if item.tipo == "campo":
                item.rrf -= 0.03
                if field_named and item.campo and query_tokens & set(tokenize(item.campo)):
                    item.rrf += 0.08
            if query_tokens:
                section_tokens = set(tokenize(item.seccion)) - SECTION_GENERIC
                overlap = len(query_section_tokens & section_tokens)
                if overlap:
                    # En preguntas muy cortas el título de la sección es la
                    # señal más fiable; en preguntas largas pesa menos.
                    weight = 0.06 if len(query_section_tokens) <= 2 else 0.02
                    item.rrf += min(0.12, weight * overlap)
            if section_number_in_query(item.seccion_numero, query):
                item.rrf += 0.10
            text_tokens = set(tokenize(item.text))
            lexical_overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            item.rrf += 0.04 * lexical_overlap

        ranked = sorted(candidates.values(), key=lambda item: item.rrf, reverse=True)[:k]
        top_rrf = ranked[0].rrf if ranked else 1.0
        for item in ranked:
            item.score = round(item.rrf / top_rrf, 4) if top_rrf else 0.0
        return ranked

    @staticmethod
    def _matches(chunk: dict[str, Any], filtros: Any) -> bool:
        if filtros is None:
            return True
        if getattr(filtros, "tipo", None) and chunk.get("tipo") != filtros.tipo:
            return False
        if getattr(filtros, "seccion", None):
            raw = str(filtros.seccion).strip()
            if re.fullmatch(r"\d+(?:\.\d+)*", raw):
                number = str(chunk.get("seccion_numero") or "")
                if not (number == raw or number.startswith(raw + ".")):
                    return False
            else:
                needle = normalize_for_match(raw)
                haystack = normalize_for_match(
                    f"{chunk.get('seccion', '')} {chunk.get('seccion_corta', '')}"
                )
                if needle not in haystack:
                    return False
        if getattr(filtros, "pagina", None):
            start = int(chunk.get("pagina") or 0)
            end = int(chunk.get("pagina_fin") or start)
            if not start <= filtros.pagina <= end:
                return False
        return True

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
