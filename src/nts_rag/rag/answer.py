from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from nts_rag.config import DOC_TITLE, Settings
from nts_rag.rag.formatting import clip_markdown, format_chunk_markdown
from nts_rag.rag.retrieve import RetrievedChunk, Retriever
from nts_rag.text import expand_tokens, tokenize

logger = logging.getLogger(__name__)

# Un solo cliente reutiliza la conexión en las tres llamadas de cada consulta.
_client = httpx.Client()

NO_ANSWER = "No se encontró esa información en la normativa cargada."

GENERIC_QUERY_TERMS = {
    "ano",
    "aplicacion",
    "aplicar",
    "contra",
    "cual",
    "cuales",
    "cuando",
    "cuanta",
    "cuanto",
    "debe",
    "dice",
    "dosis",
    "donde",
    "esquema",
    "esta",
    "hacer",
    "hasta",
    "incluida",
    "incluido",
    "incluye",
    "incluyen",
    "informacion",
    "nina",
    "nino",
    "norma",
    "nts",
    "para",
    "persona",
    "puede",
    "que",
    "recibe",
    "recibir",
    "segun",
    "sobre",
    "tiempo",
    "tipo",
    "usa",
    "uso",
    "vacuna",
    "vacunas",
    "vacunacion",
}

GENERIC_STEMS = {token for term in GENERIC_QUERY_TERMS for token in tokenize(term)}

SYSTEM_PROMPT = """Eres un asistente documental especializado en la normativa de vacunación \
del Perú.
Responde en español, de forma breve y literal, usando únicamente los fragmentos entregados.
Cada afirmación debe llevar una cita con el formato [NTS 246, p. <página>, sección <sección>].
No inventes edades, dosis, intervalos, vías, temperaturas ni contraindicaciones.
No completes información con conocimiento general.
Si los fragmentos no contienen la respuesta, responde exactamente:
"No se encontró esa información en la normativa cargada."
Responde en Markdown. Usa listas con viñetas cuando haya varios elementos y tablas Markdown
cuando los datos sean tabulares. No devuelvas HTML.
"""

RERANK_SYSTEM = """Eres un recuperador de documentos en español. Recibes una pregunta y una \
lista numerada de fragmentos de una norma técnica de vacunación. Devuelve únicamente un array \
JSON con los números de los fragmentos ordenados de más a menos relevante para responder la \
pregunta. Ejemplo: [3, 1, 2]. No expliques nada ni añadas texto.
"""

VERIFY_SYSTEM = """Eres un verificador de respuestas. Recibes una pregunta, una respuesta y los \
fragmentos fuente. Responde exactamente "OK" si todas las afirmaciones de la respuesta están \
contenidas o se deducen directamente de los fragmentos, y "NO" en caso contrario. No añadas \
ninguna otra palabra.
"""


@dataclass
class AnswerResult:
    text: str
    mode: str


def _pages(chunk: RetrievedChunk) -> str:
    if chunk.pagina_fin > chunk.pagina:
        return f"pp. {chunk.pagina}–{chunk.pagina_fin}"
    return f"p. {chunk.pagina}"


def _citation(chunk: RetrievedChunk) -> str:
    section = chunk.seccion or "sin sección"
    return f"[{DOC_TITLE}, {_pages(chunk)}, sección {section}]"


def _extractive(chunks: list[RetrievedChunk]) -> AnswerResult:
    if not chunks:
        return AnswerResult(
            text="No se encontró esa información en la normativa cargada.",
            mode="extractivo",
        )
    chunk = chunks[0]
    text = "\n".join(
        [
            "**Según la normativa cargada:**",
            "",
            format_chunk_markdown(chunk.text),
            "",
            f"*Fuente: {_citation(chunk)}*",
        ]
    )
    return AnswerResult(text=text.strip(), mode="extractivo")


def _significant_terms(query: str) -> set[str]:
    return {
        token
        for token in tokenize(query)
        if len(token) >= 4 and token not in GENERIC_STEMS
    }


def is_answerable(query: str, chunks: list[RetrievedChunk]) -> bool:
    """Evita responder cuando la pregunta introduce términos ajenos al corpus.

    Se considera evidencia suficiente que un término específico aparezca en la
    sección recuperada, o que al menos dos aparezcan en el texto. Así una
    pregunta sobre un tema que solo se menciona de pasada (por ejemplo,
    "diabetes" en una lista de comorbilidades) no se toma como respondible.
    """

    terms = expand_tokens(_significant_terms(query))
    if not terms:
        return True
    needed = 1 if len(terms) == 1 else 2
    for chunk in chunks:
        if terms & set(tokenize(chunk.seccion)):
            return True
        if len(terms & set(tokenize(chunk.text))) >= needed:
            return True
    return False



def _context(chunks: list[RetrievedChunk], max_chars: int = 24_000) -> str:
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[Fuente {index}] {DOC_TITLE}, {_pages(chunk)}, "
            f"sección {chunk.seccion or 'sin sección'}\n"
            f"{clip_markdown(chunk.text, 4000)}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _chat(
    settings: Settings,
    system: str,
    user: str,
    *,
    max_tokens: int = 700,
    temperature: float = 0.0,
) -> str | None:
    """Una llamada al LLM compatible con OpenAI; None si falla o no hay endpoint."""

    if not settings.llm_base_url:
        return None
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    payload = {
        "model": settings.llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        response = _client.post(url, json=payload, headers=headers, timeout=settings.llm_timeout)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("El LLM no respondió: %s", exc)
        return None


def rerank_with_llm(
    settings: Settings, query: str, candidates: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Reordena los candidatos con el LLM; si falla, conserva el orden original."""

    if not settings.llm_base_url or len(candidates) <= 1:
        return candidates[:top_k]

    items = "\n".join(
        f"[{index}] p. {c.pagina}, {c.seccion or 'sin sección'}: "
        f"{clip_markdown(c.text, 600)}"
        for index, c in enumerate(candidates, start=1)
    )
    text = _chat(
        settings,
        RERANK_SYSTEM,
        f"Pregunta: {query}\n\nFragmentos:\n{items}",
        max_tokens=200,
    )
    if not text:
        return candidates[:top_k]

    match = re.search(r"\[[0-9,\s]*\]", text)
    if not match:
        return candidates[:top_k]
    try:
        order = [int(n) for n in match.group(0).strip("[]").split(",") if n.strip()]
    except ValueError:
        return candidates[:top_k]

    ranked = [candidates[i - 1] for i in dict.fromkeys(order) if 1 <= i <= len(candidates)]
    ranked += [c for c in candidates if c not in ranked]
    return ranked[:top_k]


def _verify_grounded(
    settings: Settings, query: str, answer: str, chunks: list[RetrievedChunk]
) -> bool:
    """Comprueba con el LLM que la respuesta se apoya en los fragmentos."""

    if not settings.verify or not settings.llm_base_url:
        return True
    verdict = _chat(
        settings,
        VERIFY_SYSTEM,
        (
            f"Pregunta: {query}\n\nRespuesta:\n{answer}\n\n"
            f"Fragmentos fuente:\n{_context(chunks, 12_000)}"
        ),
        max_tokens=5,
    )
    return (verdict or "").strip().upper().startswith("OK")


def _llm_answer(
    settings: Settings, query: str, chunks: list[RetrievedChunk]
) -> AnswerResult | None:
    text = _chat(
        settings,
        SYSTEM_PROMPT,
        f"Pregunta: {query}\n\n"
        f"Fragmentos disponibles:\n{_context(chunks)}\n\n"
        "Responde citando la página y la sección.",
    )
    if not text:
        return None
    return AnswerResult(text=text, mode="llm")


def generate_answer(settings: Settings, query: str, chunks: list[RetrievedChunk]) -> AnswerResult:
    if not chunks or not is_answerable(query, chunks):
        return _extractive([])
    result = _llm_answer(settings, query, chunks)
    if result is None:
        return _extractive(chunks)
    # Si el LLM se abstiene o su respuesta no está respaldada por las fuentes,
    # se devuelve el fragmento verbatim (sin riesgo de alucinación).
    if result.text.strip() == NO_ANSWER:
        return _extractive(chunks)
    if not _verify_grounded(settings, query, result.text, chunks):
        logger.warning("Respuesta del LLM no verificada; se usa modo extractivo")
        return _extractive(chunks)
    return result


def run_query(
    settings: Settings,
    retriever: Retriever,
    query: str,
    *,
    k: int = 6,
    filtros=None,
) -> tuple[list[RetrievedChunk], AnswerResult]:
    """Pipeline completa: recuperar → re-rankea (LLM) → responder con verificación."""

    pool = retriever.search(query, k=max(k * 3, 12), filtros=filtros)
    if settings.llm_base_url and settings.rerank:
        pool = rerank_with_llm(settings, query, pool, top_k=k)
    chunks = pool[:k]
    return chunks, generate_answer(settings, query, chunks)
