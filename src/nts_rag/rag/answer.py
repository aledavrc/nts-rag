from __future__ import annotations

import contextvars
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from nts_rag.config import Settings
from nts_rag.rag.abstain import Verdict, judge
from nts_rag.rag.citations import citation, document_label, pages
from nts_rag.rag.evidence import Evidence, gather_evidence
from nts_rag.rag.fichas import get_matrix
from nts_rag.rag.formatting import clip_markdown, format_chunk_markdown
from nts_rag.rag.interpret import Plan, deterministic_plan, plan_query
from nts_rag.rag.retrieve import RetrievedChunk, Retriever
from nts_rag.rag.structured import Structured, structured_answer
from nts_rag.text import expand_tokens, tokenize

logger = logging.getLogger(__name__)

# Un solo cliente reutiliza la conexión en las llamadas de cada consulta.
_client = httpx.Client()

# Llamadas al LLM de la consulta en curso: [(tokens de entrada, tokens de salida)].
_USAGE: contextvars.ContextVar[list[tuple[int, int]] | None] = contextvars.ContextVar(
    "nts_rag_usage", default=None
)
# Máximo de llamadas de la consulta en curso (Fase 4: ≤ 3). None = sin límite.
_BUDGET: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "nts_rag_budget", default=None
)

# Presupuesto de tokens por intención (§8.3) para el contexto del redactor. Se
# estima a 4 caracteres por token y se reservan ~600 tokens para el prompt.
TOKEN_BUDGET = {
    "lookup": 2000,
    "edad": 4000,
    "conjunto": 6000,
    "comparacion": 6000,
    "comparacion_normas": 6000,
    "comun": 6000,
    "global": 8000,
    "max_atributo": 8000,
}
DEFAULT_TOKEN_BUDGET = 12_000
PROMPT_TOKENS = 600


def context_chars(plan: Plan | None) -> int:
    budget = TOKEN_BUDGET.get(plan.intencion, DEFAULT_TOKEN_BUDGET) if plan else None
    return ((budget or DEFAULT_TOKEN_BUDGET) - PROMPT_TOKENS) * 4

NO_ANSWER = "No se encontró esa información en la normativa cargada."
# Una tabla de la matriz puede citar muchas celdas (vía y sitio: 24).
MAX_STRUCTURED_SOURCES = 30

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
    # Verbos de administración y de pertenencia: forman la pregunta ("¿qué le
    # toca?", "¿se puede vacunar?") y no son un tema que deba constar en la norma.
    "administra",
    "administrar",
    "aplica",
    "aplican",
    "coloca",
    "colocan",
    "colocar",
    "corresponde",
    "corresponden",
    "deben",
    "necesita",
    "necesitan",
    "pone",
    "ponen",
    "poner",
    "previene",
    "prevenir",
    "protege",
    "proteger",
    "tener",
    "toca",
    "tocan",
    "vacunar",
}

GENERIC_STEMS = {token for term in GENERIC_QUERY_TERMS for token in tokenize(term)}

SYSTEM_PROMPT = """Eres un asistente documental especializado en la normativa de vacunación \
del Perú.
Responde en español, de forma breve y literal, usando únicamente los fragmentos entregados.
Cada fragmento empieza con una etiqueta [c1], [c2]… Cita cada afirmación con la etiqueta del
fragmento que la sustenta, por ejemplo [c2] o [c1][c3]. No escribas tú las páginas, las
secciones ni el nombre de la norma: el sistema sustituye las etiquetas por la cita exacta.
Si un fragmento pertenece a una norma derogada, dilo expresamente y no lo presentes como vigente.
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

# Instrucción para el redactor según el veredicto: el código decide el estado; el
# LLM solo lo explica.
STATE_INSTRUCTIONS = {
    "conflicto": "Las fuentes discrepan: expón cada valor con su etiqueta y no elijas uno.",
    "premisa_invalida": (
        "La pregunta da por cierto algo que la norma no sostiene: corrígelo primero, con citas."
    ),
    "metrica_no_comparable": (
        "Da los valores que declara cada fuente, pero no declares un ganador: la norma no "
        "compara esa métrica."
    ),
    "evidencia_insuficiente": "Responde solo lo que consta y di qué no consta.",
}

MOTIVO_TEXTO = {
    "fuera_de_alcance": "La pregunta queda fuera de lo que regulan las normas cargadas.",
    "sin_evidencia": "No hay fragmentos de la norma que sustenten una respuesta.",
    "requiere_conocimiento_externo": (
        "La norma no establece ese juicio ni esa indicación clínica; requiere criterio "
        "profesional o fuentes externas."
    ),
    "ambiguedad": (
        "La pregunta admite varias lecturas. Indica la vacuna, la edad o el grupo de población "
        "al que se refiere."
    ),
    "conflicto": "Las fuentes de la norma no coinciden; se muestran ambas.",
    "premisa_invalida": "La pregunta parte de un supuesto que la norma no sostiene.",
    "metrica_no_comparable": (
        "La norma da los valores de cada vacuna, pero no los compara ni establece cuál es superior."
    ),
    "evidencia_insuficiente": "Solo una parte de lo pedido consta en la norma.",
}


@dataclass
class AnswerResult:
    text: str
    mode: str
    estado: str = "respondida"
    motivo: str | None = None
    reformulacion: str | None = None
    supuestos: list[str] = field(default_factory=list)
    faltan_fuentes: list[str] = field(default_factory=list)
    conflictos: list[dict[str, Any]] = field(default_factory=list)
    llamadas: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    plan: dict[str, Any] | None = None


# Alias conservados: los tests y otros módulos los usan con estos nombres.
_pages = pages
_document_label = document_label
_citation = citation


def _extractive(chunks: list[RetrievedChunk], *, max_fragments: int = 1) -> AnswerResult:
    if not chunks:
        return AnswerResult(text=NO_ANSWER, mode="extractivo", estado="abstenida")
    parts = ["**Según la normativa cargada:**"]
    for chunk in chunks[:max_fragments]:
        parts += ["", format_chunk_markdown(chunk.text), "", f"*Fuente: {_citation(chunk)}*"]
    return AnswerResult(text="\n".join(parts).strip(), mode="extractivo")


def _significant_terms(query: str) -> set[str]:
    return {
        token
        for token in tokenize(query)
        if len(token) >= 4 and token not in GENERIC_STEMS and not token.startswith("edad")
    }


def is_answerable(query: str, chunks: list[RetrievedChunk]) -> bool:
    """Evita responder cuando la pregunta introduce términos ajenos al corpus.

    Se considera evidencia suficiente que un término específico aparezca en la
    sección recuperada, o que al menos dos aparezcan en el texto. Así una
    pregunta sobre un tema que solo se menciona de pasada (por ejemplo,
    "diabetes" en una lista de comorbilidades) no se toma como respondible.
    """

    # Cada término de la pregunta cuenta una vez, aunque se cumpla con un sinónimo
    # ("cepas" con "serotipos"): expandir no debe subir el listón.
    groups = [expand_tokens({term}) for term in _significant_terms(query)]
    if not groups:
        return True
    needed = 1 if len(groups) == 1 else 2
    for chunk in chunks:
        section = set(tokenize(chunk.seccion))
        if any(group & section for group in groups):
            return True
        text = set(tokenize(chunk.text))
        if sum(bool(group & text) for group in groups) >= needed:
            return True
    return False


def _context(chunks: list[RetrievedChunk], max_chars: int = 24_000) -> str:
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[c{index}] {_document_label(chunk)}, {_pages(chunk)}, "
            f"sección {chunk.seccion or 'sin sección'}\n"
            f"{clip_markdown(chunk.text, 4000)}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


_PLACEHOLDER_RE = re.compile(r"\[\s*c(\d+)\s*\]", re.IGNORECASE)
# Una cita escrita de memoria ("[NTS 246, p. 24, …]") se descarta (§7.3).
_MEMORY_CITATION_RE = re.compile(r"\[(?:NTS|Fuente|p\.|pág)[^\]]*\]", re.IGNORECASE)


def render_citations(text: str, chunks: list[RetrievedChunk]) -> tuple[str, int]:
    """Sustituye [cN] por la cita de los metadatos. Devuelve (texto, citas válidas)."""

    valid = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal valid
        index = int(match.group(1))
        if 1 <= index <= len(chunks):
            valid += 1
            return _citation(chunks[index - 1])
        return ""

    text = _MEMORY_CITATION_RE.sub("", text)
    return _PLACEHOLDER_RE.sub(replace, text).strip(), valid


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
    usage = _USAGE.get()
    budget = _BUDGET.get()
    if usage is not None and budget is not None and len(usage) >= budget:
        logger.info("Presupuesto de %s llamadas agotado; se omite la llamada", budget)
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
    # La llamada cuenta aunque falle: consume presupuesto y latencia.
    if usage is not None:
        usage.append((0, 0))
    try:
        response = _client.post(url, json=payload, headers=headers, timeout=settings.llm_timeout)
        response.raise_for_status()
        data = response.json()
        if usage is not None:
            tokens = data.get("usage") or {}
            usage[-1] = (
                int(tokens.get("prompt_tokens") or 0),
                int(tokens.get("completion_tokens") or 0),
            )
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
    settings: Settings,
    query: str,
    chunks: list[RetrievedChunk],
    verdict: Verdict | None = None,
    plan: Plan | None = None,
) -> AnswerResult | None:
    instruction = STATE_INSTRUCTIONS.get(verdict.motivo or "", "") if verdict else ""
    extra = ""
    if verdict and verdict.premisas:
        extra += "\nPremisas no sostenidas: " + " ".join(verdict.premisas)
    if verdict and verdict.conflictos:
        extra += "\nDiscrepancias detectadas: " + "; ".join(
            " vs ".join(c["valores"]) for c in verdict.conflictos
        )
    text = _chat(
        settings,
        SYSTEM_PROMPT,
        f"Pregunta: {query}\n\n"
        f"Fragmentos disponibles:\n{_context(chunks, context_chars(plan))}\n\n"
        f"{instruction}{extra}\n"
        "Responde citando con las etiquetas [cN].",
    )
    if not text:
        return None
    if text.strip() == NO_ANSWER:
        return AnswerResult(text=NO_ANSWER, mode="llm")
    rendered, citations = render_citations(text, chunks)
    if citations == 0:
        # Sin una sola cita resoluble no se acepta la redacción.
        logger.warning("La respuesta del LLM no trae citas [cN]; se usa modo extractivo")
        return None
    return AnswerResult(text=rendered, mode="llm")


def _notes(verdict: Verdict, plan: Plan) -> list[str]:
    notes: list[str] = []
    if verdict.motivo and verdict.estado == "parcial":
        notes.append(f"> **Respuesta parcial.** {MOTIVO_TEXTO[verdict.motivo]}")
    for premisa in verdict.premisas:
        notes.append(f"> {premisa}")
    for conflicto in verdict.conflictos:
        notes.append("> Discrepancia: " + " · ".join(conflicto["valores"]))
    if verdict.faltan_fuentes:
        notes.append("> No consta: " + "; ".join(verdict.faltan_fuentes))
    if plan.pregunta_normalizada:
        notes.append(f"> Pregunta interpretada como: *{plan.pregunta_normalizada}*")
    return notes


def generate_answer(
    settings: Settings,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    plan: Plan | None = None,
    context: list[RetrievedChunk] | None = None,
    verdict: Verdict | None = None,
    structured: Structured | None = None,
) -> AnswerResult:
    plan = plan or deterministic_plan(query)
    if verdict is None:
        verdict = judge(plan, chunks, answerable=bool(chunks) and is_answerable(query, chunks))
    context = context or chunks

    def finish(result: AnswerResult) -> AnswerResult:
        result.estado = verdict.estado
        result.motivo = verdict.motivo
        result.supuestos = list(plan.supuestos)
        result.faltan_fuentes = list(verdict.faltan_fuentes)
        result.conflictos = list(verdict.conflictos)
        result.reformulacion = plan.pregunta_normalizada
        result.plan = plan.to_dict()
        return result

    if verdict.estado in {"abstenida", "aclaracion"}:
        text = NO_ANSWER if verdict.motivo in {"fuera_de_alcance", "sin_evidencia"} else ""
        reason = MOTIVO_TEXTO.get(verdict.motivo or "", "")
        body = "\n\n".join(part for part in (text, reason) if part)
        return finish(AnswerResult(text=body, mode="extractivo", estado=verdict.estado))

    notes = _notes(verdict, plan)

    # Ruta estructurada (comparación, lista, diff, valencias): plantilla con cita
    # por celda y 0 llamadas (§8.3). Si la pregunta tenía otra parte (subconsulta),
    # se añade su fragmento.
    if structured is not None:
        parts = [*notes, "" if notes else None, structured.text]
        cell_ids = {cell.chunk["id"] for cell in structured.cells}
        extra = [c for c in chunks if c.id not in cell_ids]
        if plan.subconsultas and extra:
            fragment = _extractive(extra[:1]).text
            parts += ["", fragment.replace("**Según la normativa cargada:**", "**Además:**")]
        text = "\n".join(part for part in parts if part is not None).strip()
        return finish(AnswerResult(text=text, mode="estructurado"))

    result = _llm_answer(settings, query, context, verdict, plan)
    if result is not None and result.text.strip() != NO_ANSWER:
        if _verify_grounded(settings, query, result.text, context):
            if notes:
                result.text = "\n".join(notes) + "\n\n" + result.text
            return finish(result)
        logger.warning("Respuesta del LLM no verificada; se usa modo extractivo")

    # Extractivo: el fragmento verbatim. Si la pregunta pide varias cosas (edad,
    # conjunto, comparación), un fragmento por fuente principal.
    fragments = 1 if plan.intencion == "lookup" and verdict.estado == "respondida" else 3
    extractive = _extractive(chunks, max_fragments=fragments)
    if notes:
        extractive.text = "\n".join(notes) + "\n\n" + extractive.text
    return finish(extractive)


def run_query(
    settings: Settings,
    retriever: Retriever,
    query: str,
    *,
    k: int = 6,
    filtros=None,
) -> tuple[list[RetrievedChunk], AnswerResult]:
    """Pipeline (§6.1): plan → validar → evidencia → veredicto → redactar → verificar.

    Con LLM, el plan sustituye al re-rankeo (D15): 2–3 llamadas en vez de 3.
    Sin LLM, el plan es determinista y la consulta no hace ninguna llamada.
    """

    usage: list[tuple[int, int]] = []
    token = _USAGE.set(usage)
    budget_token = _BUDGET.set(settings.llm_max_calls)
    try:
        chat = None
        if settings.llm_base_url:
            def chat(system: str, user: str) -> str | None:
                return _chat(settings, system, user, max_tokens=500)

        # Presupuesto (Fase 4): se reservan la redacción y, si está activa, la
        # verificación; el plan usa lo que queda (con 3: un intento, sin reintento).
        reserve = 1 + (1 if settings.verify else 0)
        attempts = max(0, min(2, settings.llm_max_calls - reserve))
        plan, evidence = plan_and_gather(
            retriever, query, k=k, filtros=filtros, chat=chat, attempts=attempts
        )
        chunks = evidence.fuentes
        # El re-rankeo solo entra si el plan del LLM falló y aún cabe en el presupuesto.
        rerank_fits = len(usage) + 1 + reserve <= settings.llm_max_calls
        if settings.llm_base_url and settings.rerank and plan.origen != "llm" and rerank_fits:
            chunks = rerank_with_llm(settings, query, chunks, top_k=len(chunks))
        # La reformulación del plan acerca la pregunta al vocabulario de la norma
        # ("¿de qué está hecha?" → "composición"): el chequeo de alcance la usa.
        checked = plan.pregunta_normalizada or query
        # Si la matriz resolvió la pregunta (comparación, diff, lista), está en el
        # alcance aunque sus palabras ("nuevas", "incorpora") no estén en las celdas.
        answerable = bool(chunks) and (
            evidence.estructurada is not None or is_answerable(checked, chunks)
        )
        verdict = judge(
            plan,
            chunks,
            answerable=answerable,
            chunks=getattr(retriever, "chunks", None) or [],
            faltantes=evidence.faltantes,
        )
        result = generate_answer(
            settings,
            query,
            chunks,
            plan=plan,
            context=evidence.contexto,
            verdict=verdict,
            structured=evidence.estructurada,
        )
    finally:
        _USAGE.reset(token)
        _BUDGET.reset(budget_token)
    result.llamadas = len(usage)
    result.tokens_entrada = sum(entrada for entrada, _ in usage)
    result.tokens_salida = sum(salida for _, salida in usage)
    return chunks, result


def plan_and_gather(
    retriever: Any,
    query: str,
    *,
    k: int = 6,
    filtros=None,
    chat=None,
    attempts: int = 2,
) -> tuple[Plan, Evidence]:
    """Plan + evidencia: lo que mide `evaluar` como recuperación del sistema."""

    plan = plan_query(query, chat, attempts=attempts)
    if filtros is not None and getattr(filtros, "vigencia", None):
        plan.version = "ambas" if filtros.vigencia == "todas" else filtros.vigencia
    evidence = gather_evidence(retriever, query, plan, k=k, filtros=filtros)
    structured = structured_answer(plan, get_matrix(retriever))
    if structured is not None and structured.cells:
        # Las celdas de la matriz son las fuentes principales; después, la evidencia.
        cell_sources = structured.fuentes
        ids = {chunk.id for chunk in cell_sources}
        rest = [chunk for chunk in evidence.fuentes if chunk.id not in ids]
        limit = min(max(k, len(cell_sources)), MAX_STRUCTURED_SOURCES)
        evidence.fuentes = (cell_sources + rest)[:limit]
        evidence.estructurada = structured
    return plan, evidence
