"""Evidencia guiada por el plan (§6.1): 0 llamadas al LLM.

1. Búsqueda híbrida sobre un pool amplio y reordenación por el plan (vacunas,
   campos, edad; las tablas operativas se penalizan en la ruta de edad).
2. *Slots* deterministas que garantizan cobertura: el campo exacto de cada vacuna
   nombrada, la ficha de cada vacuna, el calendario para la edad, el Anexo 2 para
   rescate y límites, el barrido de un campo para superlativos y una búsqueda por
   subconsulta. Así una comparación ya no depende de que un solo embedding traiga
   a todas las entidades (causa raíz 6).
3. Diversidad: como mucho dos fuentes por sección antes de repetir.
4. `expand_context` (D12): hermanos de la sección para el redactor (ficha
   completa, filas de la misma tabla que encajan con la edad), con presupuesto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nts_rag.catalogo import CANONICAL_FIELDS, FICHAS, is_domain_section
from nts_rag.rag.interpret import Plan
from nts_rag.rag.retrieve import RetrievedChunk
from nts_rag.schemas import Filtros
from nts_rag.text import expand_tokens, normalize_for_match, tokenize

POOL = 48
MAX_SOURCES = 10
MAX_PER_SECTION = 2
CONTEXT_BUDGET = 16_000
RESCATE_SECTION = "dosis de rescate"
VIAS_SECTION = "vias y tecnicas de administracion"
CALENDAR_SECTION = "6.2"
DOCUMENT_FOR_VIGENCIA = {"vigente": "nts-246-2026", "derogada": "nts-196-2022"}
# Campos que responde el calendario (6.2.x), no un barrido de fichas.
CALENDAR_FIELDS = {"ESQUEMA_DE_VACUNACION", "DOSIS"}


@dataclass
class Evidence:
    fuentes: list[RetrievedChunk]
    contexto: list[RetrievedChunk] = field(default_factory=list)
    # Slots pedidos por el plan que no tienen fuente: (vacuna, campo, documento).
    faltantes: list[tuple[str, str, str]] = field(default_factory=list)
    # Respuesta por plantilla de la matriz, si la intención la tiene (Fase 3).
    estructurada: Any = None


def _from_dict(chunk: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(id=chunk["id"], text=chunk["texto"], metadata=chunk)


def _vaccines(chunk: RetrievedChunk) -> set[str]:
    meta = chunk.metadata
    found = {slug for slug in str(meta.get("vacunas") or "").split(",") if slug}
    if meta.get("vacuna_slug"):
        found.add(str(meta["vacuna_slug"]))
    return found


def _age_fits(meta: dict[str, Any], edad: int) -> bool | None:
    low, high = meta.get("edad_min_meses"), meta.get("edad_max_meses")
    if low is None and high is None:
        return None
    return (low is None or int(low) <= edad) and (high is None or edad <= int(high))


def _operational_annex(chunk: RetrievedChunk) -> bool:
    """Anexos que no son de dominio: programación, presupuesto, indicadores, registros."""

    section = normalize_for_match(chunk.seccion_ruta)
    return "anexo" in section and not is_domain_section(chunk.seccion_ruta, chunk.seccion_numero)


def rescore(chunks: list[RetrievedChunk], plan: Plan) -> list[RetrievedChunk]:
    """Ajusta el RRF con el plan y reordena. Refuerzos pequeños y deterministas."""

    entidades = set(plan.entidades)
    for chunk in chunks:
        vacunas = _vaccines(chunk)
        if entidades:
            if vacunas & entidades:
                chunk.rrf += 0.03
            elif chunk.tipo == "campo":
                # Campo de otra vacuna: contaminación en una pregunta por vacuna.
                chunk.rrf -= 0.03
        # Sin vacuna nombrada, reforzar el campo premiaría el mismo campo de las 24
        # fichas a la vez: solo se refuerza el campo de la vacuna pedida.
        if entidades and chunk.tipo == "campo" and chunk.campo in plan.campos:
            if vacunas & entidades:
                chunk.rrf += 0.04
        if plan.edad_meses is not None:
            fits = _age_fits(chunk.metadata, plan.edad_meses)
            if fits is True:
                chunk.rrf += 0.05
            elif fits is False:
                chunk.rrf -= 0.03
            # Señuelo p. 121: tablas de programación con edades en el título.
            if _operational_annex(chunk):
                chunk.rrf -= 0.05
    ranked = sorted(chunks, key=lambda item: item.rrf, reverse=True)
    top = ranked[0].rrf if ranked else 0.0
    for chunk in ranked:
        chunk.score = round(chunk.rrf / top, 4) if top else 0.0
    return ranked


def _foreign_ficha(chunk: RetrievedChunk, plan: Plan) -> bool:
    """Ficha (6.1.1.x) de una vacuna que la pregunta no nombra, cuando nombra alguna.

    Es contaminación (Fase 2, objetivo 0): "composición de la hepatitis A" no debe
    traer la ficha de la hepatitis B. Las filas de calendario que mencionan otras
    vacunas no cuentan: no son la ficha de esa vacuna.
    """

    if not plan.entidades:
        return False
    number = chunk.seccion_numero
    slug = FICHAS.get(chunk.documento_id, {}).get(number) if number else None
    return slug is not None and slug not in plan.entidades


def _section_key(chunk: RetrievedChunk) -> tuple[str, str]:
    return (chunk.documento_id, chunk.seccion_numero or chunk.seccion_ruta)


def diversify(chunks: list[RetrievedChunk], limit: int, per_section: int = MAX_PER_SECTION):
    """Toma en orden, con tope por sección; lo apartado rellena si sobra hueco."""

    chosen: list[RetrievedChunk] = []
    deferred: list[RetrievedChunk] = []
    counts: dict[tuple[str, str], int] = {}
    for chunk in chunks:
        key = _section_key(chunk)
        if counts.get(key, 0) < per_section:
            chosen.append(chunk)
            counts[key] = counts.get(key, 0) + 1
        else:
            deferred.append(chunk)
        if len(chosen) >= limit:
            return chosen
    return (chosen + deferred)[:limit]


def _filtros(base: Filtros | None, plan: Plan, **update: Any) -> Filtros:
    values = base.model_dump() if base is not None else {}
    if not values.get("vigencia") and not values.get("documento"):
        values["vigencia"] = "todas" if plan.version == "ambas" else plan.version
    values.update({key: value for key, value in update.items() if value is not None})
    if values.get("documento"):
        values["vigencia"] = None
    return Filtros(**values)


def _best(retriever: Any, query: str, plan: Plan, filtros: Filtros) -> RetrievedChunk | None:
    results = rescore(retriever.search(query, k=POOL, filtros=filtros), plan)
    return results[0] if results else None


def _best_in(
    retriever: Any, query: str, plan: Plan, filtros: Filtros | None, **update: Any
) -> RetrievedChunk | None:
    return _best(retriever, query, plan, _filtros(filtros, plan, **update))


def _documents(plan: Plan, filtros: Filtros | None) -> list[str]:
    if filtros is not None and filtros.documento:
        return [filtros.documento]
    return [DOCUMENT_FOR_VIGENCIA[vigencia] for vigencia in plan.vigencias]


def _field_sweep(chunks: list[dict[str, Any]], query: str, campos: list[str], docs: list[str]):
    """Barrido de un campo en todas las fichas (§6.4): superlativos y propiedades."""

    terms = {t for t in expand_tokens(set(tokenize(query))) if len(t) >= 4}
    scored = []
    for chunk in chunks:
        if chunk["tipo"] != "campo" or chunk["campo"] not in campos:
            continue
        if chunk["documento_id"] not in docs:
            continue
        overlap = len(terms & set(tokenize(chunk["texto"])))
        if overlap:
            scored.append((overlap, chunk))
    scored.sort(key=lambda item: -item[0])
    return [_from_dict(chunk) for _, chunk in scored[:6]]


def gather_evidence(
    retriever: Any,
    query: str,
    plan: Plan,
    *,
    k: int = 6,
    filtros: Filtros | None = None,
) -> Evidence:
    """Fuentes para una pregunta según su plan. No llama al LLM."""

    chunks: list[dict[str, Any]] = getattr(retriever, "chunks", None) or []
    docs = _documents(plan, filtros)
    main = rescore(retriever.search(query, k=POOL, filtros=_filtros(filtros, plan)), plan)

    slots: list[RetrievedChunk] = []
    faltantes: list[tuple[str, str, str]] = []
    index = {
        (c["documento_id"], c["vacuna_slug"], c["campo"]): c
        for c in chunks
        if c["tipo"] == "campo"
    }

    # 1. Campo exacto de cada vacuna nombrada, en cada norma consultada.
    for slug in plan.entidades:
        for doc in docs:
            if slug not in FICHAS.get(doc, {}).values():
                if plan.version == "ambas" or (filtros and filtros.documento):
                    faltantes.append((slug, "FICHA", doc))
                continue
            for campo in plan.campos:
                chunk = index.get((doc, slug, campo))
                if chunk:
                    slots.append(_from_dict(chunk))
                elif chunks:
                    faltantes.append((slug, campo, doc))
            # Sin campo pedido: lo mejor de la ficha para la pregunta.
            if not plan.campos and plan.intencion != "lookup":
                numero = {v: n for n, v in FICHAS[doc].items()}[slug]
                best = _best_in(retriever, query, plan, filtros, documento=doc, seccion=numero)
                if best:
                    slots.append(best)

    # 2. Ruta de edad: el calendario siempre; el Anexo 2 si hay rescate o vacuna.
    # En una pregunta de rescate el Anexo 2 es la respuesta: va antes que la ficha.
    ficha_slots, slots = slots, []
    for doc in docs:
        if plan.edad_meses is not None:
            best = _best_in(
                retriever, query, plan, filtros, documento=doc, seccion=CALENDAR_SECTION
            )
            if best:
                slots.append(best)
        if plan.rescate or (plan.edad_meses is not None and plan.entidades):
            rows = [
                _from_dict(c)
                for c in chunks
                if c["documento_id"] == doc
                and c["tipo"] == "fila"
                and RESCATE_SECTION in normalize_for_match(c["seccion"])
                and c["vacuna_slug"] in plan.entidades
            ]
            if rows:
                slots.extend(rows[:2])
            else:
                best = _best_in(
                    retriever, query, plan, filtros, documento=doc, seccion=RESCATE_SECTION
                )
                if best:
                    slots.append(best)
    slots = [*slots, *ficha_slots] if plan.rescate else [*ficha_slots, *slots]

    # 3. Superlativo o propiedad global sin vacuna nombrada: barrido del campo
    # ("¿cuáles están contraindicadas en el embarazo?"). Los campos de calendario
    # no se barren: "¿qué vacunas en el embarazo?" la responde su tabla (6.2.1),
    # no las 24 fichas.
    sweep_fields = [campo for campo in plan.campos if campo not in CALENDAR_FIELDS]
    if sweep_fields and not plan.entidades and plan.intencion in {"max_atributo", "global"}:
        slots.extend(_field_sweep(chunks, query, sweep_fields, docs))

    # 4. Vía y sitio sin vacuna nombrada: el Anexo de vías y técnicas es la
    # referencia general ("¿cuándo se deja de vacunar en el vasto externo?").
    if "VIA_Y_SITIO" in plan.campos and not plan.entidades:
        for doc in docs:
            best = _best_in(retriever, query, plan, filtros, documento=doc, seccion=VIAS_SECTION)
            if best:
                slots.append(best)

    # 5. Una fuente por subconsulta.
    for subquery in plan.subconsultas:
        best = _best(retriever, subquery, plan, _filtros(filtros, plan))
        if best:
            slots.append(best)

    seen: set[str] = set()

    def unique(items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        out = []
        for chunk in items:
            if chunk.id not in seen and not _foreign_ficha(chunk, plan):
                seen.add(chunk.id)
                out.append(chunk)
        return out

    slot_chunks = unique(slots)
    rest = unique(main)
    limit = min(max(k, len(slot_chunks)), MAX_SOURCES)
    # Los slots garantizan cobertura y van primero; el resto, diversificado.
    fuentes = diversify(slot_chunks + diversify(rest, POOL), limit)
    return Evidence(
        fuentes=fuentes,
        contexto=expand_context(retriever, fuentes, plan),
        faltantes=faltantes,
    )


def expand_context(
    retriever: Any, fuentes: list[RetrievedChunk], plan: Plan, budget: int = CONTEXT_BUDGET
) -> list[RetrievedChunk]:
    """Fuentes + hermanos de su sección (D12), sin pasar el presupuesto de caracteres.

    * Ficha (campo, tabla o texto de un 6.1.1.x): sus campos canónicos en orden.
    * Fila: las demás filas de la misma tabla; con edad, solo las que encajan.
    """

    chunks: list[dict[str, Any]] = getattr(retriever, "chunks", None) or []
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    used = 0

    def add(chunk: RetrievedChunk) -> bool:
        nonlocal used
        if chunk.id in seen:
            return True
        if used + len(chunk.text) > budget and out:
            return False
        seen.add(chunk.id)
        out.append(chunk)
        used += len(chunk.text)
        return True

    for chunk in fuentes:
        add(chunk)
    order = {campo: position for position, campo in enumerate(CANONICAL_FIELDS)}
    for chunk in fuentes:
        doc, number = chunk.documento_id, chunk.seccion_numero
        if number and number in FICHAS.get(doc, {}):
            siblings = sorted(
                (
                    c
                    for c in chunks
                    if c["tipo"] == "campo"
                    and c["documento_id"] == doc
                    and c["seccion_numero"] == number
                ),
                key=lambda c: order.get(c["campo"], 99),
            )
        elif chunk.tipo == "fila":
            table_id = chunk.id.rsplit("-fila-", 1)[0]
            siblings = [
                c
                for c in chunks
                if c["tipo"] == "fila"
                and c["id"].rsplit("-fila-", 1)[0] == table_id
                and (plan.edad_meses is None or _age_fits(c, plan.edad_meses) is not False)
            ]
        else:
            continue
        for sibling in siblings:
            if not add(_from_dict(sibling)):
                return out
    return out
