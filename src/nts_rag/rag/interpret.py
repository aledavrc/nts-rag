"""Plan de consulta (D15, §7.2): el LLM interpreta, el código valida.

`plan_query` devuelve un `Plan` con intención, vacunas, campos, edad, versión y
subconsultas. Con LLM configurado, el plan lo propone el LLM (una llamada que
sustituye al re-rankeo) y se valida contra los catálogos cerrados; si el plan no
es válido se reintenta una vez y, si vuelve a fallar, se usa el plan determinista,
que se construye con los mismos catálogos y no cuesta llamadas.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from nts_rag.catalogo import (
    CANONICAL_FIELDS,
    FIELD_QUERY_PATTERNS,
    POPULATION_PATTERN,
    VACUNA_NOMBRES,
    VACUNA_SLUGS,
    detect_vaccines,
)
from nts_rag.text import age_range_months, detect_vigencias, normalize_for_match, tokenize

logger = logging.getLogger(__name__)

INTENCIONES = (
    "lookup",
    "edad",
    "conjunto",
    "comparacion",
    "comparacion_normas",
    "comun",
    "max_atributo",
    "global",
    "evaluacion",
)
VERSIONES = ("vigente", "derogada", "ambas")
MAX_SUBCONSULTAS = 4
MAX_ENTIDADES = 6
MAX_EDAD_MESES = 100 * 12

# Rescate o límite de edad: la respuesta está en el Anexo 2 (dosis de rescate).
_RESCATE_RE = re.compile(
    r"rescate|sin vacunas|ninguna vacuna|no (?:ha |han )?recibid|no recibio|"
    r"incomplet|atrasad|recuperar|recuperabl|completar|interrump|no tiene (?:sus |ninguna )?vacuna|"
    r"edad maxima|hasta que edad|limite de edad"
)
# Juicios de valor que la norma no hace: no se responden con el corpus.
_EVALUACION_RE = re.compile(
    r"\b(?:la )?(?:mejor|peor)(?:es)?\b|mas (?:recomendable|segura|eficaz|efectiva|importante)|"
    r"\bconviene\b|\brecomiendas?\b|vale la pena"
)
# Superlativos sobre un atributo contable: se responde con los valores, sin
# declarar ganador (`metrica_no_comparable`).
_MAX_ATRIBUTO_RE = re.compile(
    r"\bmas (?:cepas|serotipos|dosis|componentes|antigenos)\b|"
    r"\bmenos (?:cepas|serotipos|dosis|componentes|antigenos)\b|"
    r"\bmayor (?:numero|cantidad) de\b"
)
_COMUN_RE = re.compile(r"\bcomparten\b|\ben comun\b|\bambas tienen\b|\btodas tienen\b")
_COMPARA_RE = re.compile(r"\bcompar|\bdiferencia|\bversus\b|\bvs\b|\bdistingu")
_GLOBAL_RE = re.compile(
    r"\bque vacunas\b|\bcuales vacunas\b|\bcuales son las vacunas\b|"
    r"\bvacunas (?:vivas|inactivadas|orales)\b"
)
# Medicamentos y tratamientos: la norma no establece su posología (§7.1:
# respuesta documental, no clínica).
_TERAPIA_RE = re.compile(
    r"\b(?:paracetamol|ibuprofeno|antibiotic\w*|antipiretic\w*|analgesic\w*|"
    r"medicamento\w*|farmaco\w*|tratamiento)\b"
)
_DOSIFICAR_RE = re.compile(r"\bcuant[oa]s?\b|\bdosis\b|\bmg\b|\bdar\b|\bse le da\b|\btomar\b")
_INTERROGATIVO = (
    r"(?:cu[aá]les?|qu[eé]|cu[aá]ndo|c[oó]mo|d[oó]nde|cu[aá]nt[oa]s?|por qu[eé]|hasta)"
)


@dataclass
class Plan:
    """Contrato del plan (§7.2). `version` = vigente | derogada | ambas."""

    pregunta: str
    intencion: str = "lookup"
    entidades: list[str] = field(default_factory=list)
    campos: list[str] = field(default_factory=list)
    edad_meses: int | None = None
    version: str = "vigente"
    subconsultas: list[str] = field(default_factory=list)
    supuestos: list[str] = field(default_factory=list)
    premisas_no_soportadas: list[str] = field(default_factory=list)
    requiere_aclaracion: bool = False
    rescate: bool = False
    terapia: bool = False
    pregunta_normalizada: str | None = None
    origen: str = "determinista"
    avisos: list[str] = field(default_factory=list)

    @property
    def vigencias(self) -> tuple[str, ...]:
        if self.version == "ambas":
            return ("vigente", "derogada")
        return (self.version,)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["entidades"] = [{"slug": slug} for slug in self.entidades]
        return data


def _detect_fields(normalized: str) -> list[str]:
    return [
        canonical for canonical, pattern in FIELD_QUERY_PATTERNS if re.search(pattern, normalized)
    ]


def _detect_age(question: str, normalized: str) -> int | None:
    """Edad del vacunado en meses, solo si la pregunta la expresa como edad."""

    if not re.search(r"\b\d{1,2}\s*(?:anos?|meses|mes)\b|recien nacido", normalized):
        return None
    if re.search(r"\bsemanas?\b", normalized) and not re.search(r"\banos?\b|\bmeses\b", normalized):
        return None
    ages = age_range_months(question)
    return ages[0] if ages else None


def _split_clauses(question: str) -> list[str]:
    """Oraciones y cláusulas unidas por "y" + interrogativo.

    "difteria y tétanos" no se parte: solo se corta un "y" seguido de otra
    pregunta ("…y cuáles están contraindicadas…").
    """

    parts: list[str] = []
    for sentence in re.split(r"[?¿;.]+", question):
        sentence = sentence.strip(" ,")
        if not sentence:
            continue
        pieces = re.split(
            rf"\s+y\s+(?={_INTERROGATIVO}\b)",
            sentence,
            flags=re.IGNORECASE,
        )
        parts.extend(piece.strip(" ,") for piece in pieces if piece.strip(" ,"))
    return parts


def _field_words(campos: list[str]) -> str:
    return " ".join(campo.replace("_", " ").lower() for campo in campos)


def deterministic_plan(question: str) -> Plan:
    """Plan sin LLM: catálogos + expresiones regulares. Es también el respaldo."""

    normalized = normalize_for_match(question)
    vigencias = detect_vigencias(question)
    version = "ambas" if len(vigencias) > 1 else vigencias[0]
    entidades = detect_vaccines(question)[:MAX_ENTIDADES]
    campos = _detect_fields(normalized)
    edad = _detect_age(question, normalized)
    rescate = bool(_RESCATE_RE.search(normalized))
    terapia = bool(_TERAPIA_RE.search(normalized) and _DOSIFICAR_RE.search(normalized))

    if _EVALUACION_RE.search(normalized):
        intencion = "evaluacion"
    elif version == "ambas":
        intencion = "comparacion_normas"
    elif _MAX_ATRIBUTO_RE.search(normalized):
        intencion = "max_atributo"
    elif _COMUN_RE.search(normalized) and len(entidades) >= 2:
        intencion = "comun"
    elif _COMPARA_RE.search(normalized) and len(entidades) >= 2:
        intencion = "comparacion"
    elif edad is not None:
        intencion = "edad"
    elif len(entidades) >= 2:
        intencion = "conjunto"
    elif not entidades and _GLOBAL_RE.search(normalized):
        intencion = "global"
    else:
        intencion = "lookup"

    if intencion == "comun" and re.search(r"\balergi", normalized):
        campos = list(dict.fromkeys([*campos, "CONTRAINDICACIONES", "COMPOSICION"]))
    if intencion == "max_atributo" and "COMPOSICION" not in campos:
        campos.append("COMPOSICION")

    subconsultas: list[str] = []
    # Una cláusula solo es subconsulta si se sostiene sola ("¿cómo se completa?"
    # no dice qué completar).
    clauses = [
        clause
        for clause in _split_clauses(question)
        if len([t for t in tokenize(clause) if len(t) >= 4]) >= 2
    ]
    if len(clauses) > 1:
        subconsultas.extend(clauses)
    multi = {"conjunto", "comparacion", "comun", "comparacion_normas"}
    if intencion in multi and len(entidades) > 1:
        words = _field_words(campos)
        subconsultas.extend(f"{words} {VACUNA_NOMBRES[slug]}".strip() for slug in entidades)
    subconsultas = list(dict.fromkeys(subconsultas))[:MAX_SUBCONSULTAS]

    significant = [
        token for token in tokenize(question) if len(token) >= 4 and not token.startswith("edad")
    ]
    requiere_aclaracion = bool(
        campos
        and not entidades
        and edad is None
        and not re.search(POPULATION_PATTERN, normalized)
        and intencion == "lookup"
        and len(significant) <= 3
    )

    supuestos: list[str] = []
    if version == "vigente" and not re.search(r"\b2026\b|\b246\b|vigente", normalized):
        supuestos.append("Se consulta la norma vigente (NTS 246-2026).")
    if edad is not None:
        supuestos.append(f"Edad interpretada: {edad} meses.")

    return Plan(
        pregunta=question,
        intencion=intencion,
        entidades=entidades,
        campos=campos,
        edad_meses=edad,
        version=version,
        subconsultas=subconsultas,
        supuestos=supuestos,
        requiere_aclaracion=requiere_aclaracion,
        rescate=rescate,
        terapia=terapia,
    )


def validate_plan(data: dict) -> list[str]:
    """Errores del plan frente a los catálogos y límites (§7.2). Vacío = válido."""

    errors: list[str] = []
    if data.get("intencion") not in INTENCIONES:
        errors.append(f"intencion desconocida: {data.get('intencion')!r}")
    entidades = data.get("entidades") or []
    if not isinstance(entidades, list) or len(entidades) > MAX_ENTIDADES:
        errors.append(f"entidades: máximo {MAX_ENTIDADES}")
        entidades = []
    for entidad in entidades:
        slug = entidad.get("slug") if isinstance(entidad, dict) else entidad
        if slug not in VACUNA_SLUGS:
            errors.append(f"vacuna fuera del catálogo: {slug!r}")
    for campo in data.get("campos") or []:
        if campo not in CANONICAL_FIELDS:
            errors.append(f"campo fuera del catálogo: {campo!r}")
    if data.get("version", "vigente") not in VERSIONES:
        errors.append(f"version inválida: {data.get('version')!r}")
    edad = data.get("edad_meses")
    if edad is not None and (not isinstance(edad, int) or not 0 <= edad <= MAX_EDAD_MESES):
        errors.append(f"edad_meses fuera de rango: {edad!r}")
    subconsultas = data.get("subconsultas") or []
    if not isinstance(subconsultas, list) or len(subconsultas) > MAX_SUBCONSULTAS:
        errors.append(f"subconsultas: máximo {MAX_SUBCONSULTAS}")
    return errors


PLAN_SYSTEM = f"""Eres el planificador de consultas de un sistema documental sobre la \
normativa de vacunación del Perú (NTS 246-2026 vigente y NTS 196-2022 derogada).
No respondes la pregunta: devuelves únicamente un objeto JSON con este contrato:
{{"pregunta_normalizada": str, "intencion": una de {list(INTENCIONES)},
 "entidades": [{{"slug": vacuna}}], "campos": [campo], "edad_meses": int o null,
 "version": una de {list(VERSIONES)}, "subconsultas": [str] (máximo {MAX_SUBCONSULTAS}),
 "supuestos": [str], "premisas_no_soportadas": [str], "requiere_aclaracion": bool}}
Vacunas válidas (slug): {", ".join(VACUNA_SLUGS)}.
Campos válidos: {", ".join(CANONICAL_FIELDS)}.
Reglas: "version" es "vigente" salvo que la pregunta nombre la norma 2022/196/anterior
("derogada") o compare normas ("ambas"). "evaluacion" es para juicios de valor que la
norma no hace (mejor, más segura). "max_atributo" para superlativos sobre algo
contable (más cepas). "comun" si pregunta qué comparten varias vacunas. Anota en
"premisas_no_soportadas" lo que la pregunta da por cierto y podría no serlo.
"""


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _plan_from_llm(question: str, data: dict, fallback: Plan) -> Plan:
    entidades = [
        entidad.get("slug") if isinstance(entidad, dict) else entidad
        for entidad in data.get("entidades") or []
    ]
    normalizada = (data.get("pregunta_normalizada") or "").strip() or None
    return Plan(
        pregunta=question,
        intencion=data["intencion"],
        entidades=[slug for slug in entidades if slug],
        campos=list(data.get("campos") or []),
        edad_meses=data.get("edad_meses"),
        version=data.get("version") or "vigente",
        subconsultas=[str(s) for s in data.get("subconsultas") or []],
        supuestos=[str(s) for s in data.get("supuestos") or []],
        premisas_no_soportadas=[str(s) for s in data.get("premisas_no_soportadas") or []],
        requiere_aclaracion=bool(data.get("requiere_aclaracion")),
        # Señales deterministas que el contrato del LLM no trae.
        rescate=fallback.rescate,
        terapia=fallback.terapia,
        pregunta_normalizada=normalizada if normalizada != question else None,
        origen="llm",
    )


def plan_query(question: str, chat: Callable[[str, str], str | None] | None = None) -> Plan:
    """Plan de la pregunta. `chat(system, user)` es la llamada al LLM, si la hay."""

    fallback = deterministic_plan(question)
    if chat is None:
        return fallback

    user = f"Pregunta: {question}"
    errors: list[str] = []
    for _ in range(2):
        text = chat(PLAN_SYSTEM, user if not errors else f"{user}\n\nCorrige: {'; '.join(errors)}")
        data = _parse_json(text or "")
        if data is None:
            errors = ["la respuesta no era un objeto JSON"]
            continue
        errors = validate_plan(data)
        if not errors:
            return _plan_from_llm(question, data, fallback)
    logger.warning("Plan del LLM inválido (%s); se usa el plan determinista", "; ".join(errors))
    fallback.avisos.append("El plan del LLM no pasó la validación; se usó el plan determinista.")
    return fallback
