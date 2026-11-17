"""Veredicto de la consulta (D16, §7.3): lo decide el código, no el LLM.

Estados `respondida`, `parcial`, `abstenida`, `aclaracion`, con un motivo de la
taxonomía. Se decide con cobertura de slots, conflictos entre fuentes y
presuposiciones, en este orden: primero lo que impide responder, después lo que
obliga a responder con reservas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nts_rag.catalogo import ALLERGENS, VACUNA_NOMBRES
from nts_rag.rag.interpret import Plan
from nts_rag.rag.retrieve import RetrievedChunk
from nts_rag.text import age_range_months, normalize_for_match

ESTADOS = ("respondida", "parcial", "abstenida", "aclaracion")
MOTIVOS = (
    "fuera_de_alcance",
    "sin_evidencia",
    "evidencia_insuficiente",
    "conflicto",
    "premisa_invalida",
    "metrica_no_comparable",
    "requiere_conocimiento_externo",
    "ambiguedad",
)
_TIMING_RE = re.compile(r"\bcuando\b|que edad|a partir de|hasta que")


@dataclass
class Verdict:
    estado: str
    motivo: str | None = None
    faltan_fuentes: list[str] = field(default_factory=list)
    conflictos: list[dict[str, Any]] = field(default_factory=list)
    premisas: list[str] = field(default_factory=list)


def _ranges_overlap(a: tuple[int | None, int | None], b: tuple[int | None, int | None]) -> bool:
    low_a, high_a = a[0] or 0, a[1] if a[1] is not None else 10**6
    low_b, high_b = b[0] or 0, b[1] if b[1] is not None else 10**6
    return low_a <= high_b and low_b <= high_a


def _format_age(low: int | None, high: int | None) -> str:
    def one(months: int) -> str:
        if months == 0:
            return "recién nacido"
        if months % 12 == 0 and months >= 24:
            return f"{months // 12} años"
        return "1 mes" if months == 1 else f"{months} meses"

    if low == high and low is not None:
        return one(low)
    return f"{one(low or 0)} a {one(high) if high is not None else '…'}"


def find_conflicts(plan: Plan, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Edad de la ficha (ESQUEMA) frente a las edades del calendario 6.2.x.

    Solo para preguntas de cuándo/a qué edad: si la ficha fija una edad que no se
    solapa con ninguna fila del calendario para la misma vacuna, son dos fuentes
    que discrepan (p. ej. anticuerpo monoclonal: "recién nacido" frente a "2
    meses" y "1 a 6 meses").
    """

    conflicts: list[dict[str, Any]] = []
    for slug in plan.entidades:
        for vigencia in plan.vigencias:
            ficha = next(
                (
                    c
                    for c in chunks
                    if c["tipo"] == "campo"
                    and c["vacuna_slug"] == slug
                    and c["campo"] == "ESQUEMA_DE_VACUNACION"
                    and c["vigencia"] == vigencia
                ),
                None,
            )
            if ficha is None:
                continue
            ficha_range = age_range_months(ficha["texto"])
            rows = [
                c
                for c in chunks
                if c["tipo"] == "fila"
                and c["vigencia"] == vigencia
                and str(c["seccion_numero"]).startswith("6.2")
                and slug in str(c.get("vacunas") or "").split(",")
                and (c.get("edad_min_meses") is not None or c.get("edad_max_meses") is not None)
            ]
            if ficha_range is None or not rows:
                continue
            row_ranges = [(c["edad_min_meses"], c["edad_max_meses"]) for c in rows]
            if any(_ranges_overlap(ficha_range, r) for r in row_ranges):
                continue
            conflicts.append(
                {
                    "vacuna": slug,
                    "campo": "ESQUEMA_DE_VACUNACION",
                    "valores": [
                        f"ficha {ficha['seccion_numero']}: {_format_age(*ficha_range)}",
                        *dict.fromkeys(
                            f"calendario {c['seccion_numero']}: "
                            f"{_format_age(c['edad_min_meses'], c['edad_max_meses'])}"
                            for c in rows
                        ),
                    ],
                    "fuentes": [ficha["id"], *[c["id"] for c in rows]],
                }
            )
    return conflicts


def check_shared_allergen(plan: Plan, chunks: list[dict[str, Any]]) -> list[str]:
    """Presuposición de "alergia común": la norma debe declararla en todas."""

    declared: dict[str, set[str]] = {}
    for slug in plan.entidades:
        texts = [
            normalize_for_match(c["texto"])
            for c in chunks
            if c["tipo"] == "campo"
            and c["vacuna_slug"] == slug
            and c["vigencia"] in plan.vigencias
            and c["campo"] in {"CONTRAINDICACIONES", "COMPOSICION"}
        ]
        declared[slug] = {a for a in ALLERGENS if any(re.search(rf"\b{a}", t) for t in texts)}
    if not declared:
        return []
    shared = set.intersection(*declared.values())
    if shared and all(declared.values()):
        return []
    detail = "; ".join(
        f"{VACUNA_NOMBRES.get(slug, slug)}: {', '.join(sorted(found)) or 'no declara alérgenos'}"
        for slug, found in declared.items()
    )
    return [f"La norma no declara un alérgeno común a todas las vacunas nombradas ({detail})."]


def judge(
    plan: Plan,
    fuentes: list[RetrievedChunk],
    *,
    answerable: bool,
    chunks: list[dict[str, Any]] | None = None,
    faltantes: list[tuple[str, str, str]] | None = None,
) -> Verdict:
    chunks = chunks or []
    if plan.intencion == "evaluacion" or plan.terapia:
        return Verdict("abstenida", "requiere_conocimiento_externo")
    if not fuentes:
        has_slots = bool(plan.entidades or plan.campos or plan.edad_meses is not None)
        return Verdict("abstenida", "sin_evidencia" if has_slots else "fuera_de_alcance")
    if not answerable:
        return Verdict("abstenida", "fuera_de_alcance")
    if plan.requiere_aclaracion:
        return Verdict("aclaracion", "ambiguedad")

    premisas = list(plan.premisas_no_soportadas)
    if plan.intencion == "comun" and "CONTRAINDICACIONES" in plan.campos:
        premisas += check_shared_allergen(plan, chunks)
    if premisas:
        return Verdict("parcial", "premisa_invalida", premisas=premisas)

    timing = bool(
        _TIMING_RE.search(normalize_for_match(plan.pregunta))
        or "ESQUEMA_DE_VACUNACION" in plan.campos
    )
    conflictos = find_conflicts(plan, chunks) if timing else []
    if conflictos:
        return Verdict("parcial", "conflicto", conflictos=conflictos)

    # En una comparación entre normas, que una vacuna no exista en la 2022 es un
    # dato ("no consta"), no una falta de evidencia.
    missing = [
        f"{VACUNA_NOMBRES.get(slug, slug)} · {campo} · {doc}"
        for slug, campo, doc in faltantes or []
        if not (plan.intencion == "comparacion_normas" and campo == "FICHA")
    ]
    if missing:
        return Verdict("parcial", "evidencia_insuficiente", faltan_fuentes=missing)
    if plan.intencion == "max_atributo":
        return Verdict("parcial", "metrica_no_comparable")
    return Verdict("respondida")
