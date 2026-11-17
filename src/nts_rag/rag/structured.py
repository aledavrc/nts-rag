"""Rutas estructuradas sobre la matriz (§6.3, §8.3): 0 llamadas, cobertura garantizada.

| Intención                         | Operación          | Garantía                 |
|-----------------------------------|--------------------|--------------------------|
| comparación / conjunto / común    | `compare`          | todas las vacunas × campos |
| global por propiedad              | `filter`           | lista completa           |
| vía y sitio sin vacuna            | `render_sites`     | las 24 fichas            |
| superlativo (más cepas…)          | `valences`         | todas las que declaran   |
| comparación de normas             | `diff` / `catalog_diff` | celda a celda, con citas |

La respuesta es una plantilla (tabla con cita por celda, D11). Las celdas usadas
son las fuentes de la respuesta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nts_rag.rag.fichas import (
    CURRENT,
    FIELD_LABELS,
    PREVIOUS,
    RESCATE,
    TIPO_LABELS,
    Cell,
    Matrix,
    labels,
    name,
    render_compare,
    render_diff,
    render_list,
    render_sites,
    render_valences,
)
from nts_rag.rag.interpret import Plan
from nts_rag.text import expand_tokens, normalize_for_match, tokenize

# Campos por defecto al comparar vacunas sin campo pedido.
DEFAULT_COMPARE = [
    "TIPO_DE_INMUNIZACION",
    "COMPOSICION",
    "ESQUEMA_DE_VACUNACION",
    "DOSIS",
    "VIA_Y_SITIO",
    "CONSERVACION",
    "CONTRAINDICACIONES",
]
# Campos que se pueden filtrar como propiedad global. Esquema y dosis los
# responde el calendario (6.2.x), no un filtro de fichas.
PROPERTY_FIELDS = {
    "TIPO_DE_INMUNIZACION",
    "CONTRAINDICACIONES",
    "COMPOSICION",
    "PRESENTACION",
    "CONSERVACION",
    "JERINGAS",
    "EFECTOS_POSTVACUNALES",
    "INDICACIONES",
}
# Palabras que nombran el campo o la pregunta, no la propiedad buscada.
_FIELD_WORDS = {
    token
    for word in (
        "vacuna vacunas que cuales son esta estan contraindicada contraindicadas "
        "contraindicacion contraindicaciones tipo virus composicion contiene presentacion "
        "conservacion jeringa jeringas efectos indicacion indicaciones aplica aplican "
        "tienen tiene llevan lleva"
    ).split()
    for token in tokenize(word)
}


@dataclass
class Structured:
    kind: str
    text: str
    cells: list[Cell] = field(default_factory=list)

    @property
    def fuentes(self):
        seen, out = set(), []
        for cell in self.cells:
            if cell.chunk["id"] not in seen:
                seen.add(cell.chunk["id"])
                out.append(cell.retrieved)
        return out


def _doc(plan: Plan) -> str:
    return PREVIOUS if plan.version == "derogada" else CURRENT


def _property_terms(question: str) -> set[str]:
    terms = {
        token
        for token in tokenize(question)
        if len(token) >= 4 and token not in _FIELD_WORDS and not token.startswith("edad")
    }
    return expand_tokens(terms)


def _filter_answer(plan: Plan, matrix: Matrix) -> Structured | None:
    doc = _doc(plan)
    campos = [campo for campo in plan.campos if campo in PROPERTY_FIELDS]
    if not campos:
        return None
    campo = campos[0]
    if campo == "TIPO_DE_INMUNIZACION":
        wanted = labels(plan.pregunta, TIPO_LABELS)
        if not wanted:
            return None
        cells = matrix.filter(campo, doc, lambda c: wanted <= labels(c.valor, TIPO_LABELS))
        title = "Vacunas de tipo " + ", ".join(sorted(w.replace("_", " ") for w in wanted))
    else:
        terms = _property_terms(plan.pregunta)
        if not terms:
            return None
        cells = matrix.filter(campo, doc, lambda c: bool(terms & set(tokenize(c.valor))))
        asked = sorted(
            {w for w in re.findall(r"\w+", normalize_for_match(plan.pregunta))
             if set(tokenize(w)) & terms}
        )
        title = f"Vacunas cuyo campo «{FIELD_LABELS[campo].lower()}» menciona «{', '.join(asked)}»"
    if not cells:
        return None
    return Structured("filtro", render_list(title, cells, campo), cells)


def structured_answer(plan: Plan, matrix: Matrix) -> Structured | None:
    """Respuesta por plantilla si la intención la resuelve la matriz; None si no."""

    if not matrix:
        return None
    question = normalize_for_match(plan.pregunta)

    if plan.intencion == "comparacion_normas":
        if plan.entidades:
            campos = list(plan.campos)
            if plan.rescate:
                campos.append(RESCATE)
                if re.search(r"\bedad\b", question):
                    campos.append("ESQUEMA_DE_VACUNACION")
            only_changes = not campos
            rows = [row for slug in plan.entidades for row in matrix.diff(slug, campos or None)]
            table, cells = render_diff(rows, only_changes=only_changes)
            intro = "**Comparación NTS 196-2022 (derogada) → NTS 246-2026 (vigente):**"
            return Structured("diff", f"{intro}\n\n{table}", cells)
        rows = matrix.catalog_diff()
        if not rows:
            return None
        nuevas = [row for row in rows if row.clase == "nuevo"]
        eliminadas = [row for row in rows if row.clase == "eliminado"]
        lines = [f"**Fichas nuevas en la NTS 246-2026** ({len(nuevas)}):", ""]
        lines += [f"- **{name(r.slug)}** {r.actual.cita if r.actual else ''}" for r in nuevas]
        if eliminadas:
            lines += ["", f"**Fichas que ya no están** ({len(eliminadas)}):", ""]
            lines += [
                f"- {name(r.slug)} {r.anterior.cita if r.anterior else ''}" for r in eliminadas
            ]
        cells = [cell for row in rows for cell in (row.anterior, row.actual) if cell]
        return Structured("diff_catalogo", "\n".join(lines), cells)

    if plan.intencion in {"comparacion", "conjunto", "comun"} and len(plan.entidades) >= 2:
        campos = plan.campos or DEFAULT_COMPARE
        table, cells = render_compare(matrix, plan.entidades, campos, _doc(plan))
        return Structured("comparacion", table, cells)

    if plan.intencion == "max_atributo" and not plan.entidades:
        rows = matrix.valences(_doc(plan))
        if not rows:
            return None
        table, cells = render_valences(rows)
        intro = "**Valencia que declara cada ficha (serotipos, serogrupos o cepas):**"
        return Structured("valencias", f"{intro}\n\n{table}", cells)

    # "¿Qué vacunas se aplican en el muslo y cuáles en el deltoides?": las 24.
    asks_list = re.search(r"\bque vacunas\b|\bcuales\b|\ben que vacunas\b", question)
    if not plan.entidades and "VIA_Y_SITIO" in plan.campos and asks_list:
        table, cells = render_sites(matrix, _doc(plan))
        return Structured("sitios", table, cells)

    if plan.intencion == "global" and not plan.entidades:
        return _filter_answer(plan, matrix)
    return None
