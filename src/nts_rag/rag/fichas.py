"""Matriz determinista vacuna × campo × versión (D3, §6.4) y diff entre normas (D10).

Cada celda es el valor de un campo canónico de una ficha (o la fila de la vacuna en
el Anexo 2, campo `RESCATE`) en una norma, con el chunk que la cita. Sobre la
matriz se resuelven sin LLM y con cobertura completa:

* `compare`  — varias vacunas × campos en una norma;
* `filter`   — vacunas cuyo campo cumple una propiedad (lista completa);
* `diff`     — una vacuna, campo a campo, entre la NTS 196-2022 y la NTS 246-2026;
* `catalog_diff` — fichas nuevas y eliminadas;
* `valences` — valencia declarada (serotipos, serogrupos, cepas) por vacuna.

Una celda ausente es `no_aplica` (p. ej. jeringas en una vacuna oral) o
`no_consta`; nunca se rellena.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nts_rag.catalogo import CANONICAL_FIELDS, FICHAS, NO_APLICA, VACUNA_NOMBRES
from nts_rag.rag.citations import short_citation
from nts_rag.rag.retrieve import RetrievedChunk
from nts_rag.text import normalize_for_match, strip_accents, tokenize

CURRENT, PREVIOUS = "nts-246-2026", "nts-196-2022"
RESCATE = "RESCATE"
FIELDS = (*CANONICAL_FIELDS, RESCATE)
DIFF_CLASSES = (
    "cambio_real",
    "redaccional",
    "movido",
    "nuevo",
    "eliminado",
    "sin_cambio",
    "no_consta",
    "no_aplica",
)
FIELD_LABELS = {
    "TIPO_DE_INMUNIZACION": "Tipo",
    "DESCRIPCION": "Descripción",
    "COMPOSICION": "Composición",
    "PRESENTACION": "Presentación",
    "INDICACIONES": "Indicaciones",
    "ESQUEMA_DE_VACUNACION": "Esquema",
    "DOSIS": "Dosis",
    "CONSERVACION": "Conservación",
    "VIA_Y_SITIO": "Vía y sitio",
    "JERINGAS": "Jeringas",
    "EFECTOS_POSTVACUNALES": "Efectos post-vacunales",
    "CONTRAINDICACIONES": "Contraindicaciones",
    "USO_SIMULTANEO": "Uso simultáneo",
    RESCATE: "Dosis de rescate (Anexo 2)",
}
DOC_LABELS = {CURRENT: "NTS 246-2026", PREVIOUS: "NTS 196-2022"}
_RESCATE_SECTION = "dosis de rescate"


# --- Valores normalizados -------------------------------------------------------

TIPO_LABELS: tuple[tuple[str, str], ...] = (
    ("viva_atenuada", r"atenuad"),
    ("inactivada", r"inactivad|muert"),
    ("conjugada", r"conjugad"),
    ("recombinante", r"recombinant"),
    ("toxoide", r"toxoide"),
    ("subunidad", r"subunidad|acelular"),
    ("polisacarido", r"polisacarid"),
    ("pasiva", r"\bpasiva\b|anticuerpo monoclonal"),
)
VIA_LABELS: tuple[tuple[str, str], ...] = (
    ("intramuscular", r"intramuscular"),
    ("subcutanea", r"subcutane"),
    ("intradermica", r"intraderm"),
    ("oral", r"\boral\b|\bboca\b"),
)
SITE_LABELS: tuple[tuple[str, str], ...] = (
    ("muslo", r"vasto externo|\bmuslo\b"),
    ("deltoides", r"deltoides"),
    ("boca", r"\bboca\b|\boral\b"),
)
# Palabras de valencia. "pentavalente"/"hexavalente" nombran combinaciones de
# enfermedades, no cepas: no cuentan.
VALENCE_WORDS = {"bivalente": 2, "trivalente": 3, "tetravalente": 4, "nonavalente": 9}
_VALENCE_RE = re.compile(r"\bpcv ?(\d{1,2})\b|\b(\d{1,2}) ?valente\b")


def labels(text: str, vocabulary: tuple[tuple[str, str], ...]) -> set[str]:
    normalized = normalize_for_match(text)
    return {label for label, pattern in vocabulary if re.search(pattern, normalized)}


def valence(text: str) -> int | None:
    normalized = normalize_for_match(text)
    match = _VALENCE_RE.search(normalized)
    if match:
        return int(match.group(1) or match.group(2))
    for word, count in VALENCE_WORDS.items():
        if re.search(rf"\b{word}\b", normalized):
            return count
    return None


_REMISSION_RE = re.compile(
    r"\bver:?\s*(?:dosis de rescate en |casos especiales en )?"
    r"(?:anexo\s*\d+|item\s*\d+(?:\.\d+)*)"
    r"(?:\s*y\s*casos especiales en\s*item\s*\d+(?:\.\d+)*)?\.?"
)


def remissions(text: str) -> list[str]:
    """Remisiones a otra parte de la norma ("Ver Anexo 2", "casos especiales 6.3")."""

    raw = strip_accents(text).lower()
    phrases = " ".join(_REMISSION_RE.findall(raw))
    found = re.findall(r"anexo\s*(\d+)|item\s*(\d+(?:\.\d+)*)", phrases)
    return [f"Anexo {a}" if a else f"ítem {b}" for a, b in found]


# --- Diff -----------------------------------------------------------------------

_ORDINAL_RE = re.compile(r"\b\d+\s*(?:era|ra|da|do|er|ro|ta|to|°|º)\b")
_DIFF_FILLER = {"vacuna", "dosis", "una", "uno", "vez", "su", "lo", "los", "del", "al"}


def _clean_for_diff(text: str) -> str:
    raw = strip_accents(text.replace("µg", "mcg").replace("μg", "mcg")).lower()
    raw = _REMISSION_RE.sub(" ", raw)
    raw = re.sub(r"(\d)\s*cc\b", r"\1 ml", raw)
    raw = re.sub(r"\bcero\b", "0", raw)
    # "25G" y "25 G" son lo mismo: separar cifra y unidad.
    raw = re.sub(r"(\d)([a-z])", r"\1 \2", raw)
    return _canonical_ages(raw)


# Límites de edad equivalentes: "menores de 5 años" = "hasta los 4 años 11 meses 29
# días". "Hasta los 5 años 0 días" NO es lo mismo (§10: 4a11m29d ≠ 5a), así que
# solo se unifican las formas "menor de N" y "N-1 años 11 meses 29 días".
_UNDER_RE = re.compile(r"\b(?:hasta (?:los )?)?menor(?:es)? de (\d{1,2}) anos?\b")
_ELEVEN_29_RE = re.compile(
    r"\b(?:hasta (?:los )?)?(\d{1,2}) anos?,? ?(?:y )?11 ?meses,? ?(?:y )?29 ?dias\b"
)
_MONTHS_29_RE = re.compile(r"\b(?:hasta (?:los )?)?11 ?meses,? ?(?:y )?29 ?dias\b")


def _canonical_ages(raw: str) -> str:
    raw = _UNDER_RE.sub(lambda m: f" edadmenor{m.group(1)} ", raw)
    raw = _ELEVEN_29_RE.sub(lambda m: f" edadmenor{int(m.group(1)) + 1} ", raw)
    return _MONTHS_29_RE.sub(" edadmenor1 ", raw)


# D10: cada tipo de campo se compara por lo que significa.
# Familia de inmunización: lo que cambia el tipo de verdad.
_TIPO_FAMILIES = {"viva_atenuada", "inactivada", "pasiva"}
# Campos cuyo contenido son cifras (temperaturas, tiempos, volúmenes, calibres).
_NUMERIC_FIELDS = {"CONSERVACION", "DOSIS", "JERINGAS"}
# Campos que son listas: quitar o añadir un elemento es un cambio real.
_LIST_FIELDS = {"CONTRAINDICACIONES", "EFECTOS_POSTVACUNALES", "USO_SIMULTANEO"}


def _items(clean: str) -> list[set[str]]:
    parts = re.split(r"[.;•]|\s-\s", clean)
    return [tokens for part in parts if (tokens := _content(part))]


def _unmatched_items(a: str, b: str) -> bool:
    """Algún elemento de una lista no tiene equivalente (Jaccard ≥ 0,5) en la otra."""

    items_a, items_b = _items(a), _items(b)

    def matched(item: set[str], others: list[set[str]]) -> bool:
        return any(len(item & other) / len(item | other) >= 0.5 for other in others)

    return any(not matched(i, items_b) for i in items_a) or any(
        not matched(i, items_a) for i in items_b
    )


def _numbers(clean: str) -> set[str]:
    """Cifras del texto, sin ordinales de dosis ("1ra", "2do") ni ceros a la izquierda.

    Los límites de edad canónicos (`edadmenor5`) cuentan como una cifra propia.
    """

    out = set(re.findall(r"edadmenor\d+", clean))
    without_ordinals = _ORDINAL_RE.sub(" ", re.sub(r"edadmenor\d+", " ", clean))
    for token in re.findall(r"\d+(?:[.,]\d+)?", without_ordinals):
        value = float(token.replace(",", "."))
        out.add(f"{value:g}")
    return out


def _content(clean: str) -> set[str]:
    return {
        token
        for token in tokenize(clean)
        if not token.isdigit() and not token.startswith("edad") and token not in _DIFF_FILLER
    }


def classify_change(old: str, new: str, campo: str | None = None) -> str:
    """Tipo de cambio entre dos valores de la misma celda en dos normas (D10).

    * sin_cambio  — iguales tras quitar remisiones y normalizar unidades;
    * movido      — un lado solo remite a otra parte de la norma;
    * cambio_real — cambia una cifra, la familia del tipo, un elemento de una
      lista o el contenido cambia mucho;
    * redaccional — mismo contenido con otras palabras.

    Por campo: el tipo se compara por familia (viva atenuada / inactivada /
    pasiva); en conservación, dosis y jeringas mandan las cifras; en las listas
    (contraindicaciones, efectos, uso simultáneo) cada elemento debe tener
    equivalente. En el resto, Jaccard ≥ 0,5 y ≤ 6 palabras distintas.
    """

    a, b = _clean_for_diff(old), _clean_for_diff(new)
    if normalize_for_match(a) == normalize_for_match(b):
        return "sin_cambio"
    ta, tb = _content(a), _content(b)
    if not ta or not tb:
        return "movido"
    numbers_a, numbers_b = _numbers(a), _numbers(b)
    # Movido: la versión nueva conserva solo parte del contenido (palabras y cifras)
    # y remite a otra sección por el resto ("Ver: Dosis de rescate en Anexo 2").
    if (
        remissions(new)
        and tb <= ta
        and numbers_b <= numbers_a
        and (tb < ta or numbers_b < numbers_a)
    ):
        return "movido"
    # Composición: añadir detalle (volumen, excipientes, sustrato) sin quitar nada
    # de lo que había no cambia la vacuna.
    if campo == "COMPOSICION" and numbers_a <= numbers_b and len(ta & tb) / len(ta) >= 0.6:
        return "redaccional"
    if numbers_a != numbers_b:
        return "cambio_real"
    if campo == "TIPO_DE_INMUNIZACION":
        families_a = labels(old, TIPO_LABELS) & _TIPO_FAMILIES
        families_b = labels(new, TIPO_LABELS) & _TIPO_FAMILIES
        return "redaccional" if families_a == families_b else "cambio_real"
    if campo in _NUMERIC_FIELDS:
        return "redaccional"
    if campo in _LIST_FIELDS and _unmatched_items(a, b):
        return "cambio_real"
    jaccard = len(ta & tb) / len(ta | tb)
    return "redaccional" if jaccard >= 0.5 and len(ta ^ tb) <= 6 else "cambio_real"


# --- Matriz ---------------------------------------------------------------------


@dataclass
class Cell:
    slug: str
    campo: str
    documento_id: str
    valor: str
    chunk: dict[str, Any]

    @property
    def retrieved(self) -> RetrievedChunk:
        return RetrievedChunk(id=self.chunk["id"], text=self.chunk["texto"], metadata=self.chunk)

    @property
    def cita(self) -> str:
        return short_citation(self.retrieved)


@dataclass
class DiffRow:
    slug: str
    campo: str
    clase: str
    anterior: Cell | None
    actual: Cell | None


def _value(chunk: dict[str, Any]) -> str:
    text = chunk["texto"]
    if chunk["tipo"] == "campo":
        return text.split("\n\n", 1)[1].strip() if "\n\n" in text else text
    # Fila: la última línea de la tabla, sin la primera celda (el nombre).
    line = [row for row in text.splitlines() if row.strip().startswith("|")][-1]
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return " ".join(cell for cell in cells[1:] if cell)


class Matrix:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.cells: dict[tuple[str, str, str], Cell] = {}
        for chunk in chunks:
            slug = chunk.get("vacuna_slug")
            if not slug:
                continue
            if chunk["tipo"] == "campo" and chunk.get("campo"):
                campo = chunk["campo"]
            elif chunk["tipo"] == "fila" and _RESCATE_SECTION in normalize_for_match(
                chunk["seccion"]
            ):
                campo = RESCATE
            else:
                continue
            key = (slug, campo, chunk["documento_id"])
            if key in self.cells:  # continuación de la misma celda
                self.cells[key].valor = f"{self.cells[key].valor} {_value(chunk)}".strip()
            else:
                self.cells[key] = Cell(slug, campo, chunk["documento_id"], _value(chunk), chunk)

    def __bool__(self) -> bool:
        return bool(self.cells)

    def get(self, slug: str, campo: str, doc: str = CURRENT) -> Cell | None:
        return self.cells.get((slug, campo, doc))

    def status(self, slug: str, campo: str, doc: str = CURRENT) -> str:
        if slug not in FICHAS.get(doc, {}).values():
            return "sin_ficha"
        if self.get(slug, campo, doc):
            return "presente"
        if slug in NO_APLICA.get(campo, set()):
            return "no_aplica"
        return "no_consta"

    @staticmethod
    def slugs(doc: str = CURRENT) -> list[str]:
        return list(FICHAS.get(doc, {}).values())

    def ficha(self, slug: str, doc: str = CURRENT) -> list[tuple[str, Cell | None, str]]:
        return [
            (campo, self.get(slug, campo, doc), self.status(slug, campo, doc)) for campo in FIELDS
        ]

    def filter(self, campo: str, doc: str, predicate) -> list[Cell]:
        return [
            cell
            for slug in self.slugs(doc)
            if (cell := self.get(slug, campo, doc)) is not None and predicate(cell)
        ]

    def diff_cell(self, slug: str, campo: str) -> DiffRow:
        old, new = self.get(slug, campo, PREVIOUS), self.get(slug, campo, CURRENT)
        old_status = self.status(slug, campo, PREVIOUS)
        new_status = self.status(slug, campo, CURRENT)
        if old_status == "sin_ficha" and new_status != "sin_ficha":
            clase = "nuevo"
        elif new_status == "sin_ficha" and old_status != "sin_ficha":
            clase = "eliminado"
        elif old and new:
            clase = classify_change(old.valor, new.valor, campo)
        elif "no_aplica" in (old_status, new_status) and not (old or new):
            clase = "no_aplica"
        else:
            clase = "no_consta"
        return DiffRow(slug, campo, clase, old, new)

    def diff(self, slug: str, campos: list[str] | None = None) -> list[DiffRow]:
        return [self.diff_cell(slug, campo) for campo in (campos or list(FIELDS))]

    def catalog_diff(self) -> list[DiffRow]:
        current, previous = set(self.slugs(CURRENT)), set(self.slugs(PREVIOUS))
        rows = []
        for slug in self.slugs(CURRENT):
            if slug not in previous:
                rows.append(DiffRow(slug, "FICHA", "nuevo", None, self._anchor(slug, CURRENT)))
        for slug in self.slugs(PREVIOUS):
            if slug not in current:
                rows.append(DiffRow(slug, "FICHA", "eliminado", self._anchor(slug, PREVIOUS), None))
        return rows

    def _anchor(self, slug: str, doc: str) -> Cell | None:
        """Celda representativa de una ficha: su tipo o, si falta, la primera."""

        for campo in ("TIPO_DE_INMUNIZACION", "DESCRIPCION", *FIELDS):
            if cell := self.get(slug, campo, doc):
                return cell
        return None

    def valences(self, doc: str = CURRENT) -> list[tuple[str, int, Cell]]:
        out = []
        for slug in self.slugs(doc):
            for campo in ("COMPOSICION", "DESCRIPCION"):
                cell = self.get(slug, campo, doc)
                count = valence(cell.valor) if cell else None
                if cell and count:
                    out.append((slug, count, cell))
                    break
        return sorted(out, key=lambda item: -item[1])


def get_matrix(retriever: Any) -> Matrix:
    """Matriz del retriever, construida una vez y guardada en él."""

    matrix = getattr(retriever, "_matriz", None)
    if matrix is None:
        matrix = Matrix(getattr(retriever, "chunks", None) or [])
        try:
            retriever._matriz = matrix
        except AttributeError:
            pass
    return matrix


# --- Render (D11: tabla con cita por celda) ---------------------------------------


def _clip(text: str, limit: int = 260) -> str:
    text = " ".join(text.replace("|", "/").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _cell_text(cell: Cell | None, status: str, limit: int = 260) -> str:
    if cell is not None:
        return f"{_clip(cell.valor, limit)} {cell.cita}"
    return {"no_aplica": "*no aplica*", "sin_ficha": "*sin ficha en esta norma*"}.get(
        status, "*no consta*"
    )


def name(slug: str) -> str:
    return VACUNA_NOMBRES.get(slug, slug)


def render_compare(
    matrix: Matrix, slugs: list[str], campos: list[str], doc: str = CURRENT
) -> tuple[str, list[Cell]]:
    header = "| Campo | " + " | ".join(name(slug) for slug in slugs) + " |"
    lines = [header, "|" + " --- |" * (len(slugs) + 1)]
    cells: list[Cell] = []
    for campo in campos:
        row = []
        for slug in slugs:
            cell = matrix.get(slug, campo, doc)
            row.append(_cell_text(cell, matrix.status(slug, campo, doc)))
            if cell:
                cells.append(cell)
        lines.append(f"| {FIELD_LABELS[campo]} | " + " | ".join(row) + " |")
    return "\n".join(lines), cells


def render_list(title: str, cells: list[Cell], campo: str) -> str:
    lines = [f"**{title}** ({len(cells)}):", ""]
    lines += [f"- **{name(cell.slug)}** — {_clip(cell.valor, 180)} {cell.cita}" for cell in cells]
    return "\n".join(lines)


def render_field_table(matrix: Matrix, campo: str, doc: str = CURRENT) -> tuple[str, list[Cell]]:
    """Un campo de las 24 fichas, incluidas las que no lo declaran."""

    lines = [f"| Vacuna | {FIELD_LABELS[campo]} |", "| --- | --- |"]
    cells = []
    for slug in matrix.slugs(doc):
        cell = matrix.get(slug, campo, doc)
        if cell:
            cells.append(cell)
        lines.append(f"| {name(slug)} | {_cell_text(cell, matrix.status(slug, campo, doc), 220)} |")
    return "\n".join(lines), cells


def render_sites(matrix: Matrix, doc: str = CURRENT) -> tuple[str, list[Cell]]:
    """Vía y sitio de las 24 fichas, agrupadas por sitio (muslo, deltoides, boca)."""

    lines = ["| Vacuna | Vía | Sitio | Fuente |", "| --- | --- | --- | --- |"]
    cells = []
    for slug in matrix.slugs(doc):
        cell = matrix.get(slug, "VIA_Y_SITIO", doc)
        if cell is None:
            lines.append(f"| {name(slug)} | *no consta* | | |")
            continue
        cells.append(cell)
        vias = ", ".join(sorted(labels(cell.valor, VIA_LABELS))) or "—"
        sitios = ", ".join(sorted(labels(cell.valor, SITE_LABELS))) or _clip(cell.valor, 80)
        lines.append(f"| {name(slug)} | {vias} | {sitios} | {cell.cita} |")
    return "\n".join(lines), cells


DIFF_TEXT = {
    "cambio_real": "**cambió**",
    "redaccional": "redacción",
    "movido": "movido",
    "nuevo": "**nueva en 2026**",
    "eliminado": "**eliminada en 2026**",
    "sin_cambio": "sin cambio",
    "no_consta": "no consta",
    "no_aplica": "no aplica",
}


def render_diff(rows: list[DiffRow], *, only_changes: bool = False) -> tuple[str, list[Cell]]:
    lines = [
        "| Vacuna | Campo | Cambio | NTS 196-2022 (derogada) | NTS 246-2026 (vigente) |",
        "| --- | --- | --- | --- | --- |",
    ]
    cells: list[Cell] = []
    changes = {"cambio_real", "movido", "nuevo", "eliminado"}
    shown = [row for row in rows if not only_changes or row.clase in changes]
    for row in shown or rows:
        for cell in (row.anterior, row.actual):
            if cell:
                cells.append(cell)
        campo = FIELD_LABELS.get(row.campo, "Ficha")
        old = _cell_text(row.anterior, "sin_ficha" if row.clase == "nuevo" else "no_consta", 200)
        new = _cell_text(row.actual, "sin_ficha" if row.clase == "eliminado" else "no_consta", 200)
        lines.append(f"| {name(row.slug)} | {campo} | {DIFF_TEXT[row.clase]} | {old} | {new} |")
    return "\n".join(lines), cells


def render_valences(rows: list[tuple[str, int, Cell]]) -> tuple[str, list[Cell]]:
    lines = ["| Vacuna | Valencia declarada | Fuente |", "| --- | --- | --- |"]
    for slug, count, cell in rows:
        lines.append(f"| {name(slug)} | {count} | {cell.cita} |")
    return "\n".join(lines), [cell for _, _, cell in rows]
