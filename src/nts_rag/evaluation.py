"""Métricas de evaluación de la recuperación y de la decisión de responder.

Las funciones de este módulo son puras (no tocan ChromaDB ni el LLM) para que
puedan probarse con chunks sintéticos. `cli.cmd_evaluar` las orquesta.

Formatos de fila aceptados (JSONL):

* Heredado (`eval/preguntas.jsonl`): `pagina` + `seccion`; `pagina: null`
  marca una pregunta fuera de alcance.
* v2 (`eval/agregacion.jsonl`, `eval/ambiguedades.jsonl`, …): `tipo`,
  `fuentes_esperadas` (lista de `{pagina, pagina_fin?, seccion, documento?}`),
  `estado_esperado` y, opcionalmente, `motivo_esperado`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from nts_rag.config import DOC_ID
from nts_rag.rag.retrieve import RetrievedChunk
from nts_rag.text import normalize_for_match

ESTADOS = ("respondida", "parcial", "abstenida", "aclaracion")
# Estados en los que el sistema no debe entregar una respuesta directa.
NO_RESPONDE = {"abstenida", "aclaracion"}


def expected_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "fuentes_esperadas" in row:
        return list(row["fuentes_esperadas"] or [])
    if row.get("pagina") is not None:
        return [{"pagina": row["pagina"], "seccion": row.get("seccion")}]
    return []


def row_type(row: dict[str, Any]) -> str:
    if row.get("tipo"):
        return str(row["tipo"])
    return "lookup" if expected_sources(row) else "negativa"


def expected_state(row: dict[str, Any]) -> str:
    if row.get("estado_esperado"):
        return str(row["estado_esperado"])
    return "respondida" if expected_sources(row) else "abstenida"


def _section_matches(expected: str, chunk: RetrievedChunk) -> bool:
    """El numeral o título esperado aparece como unidad completa en la sección.

    Se exige frontera para que `6.1.1.1` no case con `6.1.1.10` ni `ANEXO 1`
    con `ANEXO 10`; una subsección (`6.2.2` dentro de `6.2`) sí cuenta.
    """

    needle = normalize_for_match(expected)
    if not needle:
        return True
    haystack = normalize_for_match(f"{chunk.seccion} {chunk.seccion_ruta}")
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def source_matches(chunk: RetrievedChunk, source: dict[str, Any]) -> bool:
    documento = source.get("documento") or DOC_ID
    if (chunk.metadata.get("documento_id") or DOC_ID) != documento:
        return False
    start = source.get("pagina")
    if start is not None:
        end = source.get("pagina_fin") or start
        if not (chunk.pagina <= end and chunk.pagina_fin >= start):
            return False
    return _section_matches(str(source.get("seccion") or ""), chunk)


@dataclass
class RetrievalScore:
    hit: bool
    recall: float
    completa: bool
    precision: float
    encontradas: list[bool]


def score_retrieval(
    chunks: list[RetrievedChunk], sources: list[dict[str, Any]], k: int
) -> RetrievalScore:
    top = chunks[:k]
    found = [any(source_matches(chunk, source) for chunk in top) for source in sources]
    relevant = sum(any(source_matches(chunk, source) for source in sources) for chunk in top)
    return RetrievalScore(
        hit=any(found),
        recall=sum(found) / len(found) if found else 0.0,
        completa=bool(found) and all(found),
        precision=relevant / len(top) if top else 0.0,
        encontradas=found,
    )


def contamination(plan: Any, chunks: list[RetrievedChunk]) -> int:
    """Fuentes que no debían estar (Fase 2, objetivo 0).

    * Norma derogada sin que la pregunta la pida.
    * Campo de ficha de una vacuna que la pregunta no nombra, cuando nombra alguna.
    """

    vigencias = set(plan.vigencias)
    entidades = set(plan.entidades)
    count = 0
    for chunk in chunks:
        if chunk.vigencia not in vigencias:
            count += 1
        elif entidades and chunk.tipo == "campo":
            if chunk.metadata.get("vacuna_slug") not in entidades:
                count += 1
    return count


def silent_substitution(plan: Any, chunks: list[RetrievedChunk], answer: Any) -> bool:
    """Fase 4 (objetivo 0): se respondió sin avisar con algo distinto de lo pedido.

    * La respuesta es `respondida` y alguna vacuna nombrada no tiene fuente propia.
    * El plan reformuló la pregunta y la respuesta no lo muestra.
    """

    from nts_rag.rag.abstain import uncovered_entities

    if answer.estado == "respondida" and plan.intencion != "comparacion_normas":
        if uncovered_entities(plan, chunks):
            return True
    reformulada = getattr(answer, "reformulacion", None)
    return bool(reformulada) and "Pregunta interpretada como" not in answer.text


def coverage(row: dict[str, Any], chunks: list[RetrievedChunk]) -> tuple[bool, float] | None:
    """Fase 3: vacunas o celdas esperadas entre todas las fuentes (no solo el top-k).

    Devuelve (completa, precisión de la lista) o None si la fila no lo pide.
    """

    if row.get("celdas_esperadas"):
        expected = {tuple(cell) for cell in row["celdas_esperadas"]}
        found = {
            (c.metadata.get("vacuna_slug"), c.campo) for c in chunks if c.tipo == "campo"
        }
        hit = expected & found
        return hit == expected, len(hit) / len(found) if found else 0.0
    if row.get("vacunas_esperadas"):
        expected = set(row["vacunas_esperadas"])
        found = {
            str(c.metadata.get("vacuna_slug"))
            for c in chunks
            if c.metadata.get("vacuna_slug") and c.tipo in {"campo", "tabla", "texto"}
        }
        hit = expected & found
        return hit == expected, len(hit) / len(found) if found else 0.0
    return None


def citations_ok(row: dict[str, Any], answer_text: str) -> bool | None:
    """Cada numeral esperado aparece citado en la respuesta (contradicciones)."""

    expected = row.get("citas_esperadas")
    if not expected:
        return None
    return all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", answer_text) for n in expected)


@dataclass
class RowResult:
    row: dict[str, Any]
    tipo: str
    por_k: dict[int, RetrievalScore] = field(default_factory=dict)
    estado: str | None = None
    motivo: str | None = None
    reformulacion: RetrievalScore | None = None
    contaminacion: int | None = None
    cobertura: tuple[bool, float] | None = None
    citas: bool | None = None
    sustitucion: bool | None = None
    llamadas: int | None = None
    tokens: int | None = None

    @property
    def estado_esperado(self) -> str:
        return expected_state(self.row)

    @property
    def decision_ok(self) -> bool | None:
        """Acierta en responder o no responder (lo único que mide la v1)."""

        if self.estado is None:
            return None
        return (self.estado in NO_RESPONDE) == (self.estado_esperado in NO_RESPONDE)

    @property
    def estado_ok(self) -> bool | None:
        if self.estado is None:
            return None
        return self.estado == self.estado_esperado


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: list[RowResult], ks: list[int]) -> dict[str, Any]:
    """Agrega métricas: recuperación en filas con fuentes y estado en todas."""

    con_fuentes = [r for r in results if expected_sources(r.row)]
    summary: dict[str, Any] = {"preguntas": len(results), "con_fuentes": len(con_fuentes)}
    for k in ks:
        scores = [r.por_k[k] for r in con_fuentes if k in r.por_k]
        summary[f"@{k}"] = {
            "hit": sum(s.hit for s in scores),
            "recall": _mean([s.recall for s in scores]),
            "completas": sum(s.completa for s in scores),
            "precision": _mean([s.precision for s in scores]),
        }

    con_estado = [r for r in results if r.estado is not None]
    deben_abstenerse = [r for r in con_estado if r.estado_esperado in NO_RESPONDE]
    deben_responder = [r for r in con_estado if r.estado_esperado not in NO_RESPONDE]
    summary["estado"] = {
        "evaluadas": len(con_estado),
        "decision_correcta": sum(bool(r.decision_ok) for r in con_estado),
        "estado_exacto": sum(bool(r.estado_ok) for r in con_estado),
        "abstiene": [
            sum(r.estado in NO_RESPONDE for r in deben_abstenerse),
            len(deben_abstenerse),
        ],
        "abstencion_indebida": [
            sum(r.estado in NO_RESPONDE for r in deben_responder),
            len(deben_responder),
        ],
        "motivo_exacto": sum(
            r.motivo is not None and r.motivo == r.row.get("motivo_esperado")
            for r in con_estado
            if r.row.get("motivo_esperado")
        ),
        "con_motivo_esperado": sum(bool(r.row.get("motivo_esperado")) for r in con_estado),
    }

    medidas = [r.contaminacion for r in results if r.contaminacion is not None]
    summary["contaminacion"] = [sum(medidas), len(medidas)] if medidas else None
    cubiertas = [r.cobertura for r in results if r.cobertura is not None]
    summary["cobertura"] = (
        {
            "completas": [sum(ok for ok, _ in cubiertas), len(cubiertas)],
            "precision": _mean([precision for _, precision in cubiertas]),
        }
        if cubiertas
        else None
    )
    citadas = [r.citas for r in results if r.citas is not None]
    summary["citas"] = [sum(citadas), len(citadas)] if citadas else None
    sustituciones = [r.sustitucion for r in results if r.sustitucion is not None]
    summary["sustitucion_silenciosa"] = (
        [sum(sustituciones), len(sustituciones)] if sustituciones else None
    )
    llamadas = [r.llamadas for r in results if r.llamadas is not None]
    tokens = [r.tokens for r in results if r.tokens is not None]
    summary["coste"] = (
        {
            "llamadas_media": _mean(llamadas),
            "llamadas_max": max(llamadas),
            "tokens_media": _mean(tokens),
        }
        if llamadas
        else None
    )

    reformuladas = [r.reformulacion for r in results if r.reformulacion is not None]
    summary["reformulacion_util"] = (
        [sum(s.hit for s in reformuladas), len(reformuladas)] if reformuladas else None
    )
    return summary


def summarize_by_type(results: list[RowResult], ks: list[int]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[RowResult]] = defaultdict(list)
    for result in results:
        groups[result.tipo].append(result)
    return {tipo: summarize(rows, ks) for tipo, rows in sorted(groups.items())}


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def format_summary(summary: dict[str, Any], ks: list[int], *, title: str) -> str:
    lines = [f"== {title} ({summary['preguntas']} preguntas) =="]
    n = summary["con_fuentes"]
    if n:
        for k in ks:
            m = summary[f"@{k}"]
            lines.append(
                f"  @{k:<3} hit {m['hit']}/{n} · recall {_fmt(m['recall'])} · "
                f"completas {m['completas']}/{n} · precision {_fmt(m['precision'])}"
            )
    estado = summary["estado"]
    if estado["evaluadas"]:
        abst, abst_n = estado["abstiene"]
        indeb, indeb_n = estado["abstencion_indebida"]
        lines.append(
            f"  decisión responder/abstenerse {estado['decision_correcta']}/{estado['evaluadas']}"
            f" · estado exacto {estado['estado_exacto']}/{estado['evaluadas']}"
        )
        if abst_n:
            lines.append(f"  abstiene cuando debe {abst}/{abst_n}")
        if indeb_n:
            lines.append(f"  se abstiene sin deber {indeb}/{indeb_n}")
        if estado["con_motivo_esperado"]:
            lines.append(
                f"  motivo exacto {estado['motivo_exacto']}/{estado['con_motivo_esperado']}"
            )
    if summary.get("contaminacion"):
        total, preguntas = summary["contaminacion"]
        lines.append(f"  contaminación {total} fuentes en {preguntas} preguntas (@{ks[0]})")
    if summary.get("cobertura"):
        ok, n = summary["cobertura"]["completas"]
        precision = _fmt(summary["cobertura"]["precision"])
        lines.append(f"  cobertura de vacunas/celdas {ok}/{n} · precisión de la lista {precision}")
    if summary.get("citas"):
        lines.append("  ambas citas en la respuesta {}/{}".format(*summary["citas"]))
    if summary.get("sustitucion_silenciosa"):
        lines.append("  sustitución silenciosa {}/{}".format(*summary["sustitucion_silenciosa"]))
    if summary.get("coste"):
        coste = summary["coste"]
        lines.append(
            f"  llamadas LLM media {_fmt(coste['llamadas_media'])} · máx {coste['llamadas_max']}"
            f" · tokens media {_fmt(coste['tokens_media'])}"
        )
    reformulacion = summary["reformulacion_util"]
    if reformulacion:
        lines.append(f"  reformulación útil {reformulacion[0]}/{reformulacion[1]}")
    else:
        lines.append("  reformulación útil n/d (sin paso de reformulación)")
    return "\n".join(lines)
