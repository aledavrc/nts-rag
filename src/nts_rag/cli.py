from __future__ import annotations

import argparse
import json
from collections import Counter

from nts_rag.config import get_settings
from nts_rag.evaluation import (
    RowResult,
    citations_ok,
    contamination,
    coverage,
    expected_sources,
    format_summary,
    row_type,
    score_retrieval,
    summarize,
    summarize_by_type,
)
from nts_rag.index import (
    build_index,
    corpus_lint,
    freshness_problems,
    get_collection,
    load_chunks,
)
from nts_rag.rag.answer import NO_ANSWER, plan_and_gather, run_query
from nts_rag.rag.embeddings import Embedder
from nts_rag.rag.retrieve import Retriever


def cmd_indexar(args: argparse.Namespace) -> None:
    settings = get_settings()
    build_index(settings, reset=not args.sin_reiniciar)


def _print_chunks(settings, question: str, k: int) -> None:
    retriever = Retriever(settings)
    chunks = retriever.search(question, k=k)
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- [{index}] score={chunk.score} ---")
        print(f"p. {chunk.pagina} | {chunk.seccion} | {chunk.tipo}")
        print(" ".join(chunk.text.split())[:900])


def cmd_consultar(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.sin_rerank:
        settings = settings.model_copy(update={"rerank": False})
    retriever = Retriever(settings)
    chunks, result = run_query(settings, retriever, args.pregunta, k=args.k)
    if args.json:
        print(
            json.dumps(
                {
                    "respuesta": result.text,
                    "modo": result.mode,
                    "fuentes": [
                        {
                            "pagina": chunk.pagina,
                            "pagina_fin": chunk.pagina_fin,
                            "seccion": chunk.seccion,
                            "tipo": chunk.tipo,
                            "score": chunk.score,
                            "fragmento": " ".join(chunk.text.split())[:500],
                        }
                        for chunk in chunks
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.render:
        from rich.console import Console
        from rich.markdown import Markdown

        Console().print(Markdown(result.text.replace("<br>", " · ")))
    else:
        print(result.text)
    print(f"\n[modo: {result.mode}]")
    for index, chunk in enumerate(chunks, start=1):
        print(f"[{index}] p. {chunk.pagina} | {chunk.seccion} | {chunk.tipo}")


def cmd_buscar(args: argparse.Namespace) -> None:
    _print_chunks(get_settings(), args.consulta, args.k)


def _preview_vector(vector, width: int = 6) -> list[float]:
    return [round(float(value), 4) for value in vector[:width]]


def cmd_inspeccionar(args: argparse.Namespace) -> None:
    """Estructura del índice y vectores que activa una pregunta concreta."""

    settings = get_settings()
    collection = get_collection(settings)
    total = collection.count()
    if total == 0:
        raise SystemExit(
            f"La colección {settings.chroma_collection!r} está vacía: ejecuta `make index`."
        )

    metadatas = collection.get(limit=total, include=["metadatas"])["metadatas"]
    dimension = len(collection.get(limit=1, include=["embeddings"])["embeddings"][0])
    indice = {
        "coleccion": collection.name,
        "ruta": str(settings.chroma_dir),
        "registros": total,
        "dimension": dimension,
        "espacio": "cosine",
        "embedding_model": settings.embedding_model,
        "metadatos": sorted({key for meta in metadatas for key in meta}),
        "tipos": dict(sorted(Counter(meta.get("tipo") for meta in metadatas).items())),
    }

    data = collection.query(
        query_embeddings=[Embedder(settings).embed_query(args.pregunta)],
        n_results=min(args.k, total),
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    resultados = []
    for chunk_id, text, meta, distance, embedding in zip(
        data["ids"][0],
        data["documents"][0],
        data["metadatas"][0],
        data["distances"][0],
        data["embeddings"][0],
        strict=False,
    ):
        meta = meta or {}
        resultados.append(
            {
                "id": chunk_id,
                "similitud": round(1.0 - float(distance), 4),
                "pagina": meta.get("pagina"),
                "pagina_fin": meta.get("pagina_fin"),
                "seccion": meta.get("seccion"),
                "seccion_corta": meta.get("seccion_corta"),
                "tipo": meta.get("tipo"),
                "campo": meta.get("campo"),
                "vector": _preview_vector(embedding),
                "fragmento": " ".join(str(text).split())[:300],
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "indice": indice,
                    "pregunta": args.pregunta,
                    "resultados": resultados,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("== índice ==")
    print(f"colección : {indice['coleccion']}  ({indice['ruta']})")
    print(f"registros : {indice['registros']}")
    print(f"dimensión : {indice['dimension']}  (espacio {indice['espacio']})")
    print(f"modelo    : {indice['embedding_model']}")
    print(f"metadatos : {', '.join(indice['metadatos'])}")
    print(f"tipos     : {' · '.join(f'{k}={v}' for k, v in indice['tipos'].items())}")

    print(f"\n== consulta ==\n{args.pregunta}")
    for position, result in enumerate(resultados, start=1):
        vector = "" if args.sin_vector else (
            f"  vec={result['vector']} … ({indice['dimension']}d)"
        )
        print(
            f"\n[{position}] sim={result['similitud']:.4f} · "
            f"p.{result['pagina']}-{result['pagina_fin']} · "
            f"{result['seccion_corta'] or result['seccion']} · {result['tipo']}"
            + (f" · {result['campo']}" if result["campo"] else "")
        )
        print(f"  id : {result['id']}{vector}")
        print(f"  txt: {result['fragmento']}")


def _system_state(result) -> tuple[str, str | None]:
    """Estado y motivo que devuelve hoy el sistema.

    La v1 solo distingue responder de abstenerse; cuando `AnswerResult` tenga
    `estado`/`motivo` (Fase 2) se usarán directamente.
    """

    estado = getattr(result, "estado", None)
    if estado is None:
        estado = "abstenida" if result.text.strip() == NO_ANSWER else "respondida"
    return estado, getattr(result, "motivo", None)


def cmd_evaluar(args: argparse.Namespace) -> None:
    settings = get_settings()
    retriever = Retriever(settings)
    ks = sorted(set(args.k))
    max_k = ks[-1]

    results: list[RowResult] = []
    for path in args.archivo:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        print(f"\n## {path}")
        for row in rows:
            question = row["pregunta"]
            result = RowResult(row=row, tipo=row_type(row))
            sources = expected_sources(row)
            # Se mide la recuperación del sistema completo (plan + evidencia), la
            # misma que usa /consulta; `--solo-busqueda` mide el buscador híbrido.
            if args.solo_busqueda:
                chunks = retriever.search(question, k=max_k)
            else:
                plan, evidence = plan_and_gather(retriever, question, k=max_k)
                chunks = evidence.fuentes
                result.contaminacion = contamination(plan, chunks[: ks[0]])
                result.cobertura = coverage(row, chunks)
            for k in ks:
                result.por_k[k] = score_retrieval(chunks, sources, k)

            if not args.sin_respuesta:
                _, answer = run_query(settings, retriever, question, k=settings.top_k)
                result.estado, result.motivo = _system_state(answer)
                result.citas = citations_ok(row, answer.text)
                reformulada = getattr(answer, "reformulacion", None)
                if reformulada and sources:
                    result.reformulacion = score_retrieval(
                        retriever.search(reformulada, k=max_k), sources, max_k
                    )
            results.append(result)

            if sources:
                score = result.por_k[ks[0]]
                marca = "OK" if score.completa else ("PARCIAL" if score.hit else "FALLA")
                detalle = f"{sum(score.encontradas)}/{len(sources)} fuentes@{ks[0]}"
            else:
                marca, detalle = "NEG", "sin fuentes"
            if result.estado is not None:
                detalle += f" · estado {result.estado} (esperado {result.estado_esperado})"
            print(f"[{marca}] {result.tipo} · {detalle} | {question[:80]}")
            if sources and not result.por_k[ks[0]].completa:
                encontrados = ", ".join(
                    f"p.{c.pagina}-{c.pagina_fin} {c.seccion_numero or c.seccion[:20]}"
                    for c in chunks[: ks[0]]
                )
                print(f"        recuperado: {encontrados}")

    resumen = summarize(results, ks)
    por_tipo = summarize_by_type(results, ks) if args.por_tipo else {}
    if args.json:
        print(json.dumps({"total": resumen, "por_tipo": por_tipo}, ensure_ascii=False, indent=2))
        return

    print()
    print(format_summary(resumen, ks, title="total"))
    for tipo, summary in por_tipo.items():
        print(format_summary(summary, ks, title=tipo))
    # Línea compatible con la salida anterior (Hit@k de lookups).
    n = resumen["con_fuentes"]
    if n:
        hits = resumen[f"@{ks[0]}"]["hit"]
        print(f"\nHit@{ks[0]}: {hits}/{n} = {hits / n:.2f}")


def _matrix():
    from nts_rag.rag.fichas import Matrix

    settings = get_settings()
    return Matrix(load_chunks(settings.chunks_path))


def _slug(value: str) -> str:
    from nts_rag.catalogo import VACUNA_SLUGS, detect_vaccines

    if value in VACUNA_SLUGS:
        return value
    found = detect_vaccines(value)
    if not found:
        raise SystemExit(f"Vacuna no reconocida: {value!r}. Slugs: {', '.join(VACUNA_SLUGS)}")
    return found[0]


def _campos(values: list[str] | None) -> list[str] | None:
    from nts_rag.rag.fichas import FIELDS

    if not values:
        return None
    campos = [value.upper() for value in values]
    unknown = [campo for campo in campos if campo not in FIELDS]
    if unknown:
        raise SystemExit(f"Campos desconocidos: {unknown}. Válidos: {', '.join(FIELDS)}")
    return campos


def cmd_ficha(args: argparse.Namespace) -> None:
    from nts_rag.rag.fichas import CURRENT, FIELD_LABELS, PREVIOUS, _cell_text, name

    matrix, slug = _matrix(), _slug(args.vacuna)
    doc = PREVIOUS if args.norma == "2022" else CURRENT
    print(f"# {name(slug)} — {'NTS 196-2022' if doc == PREVIOUS else 'NTS 246-2026'}\n")
    for campo, cell, status in matrix.ficha(slug, doc):
        print(f"- **{FIELD_LABELS[campo]}**: {_cell_text(cell, status, 400)}")


def cmd_comparar(args: argparse.Namespace) -> None:
    from nts_rag.rag.fichas import CURRENT, PREVIOUS, render_compare
    from nts_rag.rag.structured import DEFAULT_COMPARE

    matrix = _matrix()
    doc = PREVIOUS if args.norma == "2022" else CURRENT
    slugs = [_slug(value) for value in args.vacunas]
    table, _ = render_compare(matrix, slugs, _campos(args.campo) or DEFAULT_COMPARE, doc)
    print(table)


def cmd_diff(args: argparse.Namespace) -> None:
    from nts_rag.rag.fichas import render_diff

    matrix = _matrix()
    if args.evaluar:
        for path in args.evaluar:
            rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
            nuevas = {row.slug: row.clase for row in matrix.catalog_diff()}
            hits = 0
            for row in rows:
                if row["campo"] == "FICHA":
                    got = nuevas.get(row["vacuna"], "sin_cambio")
                else:
                    got = matrix.diff_cell(row["vacuna"], row["campo"]).clase
                hits += got == row["esperado"]
                if got != row["esperado"]:
                    print(f"[FALLA] {row['vacuna']}/{row['campo']}: {got} ≠ {row['esperado']}")
            print(f"{path}: precisión {hits}/{len(rows)} = {hits / len(rows):.2f}")
        return
    if args.vacuna:
        rows = matrix.diff(_slug(args.vacuna), _campos(args.campo))
        table, _ = render_diff(rows, only_changes=args.solo_cambios)
    else:
        table, _ = render_diff(matrix.catalog_diff())
    print(table)


def cmd_doctor(args: argparse.Namespace) -> None:
    import importlib.util
    import sys

    settings = get_settings()
    print(f"Python {sys.version.split()[0]} · {sys.executable}")
    print(f"Directorio: {settings.data_dir.resolve().parent}")

    def check(label: str, condition: bool, hint: str) -> None:
        print(f"[{'OK   ' if condition else 'FALTA'}] {label}")
        if not condition and hint:
            print(f"        -> {hint}")

    chroma_ready = settings.chroma_dir.is_dir() and any(settings.chroma_dir.iterdir())
    cache_ready = settings.fastembed_cache.is_dir() and any(settings.fastembed_cache.iterdir())

    print("\nConsulta (API):")
    check(
        "Índice ChromaDB en data/index/chroma",
        chroma_ready,
        "make index (reconstruye desde los chunks)",
    )
    check("Modelo de embeddings", cache_ready, "se descarga en la primera consulta")
    check(
        "Dependencias de consulta",
        importlib.util.find_spec("chromadb") is not None,
        "make install",
    )

    print("\nDatos de entrada:")
    check(
        f"Chunks en {settings.chunks_path}",
        settings.chunks_path.exists(),
        "make chunks",
    )

    print("\nFrescura del índice (hashes del manifiesto):")
    problems = freshness_problems(settings)
    check("Índice al día con el corpus y los chunks", not problems, "make chunks && make index")
    for problem in problems:
        print(f"        · {problem}")
    if settings.chunks_path.exists():
        errors = corpus_lint(load_chunks(settings.chunks_path))
        check("corpus_lint (fichas, campos y páginas)", not errors, "revisa los Markdown")
        for error in errors[:10]:
            print(f"        · {error}")

    if chroma_ready:
        print("\nListo para consultar: make serve (el modelo se descarga en la primera consulta)")
    else:
        print("\nAún no se puede consultar: falta el índice (make index).")


def cmd_servir(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "nts_rag.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nts-rag", description="Consulta RAG de la NTS 246")
    sub = parser.add_subparsers(dest="command", required=True)

    indexar = sub.add_parser("indexar", help="Indexa los chunks en ChromaDB")
    indexar.add_argument("--sin-reiniciar", action="store_true")
    indexar.set_defaults(func=cmd_indexar)

    consultar = sub.add_parser("consultar", help="Pregunta al RAG")
    consultar.add_argument("pregunta")
    consultar.add_argument("-k", type=int, default=6)
    consultar.add_argument("--json", action="store_true")
    consultar.add_argument(
        "--render", action="store_true", help="Renderiza el Markdown en terminal"
    )
    consultar.add_argument(
        "--sin-rerank",
        action="store_true",
        help="Omite el re-rankeo con el LLM (ahorra una llamada)",
    )
    consultar.set_defaults(func=cmd_consultar)

    buscar = sub.add_parser("buscar", help="Solo recuperación, sin LLM")
    buscar.add_argument("consulta")
    buscar.add_argument("-k", type=int, default=6)
    buscar.set_defaults(func=cmd_buscar)

    inspeccionar = sub.add_parser(
        "inspeccionar", help="Estructura del índice vectorial y vectores de una pregunta"
    )
    inspeccionar.add_argument("pregunta")
    inspeccionar.add_argument("-k", type=int, default=5)
    inspeccionar.add_argument("--json", action="store_true", help="Salida JSON")
    inspeccionar.add_argument(
        "--sin-vector",
        action="store_true",
        help="Oculta las primeras componentes del vector",
    )
    inspeccionar.set_defaults(func=cmd_inspeccionar)

    evaluar = sub.add_parser(
        "evaluar", help="Mide recuperación (hit, recall, completas, precisión) y abstención"
    )
    evaluar.add_argument(
        "--archivo", nargs="+", default=["eval/preguntas.jsonl"], help="Uno o varios JSONL"
    )
    evaluar.add_argument(
        "-k", type=int, nargs="+", default=[5], help="Uno o varios cortes, p. ej. -k 5 10 20"
    )
    evaluar.add_argument("--por-tipo", action="store_true", help="Desglosa por tipo de pregunta")
    evaluar.add_argument(
        "--sin-respuesta",
        action="store_true",
        help="Solo recuperación: no ejecuta la respuesta ni mide el estado",
    )
    evaluar.add_argument("--json", action="store_true", help="Resumen en JSON")
    evaluar.add_argument(
        "--solo-busqueda",
        action="store_true",
        help="Mide el buscador híbrido sin plan ni slots (comparación con fases previas)",
    )
    evaluar.set_defaults(func=cmd_evaluar)

    servir = sub.add_parser("servir", help="Levanta la API FastAPI")
    servir.add_argument("--host", default="127.0.0.1")
    servir.add_argument("--port", type=int, default=8000)
    servir.add_argument("--reload", action="store_true")
    servir.set_defaults(func=cmd_servir)

    ficha = sub.add_parser("ficha", help="Los 13 campos de una vacuna, con cita por celda")
    ficha.add_argument("vacuna", help="slug (bcg, spr…) o nombre")
    ficha.add_argument("--norma", choices=["2026", "2022"], default="2026")
    ficha.set_defaults(func=cmd_ficha)

    comparar = sub.add_parser("comparar", help="Tabla de varias vacunas × campos")
    comparar.add_argument("vacunas", nargs="+")
    comparar.add_argument("--campo", nargs="+", help="Campos canónicos (por defecto, 7)")
    comparar.add_argument("--norma", choices=["2026", "2022"], default="2026")
    comparar.set_defaults(func=cmd_comparar)

    diff = sub.add_parser("diff", help="Cambios entre la NTS 196-2022 y la NTS 246-2026")
    diff.add_argument("vacuna", nargs="?", help="Sin vacuna: fichas nuevas y eliminadas")
    diff.add_argument("--campo", nargs="+")
    diff.add_argument("--solo-cambios", action="store_true")
    diff.add_argument(
        "--evaluar", nargs="+", metavar="JSONL", help="Precisión frente a un set etiquetado"
    )
    diff.set_defaults(func=cmd_diff)

    doctor = sub.add_parser("doctor", help="Comprueba qué falta para consultar")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
