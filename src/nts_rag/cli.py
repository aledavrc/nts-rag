from __future__ import annotations

import argparse
import json
from collections import Counter

from nts_rag.config import get_settings
from nts_rag.index import build_index, get_collection
from nts_rag.rag.answer import run_query
from nts_rag.rag.embeddings import Embedder
from nts_rag.rag.retrieve import Retriever
from nts_rag.text import normalize_for_match


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


def cmd_evaluar(args: argparse.Namespace) -> None:
    settings = get_settings()
    retriever = Retriever(settings)
    rows = [
        json.loads(line)
        for line in open(args.archivo, encoding="utf-8")
        if line.strip()
    ]
    hits = 0
    evaluables = 0
    for row in rows:
        page = row.get("pagina")
        if page is None:
            continue
        evaluables += 1
        expected_section = normalize_for_match(row.get("seccion", ""))
        chunks = retriever.search(row["pregunta"], k=args.k)
        hit = any(
            chunk.pagina <= page <= chunk.pagina_fin
            and (
                not expected_section
                or expected_section
                in normalize_for_match(f"{chunk.seccion} {chunk.seccion_ruta}")
            )
            for chunk in chunks
        )
        hits += int(hit)
        estado = "OK" if hit else "FALLA"
        print(f"[{estado}] p.{page} | {row['pregunta'][:90]}")
        if not hit:
            encontrados = ", ".join(
                f"p.{c.pagina}-{c.pagina_fin}" for c in chunks[:5]
            )
            print(f"        recuperado: {encontrados}")
    if evaluables:
        print(f"\nHit@{args.k}: {hits}/{evaluables} = {hits / evaluables:.2f}")
    print(f"Preguntas fuera de alcance: {len(rows) - evaluables}")


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
    check("Modelo de embeddings", cache_ready, "se descarga en la primera indexación")
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

    if chroma_ready and cache_ready:
        print("\nListo para consultar: make serve")
    else:
        print("\nAún no se puede consultar: falta el índice o el modelo de embeddings.")


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

    evaluar = sub.add_parser("evaluar", help="Mide Hit@k sobre un archivo JSONL")
    evaluar.add_argument("--archivo", default="eval/preguntas.jsonl")
    evaluar.add_argument("-k", type=int, default=5)
    evaluar.set_defaults(func=cmd_evaluar)

    servir = sub.add_parser("servir", help="Levanta la API FastAPI")
    servir.add_argument("--host", default="127.0.0.1")
    servir.add_argument("--port", type=int, default=8000)
    servir.add_argument("--reload", action="store_true")
    servir.set_defaults(func=cmd_servir)

    doctor = sub.add_parser("doctor", help="Comprueba qué falta para consultar")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
