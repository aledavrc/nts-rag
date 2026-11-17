#!/usr/bin/env python3
"""
md_to_vector.py — convierte uno o más Markdown (generados con la guía de este
procedimiento) en fragmentos listos para una base de datos vectorial.

Uso:
    python3 md_to_vector.py a.md b.md -o chunks.jsonl
    python3 md_to_vector.py *.md --max-chars 1800 --overlap 120
    python3 md_to_vector.py a.md --format csv -o chunks.csv
    python3 md_to_vector.py a.md --report-only

Cada fragmento incluye:
    id, source, doc_title, section, section_number, pages, page_start, page_end,
    type, part, parts, text, embed_text, chars

- `text` es el contenido limpio (sin marcadores de página ni énfasis Markdown).
- `embed_text` agrega la ruta de sección y las páginas, para que el fragmento
  conserve el sujeto de la consulta al generar el embedding.
- Las tablas se mantienen como tablas Markdown y no se cortan salvo que superen
  `--max-chars`; en ese caso se parten por filas repitiendo el encabezado.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PAGE_RE = re.compile(r"^>\s*_pág[s]?\.\s*([0-9]+(?:\s*[–-]\s*[0-9]+)?)")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TABLE_SEP_RE = re.compile(r"^\|[\s:\-|]+\|$")
SECTION_NUM_RE = re.compile(r"^\s*((?:\d+\.)+\d+|\d+)\b")
COMPLETION_RE = re.compile(
    r"Tabla completada con (?:la pág\.|las págs\.)\s*([0-9][0-9,\syea–\-]*)",
    re.I,
)

TAG_RE = re.compile(r"<[^>]+>")


def parse_pages(token):
    token = token.strip()
    nums = [int(x) for x in re.findall(r"\d+", token)]
    if not nums:
        return []
    m = re.search(r"(\d+)\s*[–-]\s*(\d+)", token)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        extra = [n for n in nums if n < a or n > b]
        return sorted(set(extra + list(range(a, b + 1))))
    return sorted(set(nums))


def clean_text(text):
    """Quita énfasis, etiquetas HTML y escapes, conservando el contenido."""
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = text.replace("<sup>", "^").replace("</sup>", "")
    text = text.replace("<sub>", "_").replace("</sub>", "")
    text = TAG_RE.sub("", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = text.replace("`", "")
    text = text.replace("\\*", "*").replace("\\|", "|").replace("\\_", "_")
    return text.strip()


def section_path(stack):
    return " > ".join(t for _, t in stack)


def section_number(stack):
    for _, t in reversed(stack):
        m = SECTION_NUM_RE.match(t)
        if m:
            return m.group(1)
    return ""


def split_sentences(text):
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [p for p in parts if p]


def chunk_text(text, max_chars, overlap):
    """Divide un texto en fragmentos <= max_chars, por oraciones."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for sent in split_sentences(text):
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur += " " + sent
        else:
            out.append(cur)
            tail = ""
            if overlap > 0:
                # arrastra el final del fragmento anterior
                tail = cur[-overlap:]
                cut = tail.find(" ")
                tail = tail[cut + 1:] if cut != -1 else ""
            cur = (tail + " " + sent).strip() if tail else sent
    if cur:
        out.append(cur)
    return out


def split_table(rows, max_chars):
    """Parte una tabla por filas repitiendo el encabezado."""
    header, sep, body = rows[0], rows[1], rows[2:]
    base = len(header) + len(sep) + 2
    if base + sum(len(r) + 1 for r in body) <= max_chars:
        return [rows]
    chunks, cur, size = [], [], base
    for r in body:
        if cur and size + len(r) + 1 > max_chars:
            chunks.append([header, sep] + cur)
            cur, size = [], base
        cur.append(r)
        size += len(r) + 1
    if cur:
        chunks.append([header, sep] + cur)
    return chunks


def table_width_error(rows):
    def ncols(line):
        p = re.split(r"(?<!\\)\|", line.strip())
        if p and p[0] == "":
            p = p[1:]
        if p and p[-1] == "":
            p = p[:-1]
        return len(p)
    n = ncols(rows[0])
    bad = [i for i, r in enumerate(rows[1:], start=2) if ncols(r) != n]
    return bad


def parse_blocks(path, include_preamble=False):
    """Devuelve bloques con su sección/páginas y avisos de estructura."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    stack = []
    doc_title = ""
    current_pages = []
    blocks = []
    warnings = []
    last_table = None
    seen_page = False
    i = 0

    def add(kind, text, pages, **extra):
        if not seen_page and not include_preamble:
            return None
        block = {
            "kind": kind,
            "text": text,
            "section": section_path(stack),
            "section_number": section_number(stack),
            "pages": sorted(set(pages)),
        }
        block.update(extra)
        blocks.append(block)
        return block

    while i < len(lines):
        line = lines[i]

        m = PAGE_RE.match(line)
        if m:
            pages = parse_pages(m.group(1))
            seen_page = True
            # "continuación de la tabla …" / "continuación del listado …": la
            # página continúa la última tabla aunque no traiga filas propias.
            lowered = line.lower()
            continues = any(
                marker in lowered
                for marker in ("continuación de la tabla", "continuación del listado")
            )
            if continues and last_table is not None:
                last_table["pages"] = sorted(set(last_table["pages"]) | set(pages))
            current_pages = pages
            i += 1
            continue

        m = HEAD_RE.match(line)
        if m:
            lvl = len(m.group(1))
            title = clean_text(m.group(2).strip())
            stack = [(level, t) for level, t in stack if level < lvl] + [(lvl, title)]
            if lvl == 1 and not doc_title:
                doc_title = title
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        # Tabla
        if line.startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
            j = i
            rows = []
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(lines[j])
                j += 1
            if len(rows) < 2:
                i = j
                continue
            bad = table_width_error(rows)
            if bad:
                warnings.append(
                    f"tabla con filas de distinto ancho en líneas {i + 1}+ (filas {bad[:5]})"
                )
            pages = list(current_pages)
            # nota de completado inmediatamente después
            k = j
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines):
                cm = COMPLETION_RE.search(lines[k])
                if cm:
                    extra = cm.group(1).replace(" y ", "–").replace(", ", "–")
                    pages = sorted(set(pages) | set(parse_pages(extra)))
            last_table = add("table", "\n".join(rows), pages, rows=rows)
            i = j
            continue

        # Nota de tabla o marcador de continuación ya tratado arriba
        if line.lstrip().startswith("*") and "Tabla completada" in line:
            i += 1
            continue

        # Párrafo / lista
        j = i
        buf = []
        while (j < len(lines) and lines[j].strip()
               and not HEAD_RE.match(lines[j])
               and not lines[j].startswith("|")
               and not PAGE_RE.match(lines[j])):
            buf.append(lines[j].strip())
            j += 1
        if buf:
            add("text", "\n".join(buf), list(current_pages))
        i = j

    return doc_title, blocks, warnings


def main():
    ap = argparse.ArgumentParser(
        description="Convierte Markdown a fragmentos JSONL/JSON/CSV para una base vectorial.")
    ap.add_argument("files", nargs="+", help="archivos .md a convertir")
    ap.add_argument("-o", "--output", default="chunks.jsonl",
                    help="archivo de salida (default: chunks.jsonl)")
    ap.add_argument("--format", choices=["jsonl", "json", "csv"], default="jsonl")
    ap.add_argument("--max-chars", type=int, default=1800,
                    help="tamaño máximo de cada fragmento en caracteres (default: 1800)")
    ap.add_argument("--overlap", type=int, default=0,
                    help="solapamiento entre fragmentos de texto (default: 0)")
    ap.add_argument("--table-max-chars", type=int, default=6000,
                    help="tamaño máximo de una tabla antes de partirla por filas (default: 6000)")
    ap.add_argument("--report", default=None,
                    help="archivo de reporte/auditoría (default: <output>.report.json)")
    ap.add_argument("--report-only", action="store_true",
                    help="solo genera el reporte, no los fragmentos")
    ap.add_argument("--keep-markdown", action="store_true",
                    help="incluye el Markdown original de las tablas en el campo markdown")
    ap.add_argument("--include-preamble", action="store_true",
                    help="incluye el texto anterior al primer marcador de página")
    args = ap.parse_args()

    all_chunks = []
    report = {"generated": datetime.now(UTC).isoformat(),
              "config": {"max_chars": args.max_chars, "overlap": args.overlap},
              "files": [], "totals": {}}

    for path in args.files:
        p = Path(path)
        if not p.exists():
            print(f"ERROR: no existe {p}", file=sys.stderr)
            sys.exit(1)
        chunks, warnings = make_chunks_for_file(
            str(p),
            args.max_chars,
            args.overlap,
            args.keep_markdown,
            args.include_preamble,
            args.table_max_chars,
        )
        all_chunks.extend(chunks)
        by_type = {}
        pages = set()
        for c in chunks:
            by_type[c["type"]] = by_type.get(c["type"], 0) + 1
            pages.update(c["pages"] or [])
        report["files"].append({
            "file": p.name,
            "chunks": len(chunks),
            "by_type": by_type,
            "pages": f"{min(pages)}-{max(pages)}" if pages else "",
            "no_pages": sum(1 for c in chunks if not c["pages"]),
            "no_section": sum(1 for c in chunks if not c["section"]),
            "max_chars": max((c["chars"] for c in chunks), default=0),
            "oversize": sum(
                1
                for c in chunks
                if c["chars"] > (args.table_max_chars if c["type"] == "table" else args.max_chars)
            ),
            "warnings": warnings,
        })
        print(f"{p.name}: {len(chunks)} fragmentos {by_type}")

    report["totals"] = {
        "chunks": len(all_chunks),
        "by_type": {t: sum(1 for c in all_chunks if c["type"] == t)
                    for t in {c["type"] for c in all_chunks}},
        "chars": sum(c["chars"] for c in all_chunks),
    }
    rep_path = Path(args.report) if args.report else Path(args.output + ".report.json")
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte: {rep_path}")

    if args.report_only:
        return

    out = Path(args.output)
    if args.format == "jsonl":
        with out.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    elif args.format == "json":
        out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        rows = []
        for c in all_chunks:
            r = dict(c)
            r["pages"] = ",".join(str(p) for p in (c["pages"] or []))
            rows.append(r)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Fragmentos: {out} ({len(all_chunks)} registros)")


def make_chunks_for_file(
    path, max_chars, overlap, keep_markdown, include_preamble=False, table_max_chars=None
):
    """Igual que make_chunks pero conservando sección/páginas por fragmento."""
    table_max_chars = table_max_chars or max_chars
    doc_title, blocks, warnings = parse_blocks(path, include_preamble)
    source = Path(path).name
    chunks = []
    for b in blocks:
        text = clean_text(b["text"])
        if not text:
            continue
        if b["kind"] == "text" and not b["section"]:
            continue
        pages = b["pages"] or []
        page_label = ""
        if pages:
            page_label = f"pág. {pages[0]}" if len(pages) == 1 else f"págs. {pages[0]}–{pages[-1]}"
        context = b["section"]
        if page_label:
            context = f"{context} ({page_label})" if context else page_label

        if b["kind"] == "table":
            parts = split_table(b["rows"], table_max_chars)
            for idx, part in enumerate(parts, 1):
                t = clean_text("\n".join(part))
                if len(t) > table_max_chars:
                    warnings.append(
                        f"tabla de {len(t)} caracteres supera table-max-chars "
                        "(no se puede partir por filas)"
                    )
                chunks.append({
                    "type": "table", "part": idx, "parts": len(parts),
                    "text": t, "markdown": "\n".join(part) if keep_markdown else None,
                    "_section": b["section"], "_section_number": b["section_number"],
                    "_pages": pages, "_context": context,
                })
        else:
            parts_list = chunk_text(text, max_chars, overlap)
            for idx, part in enumerate(parts_list, 1):
                chunks.append({
                    "type": "text", "part": idx, "parts": len(parts_list),
                    "text": part, "markdown": None,
                    "_section": b["section"], "_section_number": b["section_number"],
                    "_pages": pages, "_context": context,
                })

    out = []
    seen = {}
    for c in chunks:
        context = c["_context"]
        embed = (context + "\n" + c["text"]) if context else c["text"]
        # ID estable: depende del documento, la sección, el tipo y el texto
        # completo (no de los primeros 80 caracteres), así que dos partes de una
        # tabla con el mismo encabezado no colisionan. Solo un duplicado exacto
        # recibe sufijo, en orden de aparición.
        key = f"{source}|{c['_section']}|{c['type']}|{c['text']}"
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        count = seen.get(h, 0)
        seen[h] = count + 1
        if count:
            h = f"{h}-{count + 1}"
        rec = {
            "id": h,
            "source": source,
            "doc_title": doc_title,
            "section": c["_section"],
            "section_number": c["_section_number"],
            "pages": c["_pages"],
            "page_start": c["_pages"][0] if c["_pages"] else None,
            "page_end": c["_pages"][-1] if c["_pages"] else None,
            "type": c["type"],
            "part": c["part"],
            "parts": c["parts"],
            "text": c["text"],
            "embed_text": embed,
            "chars": len(c["text"]),
        }
        if c["markdown"]:
            rec["markdown"] = c["markdown"]
        out.append(rec)
    return out, warnings


if __name__ == "__main__":
    main()
