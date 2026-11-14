#!/usr/bin/env python3
"""
ocr_pipeline.py — utilidades para convertir documentos escaneados a Markdown
con OCR de apoyo y transcripción asistida por un agente.

Subcomandos:
    prepare    Renderiza las páginas (150/300 dpi), ejecuta Tesseract y arma
               un inventario CSV de páginas.
    sheets     Genera hojas de contacto (miniaturas) para revisar orientación.
    assemble   Une los fragmentos Markdown producidos por los agentes.
    validate   Verifica cobertura de páginas, tablas, encabezados y formato.

El detalle del procedimiento está en README.md y GUIA-DE-ESTILO.md.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg, path.parent


def rp(base, value):
    """Resuelve una ruta relativa al directorio del archivo de configuración."""
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kw)


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #


def ensure_lang(tessdata_dir, lang):
    """Descarga el traineddata de tessdata_best si no existe localmente."""
    tessdata_dir = ensure_dir(tessdata_dir)
    target = Path(tessdata_dir) / f"{lang}.traineddata"
    if target.exists():
        return target
    url = f"https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata"
    print(f"  descargando {lang}.traineddata ...", flush=True)
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:  # pragma: no cover
        print(f"  ERROR: no se pudo descargar {url}: {exc}")
        return None
    return target


def ocr_page(img, out_base, tessdata_dir, lang, psm):
    cmd = ["tesseract", str(img), str(out_base)]
    if tessdata_dir:
        cmd += ["--tessdata-dir", str(tessdata_dir)]
    cmd += ["-l", lang, "--psm", str(psm)]
    proc = run(cmd)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        print(f"  aviso: tesseract falló en {img.name}: {err[-1] if err else ''}")


def first_line(text, skip=()):
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(x in s for x in skip):
            continue
        return s[:120]
    return ""


def printed_page(text):
    for line in reversed(text.splitlines()):
        s = line.strip()
        if re.fullmatch(r"\d{1,4}", s):
            return s
    return ""


def cmd_prepare(args):
    import pymupdf  # import diferido: solo se necesita aquí

    cfg, base = load_config(args.config)
    work = rp(base, cfg.get("workdir", "work"))
    tcfg = cfg.get("tesseract", {})
    lang = tcfg.get("lang", "spa")
    psm = tcfg.get("psm", 3)
    tessdata = rp(base, tcfg["tessdata_dir"]) if tcfg.get("tessdata_dir") else None
    if tessdata and shutil.which("tesseract"):
        ensure_lang(tessdata, lang)

    have_tess = shutil.which("tesseract") is not None
    if not have_tess:
        print("Aviso: tesseract no está instalado; se omite el OCR de apoyo.")

    rows = []
    for doc in cfg["documents"]:
        pdf = rp(base, doc["pdf"])
        tag = doc["tag"]
        dpi = doc.get("dpi", 150)
        detail = doc.get("detail_dpi", 300)
        start = doc.get("start_page", 1)
        d = pymupdf.open(pdf)
        img_dir = ensure_dir(work / "img150" / tag)
        det_dir = ensure_dir(work / "img300" / tag)
        ocr_dir = ensure_dir(work / "ocr" / tag)
        print(f"[{tag}] {pdf.name}: {len(d)} páginas")
        for i, page in enumerate(d):
            n = i + 1
            printed = start + i
            p150 = img_dir / f"{n:03d}.png"
            p300 = det_dir / f"{n:03d}.png"
            if not p150.exists():
                page.get_pixmap(dpi=dpi).save(p150)
            if not p300.exists():
                page.get_pixmap(dpi=detail).save(p300)
            txt = ocr_dir / f"{n:03d}.txt"
            if have_tess and not txt.exists():
                ocr_page(p150, ocr_dir / f"{n:03d}", tessdata, lang, psm)
            content = txt.read_text(encoding="utf-8", errors="replace") if txt.exists() else ""
            rows.append({
                "tag": tag,
                "page": n,
                "printed_page": printed_page(content) or printed,
                "chars": len(content.strip()),
                "detail_png": str(p300),
                "ocr_txt": str(txt) if txt.exists() else "",
                "first_line": first_line(content, skip=("NTS N", "NORMA TÉCNICA")),
            })
            if n % 10 == 0 or n == len(d):
                print(f"  {n}/{len(d)}", flush=True)

    inv = work / "inventory.csv"
    with inv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["tag"])
        w.writeheader()
        w.writerows(rows)
    print(f"Inventario: {inv}")


# --------------------------------------------------------------------------- #
# sheets
# --------------------------------------------------------------------------- #


def cmd_sheets(args):
    from PIL import Image
    import pymupdf

    cfg, base = load_config(args.config)
    work = rp(base, cfg.get("workdir", "work"))
    out = ensure_dir(work / "sheets")
    cols, rows_n = 5, 4
    tw, th = 240, 340
    for doc in cfg["documents"]:
        pdf = rp(base, doc["pdf"])
        tag = doc["tag"]
        d = pymupdf.open(pdf)
        for start in range(0, len(d), cols * rows_n):
            canvas = Image.new("RGB", (cols * tw, rows_n * th), "white")
            for k in range(cols * rows_n):
                i = start + k
                if i >= len(d):
                    break
                pix = d[i].get_pixmap(dpi=50)
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                im.thumbnail((tw - 10, th - 20))
                canvas.paste(im, ((k % cols) * tw + 5, (k // cols) * th + 5))
            path = out / f"{tag}_sheet_{start // (cols * rows_n) + 1:02d}.png"
            canvas.save(path)
            print(path)


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #


def escape_footnotes(text):
    """Evita que (*), (**), ... se interpreten como énfasis en prosa."""
    return re.sub(r"\((\*{1,4})\)", lambda m: "(" + "\\*" * len(m.group(1)) + ")", text)


def cmd_assemble(args):
    cfg, base = load_config(args.config)
    for out in cfg["outputs"]:
        chunks = []
        missing = []
        for f in out["fragments"]:
            p = rp(base, f)
            if p.exists():
                chunks.append(p.read_text(encoding="utf-8").strip())
            else:
                missing.append(str(p))
        if missing:
            print(f"Aviso: faltan {len(missing)} fragmentos para {out['name']}:")
            for m in missing:
                print("   ", m)
            if not chunks:
                continue
        intro = out.get("intro", "").strip()
        body = "\n\n".join(chunks)
        body = escape_footnotes(body)
        body = re.sub(r"\n{3,}", "\n\n", body)
        text = (intro + "\n\n" if intro else "") + body + "\n"
        dest = rp(base, out["name"])
        ensure_dir(dest.parent)
        dest.write_text(text, encoding="utf-8")
        print(f"{out['name']}: {len(text):,} caracteres, {len(chunks)} fragmentos")


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def page_set(pattern):
    pages = set()
    for m in re.finditer(r"([0-9]+)(?:[–-]([0-9]+))?", pattern):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        pages.update(range(a, b + 1))
    return pages


def table_errors(lines):
    errs = []
    def cells(line):
        parts = re.split(r"(?<!\\)\|", line.strip())
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return len(parts)
    i = 0
    while i < len(lines):
        if (lines[i].startswith("|") and i + 1 < len(lines)
                and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1])):
            n = cells(lines[i])
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                if cells(lines[j]) != n:
                    errs.append((j + 1, f"fila con {cells(lines[j])} celdas; se esperaban {n}"))
                j += 1
            i = j
        else:
            i += 1
    return errs


def cmd_validate(args):
    cfg, base = load_config(args.config)
    total_errors = 0
    for out in cfg["outputs"]:
        path = rp(base, out["name"])
        if not path.exists():
            print(f"ERROR: no existe {path}")
            total_errors += 1
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        expected = page_set(out.get("pages", ""))
        found = set()
        dups = []
        for m in re.finditer(r"^> _pág[s]?\. ([0-9–\-]+)", text, re.M):
            got = page_set(m.group(1))
            for p in got:
                if p in found:
                    dups.append(p)
                found.add(p)
        missing = sorted(expected - found)
        extra = sorted(found - expected) if expected else []
        terr = table_errors(lines)
        jumps = []
        prev = 0
        for i, l in enumerate(lines, 1):
            m = re.match(r"^(#{1,6}) ", l)
            if m:
                lv = len(m.group(1))
                if lv > prev + 1:
                    jumps.append((i, f"{prev}->{lv}"))
                prev = lv
        tags = []
        for i, l in enumerate(lines, 1):
            if l.count("<sup>") != l.count("</sup>") or l.count("<sub>") != l.count("</sub>"):
                tags.append(i)
        bold = []
        for i, l in enumerate(lines, 1):
            # tras escape_footnotes, (**) ya no existe; cuenta ** de énfasis
            if l.count("**") % 2 == 1:
                bold.append(i)
        unbolted = [i for i, l in enumerate(lines, 1)
                    if re.match(r"^- [A-ZÁÉÍÓÚÑ0-9][A-ZÁÉÍÓÚÑ0-9. ]{0,30}: ", l)]

        errs = (len(missing) + len(dups) + len(extra) + len(terr)
                + len(jumps) + len(tags) + len(bold) + len(unbolted))
        total_errors += errs
        status = "OK" if errs == 0 else f"{errs} problemas"
        print(f"{out['name']}: {status}")
        if missing:
            print("  páginas sin marcador:", missing[:20], "..." if len(missing) > 20 else "")
        if dups:
            print("  marcadores duplicados:", dups[:20])
        if extra:
            print("  páginas fuera de rango:", extra[:20])
        for line, msg in terr[:10]:
            print(f"  tabla línea {line}: {msg}")
        if jumps:
            print("  saltos de nivel de encabezado:", jumps[:10])
        if tags:
            print("  etiquetas <sup>/<sub> desbalanceadas en líneas:", tags[:10])
        if bold:
            print("  posibles negritas sin cerrar en líneas:", bold[:10])
        if unbolted:
            print("  ítems con 'TÉRMINO:' sin negrita en líneas:", unbolted[:10])
    if total_errors:
        sys.exit(1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="renderizar páginas + OCR + inventario")
    p.add_argument("-c", "--config", default="config.json")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("sheets", help="generar hojas de contacto para revisar orientación")
    p.add_argument("-c", "--config", default="config.json")
    p.set_defaults(func=cmd_sheets)

    p = sub.add_parser("assemble", help="unir fragmentos en los Markdown finales")
    p.add_argument("-c", "--config", default="config.json")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("validate", help="validar los Markdown finales")
    p.add_argument("-c", "--config", default="config.json")
    p.set_defaults(func=cmd_validate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
