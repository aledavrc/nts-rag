# Procedimiento: conversión de PDF escaneados a Markdown con OCR + agente

Este directorio contiene un procedimiento reutilizable para convertir documentos
escaneados (normas técnicas, manuales, informes) en Markdown fiel, con trazabilidad
de página y tablas aptas para bases de datos vectoriales (RAG).

Está pensado para documentos **sin capa de texto** (solo imágenes) y de extensión
media/grande (decenas o cientos de páginas), donde la precisión importa más que el
tiempo de cómputo.

---

## 1. Requisitos

| Herramienta | Uso | Instalación |
|---|---|---|
| Python 3.10+ | scripts | — |
| PyMuPDF (`pymupdf`) | renderizar páginas | `pip install pymupdf` |
| Pillow | hojas de contacto | `pip install pillow` |
| Tesseract 5 + `spa.traineddata` | OCR de apoyo | `pacman -S tesseract` (el traineddata se descarga solo) |
| Pandoc (opcional) | verificar renderizado | `pacman -S pandoc` |

El script descarga `spa.traineddata` desde `tessdata_best` si no existe en
`data/work/tessdata/`.

---

## 2. Estructura

```
pipelines/ocr/
├── README.md                 ← este documento
├── GUIA-DE-ESTILO.md         ← reglas de transcripción y RAG (para los agentes)
├── ocr_pipeline.py           ← prepare / sheets / assemble / validate
├── md_to_vector.py           ← convierte los MD en fragmentos JSONL/JSON/CSV
├── config.example.json       ← configuración de ejemplo
├── prompts/
│   ├── transcripcion.md      ← plantilla de prompt por rango de páginas
│   └── tablas.md             ← instrucciones adicionales para rangos con tablas
└── plantillas/
    ├── fragmento-ejemplo.md  ← ejemplo de salida de un fragmento
    └── rotations.example.json ← ejemplo de páginas a rotar
```

---

## 3. Flujo de trabajo

### Paso 0 — Analizar el documento

1. Abre el PDF y revisa: número de páginas, orientación, calidad del escaneo,
   presencia de tablas, formularios y páginas giradas.
2. Genera el inventario y las hojas de contacto:

   ```bash
   cp config.example.json config.json   # edita rutas y rangos
   python3 ocr_pipeline.py prepare -c config.json
   python3 ocr_pipeline.py sheets  -c config.json
   ```

   - `data/work/inventory.csv`: página, página impresa, caracteres OCR, primera línea.
   - `data/work/sheets/*.png`: miniaturas 5×4 para detectar páginas rotadas o en blanco.

3. Anota en `data/work/rotations.json` las páginas que deban rotarse (grados en sentido
   antihorario, p. ej. `{"p1": {"101": 270}}`; ver
   `plantillas/rotations.example.json`). Los agentes rotarán esas imágenes
   antes de leerlas.

### Paso 1 — Definir los rangos de transcripción

Divide el documento en rangos de 8–12 páginas **alineados con secciones**
(capítulos, anexos). Considera:

- Un rango no debe terminar en medio de una tabla grande si puedes evitarlo.
- Los rangos con muchas tablas se asignan a un agente con las instrucciones de
  `prompts/tablas.md`.
- Los anexos con formularios o tablas apaisadas van en rangos propios.
- Deja un archivo de notas por rango.

### Paso 2 — Transcribir con agentes

Para cada rango, lanza un agente con el prompt de `prompts/transcripcion.md`
(rellena los marcadores `{{...}}`). El agente:

1. Lee `GUIA-DE-ESTILO.md`.
2. Abre la imagen de cada página (150 dpi para vista general, 300 dpi para detalle).
3. Consulta el OCR de apoyo (`data/work/ocr/<tag>/NNN.txt`), sabiendo que contiene errores.
4. Escribe el fragmento en `data/work/frag/<tag>_<inicio>-<fin>.md`.
5. Escribe `data/work/frag/notas-<tag>_<inicio>-<fin>.md` con dudas e incidencias.

Consejos:

- Lanza 3–4 agentes en paralelo como máximo; cada uno con su propio rango.
- Pide respuestas finales **muy breves** (3–4 líneas) para no gastar contexto.
- Si un rango termina en una tabla que continúa, indícalo en el prompt del rango
  siguiente para que la complete en vez de duplicarla.
- Los agentes no deben tocar fragmentos de otros rangos.

### Paso 3 — Ensamblar

Configura en `config.json` la lista ordenada de fragmentos de cada salida:

```json
"outputs": [
  {
    "name": "../../corpus/markdown/documento.md",
    "pages": "1-97",
    "intro": "**NTS ...** ...",
    "fragments": ["../../data/work/frag/p1_001-010.md", "..."]
  }
]
```

Y ejecuta:

```bash
python3 ocr_pipeline.py assemble -c config.json
```

El ensamblado:

- concatena los fragmentos en orden,
- agrega la introducción,
- escapa las llamadas `(*)`, `(**)` para que no rompan el énfasis,
- colapsa líneas en blanco repetidas.

### Paso 4 — Validar

```bash
python3 ocr_pipeline.py validate -c config.json
```

Comprueba:

| Chequeo | Qué detecta |
|---|---|
| Cobertura de páginas | páginas sin marcador, duplicadas o fuera de rango |
| Tablas | filas con distinto número de celdas |
| Encabezados | saltos de nivel (`##` → `####`) |
| Etiquetas | `<sup>`/`<sub>` desbalanceados |
| Negritas | `**` sin cerrar |
| Formato continuo | ítems `TÉRMINO: ...` sin negrita (pérdida de formato entre páginas) |

Corrige en los **fragmentos** (no en el archivo final) y vuelve a ensamblar.

### Paso 5 — Revisión visual

Antes de entregar, revisa una muestra:

- [ ] La primera página (título, metadatos).
- [ ] Una página de texto denso.
- [ ] Una tabla key-value completa (6.1.1.x).
- [ ] Una tabla con subcolumnas (antineumocócica, Hepatitis B, Influenza).
- [ ] Una tabla partida entre páginas (verifica que esté unida y con su nota).
- [ ] Una tabla apaisada / formulario.
- [ ] El final del documento (bibliografía, notas al pie).

### Paso 6 — Ingesta en la base vectorial

Una vez auditados los Markdown, conviértelos a fragmentos con `md_to_vector.py`:

```bash
python3 pipelines/ocr/md_to_vector.py \
    corpus/markdown/documento-parte-1.md corpus/markdown/documento-parte-2.md \
    -o data/chunks/documento.chunks.jsonl \
    --max-chars 1800 \
    --table-max-chars 6000
```

- Acepta **cualquier cantidad de archivos**.
- Genera un JSONL (un objeto por línea) y un reporte `*.report.json`.
- `--max-chars`: tamaño de los fragmentos de texto (recomendado 1200–2000).
- `--table-max-chars`: las tablas se mantienen enteras hasta ese tamaño; si lo
  superan, se parten por filas repitiendo el encabezado (recomendado 5000–8000).
- `--overlap`: solapamiento entre fragmentos de texto (0 por defecto).
- `--format json|csv`: salida alternativa.
- `--report-only`: audita sin generar los fragmentos.

Cada registro contiene:

```json
{
  "id": "5ebed3bc312ba533",
  "source": "documento-parte-1.md",
  "doc_title": "NORMA TÉCNICA ...",
  "section": "VI. DISPOSICIONES ... > 6.1 ... > 6.1.1.10 VACUNA ANTINEUMOCÓCICA",
  "section_number": "6.1.1.10",
  "pages": [24, 25],
  "page_start": 24,
  "page_end": 25,
  "type": "table",
  "part": 1,
  "parts": 1,
  "text": "| Característica | Descripción | ...",
  "embed_text": "6.1.1.10 VACUNA ANTINEUMOCÓCICA (págs. 24–25)\n| Característica | ...",
  "chars": 3016
}
```

Recomendaciones de uso:

1. Embebe `embed_text` (incluye sección y páginas) y guarda `text` como contenido
   mostrable.
2. Usa `section`, `section_number` y `pages` como metadatos filtrables.
3. Revisa el reporte: fragmentos sin página/sección, tablas que superan el
   tamaño y advertencias de estructura.
4. No fragmentes de nuevo dentro de un registro: cada uno ya es una unidad
   autocontenida.

Recomendaciones generales (detalle en `GUIA-DE-ESTILO.md`, sección 8):

1. Fragmenta por sección: encabezado + contenido + tablas.
2. Mantén el encabezado de la sección en cada fragmento para conservar el sujeto
   («6.1.1.10 VACUNA ANTINEUMOCÓCICA»).
3. No fragmentes dentro de una tabla.
4. Conserva los marcadores `> _pág. N_` como metadato.
5. Verifica con consultas generales («esquema de vacunación de la antineumocócica»)
   que la respuesta recuperada incluya **todas** las variantes (menores/mayores
   de 5 años, pediátrica/adulto, etc.).

---

## 4. Configuración (`config.json`)

```json
{
  "workdir": "../../data/work",
  "tesseract": { "lang": "spa", "psm": 3, "tessdata_dir": "../../data/work/tessdata" },
  "documents": [
    {
      "pdf": "../../corpus/pdf/documento.pdf",
      "tag": "p1",
      "start_page": 1,
      "dpi": 150,
      "detail_dpi": 300
    }
  ],
  "outputs": [
    {
      "name": "../../corpus/markdown/documento.md",
      "pages": "1-97",
      "intro": "**NTS ...**\n\n*Documento convertido ...*",
      "fragments": ["../../data/work/frag/p1_001-010.md"]
    }
  ]
}
```

- `start_page`: número impreso de la primera página del PDF (para el inventario).
- `tag`: prefijo corto para las carpetas (`img150/p1`, `ocr/p1`, ...).
- Las rutas son relativas al directorio del `config.json`.

---

## 5. Problemas frecuentes

| Problema | Solución |
|---|---|
| Página girada 90°/180° | Anótala en `rotations.json`; el agente la rota con PIL (`im.rotate(270, expand=True)`) antes de leerla. |
| OCR con muchos errores (`N*`, `1mi`, `10 5`) | No lo uses como verdad: la imagen manda. Corrige contra la imagen. |
| Tabla que se corta entre páginas | Únela en la página donde empieza y anota «Tabla completada con la pág. N». |
| Celda con dos subcolumnas | Una sola fila con ambos contenidos etiquetados (regla RAG, sección 5.2 de la guía). |
| Pérdida de negrita al cambiar de página | Revisa el rango completo y normaliza el formato (guía, sección 6). |
| El agente devuelve demasiado texto | Pídele solo un resumen de 3–4 líneas; el contenido va al archivo. |
| Muchas páginas y poco contexto | Divide en más rangos y usa agentes en paralelo. |

---

## 6. Criterios de aceptación

- [ ] Todas las páginas tienen marcador (individual o de rango anotado).
- [ ] No hay tablas con columnas inconsistentes.
- [ ] No hay saltos de nivel de encabezado.
- [ ] Las tablas partidas están unidas y anotadas.
- [ ] Las subcolumnas están en una sola fila (regla RAG).
- [ ] El formato (negritas, listas) es continuo entre páginas.
- [ ] Las notas de conversión documentan dudas y páginas rotadas.
