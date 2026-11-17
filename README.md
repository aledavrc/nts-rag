# Consulta RAG de la NTS N° 246-MINSA/DGIESP-2026

Sistema de consulta en lenguaje natural sobre la *Norma Técnica de Salud que
establece el Esquema Nacional de Inmunizaciones*. Responde únicamente con el
contenido de los Markdown auditados e indexados de este repositorio y devuelve la
página y la sección que sustentan cada respuesta.

Este proyecto **no vuelve a hacer OCR ni validaciones de ingesta**: los Markdown
de `corpus/markdown/` (NTS 246-2026 en dos partes y NTS 196-2022) ya son la fuente
de verdad, y `pipelines/ocr/md_to_vector.py` los convierte en chunks con página,
sección y tipo. La consulta consume esos chunks directamente.

## Cómo funciona

```text
corpus/markdown/*.md
    → pipelines/ocr/md_to_vector.py (chunks con sección, páginas y tipo)
    → data/chunks/nts-246.chunks.jsonl
    → esquema v2: documento y vigencia, campos canónicos de las fichas (6.1.1.x),
      filas de las tablas de dominio, vacuna y edad normalizadas
    → embeddings FastEmbed + ChromaDB (data/index/)
    → plan de la consulta: intención, vacunas, campos, edad, versión, subconsultas
      (LLM con validador si está configurado; determinista si no)
    → evidencia: búsqueda vectorial + BM25 + RRF reordenada por el plan, más slots
      que garantizan cobertura (campo por vacuna, calendario por edad, Anexo 2…)
    → veredicto por código: respondida · parcial · abstenida · aclaración
    → comparaciones, listas, superlativos y diffs: tabla de la matriz
      vacuna × campo × versión con cita por celda (0 llamadas)
    → resto: respuesta LLM con citas [cN] y verificación, o extractivo con citas
    → API FastAPI / CLI
```

Estado actual (Fase 3): lookups 20/20, preguntas por edad y rescate con todas sus
fuentes, comparaciones y listas completas desde la matriz de fichas, diff entre la
NTS 196-2022 y la NTS 246-2026, y el estado correcto en las 52 preguntas del eval.
La precisión del diff es 0,92 en un set etiquetado a ciegas (objetivo 0,90). El
análisis y las siguientes fases están en
[`docs/CONTEXTO_MEJORAS.md`](docs/CONTEXTO_MEJORAS.md).

La recuperación combina:

- búsqueda vectorial sobre `embed_text` (sección + páginas + contenido);
- **chunks de campo**: cada ficha se parte en un fragmento por campo, con una de
  13 claves canónicas (`COMPOSICION`, `DOSIS`, `CONSERVACION`, …) y la etiqueta
  original, de modo que una pregunta sobre un dato concreto acierta el campo exacto;
- **chunks de fila** de las tablas de dominio (calendarios 6.2.x, casos especiales
  6.3.x y los anexos de administración simultánea, rescate, frascos abiertos,
  vías, ESAVI severos y errores operativos), cada uno con su encabezado;
- BM25 sobre sección y texto, con recorte ligero de plurales, stopwords del
  dominio (`niño`, `vacuna`…) y tokens de edad (`5 años` casa con `5 a 11 años`);
- fusión RRF, refuerzo de sección y de campo nombrado, y numeral citado;
- expansión mínima de términos coloquiales (`gestante` → `embarazo`,
  `gestación`, `prenatal`; `recuperar`/`completar` → `rescate`).

Por defecto solo se consulta la norma **vigente**. La NTS 196-2022 entra cuando la
pregunta la nombra (`2022`, `196`, "norma anterior") o compara normas (`comparar`,
`cambió`, `vacunas nuevas`…), y sus citas llevan "(derogada)". Los filtros de la
API acotan por sección, página, tipo de fragmento, documento o vigencia.

## Corpus

| Dato | NTS 246-2026 (vigente) | NTS 196-2022 (derogada) |
|---|---|---|
| `documento_id` | `nts-246-2026` | `nts-196-2022` |
| Páginas | 140 (97 + 43) con numeración impresa continua | 103 |
| Fichas de vacuna | 24 | 18 |
| Chunks | 1.153 | 911 |

Total: **2.064 chunks** = 1.206 base (texto y tabla) + 543 campos + 315 filas.
Entrada: `data/chunks/nts-246.chunks.jsonl` (contiene las dos normas pese al nombre).

Cada chunk lleva `documento_id`, `documento`, `version`, `vigencia`, `pagina`,
`pagina_fin`, `seccion`, `seccion_numero`, `tipo` (`texto`, `tabla`, `campo`,
`fila`) y, cuando aplica, `campo`, `campo_original`, `vacuna_slug`, `vacunas`,
`edad_min_meses` y `edad_max_meses`. El registro de documentos está en
`src/nts_rag/config.py` y los catálogos (campos, fichas, vacunas) en
`src/nts_rag/catalogo.py`. El índice embebe `embed_text` y guarda `text` como
contenido mostrable.

## Instalación

Requisitos: Python 3.12, [uv](https://docs.astral.sh/uv/) y, opcionalmente,
Docker.

```bash
cp .env.example .env
make install       # crea .venv e instala dependencias
make doctor        # comprueba qué falta
make serve         # API en http://127.0.0.1:8000
```

Sin `make`:

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
.venv/bin/python -m nts_rag.cli servir --reload
```

El índice (`data/index/`) está versionado: no hace falta construirlo para
consultar. El modelo de embeddings se descarga la primera vez que se consulta.

### Regenerar los chunks

Solo es necesario si cambian los Markdown:

```bash
make chunks        # ejecuta pipelines/ocr/md_to_vector.py
make index         # reconstruye data/index/
```

Al cambiar chunks o índice, commitea `data/chunks/`, `data/index/` y
`data/manifest.json`: los contribuidores consultan sin reindexar.

`make index` pasa `corpus_lint` (todas las fichas con sus 13 campos, ninguna
etiqueta fuera del catálogo, todas las páginas cubiertas) y escribe en el
manifiesto `schema_version` y el SHA-256 de cada Markdown y del JSONL. `make
doctor` compara esos hashes con los archivos actuales y avisa si el índice quedó
desfasado. Los IDs son estables: dependen del documento, la sección, el tipo y el
texto completo, así que dos reindexados dan los mismos IDs.

## Matriz de fichas y diff

Las 42 fichas (24 de 2026 y 18 de 2022) forman una matriz vacuna × campo × versión:
los 13 campos canónicos más `RESCATE` (la fila de la vacuna en el Anexo 2). Una
celda ausente es *no aplica* (jeringas en vacunas orales) o *no consta*; nunca se
rellena. Sobre ella se responden sin LLM, con una cita por celda:

| Pregunta | Respuesta |
|---|---|
| "Compara la conservación de la SPR y la AMA" | tabla vacunas × campos |
| "¿Qué vacunas son de virus vivos atenuados?" | lista completa (7) |
| "¿Qué vacunas están contraindicadas en el embarazo?" | lista completa (3) |
| "¿Qué vacunas se aplican en el muslo y cuáles en el deltoides?" | las 24 fichas por sitio |
| "¿Qué vacuna previene contra más cepas?" | valencias declaradas, sin ganador |
| "¿Qué vacunas nuevas incorpora la NTS 246?" | 6 fichas nuevas |
| "¿Qué cambió en el rescate de la BCG entre 2022 y 2026?" | diff de la celda con ambas citas |

El diff clasifica cada celda en `cambio_real`, `redaccional`, `movido`, `nuevo`,
`eliminado`, `sin_cambio`, `no_consta` o `no_aplica`, comparando cada campo según lo
que significa: el tipo por familia (viva atenuada, inactivada, pasiva), conservación,
dosis y jeringas por sus cifras, las listas elemento a elemento; reconoce un dato
movido a una remisión y los límites de edad equivalentes ("menores de 5 años" =
"hasta 4 años 11 meses 29 días"). Tiende a sobreavisar: puede marcar como cambio
una redacción nueva, pero en los tres sets etiquetados solo se le escapa un cambio
real.

```bash
python -m nts_rag.cli ficha bcg                      # 14 campos con cita
python -m nts_rag.cli ficha bcg --norma 2022
python -m nts_rag.cli comparar spr ama --campo CONSERVACION CONTRAINDICACIONES
python -m nts_rag.cli diff                           # fichas nuevas y eliminadas
python -m nts_rag.cli diff bcg --solo-cambios        # BCG campo a campo
python -m nts_rag.cli diff --evaluar eval/diff.jsonl eval/diff_control.jsonl eval/diff_control2.jsonl
```

## API

| Método | Ruta | Función |
|---|---|---|
| GET | `/salud` | Estado, chunks indexados y si hay LLM configurado |
| GET | `/health` | Alias de `/salud` (para healthchecks) |
| POST | `/consulta` | Pregunta → respuesta + fuentes |
| POST | `/buscar` | Solo recuperación, sin redacción (depuración) |

```bash
curl -X POST http://localhost:8000/consulta \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Cuál es la composición de la vacuna antineumocócica?"}'
```

Respuesta abreviada:

```json
{
  "respuesta": "**Según la normativa cargada:**\n\n**COMPOSICIÓN**\n\nVacuna PCV 20…",
  "fuentes": [
    {
      "chunk_id": "a07035c90dba89f9-composicion",
      "documento": "NTS N° 246-MINSA/DGIESP-2026",
      "documento_id": "nts-246-2026",
      "version": 2026,
      "vigencia": "vigente",
      "pagina": 24,
      "pagina_fin": 25,
      "seccion": "6.1.1.10. VACUNA ANTINEUMOCÓCICA (*)",
      "seccion_numero": "6.1.1.10",
      "tipo": "campo",
      "campo": "COMPOSICION",
      "campo_original": "COMPOSICIÓN",
      "vacuna_slug": "neumococo",
      "fragmento": "**COMPOSICIÓN**\n\nVacuna PCV 20…",
      "score": 1.0
    }
  ],
  "modo": "extractivo",
  "estado": "respondida",
  "motivo": null,
  "reformulacion": null,
  "supuestos": ["Se consulta la norma vigente (NTS 246-2026)."],
  "faltan_fuentes": [],
  "conflictos": [],
  "llamadas_llm": 0,
  "tokens_entrada": 0,
  "tokens_salida": 0,
  "formato": "markdown",
  "tiempo_ms": 45
}
```

Filtros opcionales en `/consulta` y `/buscar`:

```json
{
  "pregunta": "¿Qué dosis se aplica?",
  "k": 6,
  "filtros": {"seccion": "6.1.1.17", "pagina": 34, "tipo": "tabla"}
}
```

`pagina` acepta cualquier página cubierta por el chunk (`pagina`–`pagina_fin`).
`tipo` admite `texto`, `tabla`, `campo` o `fila`. `documento` (`nts-246-2026`,
`nts-196-2022`) fija una norma; `vigencia` (`vigente`, `derogada`, `todas`) manda
sobre las señales de versión de la pregunta. Sin ninguno de los dos solo se busca
en la norma vigente, salvo que la pregunta nombre la 2022 o compare normas.

## CLI

```bash
python -m nts_rag.cli doctor
python -m nts_rag.cli indexar
python -m nts_rag.cli consultar "¿Qué vacunas se aplican a gestantes?"
python -m nts_rag.cli consultar --render "¿Qué consecuencias tiene una inyección no estéril?"
python -m nts_rag.cli buscar "cadena de frío"
python -m nts_rag.cli inspeccionar "dosis de la vacuna antineumocócica" -k 3
python -m nts_rag.cli evaluar --archivo eval/preguntas.jsonl -k 5
python -m nts_rag.cli ficha spr
python -m nts_rag.cli comparar pentavalente hexavalente
python -m nts_rag.cli diff apo --solo-cambios
python -m nts_rag.cli servir --reload
```

## LLM opcional

Sin configuración, la API responde en *modo extractivo*: devuelve el fragmento
más relevante con su cita. Para respuestas redactadas se configura cualquier
servidor compatible con OpenAI (Ollama, OpenAI, etc.):

```bash
NTS_RAG_LLM_BASE_URL=http://localhost:11434/v1 \
NTS_RAG_LLM_MODEL=qwen2.5:7b \
make serve
```

Con LLM configurado, la consulta hace como mucho tres llamadas:

1. **Plan**: el LLM interpreta la pregunta (intención, vacunas, campos, edad,
   versión, subconsultas, premisas dudosas) en un JSON que el código valida contra
   los catálogos; si no valida tras un reintento, se usa el plan determinista. El
   plan **sustituye al re-rankeo**: trae filtros exactos en vez de reordenar
   candidatos. (El re-rankeo solo se usa si el plan del LLM falla y
   `NTS_RAG_RERANK=true`.)
2. **Respuesta**: el LLM redacta usando únicamente la evidencia (fuentes más los
   hermanos de su sección: ficha completa, filas de la misma tabla) y cita con
   etiquetas `[c1]`, `[c2]`… que el código sustituye por la cita exacta desde los
   metadatos. Una cita escrita de memoria se borra; una respuesta sin ninguna cita
   válida se descarta. El estado y el motivo (conflicto, premisa…) los decide el
   código antes de redactar.
3. **Verificación de fidelidad**: un pase del LLM comprueba que todas las
   afirmaciones se apoyan en las fuentes; si no, se devuelve el fragmento
   verbatim (extractivo). Se desactiva con `NTS_RAG_VERIFY=false`.

Sin LLM, el plan es determinista, la consulta no hace ninguna llamada y la respuesta
son los fragmentos verbatim con su cita (uno en los lookups, uno por fuente
principal en las preguntas por edad, conjuntos o comparaciones), precedidos de las
notas de respuesta parcial, discrepancias o premisas cuando las hay. `/consulta`
devuelve el coste en `llamadas_llm`, `tokens_entrada` y `tokens_salida`.

## Docker

La imagen solo contiene la API, ChromaDB y el modelo de embeddings. No incluye
herramientas de OCR ni Docling.

```bash
docker compose up --build -d
```

`docker-compose.yml` monta `./data/index` y `./data/chunks` (este último en solo lectura).
El índice versionado se monta tal cual; reconstruye con `make index` solo si cambias los
chunks. Si el índice no existe o no coincide con los chunks (por ejemplo, tras
`make chunks` sin `make index`), `/salud` (o `/health`) responde `degradado`; sin índice,
las consultas devuelven 503.

## Estructura

```text
nts-rag/
├── corpus/
│   ├── markdown/                 # Markdown auditado (fuente de verdad)
│   └── pdf/                      # PDF escaneados originales
├── data/
│   ├── chunks/                   # chunks para el índice (versionados)
│   ├── index/                    # ChromaDB del corpus (versionada)
│   ├── work/                     # intermedios de OCR (ignorados)
│   └── manifest.json             # metadatos del índice (versionado)
├── docs/
│   ├── PROCESO.md                # decisiones y verificaciones del OCR
│   ├── CONTEXTO_MEJORAS.md       # estado, análisis y plan de mejoras
│   └── COMANDOS.org              # recetario de pruebas de API/CLI
├── eval/                         # sets de evaluación (lookup, agregación, borde, comparativas)
├── models/                       # caché de FastEmbed (ignorada)
├── pipelines/ocr/                # procedimiento reutilizable PDF → Markdown
│   ├── README.md
│   ├── GUIA-DE-ESTILO.md
│   ├── ocr_pipeline.py
│   ├── md_to_vector.py
│   ├── config.example.json
│   ├── prompts/
│   └── plantillas/
├── src/nts_rag/
│   ├── config.py                 # NTS_RAG_* y registro de documentos
│   ├── catalogo.py               # campos canónicos, fichas y vacunas
│   ├── evaluation.py             # métricas de `evaluar`
│   ├── index.py                  # JSONL → ChromaDB, lint y manifiesto
│   ├── schemas.py                # contratos de la API
│   ├── api.py                    # FastAPI
│   ├── cli.py                    # comandos
│   ├── text.py                   # normalización y tokens
│   └── rag/
│       ├── embeddings.py         # FastEmbed / ONNX
│       ├── interpret.py          # plan de consulta (LLM + validador / determinista)
│       ├── retrieve.py           # vectorial + BM25 + RRF
│       ├── evidence.py           # slots, diversidad y expansión de sección
│       ├── abstain.py            # veredicto: estado, motivo, conflictos, premisas
│       ├── fichas.py             # matriz vacuna × campo × versión y diff
│       ├── structured.py         # respuestas por plantilla desde la matriz
│       ├── citations.py          # citas desde metadatos
│       ├── answer.py             # pipeline, citas [cN], LLM opcional / extractivo
│       └── formatting.py         # Markdown para respuestas
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## Variables de entorno

| Variable | Por defecto | Efecto |
|---|---|---|
| `NTS_RAG_CHUNKS_PATH` | `data/chunks/nts-246.chunks.jsonl` | Entrada de chunks |
| `NTS_RAG_DATA_DIR` | `data` | Índice y manifiesto |
| `NTS_RAG_MODELS_DIR` | `models` | Caché del modelo FastEmbed |
| `NTS_RAG_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings |
| `NTS_RAG_CHROMA_COLLECTION` | `nts_246` | Colección ChromaDB |
| `NTS_RAG_TOP_K` | `6` | Fuentes devueltas |
| `NTS_RAG_VECTOR_K` | `24` | Candidatos vectoriales |
| `NTS_RAG_BM25_K` | `24` | Candidatos léxicos |
| `NTS_RAG_LLM_BASE_URL` | vacío | Activa el re-rankeo y el modo redactado |
| `NTS_RAG_LLM_MODEL` | `qwen2.5:7b` | Modelo del LLM |
| `NTS_RAG_LLM_API_KEY` | `ollama` | Clave del LLM |
| `NTS_RAG_VERIFY` | `true` | Verificación de fidelidad de la respuesta |

## Evaluación

```bash
python -m nts_rag.cli evaluar -k 5
python -m nts_rag.cli evaluar --archivo eval/*.jsonl -k 5 10 20 --por-tipo
```

Por cada corte `k` mide `hit`, `recall` (fracción de fuentes esperadas
recuperadas), preguntas `completas` y `precision`; además, si el sistema responde
o se abstiene cuando debe. `--sin-respuesta` mide solo la recuperación y `--json`
devuelve el resumen en JSON.

| Set | Contenido |
|---|---|
| `eval/preguntas.jsonl` | 20 lookups + 5 fuera de alcance (regresión) |
| `eval/agregacion.jsonl` | 7 de agregación (edad, rescate, gestante) + 3 negativas |
| `eval/ambiguedades.jsonl` | 4 preguntas de borde con estado/motivo esperados + 3 negativas |
| `eval/comparativas.jsonl` | 3 comparativas 2022/2026 |
| `eval/matriz.jsonl` | 3 comparaciones nombradas, 3 listas globales y 1 contradicción |
| `eval/diff*.jsonl` | celdas 2022/2026 etiquetadas a mano: diseño y dos controles a ciegas |

`evaluar` mide la recuperación del sistema completo (plan + evidencia, la de
`/consulta`); `--solo-busqueda` mide el buscador híbrido solo. También informa la
`contaminación`: fuentes de la norma derogada sin pedirla o fichas de vacunas que la
pregunta no nombra.

Resultados actuales (modo extractivo, sin LLM), en recall@10:

| Tipo | Fase 0 | Fase 1 | Fase 2 | Fase 3 |
|---|---|---|---|---|
| lookups | 1.00 (20/20) | 1.00 (20/20) | 1.00 (20/20) | 1.00 (20/20) |
| agregación | 0.39 (2/7 completas) | 0.89 (6/7) | 1.00 (7/7) | 1.00 (7/7) |
| borde | 0.31 | 0.31 | 0.82 | 0.88 |
| comparativas | 0.33 | 0.50 | 0.67 | **1.00** |
| estado exacto | 38/45 | 39/45 | 45/45 | **52/52** (con `matriz.jsonl`) |

En la Fase 3 la abstención es correcta en 9/9 negativas sin abstenciones indebidas,
el motivo coincide en 8/8, la contaminación es 0, las comparaciones nombradas y las
listas globales están completas (6/6) y la contradicción conocida sale con ambas
citas. La precisión del diff es 0,92 en el set de control ciego (22/24). El detalle
está en
[`docs/CONTEXTO_MEJORAS.md` §9.3](docs/CONTEXTO_MEJORAS.md).
`tests/test_smoke_index.py` comprueba en CI la estructura del índice versionado
(totales, fichas, campos, lint, IDs y frescura del manifiesto).

## Limitaciones

- El sistema solo conoce las normas indexadas: NTS 246-2026 (vigente) y NTS
  196-2022 (derogada, solo bajo petición o comparación).
- En preguntas con vocabulario muy genérico la búsqueda puede no acertar el
  fragmento exacto.
- El diff entre normas acierta 0,92 y, cuando falla, casi siempre sobreavisa
  (marca «cambió» en una redacción nueva). Revisa las celdas marcadas «cambió»
  antes de citarlas como cambio normativo.
- Las respuestas por plantilla (tablas de la matriz) son exactas y citadas, pero no
  redactadas.
- El veredicto usa reglas deterministas (catálogos, edades, alérgenos): una
  pregunta fuera de alcance que comparta vocabulario con la norma puede no
  abstenerse, y una discrepancia que no sea de edad no se detecta todavía.
- El modo extractivo no redacta una síntesis; devuelve los fragmentos con su
  cita. La ruta con LLM está probada con un LLM simulado, no medida con un modelo
  real.
- Una respuesta bien formada no garantiza la exactitud del dato clínico:
  contrasta siempre con el Markdown o el PDF original.

## Mejoras esperadas

El estado actual, el análisis y el plan de las siguientes fases (plan de consulta
con LLM, abstención por cobertura, matriz de fichas, diff entre normas y coste)
están en [`docs/CONTEXTO_MEJORAS.md`](docs/CONTEXTO_MEJORAS.md). Léelo antes de
cambiar la recuperación o el esquema de chunks.
