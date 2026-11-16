# Consulta RAG de la NTS N° 246-MINSA/DGIESP-2026

Sistema de consulta en lenguaje natural sobre la *Norma Técnica de Salud que
establece el Esquema Nacional de Inmunizaciones*. Responde únicamente con el
contenido de los Markdown auditados e indexados de este repositorio y devuelve la
página y la sección que sustentan cada respuesta.

Este proyecto **no vuelve a hacer OCR ni validaciones de ingesta**: los archivos
`corpus/markdown/8265031-...-parte-1.md` y `...-parte-2.md` ya son la fuente de
verdad, y `pipelines/ocr/md_to_vector.py` los convierte en chunks con página,
sección y tipo. La consulta consume esos chunks directamente.

## Cómo funciona

```text
corpus/markdown/*.md
    → pipelines/ocr/md_to_vector.py (chunks con sección, páginas y tipo)
    → data/chunks/nts-246.chunks.jsonl
    → fragmentación por campo de las fichas (6.1.1.x)
    → embeddings FastEmbed + ChromaDB (data/index/)
    → búsqueda vectorial + BM25 + RRF
    → re-rankeo con LLM (si está configurado)
    → respuesta LLM con verificación de fidelidad, o extractivo con citas
    → API FastAPI / CLI
```

Estado actual: la recuperación acierta en preguntas de un campo o una ficha, pero
todavía falla en preguntas por edad, comparativas y dosis de rescate. El análisis y
el plan de la siguiente iteración están en
[`docs/CONTEXTO_MEJORAS.md`](docs/CONTEXTO_MEJORAS.md).

La recuperación combina:

- búsqueda vectorial sobre `embed_text` (sección + páginas + contenido);
- **chunks de campo**: cada ficha key-value se parte en un fragmento por campo
  (`COMPOSICIÓN`, `DOSIS (CANTIDAD)`, `CONSERVACIÓN DE LA VACUNA`, …), de modo
  que una pregunta sobre un dato concreto acierta el campo exacto;
- BM25 sobre sección y texto, con recorte ligero de plurales y stopwords del
  dominio (`niño`, `vacuna`, `año`…) para dar peso a los términos raros;
- fusión RRF, refuerzo de sección y de campo nombrado, y numeral citado;
- expansión mínima de términos coloquiales (`gestante` → `embarazo`,
  `gestación`, `prenatal`; `tarda` → `tiempo`, `aparición`).

Los filtros de la API permiten acotar por sección, página o tipo de fragmento.

## Corpus

| Dato | Valor |
|---|---|
| Documento | NTS N° 246-MINSA/DGIESP-2026 |
| Páginas | 140 (97 + 43) con numeración impresa continua |
| Chunks | 1004 (648 base + 356 campos de ficha) |
| Entrada | `data/chunks/nts-246.chunks.jsonl` |

La NTS 196-2022 está en `corpus/` (Markdown y PDF) pero **aún no está indexada**;
su ingesta forma parte de la siguiente iteración.

Cada chunk ya contiene `section`, `section_number`, `pages`, `type`, `text` y
`embed_text`. El índice embebe `embed_text` (sección + páginas + contenido) y
guarda `text` como contenido mostrable.

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
      "chunk_id": "f4a9864563279413-campo-3",
      "documento": "NTS N° 246-MINSA/DGIESP-2026",
      "pagina": 24,
      "pagina_fin": 25,
      "seccion": "6.1.1.10. VACUNA ANTINEUMOCÓCICA (*)",
      "seccion_numero": "6.1.1.10",
      "tipo": "campo",
      "campo": "COMPOSICIÓN",
      "fragmento": "**COMPOSICIÓN**\n\nVacuna PCV 20…",
      "score": 1.0
    }
  ],
  "modo": "extractivo",
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

## CLI

```bash
python -m nts_rag.cli doctor
python -m nts_rag.cli indexar
python -m nts_rag.cli consultar "¿Qué vacunas se aplican a gestantes?"
python -m nts_rag.cli consultar --render "¿Qué consecuencias tiene una inyección no estéril?"
python -m nts_rag.cli buscar "cadena de frío"
python -m nts_rag.cli inspeccionar "dosis de la vacuna antineumocócica" -k 3
python -m nts_rag.cli evaluar --archivo eval/preguntas.jsonl -k 5
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

Con LLM configurado, la consulta pasa por tres pasos:

1. **Re-rankeo**: se recuperan ~20 candidatos con vectorial+BM25 y el LLM los
   reordena por relevancia para quedarse con los mejores `k`. Se omite con
   `NTS_RAG_RERANK=false` o con `--sin-rerank` en la CLI.
2. **Respuesta**: el LLM redacta usando únicamente esos fragmentos, citando
   página y sección, y absteniéndose si no está la información.
3. **Verificación de fidelidad**: un pase del LLM comprueba que todas las
   afirmaciones se apoyan en las fuentes; si no, se devuelve el fragmento
   verbatim (extractivo). Se desactiva con `NTS_RAG_VERIFY=false`.

Sin LLM, la verificación y el re-rankeo se omiten y la respuesta es siempre el
fragmento verbatim con su cita.

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
├── eval/preguntas.jsonl          # 20 preguntas con respuesta + 5 fuera de alcance
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
│   ├── config.py                 # NTS_RAG_*
│   ├── index.py                  # JSONL → ChromaDB
│   ├── schemas.py                # contratos de la API
│   ├── api.py                    # FastAPI
│   ├── cli.py                    # comandos
│   ├── text.py                   # normalización y tokens
│   └── rag/
│       ├── embeddings.py         # FastEmbed / ONNX
│       ├── retrieve.py           # vectorial + BM25 + RRF
│       ├── answer.py             # LLM opcional / extractivo
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
| `NTS_RAG_VECTOR_K` | `12` | Candidatos vectoriales |
| `NTS_RAG_BM25_K` | `12` | Candidatos léxicos |
| `NTS_RAG_LLM_BASE_URL` | vacío | Activa el re-rankeo y el modo redactado |
| `NTS_RAG_LLM_MODEL` | `qwen2.5:7b` | Modelo del LLM |
| `NTS_RAG_LLM_API_KEY` | `ollama` | Clave del LLM |
| `NTS_RAG_VERIFY` | `true` | Verificación de fidelidad de la respuesta |

## Evaluación

```bash
python -m nts_rag.cli evaluar -k 5
```

Mide `Hit@k`: que la página y la sección esperadas estén entre las fuentes
recuperadas. El conjunto incluye cinco preguntas fuera de alcance para comprobar
la abstención.

Resultado actual: **Hit@1 = 20/20** y **Hit@5 = 20/20**; 4/5 preguntas fuera de
alcance se abstienen y la restante devuelve un fragmento relacionado (el
paracetamol sí se menciona en el Anexo 6).

Estos números miden solo preguntas de lookup de una ficha. Los sets de agregación,
comparación y ambigüedad están definidos en `docs/CONTEXTO_MEJORAS.md` pero todavía
no se evalúan.

## Limitaciones

- El sistema solo conoce las normas indexadas (hoy, la NTS 246-2026).
- La recuperación es vectorial + BM25 con re-rankeo LLM opcional; en preguntas
  ambiguas o con vocabulario muy genérico puede no acertar el fragmento exacto.
- Las preguntas por edad, las comparativas entre vacunas o normas y la dosis de
  rescate son el foco de la siguiente iteración.
- La abstención es heurística: una pregunta ajena que comparta vocabulario con
  la norma puede devolver un fragmento relacionado en lugar de abstenerse.
- El modo extractivo no redacta una síntesis; devuelve el fragmento más
  relevante con su cita.
- Una respuesta bien formada no garantiza la exactitud del dato clínico:
  contrasta siempre con el Markdown o el PDF original.

## Mejoras esperadas

El estado actual, el análisis y el plan de la siguiente iteración (esquema v2,
corpus 2022, preguntas por edad, comparativas, matriz de fichas, abstención y coste)
están en [`docs/CONTEXTO_MEJORAS.md`](docs/CONTEXTO_MEJORAS.md). Léelo antes de
cambiar la recuperación o el esquema de chunks.
