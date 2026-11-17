# Contexto de mejoras del RAG — NTS 246 / NTS 196

> **Traspaso.** Análisis cerrado. **Fase 0 hecha** (sets de evaluación, métricas y
> línea base en §9.3). **Fase 1 hecha**: esquema v2 reindexado (2.064 chunks, NTS
> 246-2026 + NTS 196-2022), smoke tests de §11 en verde. **Fase 2 hecha**: plan de
> consulta (LLM con validador, o determinista), evidencia por slots, expansión de
> sección, veredicto por código y citas `[cN]`; estado exacto 45/45 y contaminación
> 0. **Fase 3 hecha**: matriz vacuna × campo × versión, diff entre normas, rutas
> estructuradas y comandos `ficha`/`comparar`/`diff`; estado exacto 52/52 y
> precisión del diff **0,92 en un set ciego** (objetivo 0,90). **Fase 4 hecha**:
> presupuesto de ≤ 3 llamadas garantizado en código, presupuesto de tokens por
> intención, sustitución silenciosa 0/52, `uv.lock`, pre-commit, CI listo (pendiente de activar), registro
> de consultas y clave de API opcional. Resultados y desviaciones en §9.3 y §10.
> Fecha: 2026-11-16 · Siguiente paso: medir la ruta LLM con un modelo real y las
> complementarias de §10.

## 0. TL;DR y mandato

1. El fallo del caso guía ("niño de 5 años sin vacunas", "tabla de dosis de rescate")
   **no es del LLM**: es de recuperación y es estructural.
2. La edad desaparece en `tokenize` (`año`, `años`, `niño`, `niña` son stopwords) y
   las tablas que importan no se parten en filas.
3. Se decide una **matriz determinista `vacuna × campo × versión`** en lugar de chunks
   agregados. La comparación deja de depender del embedding.
4. Se ingesta la **NTS 196-2022** con `vigencia` por documento y **filtro por defecto a
   la vigente**; la histórica solo entra con señal explícita.
5. **Un solo reindexado** (esquema v2): corpus 2022, etiquetas canónicas, filas de
   dominio, `vacuna_slug`, edades normalizadas y metadatos de documento.
6. **El reindexado es obligatorio.** Sin `make chunks && make index` y sin pasar los
   smoke tests de §11, no hay entrega.
7. **LLM-first con guardas deterministas:** el LLM interpreta, reformula y redacta;
   el código valida entidades, cobertura, conflictos y citas. Ninguna cita la escribe
   el LLM.
8. **Coste acotado:** objetivo de 2–3 llamadas por consulta (~5–8k tokens), frente a
   las 3 actuales (~12k). Las rutas estructuradas pueden costar 0 llamadas.

**Mandato.** Si encuentras una solución mejor, aplícala. Condiciones: cumplir §9.3,
mantener las garantías de citación/abstención/"no consta", no volver a depender de
top-k para comparación o edad, y documentar la decisión. Verifica con el set
congelado; no lo ajustes para que pase.

---

## 1. Requisitos de origen

- Comparativas entre normas: diferencias de vacunas y dosis, vacunas nuevas,
  palabras *anterior/antigua/previa/2022/2026*.
- Preguntas múltiples: separar por oraciones y por "y" (solo **entre** entidades).
- Preguntas generales: comparar características de un grupo nombrado o global.
- Caso guía: niño de 5 años sin vacunas previas; priorizar, recuperable y no
  recuperable; rangos de edad (7–12, adolescencia 13+); página 80.
- Pospuesto: pesos semánticos difusos (virus–enfermedad, frío–cadena).

---

## 2. Estado actual

### 2.1 Arquitectura

```
corpus/markdown/*.md
  → pipelines/ocr/md_to_vector.py        (chunks base con sección y páginas)
  → src/nts_rag/index.py                 (normaliza + expande campos de ficha)
  → FastEmbed (ONNX) + ChromaDB
  → retrieve.py: vectorial + BM25 + RRF + refuerzos
  → answer.py: re-rankeo LLM + respuesta + verificación
  → FastAPI / CLI
```

| Archivo | Responsabilidad |
|---|---|
| `pipelines/ocr/md_to_vector.py` | Markdown → JSONL; tablas partidas solo si superan `--table-max-chars`. |
| `src/nts_rag/index.py` | `normalize_record`, `load_chunks`, `expand_field_chunks`, `build_index`. |
| `src/nts_rag/text.py` | `normalize_for_match`, `stem`, `tokenize`, `expand_tokens`. |
| `src/nts_rag/rag/retrieve.py` | Híbrido vectorial + BM25 + RRF. |
| `src/nts_rag/rag/answer.py` | Re-rankeo, prompt, citas, verificación, abstención. |
| `src/nts_rag/config.py` | `Settings`, `DOC_TITLE` hardcodeado. |
| `src/nts_rag/cli.py` | `indexar`, `consultar`, `buscar`, `inspeccionar`, `evaluar`, `servir`, `doctor`. |
| `Makefile` | `chunks` e `index`; `DOCS` solo lista los dos ficheros de 2026. |

### 2.2 Índice (v1 corregido)

| Dato | Valor |
|---|---|
| Colección | `nts_246` |
| Chunks | **1.004** = 562 texto + 86 tabla + 356 campo |
| Documentos | Solo NTS 246-2026 |
| Modelo | `paraphrase-multilingual-MiniLM-L12-v2` (FastEmbed/ONNX) |
| JSONL | 648 registros (regenerarlo da un archivo idéntico) |
| `vector_k` / `bm25_k` / `top_k` | 12 / 12 / 6 |
| Pool de `run_query` | 18 candidatos |

### 2.3 Correcciones ya aplicadas

| Cambio | Efecto |
|---|---|
| `formatting.py`: `+`/`*` solo abren viñeta al inicio de palabra | `+ 2° a + 8° C` llega intacto |
| `_kv_rows`: descarta encabezado y salta filas no-campo | Anexo 2 pasa de 0 a **10 de 13** campos |
| `build_index` en `chroma.tmp` y swap atómico | un index interrumpido no destruye el anterior |
| `/salud` compara `collection.count()` y marca `degradado`; healthcheck Docker | no se anuncia `ok` con índice parcial |
| `FIELD_INDICATORS` comparado por stems | el boost de campo funciona con «dosis»/«efectos» |
| `httpx.Client` compartido; re-rankeo sin índices repetidos | menos latencia; sin fuentes duplicadas |
| Tests de API y salud; `pipelines` en lint; `numpy` declarado | base para CI |

**Sigue igual:** edad stopword, pool 12+12, Anexo 2 como tabla de 2.690 caracteres,
2022 sin indexar, evaluación solo de lookup.

---

## 3. Evidencia

### 3.1 Caso guía

Tokens tras `tokenize` + `expand_tokens`:
`['5', 'debe', 'no', 'previa', 'priorizar', 'recibido', 'recuperabl', 'tener']`
— la edad desaparece y "no" es de las palabras más frecuentes del corpus.

| Fuente | Contenido | BM25 | Vector |
|---|---|---|---|
| `a1426dadea31de72` p.45 | 6.2.2 calendario 0–11 | 354 | 709 |
| `406d2c2e5d81d021` p.80 | Anexo 2, tabla de rescate | 109 | 260 |
| `fdd64fa0a6b49ec4` p.47 | 6.2.3 adolescencia | 584 | 464 |

Ninguna entra en el pool de 12+12, **ni con k=20**. Devuelve `p72, p121, p121, p7,
p121`; la p.121 es una tabla de **programación presupuestal**, no el calendario.

### 3.2 Incluso citando el nombre exacto

| Consulta | Anexo 2 (BM25) | Anexo 2 (vector) |
|---|---|---|
| "dosis de rescate" | 40 | 97 |
| "esquemas de vacunación con dosis de rescate" | 30 | 298 |
| "dosis de rescate BCG" (tabla) | 15 | 20 |
| "dosis de rescate BCG" (**fila BCG**) | **1** | **5** |

La tabla completa sigue fuera del pool, pero la **fila** ya se recupera: el problema
era de granularidad, no de vocabulario.

### 3.3 Comparaciones y edad

| Caso | Resultado |
|---|---|
| 6.2.2 para "qué toca a los 5 años" | **fuera incluso con k=20** |
| pentavalente vs hexavalente | **0/2** entidades |
| conservación SPR vs AMA | 1/2 (gana AMA) |
| composición neumo vs VPH | 2 fichas, **0/2 campos** |
| "vivas atenuadas" | 5/7 + ruido |
| "contraindicadas en embarazo" | **0/3** |
| "muslo vs deltoides" | **4/24** celdas |

Patrón: una consulta = un embedding; una entidad gana o la lista queda incompleta.

### 3.4 NTS 196-2022

No está indexada. Además, los numerales **no coinciden**:

| Vacuna / sección | 2022 | 2026 |
|---|---|---|
| Pentavalente | 6.1.1.3 | 6.1.1.4 |
| Antineumocócica | 6.1.1.9 | 6.1.1.10 |
| Fiebre amarilla | 6.1.1.12 | 6.1.1.14 |
| "Mayor de 5 años" | **6.2.2** | (6.2.2 = 0–11) |

Alinear por `vacuna_slug`, **nunca** por numeral. Anexo 2: p.50 (2022) vs p.80 (2026).

### 3.5 La evaluación no mide esto

`eval/preguntas.jsonl` son 20 lookups + 5 negativas; `cmd_evaluar` exige una única
sección y mide `Hit@k`. No hay agregación por edad, comparación ni versión. Por eso
"Hit@1 = 20/20" convive con el fallo real.

---

## 4. Causas raíz

| # | Causa | Evidencia |
|---|---|---|
| 1 | La edad se borra en `tokenize` (stopwords) | tokens = `['5', …]` |
| 2 | `_kv_rows` exige 2 columnas y mayúsculas; ya no es todo-o-nada | faltan HvB, HvA, Varicela (10/13) |
| 3 | Solo se parten tablas de 2 columnas | 6.2.2, 6.3.3 y Anexo 1 monolíticos |
| 4 | Tablas grandes como un chunk | Anexo 2 = 2.690 caracteres |
| 5 | Pool 12+12 y `k*3` | Anexo 2 en 109/260 |
| 6 | Una sola consulta, sin descomposición | los 6 slots se llenan del mismo tema |
| 7 | Sin eje de versión | 2022 fuera; `DOC_TITLE` fijo |
| 8 | Evaluación solo de lookup | 20/20 con el caso roto |
| 9 | Continuaciones de 1 fila sin campo | 2022 6.1.1.9, `USO SIMULTÁNEO` |

---

## 5. Decisiones

**D1 — Evaluación primero.** Congelar set y medir antes de tocar recuperación.

**D2 — Filas solo de tablas de dominio.** 6.2.x, 6.3.x y Anexos 1, 2, 3, 6, 8, 9.
Las operativas aportan ~948 filas de ruido. Se conserva la tabla completa.

**D3 — Matriz determinista, no chunks agregados.** `vacuna × campo × versión`.
Garantiza cobertura, completitud y verificación mecánica.

**D4 — 13 claves canónicas (congeladas).** Clave autodescriptiva, singular para
valor único, plural para listas.

| Canónico | Variantes | Celdas |
|---|---|---|
| `TIPO_DE_INMUNIZACION` | TIPO DE VACUNA / TIPO DE INMUNIZACIÓN | 42 |
| `DESCRIPCION` | DESCRIPCIÓN | 42 |
| `COMPOSICION` | COMPOSICIÓN | 42 |
| `PRESENTACION` | PRESENTACIÓN | 42 |
| `INDICACIONES` | INDICACIÓN / INDICACIONES | 42 |
| `ESQUEMA_DE_VACUNACION` | ESQUEMA / ESQUEMA DE VACUNACIÓN¹ | 42 |
| `DOSIS` | DOSIS / DOSIS (CANTIDAD) | 42 |
| `CONSERVACION` | CONSERVACIÓN DE LA VACUNA | 42 |
| `VIA_Y_SITIO` | VÍA Y SITIO DE ADMINISTRACIÓN / VACUNACIÓN | 42 |
| `JERINGAS` | JERINGA / JERINGA(S) A UTILIZAR | 38 |
| `EFECTOS_POSTVACUNALES` | POST-VACUNALES / POST VACUNALES / POSTVACUNALES | 42 |
| `CONTRAINDICACIONES` | CONTRAINDICACIÓN / CONTRAINDICACIONES / CONTRAINDICACION | 42 |
| `USO_SIMULTANEO` | CON OTRAS VACUNAS / DE LA VACUNA | 42² |

¹ `ESQUEMA DE VACUNACIÓN Y DOSIS` mapea a `{ESQUEMA_DE_VACUNACION, DOSIS}`.
² 41 antes de integrar la fila de continuación de 6.1.1.9; 42 con el arreglo.
`JERINGAS` falta en APO y rotavirus (orales): es **no aplica**, no `no consta`.
`EFECTOS_POSTVACUNALES` y no `EFECTOS`: evita colisión con "efectos de la cadena de
frío" y no introduce causalidad. `ESAVI` es otro concepto.

**D5 — Canonicalizar en el índice**, guardando `campo` y `campo_original`. Test que
falla con etiquetas fuera del catálogo.

**D6 — Vigencia por documento.** Registro con `documento_id`, `version`, `vigencia`.
Por defecto `vigente`.

**D7 — Señales de versión.** `2022`/`196`/`norma anterior` → histórica;
`2026`/`246`/`vigente` → actual; `comparar/cambió/nuevo/antes` → ambas. **`previa`
suelta no cuenta** ("vacunas previas" ≠ norma previa).

**D8 — Un reindexado único.** ~1.965 chunks estimados (1.208 base + 577 campo +
~180 filas). Manifiesto con `schema_version` y hashes.

**D9 — `vacuna_slug` en este reindexado.** Sin él hará falta un segundo.

**D10 — Diff con semántica.** `cambio_real`, `redaccional`, `movido`, `nuevo`,
`eliminado`, `sin_cambio`, `no_consta`, `no_aplica`. Normalizar `TIPO` a vocabulario
multi-etiqueta y resolver remisiones a Anexo 2/6.3.

**D11 — Contrato de respuesta.** Tabla con cita por celda; "no consta" y "no aplica"
explícitos; abstención referida a la norma vigente.

**D12 — Expansión por sección.** Tras el top-k, cargar los hermanos de
`seccion_numero` (ficha completa, tabla completa). Sin reindexar.

**D13 — Frescura y lint.** `fuentes: {ruta: sha256}` en el manifiesto; `doctor`
detecta desfase; `corpus_lint()` falla si falta un campo núcleo o una página.

**D14 — IDs estables.** `sha1(doc_id + seccion_numero + tipo + texto completo)`,
sufijo solo en duplicados reales. Validar con dos reindexados.

**D15 — LLM-first con guardas deterministas.** El LLM hace plan, reformulación y
redacción. El código valida y decide:

| Tarea | Quién | Por qué |
|---|---|---|
| Entender intención y reformular | LLM | lenguaje natural y premisas falsas |
| Elegir entidad/campo/edad | LLM desde catálogos cerrados | no inventa fuera del enum |
| Redactar la respuesta | LLM | lo que pide el producto |
| Validar entidades/campos/versión | **código** | un enum no se valida solo |
| Cobertura y abstención | **código** | el que responde no juzga si debía |
| Conflictos | **código** | dos celdas distintas, misma clave |
| Citas (página/sección) | **código** | salen de metadatos, no del LLM |
| Comparación, diff, edad | **código** | operaciones exactas |

El plan del LLM **sustituye al re-rankeo**: trae filtros exactos (slug, campo, edad,
versión) y no hay que reordenar 18 candidatos.

**D16 — Abstención como taxonomía.** Estados `respondida`, `parcial`, `abstenida`,
`aclaracion`; motivos `fuera_de_alcance`, `sin_evidencia`, `evidencia_insuficiente`,
`conflicto`, `premisa_invalida`, `metrica_no_comparable`,
`requiere_conocimiento_externo`, `ambiguedad`. Se decide con cobertura de slots,
conflictos y presuposiciones, no con la opinión del LLM.

**D17 — Coste acotado.** Presupuesto por intención, caché de plan y de prefijo,
abstención temprana y plantillas para respuestas estructuradas. Detalle en §8.

---

## 6. Diseño objetivo

### 6.1 Pipeline

```
consulta
  → plan LLM (JSON: intención, entidades, campos, edad, versión, supuestos)
  → validar plan (código: catálogos, límites, presuposiciones)
  → evidencia (matriz/retrieval/filtros, 0 llamadas)
  → redactar (LLM, con celdas y placeholders de cita)
  → verificar (código: fidelidad y cobertura; LLM opcional para prosa)
  → estado: respondida | parcial | abstenida | aclaracion
```

### 6.2 Metadatos por chunk (v2)

`documento_id`, `documento`, `version`, `vigencia`, `pagina`, `pagina_fin`,
`seccion`, `seccion_corta`, `seccion_numero`, `tipo` (`texto|tabla|campo|fila`),
`campo`, `campo_original`, `vacuna_slug`, `edad_min_meses`, `edad_max_meses`,
`parte`, `parte_total`.

### 6.3 Intenciones

| Intención | Motor | Garantía |
|---|---|---|
| lookup | híbrido + campo | top-k |
| edad | filtro de rango + filas 6.2.x/Anexo 2 | todas las aplicables |
| conjunto nombrado | matriz × entidades × facetas | cobertura 100% |
| global por propiedad | matriz + vocabulario | lista completa |
| global libre | barrido exhaustivo del campo | lista completa |
| comparación de normas | matriz 3 ejes + diff | diff con citas |
| evaluación/superlativo | abstención | `requiere_conocimiento_externo` |

### 6.4 Matriz y verificación

Clave `(vacuna_slug, campo, version)` → `{texto, pagina, documento_id}`.
Operaciones: `ficha`, `comparar`, `filtrar`, `diff`. Fallback: barrido del campo
(24 celdas ≈ 4.200 caracteres).

Verificación mecánica: completitud de lista, no invención, fidelidad de celda, no
sustitución y contaminación 0.

---

## 7. Reformulación, abstención y respuestas solo desde el MD

### 7.1 Principio

El sistema responde **solo lo que la norma declara**. No completa con conocimiento
externo. Una pregunta puede ser respondida como *documental* aunque no como
*clínica*; el sistema debe decirlo.

### 7.2 Reformulación en tres niveles

| Nivel | Qué cambia | Quién |
|---|---|---|
| Léxica | vocabulario → términos del corpus | determinista |
| Estructural | entidades, subconsultas, versión | LLM + validador |
| Presuposicional | corrige una premisa falsa | LLM propone, se muestra |

Contrato del plan:

```json
{
  "pregunta_normalizada": "...",
  "intencion": "max_atributo",
  "entidades": [{"slug": "neumococo"}],
  "campos": ["COMPOSICION"],
  "edad_meses": null,
  "version": "vigente",
  "subconsultas": ["..."],
  "supuestos": ["..."],
  "premisas_no_soportadas": ["..."],
  "requiere_aclaracion": false
}
```

Validador: slugs y campos en catálogo, versión válida, edad en rango, ≤4 subconsultas,
≤6 entidades. Si falla, un reintento; si vuelve a fallar, `parcial`.

### 7.3 Abstención

| Condición | Estado | Motivo |
|---|---|---|
| Ningún slot reconocible | `abstenida` | `fuera_de_alcance` |
| Slots, cero fuentes | `abstenida` | `sin_evidencia` |
| Faltan fuentes obligatorias | `parcial` | `evidencia_insuficiente` |
| Dos fuentes discrepan | `parcial` | `conflicto` |
| Presuposición no soportada | `parcial` | `premisa_invalida` |
| Pide juicio clínico | `abstenida` | `requiere_conocimiento_externo` |
| Ambigüedad material | `aclaracion` | `ambiguedad` |

Respuesta:

```json
{
  "estado": "parcial",
  "motivo": "conflicto",
  "respuesta": "Markdown con [c1], [c2]…",
  "reformulacion": "...",
  "supuestos": ["..."],
  "faltan_fuentes": ["6.2.2"],
  "conflictos": [{"campo": "ESQUEMA", "valores": ["recién nacido", "2 meses"]}]
}
```

El LLM escribe `[c1]`; el renderizador lo sustituye por
`[NTS 246-2026, p. 24, 6.1.1.10]`. Una cita escrita de memoria se descarta.

### 7.4 Preguntas de borde (ejemplos de referencia)

| Pregunta | Respuesta grounded | Estado |
|---|---|---|
| ¿Qué vacuna previene contra más cepas? | PCV20 20 serotipos; VPH 9; meningococo 4; influenza 3; VRS 2. La norma no compara eficacia | `parcial` / `metrica_no_comparable` |
| ¿Cuándo se coloca el mAb si la gestante no inició esquema? | La norma condiciona a "madre no vacunada frente a VRS": recién nacido; excepcional <6 meses. El calendario lista 2 meses y 1–6 meses | `parcial` / `conflicto` |
| ¿Cuándo cambia el vasto externo y por qué? | A los 2 años: <2 vasto externo; ≥2 deltoides; "no hay nervios ni vasos importantes" (Anexo 6) | `respondida` |
| ¿Qué alergia comparten SPR, AMA e influenza? | 2026: huevo en SPR/SR (con gelatina y neomicina) y AMA (con proteínas de pollo); influenza no la declara (sí en 2022). Componente: substrato de cultivo en embrión/huevos de pollo | `parcial` / `premisa_invalida` |

---

## 8. Coste y eficiencia

### 8.1 Perfil actual

| Paso | Llamadas | Entrada |
|---|---|---|
| Re-rankeo | 1 | ~2,7k tokens |
| Respuesta | 1 | ~6k |
| Verificación | 1 | ~3k |
| **Total** | **3** | **~12k** |

Los embeddings y Chroma son locales: el coste API es solo LLM.

### 8.2 Objetivo

| Paso | Llamadas | Tokens |
|---|---|---|
| Plan | 1 | ~1,4k (prefijo cacheado) |
| Evidencia | 0 | 0 |
| Redacción | 1 | ~3–6k |
| Verificación | 0–1 | ~1k |
| **Total** | **2–3** | **~5–8k** |

### 8.3 Palancas

| Palanca | Ahorro | Riesgo |
|---|---|---|
| Plantilla para lookup/edad/comparación/diff | 0 llamadas | respuestas menos naturales |
| Enviar celdas, no chunks completos | 50–70% de entrada | perder contexto |
| Plan en lugar de re-rankeo | ~1,3k netos (quita 2,7k, añade 1,4k) | plan erróneo → validador |
| Verificación mecánica de celdas | ~3k | solo para extractivo |
| Re-ranker local (cross-encoder) | 100% de esa llamada | CPU/latencia |
| Caché exacta + prefijo | alto en repetidas | invalidar al reindexar |
| Abstención temprana | 100% en fuera de alcance | abstenerse de más |

**Caché:** plan, expansión de sección, embeddings y respuesta exacta indexada por
`(pregunta normalizada, schema_version, hashes)`. **No** caché semántica por similitud:
dos preguntas parecidas pueden tener intenciones distintas.

**Presupuesto por intención:** lookup 2k, edad 6k, comparación 6k, global 8k, abierta
12k, fuera de alcance 0. Si se excede, se recorta el contexto (no marca `parcial`).

**Medición:** `usage` por llamada, `--explicar` con plan/tokens/ruta y promedio por
tipo en `evaluar`.

---

## 9. Evaluación

### 9.1 Sets

- `eval/preguntas.jsonl`: 20 lookups + 5 negativas (regresión).
- `eval/agregacion.jsonl`: 7 de agregación + 3 negativas.
- `eval/ambiguedades.jsonl`: las 4 preguntas de borde + 3 negativas, con `estado` y
  `motivo` esperados.
- `eval/comparativas.jsonl`: las 3 comparativas 2022/2026 del Anexo C. Las fuentes
  2022 llevan `documento: nts-196-2022`; hasta la Fase 1 no pueden recuperarse.

Formato v2 de fila: `id`, `pregunta`, `tipo`, `fuentes_esperadas`
(`[{documento?, pagina, pagina_fin?, seccion}]`, `documento` por defecto
`nts-246-2026`), `estado_esperado`, `motivo_esperado?`, `clave`. Las filas
heredadas (`pagina` + `seccion`) se leen como una única fuente esperada. La lógica
está en `src/nts_rag/evaluation.py` (pura, con tests en `tests/test_evaluation.py`).

### 9.2 Métricas

`hit@k` (compatibilidad), `recall`, `completa`, `precision`, `abstiene`,
`reformulacion_util` (la reescritura recupera las fuentes esperadas) y
`sustitucion_silenciosa = 0`. Reporte **por tipo**.

Implementadas en `evaluar` (Fase 0): `hit`, `recall`, `completas` y `precision` por
corte (`-k 5 10 20`); decisión responder/abstenerse, estado exacto, abstención
indebida y motivo exacto; `--por-tipo`, `--json`, `--sin-respuesta`. Una sección
casa como unidad completa (`6.1.1.1` ≠ `6.1.1.10`, `ANEXO 1` ≠ `ANEXO 10`) y las
páginas por solape de rangos. `reformulacion_util` se calcula cuando
`AnswerResult` exponga `reformulacion` (Fase 2); `estado`/`motivo` se leen de
`AnswerResult` cuando existan, y mientras tanto se deriva `abstenida`/`respondida`.
`sustitucion_silenciosa` queda para la Fase 4 (requiere el contrato de citas).

### 9.3 Línea base y umbrales

Baseline v1 (994): recall@5 0.29, @10 0.36, @20 0.57; completas 1/7, 1/7, 2/7.

**Línea base Fase 0** (2026-11-16, índice de 1.004 chunks, modo extractivo sin LLM):

```bash
python -m nts_rag.cli evaluar --archivo eval/preguntas.jsonl eval/agregacion.jsonl \
  eval/ambiguedades.jsonl eval/comparativas.jsonl -k 5 10 20 --por-tipo
```

| Tipo | n | recall@5 | recall@10 | recall@20 | completas @5/@10/@20 | decisión | estado exacto |
|---|---|---|---|---|---|---|---|
| lookup | 20 | 1.00 | 1.00 | 1.00 | 20/20/20 | 20/20 | 20/20 |
| agregacion | 7 | 0.39 | 0.39 | 0.54 | 2/2/2 | 6/7 | 6/7 |
| borde | 4 | 0.26 | 0.31 | 0.47 | 0/0/1 | 3/4 | 1/4 |
| comparacion | 3 | 0.25 | 0.33 | 0.33 | 0/0/0 | 3/3 | 3/3 |
| negativa | 11 | 0.75¹ | 0.75 | 0.75 | 1/1/1 | 8/11 | 8/11 |
| **total** | 45 | 0.72 | 0.74 | 0.78 | 23/23/24 | 40/45 | 38/45 |

¹ Solo 2 negativas tienen fuentes (las de "no procede": rotavirus a los 12 meses,
BCG a los 6 años).

- Abstención: 6/9 cuando debe (fallan paracetamol, "mejor vacuna" y "¿cuántas
  dosis?"); 2/36 abstenciones indebidas ("qué toca a los 5 años" y "más cepas").
  Motivo exacto 0/7: la v1 no produce motivos.
- La recuperación casi no mejora de @10 a @20 porque el pool de `search` es
  12+12 candidatos (causa raíz 5): pedir más `k` no trae más.
- Este set da recall@10 0.39 en agregación frente al 0.36 de la medición previa
  sobre 994 chunks. No son directamente comparables (set y fuentes distintos):
  **la referencia para los umbrales es esta tabla**.
- Agregación 6 usa como contraindicadas en el embarazo 6.1.1.11 (SPR), 6.1.1.12 (SR)
  y 6.1.1.14 (AMA): son las fichas cuyo campo `CONTRAINDICACIÓN` menciona
  embarazo/gestantes. La ficha 6.1.1.19 (VPH) que citaba el Anexo C no lo menciona.

**Resultado Fase 1** (2026-11-16, índice v2 de 2.064 chunks, mismo comando, modo
extractivo):

| Tipo | n | recall@5 | recall@10 | recall@20 | completas @5/@10/@20 | decisión | estado exacto |
|---|---|---|---|---|---|---|---|
| lookup | 20 | 1.00 | 1.00 | 1.00 | 20/20/20 | 20/20 | 20/20 |
| agregacion | 7 | 0.61 | **0.89** | 0.96 | 3/6/6 | 7/7 | 7/7 |
| borde | 4 | 0.26 | 0.31 | 0.47 | 0/0/1 | 3/4 | 1/4 |
| comparacion | 3 | 0.42 | 0.50 | 0.67 | 1/1/2 | 3/3 | 3/3 |
| negativa | 11 | 0.50 | 1.00 | 1.00 | 0/2/2 | 8/11 | 8/11 |
| **total** | 45 | 0.76 | 0.86 | 0.91 | 24/29/31 | 41/45 | 39/45 |

Umbrales de Fase 1:

| Métrica | Objetivo | Resultado |
|---|---|---|
| recall@10 agregación | ≥ 0.80 | **0.89** ✅ |
| 6.2.2 en toda pregunta por edad | 100% | **6/6 en top-10** ✅ (posición 1–3 salvo el caso guía, en la 6) |
| lookups | 20/20 | **20/20** ✅ |
| índice fresco / lint / IDs | 0 / 100% / idénticos | **0 desfases / 0 errores / JSONL idéntico en dos pasadas** ✅ |

- Lo que sube es lo que el diagnóstico atribuía a la estructura: filas de dominio
  (la fila BCG del Anexo 2 sale en el top-5), edad tokenizada y sinónimos de
  rescate. "¿Qué vacunas le tocan a un niño de 5 años?" ya no se abstiene.
- Las comparativas pasan de 0.33 a 0.50 en recall@10 solo con indexar la 2022 y las
  señales de versión: "dosis de rescate BCG 2022 vs 2026" recupera ambos Anexo 2.
- Sin cambios (fuera del alcance de Fase 1): borde (necesita plan y matriz),
  "vacunas nuevas" (necesita el diff de la Fase 3) y la abstención (6/9).
- **Señuelo p.121 vivo**: en el caso guía las tablas de programación del Anexo 17
  ocupan el top-5 y 6.2.2 queda en la posición 6. Excluirlas de la ruta de edad
  es trabajo de la Fase 2 (§12).

**Resultado Fase 2** (2026-11-16, mismo índice v2, modo extractivo **sin LLM**: el
plan es el determinista). `evaluar` mide ahora la recuperación del sistema completo
(plan + evidencia, la misma de `/consulta`); `--solo-busqueda` mide el buscador
híbrido solo.

| Tipo | n | recall@5 | recall@10 | completas @5/@10 | decisión | estado exacto | solo búsqueda @10 |
|---|---|---|---|---|---|---|---|
| lookup | 20 | 1.00 | 1.00 | 20/20 | 20/20 | 20/20 | 1.00 |
| agregacion | 7 | **1.00** | **1.00** | 7/7 | 7/7 | 7/7 | 0.75 |
| borde | 4 | 0.47 | 0.82 | 1/2 | 4/4 | **4/4** | 0.66 |
| comparacion | 3 | 0.67 | 0.67 | 2/2 | 3/3 | 3/3 | 0.50 |
| negativa | 11 | 1.00 | 1.00 | 2/2 | 11/11 | **11/11** | 0.50 |
| **total** | 45 | 0.91 | 0.95 | 32/33 | **45/45** | **45/45** | 0.84 |

Abstención: 9/9 cuando debe, 0/36 indebidas; motivo exacto 7/7. Umbrales:

| Fase | Métrica | Objetivo | Resultado |
|---|---|---|---|
| Fase 2 | contaminación | 0 | **0** fuentes en 45 preguntas (@5) ✅ |
| Fase 2 | expansión de sección | 13/13 | **42/42 fichas** completas (12/13 en APO y rotavirus por `NO_APLICA`) ✅ |
| Fase 1 | lookups | 20/20 | 20/20 ✅ (se mantiene) |
| Fase 3 | abstención en negativas | 5/5 | 9/9 ✅ (adelantado) |
| Fase 4 | llamadas por consulta | ≤ 3 | 3 con LLM (plan + redacción + verificación), 0 sin LLM ✅ en test |

- El caso guía ya trae 6.2.2 y Anexo 2 en el top-5: la ruta de edad añade la
  mejor fila del calendario y del Anexo 2 y penaliza los anexos operativos (el
  señuelo p. 121).
- `contaminación` cuenta fuentes de la norma derogada sin pedirla y fichas de una
  vacuna no nombrada cuando la pregunta nombra alguna. Al principio daba 6 (HvB
  junto a HVA, IPV junto a APO, hexavalente junto a pentavalente): esas fichas se
  descartan de las fuentes, no solo se penalizan.
- Siguen fallando, y es trabajo de la Fase 3: "vacunas nuevas 2026 vs 2022" (0/6,
  necesita el diff) y "vasto externo" (la fila del Anexo 6 no entra en el top-5).
  "más cepas" encuentra 2/5 fichas porque el barrido es léxico (serotipos,
  serogrupos…); la matriz con valores normalizados lo resolverá.
- **Sin LLM en este equipo** (ni Ollama ni `.env`): la ruta del plan con LLM, la
  redacción con `[cN]` y el conteo de llamadas están cubiertos con un LLM simulado
  (`tests/test_phase2.py`), no medidos con un modelo real. `reformulacion_util`
  sigue en n/d por lo mismo.

**Resultado Fase 3** (2026-11-16, mismo índice v2, sin LLM). Se añade
`eval/matriz.jsonl` (3 conjuntos nombrados de §3.3, 3 listas globales y 1
contradicción) y tres sets de diff etiquetados a mano, cada uno **antes** de ver
cómo lo clasifica el código: `eval/diff.jsonl` (diseño: 20 celdas + 6 fichas
nuevas), `eval/diff_control.jsonl` (control 1: 20 celdas) y
`eval/diff_control2.jsonl` (control 2: 24 celdas, incluidas 4 de `RESCATE`).

```bash
python -m nts_rag.cli evaluar --archivo eval/preguntas.jsonl eval/agregacion.jsonl \
  eval/ambiguedades.jsonl eval/comparativas.jsonl eval/matriz.jsonl -k 5 10 --por-tipo
python -m nts_rag.cli diff --evaluar eval/diff.jsonl eval/diff_control.jsonl   eval/diff_control2.jsonl
```

| Tipo | n | recall@5 | recall@10 | decisión | estado exacto |
|---|---|---|---|---|---|
| lookup | 20 | 1.00 | 1.00 | 20/20 | 20/20 |
| agregacion | 7 | 1.00 | 1.00 | 7/7 | 7/7 |
| borde | 4 | **0.82** | 0.88 | 4/4 | 4/4 |
| comparacion | 3 | **0.94** | **1.00** | 3/3 | 3/3 |
| conjunto (nuevo) | 3 | 1.00 | 1.00 | 3/3 | 3/3 |
| global (nuevo) | 3 | — | — | 3/3 | 3/3 |
| contradiccion (nuevo) | 1 | — | — | 1/1 | 1/1 |
| negativa | 11 | 1.00 | 1.00 | 11/11 | 11/11 |
| **total** | 52 | 0.98 | 0.99 | **52/52** | **52/52** |

Umbrales de Fase 3:

| Métrica | Objetivo | Resultado |
|---|---|---|
| cobertura nombrada | 2/2 | **3/3** conjuntos con todas las vacunas/celdas ✅ |
| cobertura global | 7 · 3 · 24 | **7/7** vivas atenuadas · **3/3** contraindicadas en el embarazo · **24/24** celdas de vía y sitio ✅ (precisión de las listas 0.92) |
| precisión del diff | ≥ 0.90 | control 2, ciego: **0.92** (22/24) ✅ · control 1: 0.90 (18/20) · diseño: 0.92 (24/26) |
| contradicciones | 1/1 con ambas citas | **1/1** (anticuerpo monoclonal: ficha 6.1.1.24 y calendario 6.2.2) ✅ |
| abstención en negativas | 5/5 | **9/9**, 0 indebidas ✅ |

- **Cómo se llegó al diff y qué cifra vale.** Primera versión (una sola medida
  de palabras para todos los campos): diseño 0.77. Reglas por campo de D10 (familia
  del tipo; cifras en conservación, dosis y jeringas; listas elemento a elemento):
  diseño 0.92, pero control 1 **0.80**, por debajo del umbral. Sus 4 fallos iban en
  la misma dirección (marcar `cambio_real` donde había redacción o un dato
  movido). Se etiquetó entonces el control 2 a ciegas (línea base 0.88, sin mirar
  qué fallaba) y se añadieron tres reglas generales: (1) `movido` cuando la versión
  nueva conserva solo parte de las palabras y cifras y remite a otra sección; (2)
  límites de edad equivalentes ("menores de 5 años" = "hasta 4a11m29d", pero no
  "hasta 5 años 0 días"); (3) en composición, añadir detalle sin quitar nada es
  redacción. Resultado: control 2 **0.92**. El control 1 sube a 0.90, pero es
  optimista porque esas reglas salen de sus fallos; la cifra honesta es la del
  control 2. Sus dos fallos (dT adulto esquema, influenza dosis) se vieron después
  de medir y no se ajustaron.
- Sigue tendiendo a sobreavisar: de los 7 fallos en los tres sets, 6 marcan un
  cambio que no lo es; el único cambio real que se escapa es BCG dosis (mismas
  cifras, otra asignación por edad).
- Cerrados por la matriz: "vacunas nuevas 2026 vs 2022" (0/6 → 6/6, diff de
  catálogo), "más cepas" (2/5 → 5/5 en el top-10, tabla de valencias) y la
  edad máxima de APO entre normas (diff de esquema y Anexo 2 con las cuatro citas).
  "vasto externo" (0/1 → 1/1) lo cierra un slot del anexo de vías y técnicas.
- La tabla de valencias incluye APO (bivalente), que la referencia de §7.4 no
  listaba: es un dato del texto, no un error.

**Resultado Fase 4** (2026-11-16): el eval completo no cambia (estado exacto
52/52, contaminación 0, cobertura 6/6) y se añaden dos métricas.

| Métrica | Objetivo | Resultado |
|---|---|---|
| llamadas por consulta | ≤ 3 | **≤ 3 garantizado** por `_chat` (tope duro por consulta). Peor caso probado con LLM simulado: plan inválido → sin reintento ni re-rankeo, redacción + verificación = 3. Sin LLM: 0 en las 52 preguntas ✅ |
| sustitución silenciosa | 0 | **0/52** ✅ (vacuna nombrada sin fuente propia respondida como tal, o reformulación no mostrada) |

**Resultado Fase 5** (2026-11-17): toda respuesta estructurada pasa por el LLM antes
de mostrarse. La matriz sigue siendo la fuente de verdad y la plantilla el respaldo.

- El eval completo no cambia: **52/52 estados**, 0 sustitución, 0 contaminación,
  recall@10 0.99 y cobertura 6/6.
- La pregunta de dos partes (“menor de 5 años … no recuperables”) baja de ~4,7k a
  ~1,6k caracteres: esquema 0–4 en tabla, umbrales de no recuperables y sin listar
  las recuperables que no se piden. Las fuentes se agrupan por sección y solo se
  listan las que la respuesta cita.
- Coste: **2,65 llamadas** y **6.214 tokens** por consulta; las estructuradas hacen
  3 llamadas. Con DeepSeek Flash: **~$1,27/1.000 consultas off-peak** y **~$2,54 en
  peak**; a 1.000/mes, **~$1,27–$2,54**.
- La redacción cae a la plantilla si la verificación falla, si omite una vacuna de
  una lista o si introduce un número que no está en la tabla.
- Se generaliza “campo de cada vacuna” (dosis, composición, conservación, rescate) a
  las 24 fichas y se elimina el bloque “Además”. La pregunta de edad con rescate
  recibe los 13 límites del Anexo 2 y responde también las “no recuperables” con
  umbrales de edad.
- `pytest`: 191; `ruff` limpio; diff 0,92 / 0,90 / 0,92.

| Fase | Métrica | Objetivo |
|---|---|---|
| Fase 1 | recall@10 agregación | ≥ 0.80 |
| Fase 1 | 6.2.2 en toda pregunta por edad | 100% |
| Fase 1 | lookups | 20/20 |
| Fase 1 | índice fresco / lint / IDs | 0 desfases / 100% / idénticos |
| Fase 2 | contaminación | 0 |
| Fase 2 | expansión de sección | 13/13 |
| Fase 3 | cobertura nombrada / global | 2/2 / 7-3-24 |
| Fase 3 | precisión del diff | ≥ 0.90 |
| Fase 3 | contradicciones | 1/1 con ambas citas |
| Fase 3 | abstención en negativas | 5/5 |
| Fase 4 | llamadas por consulta | ≤ 3 |
| Fase 4 | sustitución silenciosa | 0 |
| Fase 5 | respuestas estructuradas | 3 llamadas · plantilla de respaldo |
| Fase 5 | sustitución silenciosa con LLM | 0/52 |
| Fase 5 | coste | 2,65 llamadas · 6.214 tokens (Flash ~$1,27/1k off-peak) |

---

## 10. Plan

### Fase 0 — Evaluación ✅

1. ✅ `eval/agregacion.jsonl` (7 + 3 negativas).
2. ✅ `eval/ambiguedades.jsonl` (4 de borde + 3 negativas, con estado/motivo).
3. ✅ `cmd_evaluar`: recall, completa, precision, abstiene, reformulación, `--por-tipo`.
4. ✅ Línea base registrada en §9.3 (más `eval/comparativas.jsonl`).

### Fase 1 — Esquema v2 + reindexado ✅

**Hecho** (resultados en §9.3, smoke tests en §11):

| Archivo | Cambio aplicado |
|---|---|
| `md_to_vector.py` | ID = `sha1(fuente + sección + tipo + texto completo)`, sufijo solo en duplicados exactos; el marcador "continuación del listado" extiende las páginas de la tabla (la p. 78 de 2026 quedaba sin chunk). `split_table` revisado: sin cambios. |
| `config.py` | Registro `DOCUMENTS` (Anexo B), `INDEX_SCHEMA_VERSION = 2`, `corpus_dir`. |
| `catalogo.py` (nuevo) | 13 claves canónicas y variantes, fichas del Anexo A por norma, `NO_APLICA` (jeringas en APO/rotavirus), tablas de dominio por título y `detect_vaccines`. |
| `index.py` | Metadatos v2 (§6.2), campo canónico + `campo_original` solo en fichas, filas de dominio con `vacuna_slug`/`vacunas` y edades en meses, `corpus_lint()`, manifiesto con `schema_version`, documentos y hashes; `freshness_problems()`. |
| `text.py` | "año(s)" fuera de las stopwords + tokens de edad (`edad5a`; "5 a 11 años" → `edad5a`…`edad11a`), `age_range_months`, `age_limit_months`, sinónimos de rescate y `detect_vigencias` (D7). |
| `retrieve.py` | Vigencia vigente por defecto, filtros `documento`/`vigencia`, boost de campo por clave canónica u original. |
| `schemas.py` / `api.py` | `Filtros.documento`, `Filtros.vigencia`, `tipo=fila`; `Fuente` con documento, versión, vigencia, `campo_original` y `vacuna_slug`. |
| `answer.py` | Citas y contexto con el documento del chunk; "(derogada)" en la 2022. |
| `cli.py` | `doctor` compara hashes y pasa `corpus_lint`; `indexar` imprime el lint. |
| `Makefile` | `DOCS` incluye la 2022. |

**Desviaciones y decisiones tomadas** (Mandato de §0):

- Se adelantan dos piezas de la Fase 2 porque los smoke tests las exigen y sin
  ellas la 2022 contaminaría los lookups: el filtro `vigencia=vigente` por defecto
  y las citas por documento.
- D7 ajustado: "antes" y "nuevo" sueltos **no** son señal de versión ("antes de
  vacunar", Anexo 5). Cuentan `comparar`, `cambió`, `diferencia`, `vacunas
  nuevas`, `incorpora`, `vs`, además de 2022/196/2026/246/vigente/"norma anterior".
- La edad se resuelve con tokens sintéticos además de sacar "año(s)" de las
  stopwords: el número suelto no casaba "5 años" con la fila "5 a 11 años".
  "niño/niña" siguen siendo stopwords.
- El Anexo 2 ya no genera chunks `campo` (era 10/13 de un artefacto); sus filas son
  `fila` con vacuna y límite de edad.
- Anexo E resuelto: #1 una colección con filtro de vigencia; #2 registro en
  `config.py`; #3 los 24 slugs del Anexo A quedan congelados en `catalogo.py`; #4
  edades en **meses** inclusivos ("4a11m29d" → 59, "5 a 11 años" → 60–143); #10
  `corpus_lint` falla en CI (`tests/test_smoke_index.py`) y avisa en local
  (`indexar`, `doctor`); #11 hash del texto completo sin normalizar.
- Se mantienen el nombre del JSONL (`nts-246.chunks.jsonl`) y de la colección
  (`nts_246`) aunque ya contienen las dos normas: renombrarlos toca Docker, `.env`
  y la documentación y no aporta a la recuperación. Pendiente de decidir.
- Dos arreglos de robustez encontrados al reindexar en Windows: el swap atómico
  fallaba porque Chroma retiene `chroma.sqlite3` abierto (ahora se libera el
  cliente y el índice anterior se aparta a `.old` antes de mover el nuevo), y los
  hashes del manifiesto se calculan con saltos de línea normalizados a LF para que
  `core.autocrlf` no produzca desfases falsos entre máquinas.

Plan original:

| Archivo | Cambio |
|---|---|
| `md_to_vector.py` | IDs estables; revisar `split_table`. |
| `config.py` | Registro `DOCUMENTS`; `index_schema_version = 2`. |
| `index.py` | Documento/versión/vigencia, `vacuna_slug`, `campo` canónico + `campo_original`, filas de dominio, edades, manifiesto con hashes y `corpus_lint()`. `_kv_rows` de N columnas y sin umbral de 3 filas. |
| `text.py` | Edades (sacar `ano/anos` o tokenizar), sinónimos de rescate/completar y de versión, `FIELD_INDICATORS` canónicos. |
| `schemas.py` | `documento`, `vigencia`, `alcance`. |
| `cli.py` | `doctor` compara hashes. |
| `Makefile` | `DOCS` incluye 2022. |

**Reindexado obligatorio** y determinista (dos pasadas, mismos IDs).

### Fase 2 — Recuperación y plan ✅

**Hecho** (resultados en §9.3):

| Archivo | Cambio aplicado |
|---|---|
| `rag/interpret.py` (nuevo) | `Plan` con el contrato de §7.2. Con LLM: el plan lo propone el LLM (1 llamada, **sustituye al re-rankeo**), se valida contra los catálogos (vacunas, campos, versión, edad, ≤4 subconsultas, ≤6 entidades), un reintento con los errores y, si falla, plan determinista. Sin LLM: plan determinista con los mismos catálogos (intención, vacunas, campos, edad, versión, rescate, subconsultas por oración y por "y" + interrogativo, aclaración, juicio de valor y posología). |
| `rag/evidence.py` (nuevo) | Pool 24+24 reordenado por el plan; slots deterministas (campo exacto por vacuna y norma, ficha, calendario por edad, Anexo 2 para rescate/límites, barrido de campo, una fuente por subconsulta); fichas de vacunas no nombradas fuera; ≤2 por sección; `expand_context` con presupuesto de 16k caracteres. |
| `rag/abstain.py` (nuevo) | Veredicto por código en el orden de §7.3: juicio/posología → sin fuentes → fuera de alcance → aclaración → premisa (alérgeno común) → conflicto (edad de la ficha frente al calendario) → evidencia insuficiente → métrica no comparable → respondida. |
| `rag/answer.py` | Contrato de estado en `AnswerResult`; el LLM cita con `[cN]` y el código pone la cita desde los metadatos (las citas escritas de memoria se borran; sin ninguna cita válida, extractivo); notas de parcial/conflicto/premisa; extractivo con un fragmento por fuente principal fuera de los lookups; conteo de llamadas y tokens por consulta. |
| `retrieve.py` / `config.py` | Pool por defecto 24+24 (antes 12+12). |
| `schemas.py` / `api.py` | `/consulta` devuelve `estado`, `motivo`, `reformulacion`, `supuestos`, `faltan_fuentes`, `conflictos`, `llamadas_llm` y tokens. |
| `evaluation.py` / `cli.py` | `evaluar` mide el sistema completo, `contaminacion` y `--solo-busqueda`. |
| `index.py` | El manifiesto también registra el hash de `catalogo.py`: cambiar el catálogo deja el índice desfasado y `doctor` lo avisa. |

**Desviaciones y decisiones tomadas:**

- Planificador **determinista además del LLM**: sin LLM configurado (el caso por
  defecto) el sistema también necesita plan, y el validador vale para los dos. Es
  también el respaldo cuando el plan del LLM no valida dos veces (en lugar de
  devolver `parcial`, que penalizaría una pregunta bien formada por un fallo del
  planificador; queda un aviso en el plan).
- El filtro de slug/edad no es un filtro duro de Chroma: los chunks de texto no
  llevan edad. Se aplica como slots y como reordenación.
- "Posología de un medicamento" (paracetamol, antibiótico…) se abstiene con
  `requiere_conocimiento_externo` (§7.1, documental ≠ clínica), igual que los
  juicios de valor ("mejor vacuna").
- Verbos de la pregunta ("toca", "vacunar", "previene", "colocar"…) son genéricos
  en `is_answerable`, y un sinónimo cumple su término sin subir el listón: antes
  "¿qué le toca a un niño de 5 años?" se abstenía.
- La reformulación del plan (LLM) se usa también para el chequeo de alcance.
- Anexo E resuelto: #9 la expansión de fichas es **siempre** (los campos son
  cortos); la de filas, solo de la misma tabla y filtrada por edad.

Plan original:

| Archivo | Cambio |
|---|---|
| `retrieve.py` | ~~`vigencia=vigente` por defecto~~ (hecho en Fase 1); filtros slug/edad; pool 40–50; dedup/MMR; boost canónico; excluir tablas operativas (p. 121) de la ruta de edad. |
| `retrieve.py` | `expand_context`: hermanos de `seccion_numero` con presupuesto. |
| `interpret.py` (nuevo) | Plan LLM + validador + presuposiciones + subconsultas. |
| `abstain.py` (nuevo) | Cobertura, conflictos y veredicto. |
| `answer.py` | Citas por documento; redacción LLM; contrato de estado. |

### Fase 3 — Matriz y comparación ✅

**Hecho** (resultados en §9.3):

| Archivo | Cambio aplicado |
|---|---|
| `rag/fichas.py` (nuevo) | `Matrix` con 567 celdas `(vacuna, campo, documento)` — los 13 campos canónicos más `RESCATE` (fila del Anexo 2) —, estados `no_aplica`/`no_consta`/`sin_ficha`; valores normalizados (tipo multi-etiqueta, vía, sitio, valencia, remisiones); operaciones `ficha`, `compare`, `filter`, `diff`, `catalog_diff`, `valences`; `classify_change` con las 8 clases de D10; render de tabla con cita por celda (D11). |
| `rag/structured.py` (nuevo) | Rutas por plantilla, 0 llamadas (§8.3): comparación/conjunto/común → tabla; global por propiedad → lista completa; vía y sitio → las 24 fichas; superlativo → valencias; comparación de normas → diff de celdas o de catálogo. Las celdas son las fuentes. |
| `rag/citations.py` (nuevo) | Citas compartidas (cita completa y compacta de celda). |
| `rag/answer.py` / `evidence.py` | `modo: "estructurado"`; las celdas encabezan las fuentes (hasta 30); si la matriz resuelve la pregunta, está en el alcance; slot del anexo de vías y técnicas para vía y sitio sin vacuna. |
| `cli.py` | `ficha`, `comparar`, `diff` (con `--solo-cambios` y `--evaluar`). `evaluar` mide cobertura de vacunas/celdas, precisión de las listas y citas esperadas. |

**Desviaciones y decisiones tomadas:**

- No hay `entidades.py`: el catálogo y los alias ya viven en `catalogo.py` desde
  la Fase 1 (con los calificadores DT/dT/DPT, SR/SPR, hexavalente/VIH).
- El Anexo 2 entra en la matriz como campo `RESCATE`: las comparaciones de
  rescate y de edad máxima entre normas son diffs de celda, no de búsqueda.
- Las rutas estructuradas responden con plantilla **también con LLM**: es la
  palanca de §8.3 (0 llamadas) y evita que el redactor reordene o resuma celdas
  citadas. Coste: respuestas menos naturales.
- Anexo E resuelto: #5 los valores se normalizan **en la matriz**, no en el
  índice (no obliga a reindexar); #6 los punteros se resuelven por `vacuna_slug`
  + fila del Anexo 2; #7 el diff es una **tabla**; #8 `no_aplica` solo para los
  pares de `NO_APLICA`, el resto `no_consta`.
- El diff se afinó con tres reglas generales medidas en un set ciego nuevo
  (§9.3). Cualquier ajuste futuro necesita otro set nuevo: los tres actuales ya se
  han visto.

Plan original: `fichas.py` (matriz, valores canónicos, operaciones), `entidades.py`
(catálogo y alias), render de tabla/diff, comandos `comparar`, `ficha`, `diff`.

### Fase 4 — Endurecer ✅

**Hecho** (resultados en §9.3):

| Pieza | Cambio aplicado |
|---|---|
| Presupuesto de llamadas | `NTS_RAG_LLM_MAX_CALLS` (3). `_chat` rechaza cualquier llamada por encima del tope de la consulta y cuenta también las fallidas. Se reservan la redacción y la verificación; el plan usa lo que queda (con 3 y verificación: un intento, sin reintento). El re-rankeo solo entra si el plan del LLM falló **y** cabe. |
| Presupuesto de tokens | El contexto del redactor se limita por intención (§8.3: lookup 2k, edad 6k, comparación 6k, global 8k, resto 12k; 600 para el prompt). Las rutas estructuradas costaban 0 en la Fase 4; desde la Fase 5 hacen redacción + verificación. |
| Sustitución silenciosa | Regla en el veredicto: si la pregunta nombra una vacuna sin ninguna fuente propia, `parcial`/`evidencia_insuficiente` y se nombra. Métrica `sustitucion_silenciosa` en `evaluar` (también reformulación no mostrada). |
| Coste en `evaluar` | Media y máximo de llamadas y tokens por consulta. |
| Registro de consultas | `NTS_RAG_QUERY_LOG` (desactivado por defecto): una línea JSONL por `/consulta` con estado, motivo, intención, fuentes, llamadas, tokens y tiempo. `logs/` en `.gitignore`. |
| Auth | `NTS_RAG_API_KEY`: `/consulta` y `/buscar` exigen `X-API-Key` o `Authorization: Bearer` (comparación en tiempo constante); `/salud` y `/health` siguen abiertas para los healthchecks. |
| `uv.lock` | 96 paquetes fijados; `make install` = `uv sync --extra dev`, `make lock`. Verificado instalando en un entorno limpio solo desde el lockfile: ruff y 169 tests en verde. |
| CI | **Pendiente de activar.** Workflow listo en `docs/ci-pendiente.yml` (`uv sync --frozen`, ruff y pytest en Python 3.12 en cada PR y push a `main`; no necesita el modelo de embeddings). No se subió a `.github/workflows/` porque el token de GitHub de la autora no tiene permiso `workflow`; activarlo es moverlo ahí (README, "Calidad"). |
| pre-commit | `.pre-commit-config.yaml`: ruff (misma versión que el lockfile), YAML/JSON, conflictos, archivos grandes (≤ 30 MB por el índice) y pytest antes de cada push. |
| Tests | 169 (eran 51 al empezar): `test_phase4.py` cubre presupuesto, tope, tokens, sustitución, auth y registro. |

**Desviaciones y decisiones tomadas:**

- **Re-ranker local: no.** El plan lo condicionaba a que el eval lo pidiera, y con
  recall@10 0.99 y 52/52 estados no lo pide; añadiría CPU y latencia.
- El reintento del plan de §7.2 queda supeditado al presupuesto: con los valores
  por defecto (3 llamadas, verificación activa) no hay reintento y, si el plan del
  LLM no valida, se usa el determinista. Con `NTS_RAG_VERIFY=false` el reintento
  vuelve a caber.
- El presupuesto de tokens recorta el contexto, pero no marca `parcial` al
  recortar: la cobertura la garantizan los slots, que van primero.
- El CI se dejó **sin activar** en esta entrega (ver tabla). Cuando se active
  correrá en Python 3.12 (Linux). En este equipo el Python 3.12 de uv no se pudo
  instalar (fallo de uv en Windows), así que el entorno limpio se validó con 3.13.
- **Sigue sin medirse la ruta LLM con un modelo real**: plan, redacción `[cN]`,
  verificación y conteo de tokens están probados con un LLM simulado. (La Fase 5
  la midió con DeepSeek Flash.)

Plan original: tests, CI, `uv.lock`, pre-commit, log de consultas, medición de
tokens, re-ranker local si el eval lo pide, auth si se expone.

### Fase 5 — Redacción con LLM de respuestas estructuradas ✅

**Hecho** (resultados en §9.3):

| Pieza | Cambio aplicado |
|---|---|
| Redacción | `generate_answer` envía la tabla/lista de la matriz a `STRUCTURED_SYSTEM`; el LLM redacta la respuesta y conserva las citas. |
| Verificación | Se verifica contra el texto estructurado (no contra fragmentos sueltos), se comprueba que todo número de la respuesta aparezca en la tabla y que cada vacuna de una lista esté nombrada. Si algo falla, se devuelve la plantilla. |
| Listas de campo | `LIST_FIELDS` = todos los campos de ficha + `RESCATE`; tabla de las 24 fichas con `no consta`/`no aplica`. Cubre dosis, composición, conservación y dosis de rescate de cada vacuna. |
| Edad + rescate | Los 13 límites del Anexo 2 se anteponen al contexto del redactor; la pregunta “no recuperables” se responde con umbrales de edad. |
| Ruido | Se elimina el bloque “Además” de las respuestas estructuradas. |
| Citas | `short_citation` inline, deduplicación de citas idénticas y listado del CLI agrupado por sección; solo se muestran las secciones que la respuesta cita (`--fuentes` para el detalle). |
| Brevedad | El checklist usa las cláusulas literales de la pregunta (`question_parts`), no las subconsultas del LLM; una pregunta de dos partes no recibe secciones de más. |
| Tests | 191 (eran 176): redacción estructurada, respaldo por verificación/cobertura/números, listas de campo, límites de rescate, citas y fuentes. |

**Decisiones tomadas:**

- La respuesta negativa del diff de catálogo (“No existen…”) no tiene celdas que
  verificar y se mantiene determinista.
- El presupuesto sigue en 3 llamadas: plan + redacción + verificación. Sin LLM o
  sin verificación, la plantilla es la salida.
- El redactor de edad tiene 1.800 tokens de salida para que quepan todas las partes
  de la pregunta; el resto, 900.
- La comprobación numérica es conservadora: si el LLM reformula una cifra que no
  aparece literalmente en la tabla, se usa la plantilla.

Plan original: redactar las respuestas estructuradas con el LLM, verificar contra la
matriz, generalizar las listas de campo y responder los umbrales de rescate.

### Complementarias

Casos especiales 6.3.x como matriz (condición × vacuna; agrupar por condición/edad,
`vacuna_slug` como lista y flag `comodin` para "resto de vacunas"; `<br>` → lista),
simultaneidad Anexo 1, normalización numérica/unidades (`4a11m29d ≠ 5a`),
`/esquema?edad=` y `datos` estructurado. Contradicción conocida: SPR 6.2.2 (5–11
años) vs Anexo 2 (4a11m29d). Se pueden reordenar o sustituir si hay algo mejor.

---

## 11. Smoke tests del reindexado

Resultado del reindexado v2 (2026-11-16). Los marcados con *test* están en
`tests/test_smoke_index.py`; el resto se comprobó contra el índice y lo cubre
`evaluar`.

- ✅ total ≈ 1.965 ± 10%: **2.064** (1.206 base + 543 campo + 315 fila; 1.153 de
  2026 y 911 de 2022) — *test*;
- ✅ 100% con `vigencia` y `documento_id` (también en los metadatos de Chroma) — *test*;
- ✅ 0 etiquetas fuera de las 13 — *test*;
- ✅ 18 fichas 2022 / 24 fichas 2026 — *test*;
- ✅ manifiesto con `schema_version = 2` y hashes; 0 desfases — *test*;
- ✅ dos reindexados → mismos IDs (JSONL byte a byte idéntico) — *test*;
- ✅ `corpus_lint()` 100% — *test*;
- ✅ ninguna fuente 2022 citada como "NTS 246": se cita "NTS N° 196-MINSA/DGIESP-2022
  (derogada)" — *test* para los metadatos;
- ✅ consulta sin versión → 0 chunks de 2022 (42 preguntas del eval, top-20);
- ✅ "dosis de rescate BCG" → fila BCG del Anexo 2 en el puesto 2;
- ✅ ficha recuperada → 13/13 campos (12/13 en APO y rotavirus: `JERINGAS` no
  aplica) — *test*;
- ✅ 20 lookups siguen 20/20.

---

## 12. Trampas

| Riesgo | Mitigación |
|---|---|
| Partir por "y": "difteria y tétanos" es una vacuna | separar solo entre spans de entidades |
| `previa` como versión ("vacunas previas") | exigir "norma/versión previa" |
| DT pediátrico vs dT adulto vs DPT | alias con calificador |
| Numerales y páginas colisionan entre normas | alinear por slug; filtrar página con versión |
| Remisiones "Ver Anexo 2" | detectar y resolver el puntero; clasificar `movido` |
| "No consta" ≠ "eliminado" ≠ "no aplica" | estados distintos |
| Ausencia de contraindicación ≠ seguridad | decir "no consta" |
| Near-duplicates 2022/2026 | filtro de vigencia + dedup |
| Señuelo p.121 (programación) | excluir tablas operativas de la ruta de edad |
| Continuación de 1 fila | aceptar 1+ filas o fusionar en el chunker |
| Citas escritas por el LLM | citas desde metadatos; `[c1]` |
| Reformulación silenciosa | mostrar reformulación y supuestos |
| Caché obsoleta | invalidar por `schema_version` + hashes |

---

## Anexo A — Catálogo de vacunas

| `vacuna_slug` | 2026 | 2022 |
|---|---|---|
| `bcg` | 6.1.1.1 | 6.1.1.1 |
| `hvb` | 6.1.1.2 | 6.1.1.2 |
| `hexavalente` | 6.1.1.3 | — |
| `pentavalente` | 6.1.1.4 | 6.1.1.3 |
| `dt_pediatrico` | 6.1.1.5 | 6.1.1.4 |
| `hib` | 6.1.1.6 | 6.1.1.5 |
| `ipv` | 6.1.1.7 | 6.1.1.6 |
| `apo` | 6.1.1.8 | 6.1.1.7 |
| `rotavirus` | 6.1.1.9 | 6.1.1.8 |
| `neumococo` | 6.1.1.10 | 6.1.1.9 |
| `spr` | 6.1.1.11 | 6.1.1.10 |
| `sr` | 6.1.1.12 | — |
| `varicela` | 6.1.1.13 | 6.1.1.11 |
| `ama` | 6.1.1.14 | 6.1.1.12 |
| `dpt` | 6.1.1.15 | 6.1.1.13 |
| `dt_adulto` | 6.1.1.16 | 6.1.1.14 |
| `tdap` | 6.1.1.17 | 6.1.1.15 |
| `hva` | 6.1.1.18 | 6.1.1.16 |
| `vph` | 6.1.1.19 | 6.1.1.17 |
| `influenza` | 6.1.1.20 | 6.1.1.18 |
| `vrs` | 6.1.1.21 | — |
| `hexavalente_vih` | 6.1.1.22 | — |
| `meningococo` | 6.1.1.23 | — |
| `mab_vrs` | 6.1.1.24 | — (pasiva) |

Nuevas en 2026: hexavalente, sr, vrs, hexavalente_vih, meningococo, mab_vrs.
Trampas: `dt_pediatrico` vs `dt_adulto` vs `dpt`; `spr` vs `sr`; `hexavalente` vs
`hexavalente_vih`.

## Anexo B — Registro de documentos

| Campo | 2026 | 2022 |
|---|---|---|
| `documento_id` | `nts-246-2026` | `nts-196-2022` |
| `version` | 2026 | 2022 |
| `vigencia` | vigente | derogada |
| `paginas` | 140 | 103 |
| Markdown | `8265031-...-parte-1/2.md` | `nts-n-196-...-2022.md` |

## Anexo C — Set de evaluación

**Agregación (7):** niño de 5 años sin vacunas (6.2.2 + Anexo 2); qué toca a los 5
(6.2.2); adolescente 13 (6.2.3 + Anexo 2); recuperables/no recuperables (Anexo 2);
esquema incompleto de pentavalente (6.1.1.4 + Anexo 2); gestante y contraindicación
(6.2.1 + 6.1.1.11/12/14; ver nota en §9.3); rescate BCG (Anexo 2 + 6.1.1.1).

**Negativas (3):** rotavirus a los 12 meses (no procede); BCG a los 6 años (no
procede); ébola (fuera de alcance).

**Borde (4):** más cepas, mAb y gestante, vasto externo, alergia SPR/AMA/influenza
(§7.4).

Esquema:

```json
{"pregunta": "...", "tipo": "agregacion",
 "fuentes_esperadas": [{"pagina": 45, "seccion": "6.2.2"}], "clave": "..."}
{"pregunta": "...", "tipo": "borde",
 "estado_esperado": "parcial", "motivo_esperado": "conflicto"}
```

**Comparativas (3):** rescate BCG 2022 vs 2026; vacunas nuevas; edad máxima de APO.

## Anexo D — Reproducción

```bash
cd /home/rcaled/Projects/own/nts-rag

# Índice
.venv/bin/python -c "import chromadb; print(chromadb.PersistentClient(path='data/index/chroma').get_collection('nts_246').count())"

# Tokens del caso guía
.venv/bin/python - <<'PY'
from nts_rag.text import tokenize, expand_tokens
q = "¿Qué vacunas debe tener un niño de 5 años que no ha recibido vacunas previas?"
print(sorted(expand_tokens(set(tokenize(q)))))
PY

# Rango de las fuentes correctas (BM25 y vector)
.venv/bin/python - <<'PY'
from nts_rag.config import get_settings
from nts_rag.rag.retrieve import Retriever
from nts_rag.text import tokenize, expand_tokens
r = Retriever(get_settings())
q = "¿Qué vacunas debe tener un niño de 5 años que no ha recibido vacunas previas?"
tok = list(expand_tokens(set(tokenize(q))))
sc = r.bm25.get_scores(tok)
pos = {r.chunks[i]["id"]: k for k, i in enumerate(sorted(range(len(sc)), key=lambda i: sc[i], reverse=True), 1)}
v = r.collection.query(query_embeddings=[r.embedder.embed_query(q)], n_results=r.collection.count())
vpos = {cid: k for k, cid in enumerate(v["ids"][0], 1)}
for cid, label in [("406d2c2e5d81d021", "Anexo 2"), ("a1426dadea31de72", "6.2.2")]:
    print(label, "bm25=", pos.get(cid), "vec=", vpos.get(cid))
PY
```

## Anexo E — Decisiones abiertas

1. Una colección con filtro de versión (recomendado) vs dos colecciones.
2. Registro de documentos en `config.py` vs YAML/JSON.
3. Congelar los 24 slugs del Anexo A.
4. `edad_min/max` en meses vs días.
5. Normalizar valores en el índice o en la matriz.
6. Resolver punteros por `vacuna_slug` + fila.
7. Diff como tabla vs prosa.
8. Umbral de "no consta" y de `no_aplica`.
9. Alcance de la expansión por sección (siempre vs solo agregación).
10. Corpus lint: fallar en CI, avisar en local.
11. IDs: hash de texto completo vs normalizado.
12. Orden de las complementarias: casos especiales antes de `/esquema`.
