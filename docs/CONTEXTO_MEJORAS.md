# Contexto de mejoras del RAG — NTS 246 / NTS 196

> **Traspaso.** Análisis cerrado; correcciones previas aplicadas (índice de 1.004
> chunks). El esquema v2 y el reindexado siguen pendientes.
> Fecha: 2026-11-16 · Siguiente paso: Fases 0 y 1 de §10.

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

**Presupuesto por intención:** lookup 2k, edad 4k, comparación 6k, global 8k, abierta
12k, fuera de alcance 0. Si se excede, `parcial`.

**Medición:** `usage` por llamada, `--explicar` con plan/tokens/ruta y promedio por
tipo en `evaluar`.

---

## 9. Evaluación

### 9.1 Sets

- `eval/preguntas.jsonl`: 20 lookups + 5 negativas (regresión).
- `eval/agregacion.jsonl`: 7 de agregación + 3 negativas (nuevo).
- `eval/ambiguedades.jsonl`: las 4 preguntas de borde + negativas, con `estado` y
  `motivo` esperados (nuevo).

### 9.2 Métricas

`hit@k` (compatibilidad), `recall`, `completa`, `precision`, `abstiene`,
`reformulacion_util` (la reescritura recupera las fuentes esperadas) y
`sustitucion_silenciosa = 0`. Reporte **por tipo**.

### 9.3 Línea base y umbrales

Baseline v1 (994): recall@5 0.29, @10 0.36, @20 0.57; completas 1/7, 1/7, 2/7.

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

---

## 10. Plan

### Fase 0 — Evaluación

1. `eval/agregacion.jsonl` (7 + 3 negativas).
2. `eval/ambiguedades.jsonl` (4 de borde + negativas, con estado/motivo).
3. `cmd_evaluar`: recall, completa, precision, abstiene, reformulación, `--por-tipo`.
4. Registrar la línea base.

### Fase 1 — Esquema v2 + reindexado

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

### Fase 2 — Recuperación y plan

| Archivo | Cambio |
|---|---|
| `retrieve.py` | `vigencia=vigente` por defecto; filtros slug/edad; pool 40–50; dedup/MMR; boost canónico. |
| `retrieve.py` | `expand_context`: hermanos de `seccion_numero` con presupuesto. |
| `interpret.py` (nuevo) | Plan LLM + validador + presuposiciones + subconsultas. |
| `abstain.py` (nuevo) | Cobertura, conflictos y veredicto. |
| `answer.py` | Citas por documento; redacción LLM; contrato de estado. |

### Fase 3 — Matriz y comparación

`fichas.py` (matriz, valores canónicos, operaciones), `entidades.py` (catálogo y
alias), render de tabla/diff, comandos `comparar`, `ficha`, `diff`.

### Fase 4 — Endurecer

Tests, CI, `uv.lock`, pre-commit, log de consultas, medición de tokens, re-ranker
local si el eval lo pide, auth si se expone.

### Complementarias

Casos especiales 6.3.x como matriz (condición × vacuna; agrupar por condición/edad,
`vacuna_slug` como lista y flag `comodin` para "resto de vacunas"; `<br>` → lista),
simultaneidad Anexo 1, normalización numérica/unidades (`4a11m29d ≠ 5a`),
`/esquema?edad=` y `datos` estructurado. Contradicción conocida: SPR 6.2.2 (5–11
años) vs Anexo 2 (4a11m29d). Se pueden reordenar o sustituir si hay algo mejor.

---

## 11. Smoke tests del reindexado

- total ≈ 1.965 ± 10%;
- 100% con `vigencia` y `documento_id`;
- 0 etiquetas fuera de las 13;
- 18 fichas 2022 / 24 fichas 2026;
- manifiesto con `schema_version = 2` y hashes;
- dos reindexados → mismos IDs;
- `corpus_lint()` 100%;
- ninguna fuente 2022 citada como "NTS 246";
- consulta sin versión → 0 chunks de 2022;
- "dosis de rescate BCG" → fila en top-5 (ya se cumple);
- ficha recuperada → 13/13 campos;
- 20 lookups siguen 20/20.

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
(6.2.1 + 6.1.1.19); rescate BCG (Anexo 2 + 6.1.1.1).

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
