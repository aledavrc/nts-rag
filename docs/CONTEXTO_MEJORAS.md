# Contexto de mejoras del RAG — NTS 246 / NTS 196

> **Documento de traspaso.** Recoge el análisis completo, la evidencia medida y las
> decisiones ya tomadas para la siguiente iteración del proyecto. La implementación
> **no ha comenzado**: no se ha modificado código, datos ni configuración.
>
> Fecha: 2026-11-16 · Estado: análisis cerrado, claves canónicas aprobadas.
> Siguiente paso: implementar el esquema v2 y el reindexado único (sección 9).

## 0. TL;DR para el implementador

1. El fallo reportado ("no encuentra la tabla de dosis de rescate" / niño de 5 años
   sin vacunas) **no es del LLM**: es de recuperación y es estructural.
2. La causa principal es que la **edad desaparece de la tokenización** (`año`, `años`,
   `niño`, `niña` son stopwords) y las **tablas que importan no se parten en filas**
   (el field-chunking solo funciona con fichas de 2 columnas).
3. Se decide una **matriz determinista `vacuna × campo × versión`** en lugar de chunks
   agregados buscables. La comparación deja de ser una lotería de embeddings.
4. Se ingesta también la **NTS 196-2022**, con `vigencia` por documento y **filtro por
   defecto a la norma vigente**. La histórica solo entra con señal explícita.
5. **Un solo reindexado** (esquema v2) que incluye: corpus 2022, etiquetas canónicas,
   filas de dominio, `vacuna_slug`, edades normalizadas y metadatos de documento.
6. **El reindexado es obligatorio.** Ningún cambio de esquema se considera terminado
   sin `make chunks && make index` y sin pasar los smoke tests de §10.
7. Se añaden tres mejoras recomendadas: **expansión por sección** (§9, Fase 2),
   **frescura del índice + corpus lint** e **IDs estables** (§9, Fase 1). El resto
   (casos especiales, normalización numérica, contradicciones, API `/esquema`, CI,
   log de consultas, re-ranker local) queda como **mejoras complementarias** en §9.

---

## 0.1 Mandato para quien implementa

Este documento fija **objetivos y criterios de aceptación**, no una implementación
cerrada. Quien lo reciba tiene margen para mejorar el diseño:

- **Si encuentras una solución más adecuada** que la propuesta (otra estructura de
  datos, otro algoritmo de fusión, otra forma de partir tablas, otro modelo de
  embeddings), **aplícala**. No estás obligado a seguir la letra del plan.
- Condiciones para desviarte: (a) los criterios de aceptación de §8.3 se cumplen o
  mejoran; (b) mantienes las garantías de citación, abstención y "no consta";
  (c) no reintroduces la dependencia de top-k para preguntas de comparación o de
  edad; (d) documentas la decisión y el motivo en este archivo o en un ADR.
- **Debes reindexar la base de datos** con el esquema v2 antes de dar nada por
  terminado. Un cambio de código sin reindexado no es una entrega válida.
- Verifica con el set de evaluación congelado (§8.1) y registra los números. No
  ajustes el set para que pase.
- Si una decisión de este documento resulta errónea, cámbiala y explica por qué.

---

## 1. Requisitos de origen

Notas del usuario que originaron este trabajo:

**Comparativas entre normativas**
- Diferencia entre vacunas.
- Diferencias en dosis.
- Nuevas vacunas.
- Reconocer palabras: *anterior*, *antigua*, *previa*, *2022*, *2026*.

**Pesos semánticos entre palabras** (pospuesto, ver §8.4)
- virus ↔ enfermedad
- tratar ↔ proteger ↔ prevenir
- temperatura (°C) ↔ frío ↔ calor ↔ ruptura de cadena

**Preguntas múltiples**
- Separar oraciones al encontrar "y" o similares.
- Realizar múltiples consultas.

**Preguntas generales**
- Comparación entre características de diferentes vacunas.
- El grupo puede ser mencionado o global.

**Pregunta comparativa de características (caso guía)**
- "¿Qué vacunas debe tener un niño de 5 años que no ha recibido vacunas previas?
  ¿Qué se debe priorizar, qué es recuperable y qué no?"
- Implementar rangos de edad (7–12, adolescencia 13+).
- Considerar página 80: *rescate* / *recuperar* / *subsanar*.
- **"No está encontrando bien la tabla de dosis de rescate".**

---

## 2. Estado actual del proyecto

### 2.1 Arquitectura

```
corpus/markdown/*.md
  → pipelines/ocr/md_to_vector.py        (chunks base con sección y páginas)
  → data/chunks/nts-246.chunks.jsonl
  → src/nts_rag/index.py                 (normaliza + expande campos de ficha)
  → embeddings FastEmbed (ONNX) + ChromaDB (data/index/)
  → retrieve.py: vectorial + BM25 + RRF + refuerzos
  → answer.py: re-rankeo LLM opcional + respuesta + verificación de fidelidad
  → FastAPI / CLI
```

Puntos de entrada relevantes:

| Archivo | Responsabilidad |
|---|---|
| `pipelines/ocr/md_to_vector.py` | Markdown → JSONL. Tablas partidas solo si superan `--table-max-chars`. |
| `src/nts_rag/index.py` | `normalize_record`, `load_chunks`, `expand_field_chunks`, `build_index`. |
| `src/nts_rag/text.py` | `strip_accents`, `normalize_for_match`, `stem`, `tokenize`, `expand_tokens`. |
| `src/nts_rag/rag/retrieve.py` | Híbrido vectorial + BM25 + RRF y refuerzos. |
| `src/nts_rag/rag/answer.py` | Re-rankeo, prompt, citas, verificación, abstención. |
| `src/nts_rag/config.py` | `Settings` (NTS_RAG_*), `DOC_TITLE` hardcodeado. |
| `src/nts_rag/cli.py` | `indexar`, `consultar`, `buscar`, `inspeccionar`, `evaluar`, `servir`, `doctor`. |
| `Makefile` | `chunks` e `index`; `DOCS` solo lista los dos ficheros de 2026. |

### 2.2 Índice actual (v1)

| Dato | Valor |
|---|---|
| Colección | `nts_246` |
| Chunks | **994** = 562 texto + 86 tabla + 346 campo |
| Documentos | Solo **NTS 246-2026** |
| Modelo | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (FastEmbed/ONNX) |
| JSONL de entrada | `data/chunks/nts-246.chunks.jsonl` (648 registros) |
| `vector_k` / `bm25_k` / `top_k` | 12 / 12 / 6 |
| Pool real de `run_query` | `max(k*3, 12)` = 18 candidatos |
| BM25 | `BM25Okapi(corpus, b=0.3)` |

### 2.3 Contrato de datos

El JSONL usa claves en inglés (`section`, `section_number`, `type`, `text`,
`page_start`, `page_end`, `source`, `doc_title`). `normalize_record` las traduce a
claves internas (`seccion`, `tipo`, `texto`, `pagina`, `pagina_fin`, `campo`…).

Metadatos guardados en Chroma (`META_FIELDS`): `documento`, `pagina`, `pagina_fin`,
`seccion`, `seccion_corta`, `seccion_numero`, `tipo`, `campo`, `fuente`, `parte`,
`parte_total`.

**Deuda detectada:** `normalize_record` ignora `doc_title` del JSONL y fuerza
`DOC_TITLE` (`index.py:66`). Con dos documentos, todas las fuentes dirían "NTS 246".

---

## 3. Metodología del análisis

Todo se midió en **solo lectura**, con scripts ad-hoc contra el índice existente
(no se modificó nada). Los comandos de reproducción están en el Anexo E.

Se midieron cinco dimensiones:

1. **Tokenización** de la pregunta real (`tokenize` + `expand_tokens`).
2. **Rango de los chunks correctos** en cada recuperador (BM25 y vectorial, sobre 994).
3. **Cobertura** de la respuesta: ¿están todas las fuentes necesarias en top-k?
4. **Comportamiento comparativo**: conjuntos nombrados y preguntas globales.
5. **Estructura del corpus**: etiquetas, filas de tabla, fichas y numerales.

---

## 4. Hallazgos con evidencia

### 4.1 Caso guía: niño de 5 años sin vacunas previas

Pregunta:
> "¿Qué vacunas debe tener un niño de 5 años que no ha recibido vacunas previas?
> ¿Qué se debe priorizar, qué es recuperable y qué no?"

Tras `tokenize` + `expand_tokens` queda en:

```
['5', 'debe', 'deben', 'no', 'previa', 'priorizar', 'recibido', 'recuperabl', 'tener']
```

La edad desaparece: `año/años` son stopwords y `niño` también. "5" solo no
significa nada, y "no" es una de las palabras más frecuentes del corpus.

Rango de las fuentes correctas (994 chunks):

| Chunk | Contenido | BM25 | Vectorial |
|---|---|---|---|
| `a1426dadea31de72` p.45 | **6.2.2 calendario 0–11 años** | **352** | **~690** |
| `406d2c2e5d81d021` p.80 | **Anexo 2, tabla de rescate (13 filas)** | **107** | **285** |
| `fdd64fa0a6b49ec4` p.47 | 6.2.3 adolescencia | 582 | 505 |

El pool es de 12 + 12. **Ninguna entra, ni con k=20** (6.2.2 sigue ausente).

Lo que devuelve hoy (k=5): `p72, p121, p121, p7, p121`. La p.121 es "A. Esquema de
vacunación de la población menor de 5 años", que **es una tabla de programación
presupuestal** (criterios de programación, factor pérdida), no el calendario. Es un
señuelo verosímil: peor que abstenerse.

### 4.2 Incluso citando el nombre exacto de la tabla

| Consulta | Anexo 2 (BM25) | Anexo 2 (vector) |
|---|---|---|
| "dosis de rescate" | 30 | 92 |
| "esquemas de vacunación con dosis de rescate" | 20 | 297 |
| "dosis de rescate BCG" | **13** | **18** |

Con "dosis de rescate BCG" queda a **un puesto** de entrar en el pool de 12. Es el
control que demuestra que el problema no es de vocabulario, sino de granularidad y
de tamaño del pool.

### 4.3 Preguntas por edad: fallo total

`6.2.2` (el calendario que responde "qué toca a los 5 años") no aparece **ni con
k=20**. La razón: su título ("…DE LA NIÑA Y EL NIÑO (0 a 11 años…)") no aporta
tokens útiles porque `niña`, `niño`, `año`, `vacunación` son stopwords. El único
token de la sección es `11`, `mes`, `día`; la pregunta solo aporta `5`.

### 4.4 Comparación entre vacunas (conjunto nombrado)

| Pregunta | Esperado | Devuelto | Cobertura |
|---|---|---|---|
| pentavalente vs hexavalente | 6.1.1.4 + 6.1.1.3 | p121, p79, p20 (Hib) | **0/2** |
| conservación SPR vs AMA | 6.1.1.11 + 6.1.1.14 | 4 chunks de AMA + ruido | **1/2** |
| composición neumo vs VPH | 2 fichas, campo COMPOSICIÓN | 2 fichas, pero el único `campo` es INDICACIÓN | **0/2 campos** |

Patrón: una sola consulta = un solo embedding; una de las dos entidades gana.

### 4.5 Comparación global (por propiedad)

| Pregunta | Esperado | Devuelto | Cobertura |
|---|---|---|---|
| "¿qué vacunas son de virus vivos atenuados?" | 7 (BCG, APO, rotavirus, SPR, SR, varicela, AMA) | 5 de 7 + ruido | **5/7** |
| "¿qué vacunas están contraindicadas en el embarazo?" | 3 (SPR, SR, AMA) | p45, p125, p8, Tdap… | **0/3** |
| "¿qué vacunas van en muslo y cuáles en deltoides?" | 24 celdas VÍA/SITIO | Anexo 6 + 4 celdas | **4/24** |
| "¿qué vacunas se conservan entre +2 y +8 °C?" | 24 celdas | 5 fichas + ruido | parcial |

El caso "vivas atenuadas" funciona **de casualidad** porque la frase aparece literal.
Con "¿qué vacunas tienen el virus vivo?" cae.

### 4.6 Comparación entre normas

La NTS 196-2022 **no está indexada**, aunque su Markdown y su PDF están en
`corpus/`. Cualquier comparativa es imposible por construcción, no "difícil".

Datos verificados:

| Dato | 2022 | 2026 |
|---|---|---|
| Chunks base (`--table-max-chars 6000`) | 560 | 648 |
| Chunks de campo | ~226–231 | 346 |
| Fichas 6.1.1.x | **18** | **24** |
| Anexo 2 (rescate) | **p. 50** | **p. 80** |
| Calendario | 6.2.1 "<5 años" + 6.2.2 ">5 años" | 6.2.2 "0–11" + 6.2.3 adolescente + … |

**Colisión de numerales (crítica):**

| Vacuna / sección | 2022 | 2026 |
|---|---|---|
| Pentavalente | 6.1.1.**3** | 6.1.1.**4** |
| Antineumocócica | 6.1.1.**9** | 6.1.1.**10** |
| Fiebre amarilla | 6.1.1.**12** | 6.1.1.**14** |
| "Mayor de 5 años" | **6.2.2** | (6.2.2 = 0–11 años) |

Consecuencia: `section_number_in_query` y `filtros.seccion` se vuelven ambiguos
entre versiones. **La alineación debe ser por nombre/alias de vacuna, nunca por
numeral.**

### 4.7 La evaluación actual no mide esto

- `eval/preguntas.jsonl`: 20 preguntas dentro de alcance + 5 fuera.
- Todas las de alcance son **lookup de una ficha/un anexo** (una página, una
  sección).
- `cmd_evaluar` (`cli.py:170`) exige **una única** sección esperada y mide
  `Hit@k` binario.
- No hay preguntas de agregación por edad, ni de comparación nombrada, ni global,
  ni entre normas.

Por eso "Hit@1 = 20/20" convive con un fallo total en el caso de uso real.

### 4.8 Causas secundarias

- `stem` no une `debe`/`deben` ni `recuperable`/`recuperar`.
- `QUERY_SYNONYMS` no cubre: rescate/recuperar/completar, ni
  anterior/antigua/previa/2022/2026.
- "recuperable" aparece **0** veces en el corpus; "subsanar" **0**; "recuperar" 2.
  La norma dice "dosis de rescate" y "completar el esquema".
- `SECTION_GENERIC` y `GENERIC_QUERY_TERMS` eliminan justo los tokens que
  identifican la pregunta (`nino`, `ano`, `vacuna`, `esquema`).
- El índice de anexos (p.77) genera un chunk de campo "ANEXO 2 | Esquemas de
  vacunación con dosis de rescate": un **puntero**, no el contenido. Puede
  recuperarse y responderse como si fuera la tabla.

---

## 5. Causas raíz

| # | Causa | Evidencia | Impacto |
|---|---|---|---|
| 1 | La edad se borra en `tokenize` (stopwords `ano/anos/nino/nina`) y se aplica también al corpus | tokens de la pregunta = `['5', …]` | Toda pregunta por edad es irresoluble léxicamente |
| 2 | `_kv_rows` es todo-o-nada: exige 2 columnas, etiquetas en MAYÚSCULA y **todas** las filas válidas | la fila `Varicela` (minúscula) anula el Anexo 2 completo | 0 chunks de campo para las 13 vacunas |
| 3 | Solo se parten tablas de 2 columnas | 6.2.2 (3), 6.3.3 (4), Anexo 1 (5) quedan monolíticos | Los calendarios y casos especiales no son recuperables por fila |
| 4 | Tablas grandes como un único chunk | Anexo 2 = 2.690 caracteres; embedding promedio de 13 vacunas | BM25 lo penaliza por longitud; el vector lo diluye |
| 5 | Pool de 12+12 y `k*3` para el re-ranker | Anexo 2 en 107 (BM25) / 285 (vector) | El re-ranker no puede arreglar lo que no está en el pool |
| 6 | Una sola consulta, sin descomposición | "priorizar" + "recuperables" + "5 años" compiten | Los 6 slots se llenan de variantes del mismo tema |
| 7 | Sin eje de versión | 2022 no indexada; `DOC_TITLE` fijo | Comparativas imposibles; riesgo de citar norma derogada |
| 8 | Evaluación solo de lookup | 20/20 con el caso real roto | El benchmark oculta el problema |

---

## 6. Decisiones tomadas (con razonamiento)

### D1 — Evaluación primero

Se congela un set de preguntas de agregación/comparación y se mide la línea base
**antes** de tocar la recuperación. Sin esto no se puede distinguir una mejora real
de un cambio de comportamiento.

### D2 — Chunks por fila, pero solo de tablas de dominio

No se parten *todas* las tablas: las operativas (indicadores, programación,
formularios, carnés, consolidados) aportan **948 filas de ruido** frente a 728 del
dominio, y de esas 728 unas 546 ya son campos de ficha. Se parte:

- 6.2.x (calendarios por curso de vida)
- 6.3.x (casos especiales)
- Anexos 1, 2, 3, 6, 8 y 9

Se **conserva** la tabla completa además de las filas: la fila da precisión, la
tabla da contexto.

### D3 — Matriz determinista, no chunks agregados

Se descarta generar chunks sintéticos "todas las composiciones" y se construye una
matriz en memoria/derivada `vacuna × campo × versión`.

Razones:

- La comparación pasa a ser una **operación exacta** (lookup + diff), no una
  búsqueda probabilística.
- Garantiza **cobertura de todas las entidades nombradas** (hoy se pierde una).
- Garantiza **completitud en preguntas globales** (24 de 24, no top-6).
- Permite **verificación mecánica**: toda vacuna listada pertenece al conjunto
  filtrado; el valor de cada celda está en la página citada.
- Con dos normas, los near-duplicates (misma sección, dos años) degradan el top-k;
  la matriz los evita porque no compite por relevancia.

### D4 — 13 claves canónicas de campo (**APROBADO, congelar**)

Convención: clave autodescriptiva, **singular** para valor único, **plural** para
listas, sin acentos, mayúsculas y guion bajo.

| Canónico | Etiquetas de origen | Celdas |
|---|---|---|
| `TIPO_DE_INMUNIZACION` | TIPO DE VACUNA (41), TIPO DE INMUNIZACIÓN (1) | 42 |
| `DESCRIPCION` | DESCRIPCIÓN | 42 |
| `COMPOSICION` | COMPOSICIÓN | 42 |
| `PRESENTACION` | PRESENTACIÓN | 42 |
| `INDICACIONES` | INDICACIÓN (24), INDICACIONES (18) | 42 |
| `ESQUEMA_DE_VACUNACION` | ESQUEMA DE VACUNACIÓN (40), ESQUEMA (1), ESQUEMA DE VACUNACIÓN Y DOSIS (1)¹ | 42 |
| `DOSIS` | DOSIS (CANTIDAD) (24), DOSIS (18) | 42 |
| `CONSERVACION` | CONSERVACIÓN DE LA VACUNA | 42 |
| `VIA_Y_SITIO` | VÍA Y SITIO DE ADMINISTRACIÓN (41), VÍA Y SITIO DE VACUNACIÓN (1) | 42 |
| `JERINGAS` | JERINGAS A UTILIZAR (31), JERINGA A UTILIZAR (6), JERINGA (1) | 38 |
| `EFECTOS_POSTVACUNALES` | POST-VACUNALES (20), POST VACUNALES (18), POSTVACUNALES (4) | 42 |
| `CONTRAINDICACIONES` | CONTRAINDICACIÓN (21), CONTRAINDICACIONES (18), CONTRAINDICACION (3) | 42 |
| `USO_SIMULTANEO` | CON OTRAS VACUNAS (40), DE LA VACUNA (1) | 41 |

¹ compuesta: mapea a `{ESQUEMA_DE_VACUNACION, DOSIS}` o se marca `compuesto`.

Evidencia: **26 etiquetas distintas → 13 claves**, cubriendo **541 celdas**
(231 de 2022 + 310 de 2026) al **100%**.

**Por qué `EFECTOS_POSTVACUNALES` y no `EFECTOS`:**

- `EFECTOS` es ambigua dentro del propio corpus ("efectos de la cadena de frío",
  "efectos de la interrupción del esquema").
- `REACCIONES_ADVERSAS` implica causalidad; la norma dice "posibles efectos".
- `ESAVI` **nunca**: es otro concepto (6.8.1, Anexos 7 y 8, tasas y vigilancia).
- `EFECTOS_SECUNDARIOS` es coloquial y sugiere jerarquía que la norma no establece.
- La etiqueta original se conserva en `campo_original` y se puede seguir mostrando.

Efecto técnico útil: `tokenize("EFECTOS_POSTVACUNALES")` → `{efectos, postvacunal}`,
así que el boost de campo acierta con ambos términos.

### D5 — Normalización de etiquetas en el índice

Se canonicaliza en `expand_field_chunks` (index time), guardando `campo` (canónico)
y `campo_original` (fuente). Beneficia a la vez el boost de campo y la matriz.
Se añade un **test que falla** si aparece una etiqueta fuera del catálogo: red de
seguridad ante re-OCR o una norma nueva.

### D6 — Vigencia por documento y filtro por defecto

Registro de documentos (fuente única de verdad):

| id | corto | año | vigencia | páginas |
|---|---|---|---|---|
| `nts-246-2026` | NTS 246-2026 | 2026 | **vigente** | 140 |
| `nts-196-2022` | NTS 196-2022 | 2022 | **derogada** | 103 |

Cada chunk guarda `documento_id`, `documento`, `version` y `vigencia`.
Si no hay señal explícita, `vigencia == "vigente"`. Cuando llegue la 2027 se cambia
el registro, no el código.

### D7 — Señales de versión (con dos correcciones importantes)

- Histórica: `2022`, `196`, `nts 196`,
  `(norma|versión|normativa|nts)\s+(anterior|previa|antigua|derogada)`.
- Vigente: `2026`, `246`, `vigente`, `actual`.
- Comparación/novedad: `compar|diferenc|cambi|antes|ahora|nuev[ao]|se incorpor|se elimin|ya no|desde cu[aá]ndo`.
- Por defecto: **solo vigente**.

**Corrección 1 — `previa` no puede ser señal suelta.** La propia pregunta guía dice
"no ha recibido vacunas **previas**". Si `previa` activa la histórica, se rompe el
caso que se quiere arreglar. Solo cuenta con contexto (`norma/versión previa/anterior`).
Igual con `anterior` ("dosis anterior" ≠ "norma anterior"). `2022`/`196` son
inequívocos.

**Corrección 2 — las preguntas de novedad no mencionan el año.** "¿Qué vacunas son
nuevas?", "¿qué cambió en la BCG?", "¿desde cuándo se aplica la VPH a los niños?",
"¿ya no se usa la pentavalente?" no contienen "2022". Si solo se busca la vigente,
no se puede responder: la ausencia en 2022 **es** la respuesta.

### D8 — Un solo reindexado (esquema v2)

Se hace un único `chunks + index` que incluye todo. Estimación:

| Bloque | Chunks |
|---|---|
| Base (ambas normas) | 1.208 |
| Chunks de campo (incluye ~12 del índice de anexos) | 577 |
| Filas de dominio (sin fichas) | ~180 |
| **Total** | **~1.965** (±10%) |

Hoy: 994 → **~2×**. Si se partieran todas las tablas: ~3.460 (3,5×), un tercio de
ruido operativo.

El manifiesto guarda `schema_version`/`chunking_version`; `/salud` lo reporta y el
código rechaza (o advierte de) un índice v1.

### D9 — `vacuna_slug` ya, en este reindexado

Si se deja para después, hará falta **un segundo reindexado**. Incluirlo ahora
permite:

- `where={"vacuna_slug": "bcg"}` → todos los campos de una vacuna sin alias matching.
- Construir la matriz sin lógica de entidades en consulta.
- Alinear las dos normas por identidad, no por numeral.

### D10 — Semántica del diff

Clases por celda: `cambio_real`, `redaccional`, `movido`, `nuevo`, `eliminado`,
`sin_cambio`, `no_consta`.

Reglas:

- **"No consta" ≠ "eliminado".** Que 2022 no declare un campo no significa que la
  norma lo haya quitado.
- Los valores de `TIPO_DE_INMUNIZACION` se normalizan a un vocabulario cerrado
  **multi-etiqueta**: `{viva_atenuada, inactivada, conjugada, recombinante,
  toxoide, subunidad, pasiva}`. Sin esto, 18 de 24 vacunas aparecen como
  "cambiadas" cuando no cambió ninguna.
- Las remisiones ("Ver: Dosis de rescate en Anexo 2 y casos especiales en ítem 6.3")
  se clasifican como `movido` y se **resuelven al puntero** (fila del Anexo 2 por
  `vacuna_slug`) antes de comparar. Sin esto, el 100% de los diffs de
  `ESQUEMA_DE_VACUNACION` son falsos.

Ejemplo real de por qué: BCG 2022 tiene el calendario completo en la ficha; 2026
tiene solo la remisión al Anexo 2.

### D11 — Contrato de respuesta

- Tabla Markdown con filas = características y columnas = vacunas, **una cita por
  celda**.
- Celda vacía → `No consta en la ficha (p. X)`.
- Distinguir **"contraindicada" / "no recomendada" / "sin dato"**. Que una ficha no
  declare contraindicación **no significa que sea segura**; en embarazo la norma
  solo declara 3 (SPR, SR, AMA) y el resto debe remitirse a 6.2.1.
- Vacuna no mencionada → "no consta", sin sustituir.
- Abstención: "no consta en la **norma vigente**", no un "no consta" absoluto.

### D12 — Expansión por sección (recomendada)

El top-k devuelve fragmentos, pero la unidad documental que necesita una respuesta es
la **sección**: una ficha son 13 campos, el Anexo 2 son 13 filas, 6.3.3 son ~30
filas. Tras recuperar, se cargan los chunks hermanos del mismo `seccion_numero`
(con presupuesto de caracteres, dedup y orden por página).

- Coste bajo y **no requiere reindexar**.
- Corrige "me diste 1 campo de 13" y "media tabla".
- Aceptación: ficha recuperada → 13/13 campos en el contexto.

### D13 — Frescura del índice y corpus lint (recomendada)

El manifiesto guarda `schema_version` y `fuentes: {ruta: sha256}`. `doctor` compara
y avisa si el índice está desfasado. Además, `corpus_lint()` falla si una ficha no
tiene los campos núcleo, una fila no tiene página o un Anexo 2 no mapea a slug.

Razón: una respuesta plausible sobre un corpus viejo o incompleto es el peor error
posible, y hoy nada lo detecta.

### D14 — IDs estables (recomendada)

El ID actual (`sha1(source | contexto | texto[:80])` + sufijo por orden) cambia si el
OCR reordena filas, y rompe citas y evaluación. Se sustituye por
`sha1(doc_id + seccion_numero + tipo + texto completo)`, con sufijo solo en duplicados
reales. Se valida reindexando dos veces y comparando IDs.

---

## 7. Diseño objetivo

### 7.1 Pipeline v2

```
corpus/markdown/*.md (2022 + 2026)
  → chunks base (documento, versión, vigencia)
  → campos de ficha canónicos (13 claves) + campo_original + vacuna_slug
  → filas de dominio (tipo=fila, campo=primera celda, vacuna_slug, edad_min/max)
  → edades normalizadas en tokens BM25 + metadata
  → Chroma (una colección, filtro por defecto vigencia=vigente)
  → recuperación + expansión por sección (ficha/tabla completa)
  → router de intención
      ├─ lookup            → híbrido actual
      ├─ agregación edad    → filtro por rango de edad + filas
      ├─ conjunto nombrado  → matriz (entidades × facetas)
      ├─ global             → barrido exhaustivo del campo (24 celdas)
      ├─ comparación normas → matriz 3 ejes + diff
      └─ multi-parte        → descomposición + fan-out
  → respuesta con contrato de citas
  → verificación mecánica (completitud, fidelidad, no sustitución)
```

### 7.2 Metadatos por chunk (v2)

| Campo | Ejemplo | Uso |
|---|---|---|
| `documento_id` | `nts-246-2026` | filtro/cita |
| `documento` | `NTS N° 246-MINSA/DGIESP-2026` | cita |
| `version` | `2026` | eje de comparación |
| `vigencia` | `vigente` / `derogada` | filtro por defecto |
| `pagina`, `pagina_fin` | 80 | cita/filtro |
| `seccion`, `seccion_corta`, `seccion_numero` | 6.1.1.4 | cita |
| `tipo` | `texto` / `tabla` / `campo` / `fila` | filtro |
| `campo` | `EFECTOS_POSTVACUNALES` | matriz/boost |
| `campo_original` | `POSIBLES EFECTOS POST-VACUNALES` | trazabilidad |
| `vacuna_slug` | `pentavalente` | matriz/alineación |
| `edad_min_meses`, `edad_max_meses` | 60, 143 | filtro por edad |
| `parte`, `parte_total` | 1, 2 | trazabilidad |

### 7.3 Edades normalizadas (diseño)

- `extract_ages(text) -> {edad_5a, edad_0_11a, edad_4a11m29d…}` para BM25.
- `edad_min_meses` / `edad_max_meses` para **filtro numérico** en Chroma.
- La consulta "niño de 5 años" → 60 meses → `edad_min_meses <= 60 <= edad_max_meses`.
- Se aplica **solo en la ruta de calendario/edad**, no globalmente (las fichas no
  tienen rango de edad y quedarían fuera).
- Ojo: "4 años 11 meses 29 días" ≠ "5 años"; el cálculo debe ser en meses/días, no
  en años enteros.
- Para el Anexo 2, el rango se deriva de la precisión de cada fila ("hasta los 8
  meses 0 días" → max 8; "hasta los 6 años 11 meses 29 días" → max 83).

### 7.4 Matriz de fichas

- Construcción: desde los chunks `tipo=campo` (sin reindexar aparte).
- Clave: `(vacuna_slug, campo_canónico, version)`.
- Valor: `{texto, pagina, documento_id}`.
- Operaciones:
  - `ficha(vacuna, campos)` → celdas + citas;
  - `comparar(vacunas, campos)` → tabla N×M;
  - `filtrar(campo, predicado)` → lista completa (preguntas globales);
  - `diff(vacuna, campos)` → celdas 2022 vs 2026.
- Fallback: si una entidad o faceta no se resuelve, el barrido exhaustivo del campo
  (24 celdas ≈ 4.200 caracteres) en lugar de top-k.

### 7.5 Enrutamiento por intención

| Intención | Ejemplo | Motor | Garantía |
|---|---|---|---|
| lookup | "composición de la antineumocócica" | híbrido actual | top-k |
| edad | "niño de 5 años sin vacunas" | filtro de edad + filas 6.2.x/Anexo 2 | todas las filas aplicables |
| conjunto nombrado | "pentavalente vs hexavalente" | matriz × entidades × facetas | cobertura 100% |
| global por propiedad | "¿qué vacunas son vivas atenuadas?" | matriz + vocabulario canónico | lista completa |
| global libre | "¿qué vacunas van en el muslo?" | barrido exhaustivo del campo | lista completa |
| comparación de normas | "qué cambió en la BCG" | matriz 3 ejes + diff | diff con citas |
| multi-parte | "qué se prioriza y qué es recuperable" | descomposición + fan-out | una fuente por parte |

### 7.6 Verificación mecánica (lo que hace fiable la comparación)

1. **Completitud de lista**: toda vacuna del conjunto filtrado aparece.
2. **No invención**: toda vacuna listada pertenece al conjunto filtrado.
3. **Fidelidad de celda**: el valor está literalmente en la página citada.
4. **No sustitución**: no se responde por una vacuna ausente de la norma.
5. **Contaminación = 0**: una pregunta sin señal de versión no cita 2022.

---

## 8. Objetivos y criterios de aceptación

### 8.1 Set de evaluación nuevo

Archivo nuevo `eval/agregacion.jsonl` (no rompe `eval/preguntas.jsonl`). Esquema
retrocompatible:

```json
{"pregunta": "...", "tipo": "agregacion",
 "fuentes_esperadas": [{"pagina": 45, "seccion": "6.2.2"},
                       {"pagina": 80, "seccion": "ANEXO 2"}],
 "clave": "..."}
```

`tipo ∈ {lookup, agregacion, comparativa, negativa}`. `bloqueada_por` opcional
(p. ej. `corpus_2022`) para no contar como fallo lo que aún no es posible.

**Métricas por pregunta:**

- `hit@k` (compatibilidad);
- `recall` = esperadas encontradas / esperadas;
- `completa` = 1 solo si están todas;
- `precision` = esperadas / devueltas (que subir `k` no salga gratis);
- `abstiene` para negativas.

**Reporte por tipo**, nunca un único número global.

### 8.2 Línea base medida

7 preguntas de agregación:

| Pregunta | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|
| Niño de 5 años sin vacunas | 0.00 | 0.00 | 0.50 |
| Qué toca a los 5 años | 0.00 | 0.00 | **0.00** |
| Adolescente de 13 años | 0.50 | 0.50 | 0.50 |
| Recuperable vs no recuperable | 0.00 | 0.00 | 1.00 |
| Esquema incompleto pentavalente | 0.00 | 0.50 | 0.50 |
| Gestante: qué sí y qué no | 0.50 | 0.50 | 0.50 |
| Rescate BCG (control) | 1.00 | 1.00 | 1.00 |
| **Media** | **0.29** | **0.36** | **0.57** |
| **Completas** | **1/7** | **1/7** | **2/7** |

Comparaciones: 0/2, 1/2, 5/7, 0/3, 4/24 (ver §4.4–4.5).

### 8.3 Umbrales por fase

| Fase | Métrica | Objetivo |
|---|---|---|
| Baseline | recall@6 agregación | 0.29 |
| Fase 1 | recall@10 agregación | ≥ 0.80 |
| Fase 1 | 6.2.2 presente en toda pregunta por edad | 100% |
| Fase 1 | lookups del eval | siguen 20/20 |
| Fase 1 | índice fresco (hash de fuentes) | **0 desfases** |
| Fase 1 | corpus lint (invariantes de §9) | **100%** |
| Fase 1 | IDs estables (dos reindexados seguidos) | **idénticos** |
| Fase 2 | `contaminación` | **0** |
| Fase 2 | expansión de sección (ficha completa) | **13/13 campos** |
| Fase 3 | cobertura de conjunto nombrado | 2/2, 2/2 |
| Fase 3 | completitud global | 7/7, 3/3, 24/24 |
| Fase 3 | precisión del diff | ≥ 0.90 |
| Fase 3 | nuevas/eliminadas | exacto (6 nuevas) |
| Fase 3 | contradicciones detectadas (SPR) | **1/1 con ambas citas** |

### 8.4 Objetivo pospuesto explícitamente

Los pesos semánticos difusos (virus–enfermedad, tratar–proteger–prevenir,
temperatura–frío–cadena) quedan para el final. Mucho esfuerzo, poco retorno frente
a los puntos 1–4. Mejor un mapa explícito y pequeño que "pesos" implícitos.

---

## 9. Plan de implementación (ordenado)

### Fase 0 — Evaluación (antes de tocar recuperación)

1. Crear `eval/agregacion.jsonl` con las 7 preguntas de agregación + 3 negativas.
2. Extender `cmd_evaluar` (`cli.py:170`): `recall`, `completa`, `precision`,
   `abstiene`, `--por-tipo`, `--con-llm` (end-to-end).
3. Registrar la línea base en el commit.

**Negativas propuestas** (hoy probablemente responden de más):

- "¿Cuántas dosis de rescate de rotavirus corresponden a los 12 meses?"
  (correcto: no procede; máximo 8 meses).
- "¿Qué dosis de rescate de BCG a los 6 años?" (correcto: no procede).
- "¿Cuál es el esquema de la vacuna contra el ébola?" (fuera de alcance).

### Fase 1 — Esquema v2 + reindexado único

| Archivo | Cambio |
|---|---|
| `pipelines/ocr/md_to_vector.py` | **IDs estables**: `sha1(doc_id + seccion_numero + tipo + texto completo)` y sufijo solo en duplicados reales (hoy usa `texto[:80]` + orden, lo que rompe citas y eval si el OCR reordena). Revisar `split_table` y `--table-max-chars`. El corte por filas vive en `index.py`. |
| `src/nts_rag/config.py` | Registro `DOCUMENTS` (id, título, año, vigencia, páginas). Eliminar `DOC_TITLE` como fuente de verdad. `index_schema_version = 2`. |
| `src/nts_rag/index.py` | `normalize_record`: resolver documento desde `source`/`doc_title`, añadir `documento_id`, `version`, `vigencia`, `vacuna_slug`, `schema_version`. `_kv_rows` → parser de N columnas; etiquetas canónicas + `campo_original`. `expand_row_chunks` para filas de dominio (`tipo=fila`) y edades (`edad_min/max`). **Manifiesto con `fuentes: {ruta: sha256}`** y `corpus_lint()` que falla si una ficha no tiene los campos núcleo, una fila no tiene página o un Anexo 2 no mapea a slug. |
| `src/nts_rag/text.py` | Sacar `ano/anos` de stopwords (o sustituir por tokens de edad). `extract_ages`. Ampliar `QUERY_SYNONYMS` (rescate/recuperar/completar; anterior/antigua/2022/2026 con contexto). `FIELD_INDICATORS` → claves canónicas + `reaccion`, `adverso`, `secundario`, `postvacunal`. |
| `src/nts_rag/schemas.py` | `Fuente.documento` real, `vigencia`; request `alcance` (`vigente`/`historica`/`ambas`). |
| `src/nts_rag/cli.py` | `doctor` compara el `sha256` de los Markdown con el manifiesto y avisa si el índice está desfasado. |
| `Makefile` | `DOCS` incluye `corpus/markdown/nts-n-196-minsa-digiesp-2022.md`. |

**Reindexado obligatorio:** un solo `make chunks && make index`, ejecutado de verdad
antes de cerrar la fase. Verificación en §10. Repetir el reindexado dos veces debe
producir **los mismos IDs** (determinismo).

### Fase 2 — Recuperación

| Archivo | Cambio |
|---|---|
| `retrieve.py` | `_where` con `vigencia=vigente` por defecto; filtros `vacuna_slug` y rango de edad; subir `vector_k`/`bm25_k` a ~40–50; dedup/MMR por `(seccion, campo)`; boost de campo con claves canónicas. |
| `retrieve.py` (nuevo `expand_context`) | **Expansión por sección**: tras el top-k, cargar los chunks hermanos del mismo `seccion_numero` (ficha completa de 13 campos, tabla completa, 6.3.3 completo) con presupuesto de caracteres, dedup y orden por página. Corrige "me diste 1 campo de 13" y "media tabla" sin reindexar. |
| `router.py` (nuevo) | Intención + alcance de versión + descomposición en subconsultas por oración y por "y"/"¿" (solo **entre** spans de entidades, ver §11). |
| `answer.py` | Citas por documento (`[NTS 196-2022, p. 50]`); contexto multi-fuente (incluye la expansión de sección); abstención "en la norma vigente". |

**Aceptación de la expansión:** si el top-k trae un campo de una ficha, el contexto
final debe incluir los 13 campos de esa ficha; si trae una fila del Anexo 2, las
filas de esa tabla presentes en el documento.


### Fase 3 — Matriz, comparación y diff

| Archivo | Cambio |
|---|---|
| `fichas.py` (nuevo) | `FichaMatrix` desde chunks `campo`; vocabulario canónico de valores; `ficha`, `comparar`, `filtrar`, `diff`. |
| `entidades.py` (nuevo) | Catálogo de 24 `vacuna_slug`, alias, detección de spans (Anexo B). |
| `answer.py` | Render de tabla comparativa y de diff; contrato de "no consta". |
| `cli.py` | Comandos `comparar`, `ficha`, `diff`. |

### Fase 4 — Endurecer

- Tests de etiquetas desconocidas, contaminación, completitud, fidelidad.
- Métricas en CI con umbrales de §8.3.
- Reevaluar el modelo de embeddings (`bge-m3`/`multilingual-e5`) **solo si** tras
  Fase 1–3 sigue flojo; antes ocultaría el problema de chunking.

### Mejoras complementarias (alto valor, no bloqueantes)

No son necesarias para cerrar el caso guía, pero cubren puntos ciegos importantes.
Se pueden reordenar o sustituir si el implementador encuentra algo mejor (§0.1).
Detalle en el Anexo I.

| Mejora | Qué resuelve | Coste | Fase | Aceptación |
|---|---|---|---|---|
| **Casos especiales como matriz** (6.3.x) | `condición × vacuna → conducta`; el caso guía pide contraindicación y uso simultáneo | Medio | 3 | 8 condiciones agrupadas; `vacuna_slug` lista + flag `comodin` |
| **Simultaneidad** (Anexo 1) | "¿SPR y varicela el mismo día?" | Bajo | 3 | 1 matriz antígeno × intervalo |
| **Normalización numérica/unidades** | `0,5 ml` vs `0.5 ml`; `+2 °C a +8 °C`; `4a11m29d` vs `5a` | Medio | 1/3 | 100% de dosis y temperaturas parseadas; diff sin falsos por puntuación |
| **Detección de contradicciones** | SPR: 6.2.2 dice 5–11 años y Anexo 2 hasta 4a11m29d | Medio | 3 | conflicto marcado con ambas citas |
| **`/esquema?edad=` + `datos` estructurado** | respuesta determinista del caso guía, sin depender del LLM | Medio | 3 | 60 meses → calendario + recuperables + no recuperables |
| **Calibración de abstención** | negativas con vocabulario compartido (rotavirus a los 12 meses) | Bajo | 2/3 | 5/5 negativas se abstienen |
| **CI + `uv.lock` + pre-commit** | reproducibilidad entre máquinas | Bajo | 4 | ruff + pytest en cada push |
| **Log de consultas + feedback** | crecer el eval con fallos reales | Bajo | 4 | JSONL sin datos personales |
| **Re-ranker cross-encoder local** | sustituir el re-rankeo por LLM (coste, JSON frágil) | Medio | 4 | iguala o mejora recall sin API |
| **Auth** (bearer + CORS + rate limit) | exposición fuera de localhost | Bajo | 4 | solo si se despliega |

---

## 10. Verificación del reindexado (smoke tests)

**El reindexado es obligatorio.** Estos checks se ejecutan sobre el índice v2 real,
no sobre mocks.

- total ≈ 1.965 ± 10%;
- **100%** de chunks con `vigencia` y `documento_id`;
- **0** etiquetas de campo fuera del catálogo de 13 (falla, no warning);
- 18 fichas de 2022 y 24 de 2026;
- manifiesto con `schema_version = 2` y `fuentes: {ruta: sha256}`;
- `doctor` no reporta desfase entre Markdown y manifiesto;
- **dos reindexados consecutivos producen los mismos IDs**;
- `corpus_lint()` pasa al 100%;
- ninguna fuente de 2022 citada como "NTS 246";
- consulta sin señal de versión → **0** chunks de 2022 (test de contaminación);
- "dosis de rescate BCG" → fila del Anexo 2 en top-5 (hoy puesto 13 BM25);
- expansión de sección: una ficha recuperada trae **13/13** campos;
- los 20 lookups del eval siguen 20/20.

---

## 11. Trampas y riesgos

| Riesgo | Detalle | Mitigación |
|---|---|---|
| **Partir por "y"** | "Difteria **y** tétanos" es *una* vacuna (6.1.1.5/16); "Sarampión **y** rubeola" es *una* (6.1.1.12); "Sarampión, paperas **y** rubeola" es otra (6.1.1.11) | Detectar primero spans de vacunas conocidas y separar **entre** spans, nunca dentro |
| **`previa` como señal de versión** | "vacunas previas" = dosis anteriores | Solo `norma/versión previa`; nunca suelta |
| **DT pediátrico vs dT adulto vs DPT** | 6.1.1.5 vs 6.1.1.16 vs 6.1.1.15 | Alias con calificador de edad; match exacto de sigla primero |
| **Numerales colisionan entre normas** | 2022 6.2.2 = ">5 años"; 2026 6.2.2 = "0–11" | Alinear por `vacuna_slug`, no por numeral |
| **Página colisiona** | p.50 = Anexo 2 en 2022, otra cosa en 2026 | Filtro de página siempre con versión |
| **Remisiones a anexos** | 2026 movió el esquema a Anexo 2/6.3 | Detectar `Ver:`/`Anexo N`/`6.3` y resolver al puntero; clasificar `movido` |
| **"No consta" ≠ "eliminado"** | 2022 no declara JERINGAS en varias fichas | Estado `no_consta`, no `eliminado` |
| **Ausencia ≠ seguridad** | La ficha no declara contraindicación en embarazo | Decir "no consta", no "apta" |
| **Near-duplicates entre normas** | Misma sección en 2022 y 2026 ocupa los 12+12 slots | Filtro de vigencia por defecto + dedup |
| **Señuelo p.121** | Tabla de programación presupuestal parece calendario | Excluir tablas operativas de la ruta de edad |
| **Chunks agregados** | Reintroducen la lotería del embedding | No usarlos para comparar |
| **`DOC_TITLE` hardcodeado** | `index.py:66`, `answer.py:67,103` | Resolver por registro de documentos |
| **`.env` con API key real** | Está en `.gitignore`; no commitear ni loguear | No incluir en commits ni en la documentación |

---

## 12. Qué NO se ha hecho

- No se ha modificado código, datos, configuración ni tests.
- No se ha reindexado.
- No se ha ingestado la NTS 196-2022 en el índice.
- No se ha cambiado el esquema de chunks ni las etiquetas.
- Las únicas modificaciones de esta iteración son **este documento**.

Ya en el repositorio (commits previos):

- `3bad6b5`: corpus de la NTS 196-2022 (`corpus/markdown/...2022.md` y
  `corpus/pdf/...2022.pdf`).
- `ad8a2e5`: comando `inspeccionar` (`Makefile`, `README.md`, `docs/COMANDOS.org`,
  `src/nts_rag/cli.py`).
- `7691649`: primera versión de este documento.

El corpus 2022 está versionado pero **no indexado**: la ingesta en Chroma sigue
pendiente y forma parte de la Fase 1.

---

## Anexo A — Mapa completo de etiquetas

| Etiqueta observada | Ocurrencias | Canónico |
|---|---|---|
| COMPOSICIÓN | 42 | `COMPOSICION` |
| CONSERVACIÓN DE LA VACUNA | 42 | `CONSERVACION` |
| CONTRAINDICACIÓN | 21 | `CONTRAINDICACIONES` |
| CONTRAINDICACIONES | 18 | `CONTRAINDICACIONES` |
| CONTRAINDICACION | 3 | `CONTRAINDICACIONES` |
| DESCRIPCIÓN | 42 | `DESCRIPCION` |
| DOSIS | 18 | `DOSIS` |
| DOSIS (CANTIDAD) | 24 | `DOSIS` |
| ESQUEMA | 1 | `ESQUEMA_DE_VACUNACION` |
| ESQUEMA DE VACUNACIÓN | 40 | `ESQUEMA_DE_VACUNACION` |
| ESQUEMA DE VACUNACIÓN Y DOSIS | 1 | `ESQUEMA_DE_VACUNACION` + `DOSIS` (compuesto) |
| INDICACIÓN | 24 | `INDICACIONES` |
| INDICACIONES | 18 | `INDICACIONES` |
| JERINGA | 1 | `JERINGAS` |
| JERINGA A UTILIZAR | 6 | `JERINGAS` |
| JERINGAS A UTILIZAR | 31 | `JERINGAS` |
| POSIBLES EFECTOS POST VACUNALES | 18 | `EFECTOS_POSTVACUNALES` |
| POSIBLES EFECTOS POST-VACUNALES | 20 | `EFECTOS_POSTVACUNALES` |
| POSIBLES EFECTOS POSTVACUNALES | 4 | `EFECTOS_POSTVACUNALES` |
| PRESENTACIÓN | 42 | `PRESENTACION` |
| TIPO DE INMUNIZACIÓN | 1 | `TIPO_DE_INMUNIZACION` |
| TIPO DE VACUNA | 41 | `TIPO_DE_INMUNIZACION` |
| USO SIMULTÁNEO CON OTRAS VACUNAS | 40 | `USO_SIMULTANEO` |
| USO SIMULTÁNEO DE LA VACUNA | 1 | `USO_SIMULTANEO` |
| VÍA Y SITIO DE ADMINISTRACIÓN | 41 | `VIA_Y_SITIO` |
| VÍA Y SITIO DE VACUNACIÓN | 1 | `VIA_Y_SITIO` |

## Anexo B — Catálogo de vacunas (borrador para congelar)

| `vacuna_slug` | 2026 | 2022 |
|---|---|---|
| `bcg` | 6.1.1.1 | 6.1.1.1 |
| `hvb` | 6.1.1.2 | 6.1.1.2 |
| `hexavalente` | 6.1.1.3 | — |
| `pentavalente` | 6.1.1.4 | 6.1.1.3 |
| `dt_pediatrico` | 6.1.1.5 | 6.1.1.4 (TOXOIDE DIFTOTÉTANO PEDIÁTRICO) |
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
| `mab_vrs` | 6.1.1.24 | — (no es vacuna: inmunización pasiva) |

Nuevas en 2026: `hexavalente`, `sr`, `vrs`, `hexavalente_vih`, `meningococo`,
`mab_vrs`.

Trampas de alias: `dt_pediatrico` (6.1.1.5) vs `dt_adulto` (6.1.1.16) vs `dpt`
(6.1.1.15); `spr` (sarampión, paperas y rubeola) vs `sr` (sarampión y rubeola);
`hexavalente` vs `hexavalente_vih`.

## Anexo C — Registro de documentos

| Campo | 2026 | 2022 |
|---|---|---|
| `documento_id` | `nts-246-2026` | `nts-196-2022` |
| `documento` | NTS N° 246-MINSA/DGIESP-2026 | NTS N° 196-MINSA/DGIESP-2022 |
| `version` | 2026 | 2022 |
| `vigencia` | vigente | derogada |
| `paginas` | 140 | 103 |
| Fuentes Markdown | `8265031-...-parte-1.md`, `...-parte-2.md` | `nts-n-196-minsa-digiesp-2022.md` |

## Anexo D — Baseline de recuperación

Medido con `n_results=994`, `k=12/12`, modelo actual. "Rango" = posición en el
recuperador (1 = mejor).

| Consulta | Objetivo | BM25 | Vector |
|---|---|---|---|
| Pregunta guía | Anexo 2 (p80) | 107 | 285 |
| Pregunta guía | 6.2.2 (p45) | 352 | ~690 |
| "dosis de rescate" | Anexo 2 | 30 | 92 |
| "dosis de rescate BCG" | Anexo 2 | 13 | 18 |
| "esquemas de vacunación con dosis de rescate" | Anexo 2 | 20 | 297 |

## Anexo E — Comandos de reproducción

```bash
cd /home/rcaled/Projects/own/nts-rag

# 1) Estado del índice
.venv/bin/python -c "import chromadb; c=chromadb.PersistentClient(path='data/index/chroma').get_collection('nts_246'); print(c.count())"

# 2) Tokenización de la pregunta guía
.venv/bin/python - <<'PY'
from nts_rag.text import tokenize, expand_tokens
q = "¿Qué vacunas debe tener un niño de 5 años que no ha recibido vacunas previas?"
print(sorted(expand_tokens(set(tokenize(q)))))
PY

# 3) Rango de un chunk en BM25 y en vectorial
.venv/bin/python - <<'PY'
from nts_rag.config import get_settings
from nts_rag.rag.retrieve import Retriever
from nts_rag.text import tokenize, expand_tokens
s = get_settings(); r = Retriever(s)
q = "¿Qué vacunas debe tener un niño de 5 años que no ha recibido vacunas previas?"
tok = list(expand_tokens(set(tokenize(q))))
sc = r.bm25.get_scores(tok)
order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
pos = {r.chunks[i]["id"]: k for k, i in enumerate(order, 1)}
emb = r.embedder.embed_query(q)
v = r.collection.query(query_embeddings=[emb], n_results=994, include=["metadatas"])
vpos = {cid: k for k, cid in enumerate(v["ids"][0], 1)}
for target, label in [("406d2c2e5d81d021", "Anexo 2"), ("a1426dadea31de72", "6.2.2")]:
    print(label, "bm25=", pos.get(target), "vec=", vpos.get(target))
PY

# 4) Por qué el Anexo 2 no se parte en campos
.venv/bin/python - <<'PY'
from nts_rag.config import get_settings
from nts_rag.rag.retrieve import Retriever
from nts_rag.index import _kv_rows
r = Retriever(get_settings())
chunk = r.by_id["406d2c2e5d81d021"]
print("filas detectadas:", _kv_rows(chunk["texto"]))
print("chunks de la tabla:", [c["id"] for c in r.chunks if c["id"].startswith("406d2c2e")])
PY

# 5) Conteo de filas por bloque (reindexado)
.venv/bin/python - <<'PY'
import importlib.util, re
from collections import Counter
spec = importlib.util.spec_from_file_location("m", "pipelines/ocr/md_to_vector.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from nts_rag.index import normalize_record
DOM = re.compile(r"^(6\.1\.1\.|6\.2\.|6\.3\.|ANEXO ?(1|2|3|6|8|9)\b)", re.I)
tot = Counter()
for f in ["corpus/markdown/nts-n-196-minsa-digiesp-2022.md",
          "corpus/markdown/8265031-nts-n-246-minsa-digiesp-2026-parte-1.md",
          "corpus/markdown/8265031-nts-n-246-minsa-digiesp-2026-parte-2.md"]:
    ch, _ = m.make_chunks_for_file(f, 1800, 0, False, False, 6000)
    for c in [normalize_record(x) for x in ch]:
        if c["tipo"] != "tabla":
            continue
        lines = [l for l in c["texto"].splitlines() if l.lstrip().startswith("|")]
        body = [l for l in lines if not re.match(r"^\|[\s:\-|]+\|$", l.strip())]
        n = max(0, len(body) - 1)
        sec = c["seccion_corta"] or c["seccion"]
        tot["DOMINIO" if DOM.match(sec) else "OPERATIVO"] += n
print(dict(tot))
PY
```

## Anexo F — Preguntas de evaluación propuestas

**Agregación (7):**

| Pregunta | Fuentes esperadas |
|---|---|
| ¿Qué vacunas debe tener un niño de 5 años sin vacunas previas? ¿Qué priorizar y qué es recuperable? | 6.2.2 (p45) + Anexo 2 (p80) |
| ¿Qué vacunas corresponden a un niño de 5 años? | 6.2.2 |
| ¿Qué vacunas corresponden a un adolescente de 13 años? | 6.2.3 + Anexo 2 |
| ¿Qué vacunas son recuperables en un niño de 5 años y cuáles ya no? | Anexo 2 |
| ¿Qué se hace si un niño de 2 años tiene el esquema incompleto de pentavalente? | 6.1.1.4 + Anexo 2 |
| ¿Qué vacunas debe recibir una gestante y cuál está contraindicada? | 6.2.1 + 6.1.1.19 |
| ¿Hasta qué edad se puede aplicar la dosis de rescate de la BCG? (control) | Anexo 2 + 6.1.1.1 |

**Negativas (3):**

| Pregunta | Respuesta correcta |
|---|---|
| ¿Cuántas dosis de rescate de rotavirus a los 12 meses? | No procede (máximo 8 meses) |
| ¿Qué dosis de rescate de BCG a los 6 años? | No procede |
| ¿Cuál es el esquema de la vacuna contra el ébola? | Fuera de alcance |

**Comparativas (cuando la 2022 esté indexada):**

| Pregunta | Fuentes |
|---|---|
| ¿Qué cambió en la dosis de rescate de la BCG entre 2022 y 2026? | Anexo 2 de ambas |
| ¿Cuáles son las vacunas nuevas en la 2026 respecto de la 2022? | catálogo (6) |
| ¿Cuál es la diferencia de edad máxima para la APO entre ambas normas? | Anexo 2 de ambas |

## Anexo G — Decisiones abiertas (para el siguiente modelo)

1. **Una colección con filtro de versión** (recomendada) **vs dos colecciones**.
2. **Ubicación del registro de documentos**: `config.py` vs `docs.yaml`/JSON.
3. **Catálogo de slugs**: congelar los 24 del Anexo B (recomendado) o derivarlos del
   título en tiempo de indexado.
4. **`edad_min/max` en meses**: ¿basta, o se necesitan días para 4a11m29d vs 5a0d?
5. **Normalización de valores** (`TIPO_DE_INMUNIZACION`) en el índice o en la matriz.
6. **Resolución de punteros** a Anexo 2/6.3: por `vacuna_slug` + fila.
7. **Formato del diff**: tabla por vacuna/campo vs prosa con referencias.
8. **Umbral de "no consta"**: ¿solo cuando la celda está vacía, o también cuando no
   hay campo en esa versión?
9. **Negativas**: validar que el sistema se abstiene y no devuelve un fragmento
   relacionado.
10. **Alcance de la expansión por sección**: ¿siempre, o solo en intenciones de
    agregación/comparación? (coste de contexto frente a completitud).
11. **Corpus lint**: ¿falla el build o solo avisa? (recomendado: fallar en CI,
    avisar en local).
12. **IDs**: ¿hash del texto completo o de una versión normalizada? (el completo es
    más simple; el normalizado es más estable ante cambios de formato).
13. **Orden de las complementarias**: ¿casos especiales antes o después de la API
    `/esquema`? El caso guía sugiere resolver antes los casos especiales.

## Anexo H — Glosario mínimo

- **Ficha**: bloque 6.1.1.x con la tabla `Característica | Descripción`.
- **Campo**: una fila de la ficha (p. ej. `COMPOSICION`).
- **Fila de dominio**: fila de una tabla de calendario/casos especiales/anexo.
- **Conjunto nombrado**: comparación de vacunas mencionadas explícitamente.
- **Global**: comparación por propiedad sin nombrar vacunas ("vivas atenuadas").
- **Vigencia**: `vigente`/`derogada` por documento.
- **Contaminación**: citar la norma histórica en una pregunta de la vigente.

---

## Anexo I — Notas para las mejoras complementarias

Estas mejoras no cambian las decisiones D1–D11 ni el reindexado obligatorio. Se
documentan aquí para que el implementador no tenga que redescubrir la estructura.

### I.1 Casos especiales (6.3.1–6.3.4)

Estructura observada en 6.3.3: 4 columnas
(`POBLACIÓN OBJETIVO | EDAD | VACUNA | DOSIS Y VÍA DE ADMINISTRACIÓN`), ~30 filas,
con la población y la edad **repetidas** en cada fila. La columna `VACUNA` no
siempre contiene una vacuna:

- "Influenza y las demás vacunas según esquema".
- "Resto de Vacunas" → comodín, no una vacuna concreta.
- "Según valoración médica" → conducta, no vacuna.

Modelo propuesto:

- entidad `condicion` + `edad` + lista de `(vacuna_slug, conducta)`;
- condiciones detectadas: prematuro/bajo peso/VIH, contacto TB, hemodiálisis,
  convivientes inmunodeprimidos, esplenectomizados, oncohematológicos, TPH y
  trasplante de órganos sólidos;
- `vacuna_slug` como **lista** en la metadata y flag `comodin` para "resto";
- agrupar por `(condición, edad)` para no generar ~30 near-duplicates;
- las celdas con `<br>` deben partirse en lista: `clean_text` los convierte en
  espacios y pierde la estructura.

Aceptación: "niño de 5 años esplenectomizado" devuelve sus vacunas y dosis con cita.

### I.2 Simultaneidad (Anexo 1)

Tabla de 5 columnas (`TIPOS DE ANTÍGENOS | VACUNAS | INTERVALO ENTRE DOSIS |
OBSERVACIONES | PRECAUCIONES`). Es una matriz pequeña: vivos atenuados (intervalo
mínimo de 1 mes si no se administran simultáneamente) frente a inactivados
(cualquier intervalo). Responde "¿puedo poner SPR y varicela el mismo día?" y
alimenta la parte de "uso simultáneo" del caso guía.

### I.3 Normalización numérica y de unidades

Casos observados: `0.5 ml`, `0,5 ml`, `0.5 ml,`, `1,5 ml`, `+2 °C a +8 °C`,
`entre 2 y 8 °C`, `1,500g`, `4 años 11 meses 29 días`.

Propuesta: `parse_value(text) -> {tipo: dosis|temperatura|edad|peso, valor o rango,
unidad}` y guardar en metadata (`valor_num`, `unidad`) **solo cuando el parseo es
inequívoco**; si no, dejarlo nulo y conservar el texto. Habilita filtros por rango
y un diff sin ruido de puntuación.

Ojo: `4a11m29d` = 59,97 meses ≠ `5 años`. El cálculo debe ser en meses/días.

### I.4 Contradicciones entre fuentes

Ejemplo verificado: **SPR**. 6.2.2 (p. 46) dice "5 a 11 años: dosis única… en zonas
de brote"; Anexo 2 (p. 80) dice "hasta los 4 años 11 meses 29 días". No es un error
de OCR: son fuentes distintas con alcance distinto.

Regla: cuando dos fuentes del mismo documento dan valores distintos para el mismo
`(vacuna_slug, campo)`, marcar `conflicto` y mostrar **ambas** con su cita. No
resolver automáticamente. Añadir al eval una pregunta que exija las dos fuentes.

### I.5 API `/esquema` y salida estructurada

`POST /esquema {edad_meses, vacunado, condiciones[]}` →
`{calendario[], recuperables[], no_recuperables[], casos_especiales[], advertencias[]}`
con cita por ítem. Determinista: no usa el LLM salvo para redactar.

En `/consulta`, añadir `datos` a la respuesta:

```json
{"filas": ["TIPO_DE_INMUNIZACION", "DOSIS"],
 "columnas": ["pentavalente", "hexavalente"],
 "celdas": [{"vacuna": "pentavalente", "campo": "DOSIS",
             "valor": "0.5 ml", "pagina": 17,
             "seccion": "6.1.1.4", "documento": "nts-246-2026"}]}
```

El Markdown es una vista; el JSON es el contrato para tests y UI.

### I.6 Abstención por slots

`is_answerable` es hoy solapamiento de términos. Propuesta: extraer slots
(`vacuna`, `campo`, `edad`, `población`, `versión`) y comprobar que el conjunto
recuperado los cubre. Sin ningún slot reconocible → abstenerse. Con slots pero sin
cobertura → "no consta en la norma vigente". Evaluar con las 5 negativas actuales
más las 3 nuevas.

### I.7 Ingeniería y operación

- **CI**: `.github/workflows/ci.yml` con `uv sync --dev`, `ruff check`, `pytest`.
  El eval completo puede ser un job manual por la descarga del modelo.
- **`uv.lock`**: fijar transitivas de `chromadb` y `fastembed`.
- **pre-commit**: `ruff`.
- **Log**: JSONL `{ts, consulta, ids, modo, abstención, latencia}`; script que
  convierta fallos reales en candidatos de eval.
- **Re-ranker local**: `TextCrossEncoder` de FastEmbed (p. ej.
  `BAAI/bge-reranker-base`); medir contra el re-rankeo por LLM.
- **Auth**: bearer + CORS + rate limit solo si la API sale de localhost.
