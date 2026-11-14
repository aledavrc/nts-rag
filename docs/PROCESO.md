# Proceso de conversión — NTS N° 246 -MINSA/DGIESP-2026

Registro de decisiones, método y verificaciones para convertir dos PDF escaneados
en los archivos `corpus/markdown/8265031-nts-n-246-minsa-digiesp-2026-parte-1.md`
y `...-parte-2.md`.

---

## 1. Objetivo

Pasar el contenido de dos PDF a Markdown con estas prioridades:

1. **Fidelidad** al texto original (incluidas erratas).
2. **Trazabilidad**: saber de qué página proviene cada fragmento.
3. **Tablas utilizables** en una base vectorial (RAG).
4. **Estructura navegable** (capítulos, secciones, fichas).

---

## 2. Diagnóstico inicial

| Dato                 | Valor                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------|
| PDF parte 1          | 97 páginas                                                                                              |
| PDF parte 2          | 43 páginas (numeración impresa 98–140)                                                                  |
| Capa de texto        | **Ninguna** (páginas escaneadas)                                                                        |
| Resolución de origen | ~150 dpi (imagen 1754×1240)                                                                             |
| Orientación          | Páginas A4 verticales con `/Rotate 270`; algunas páginas con contenido apaisado                         |
| Contenido            | Texto legal, definiciones, 24 fichas de vacunas, tablas de esquemas, anexos, formularios y bibliografía |

Consecuencia: era necesario **OCR** o lectura visual. Las tablas y los
superíndices descartaban un OCR lineal simple.

---

## 3. Decisiones de formato

| Decisión                        | Alternativas                              | Criterio                                    | Elección                                                 |
|---------------------------------|-------------------------------------------|---------------------------------------------|----------------------------------------------------------|
| Marcador de página              | comentario HTML / encabezado / blockquote | visible y discreto, sin romper la jerarquía | `> _pág. N_`                                             |
| Encabezados                     | solo negritas / niveles Markdown          | navegación e índice                         | `##` capítulos, `###` X.Y, `####` X.Y.Z, `#####` X.Y.Z.W |
| Tablas simples                  | HTML / GFM                                | legibilidad y compatibilidad                | GFM                                                      |
| Tablas key-value sin encabezado | dejarlas sin encabezado / agregar uno     | claridad y RAG                              | `\| Característica \| Descripción \|`                    |
| Celdas combinadas               | HTML con rowspan / repetir valores        | GFM no soporta span                         | repetir el valor en cada fila                            |
| Superíndices                    | Unicode / etiquetas                       | fidelidad                                   | `<sup>` / `<sub>`                                        |
| Encabezado y pie repetidos      | conservarlos / omitirlos                  | ruido en 140 páginas                        | omitir, dejando constancia en la introducción            |
| Sellos y fotos                  | describirlos / omitirlos                  | no aportan contenido                        | omitir; describir solo figuras técnicas                  |

### Cambio posterior (feedback del usuario)

En una primera versión, las **subcolumnas** de una celda de valor (p. ej.
«Vacuna pediátrica» / «Vacuna para adulto») se habían convertido en filas
separadas (`COMPOSICIÓN — Vacuna pediátrica`, etc.). El usuario señaló el riesgo
para RAG: una consulta general podía recuperar solo una de las dos filas.

**Corrección aplicada:** una sola fila por característica, con ambos contenidos
en la misma celda y etiquetas en negrita:

```markdown
| ESQUEMA DE VACUNACIÓN | **Menores de 5 años (3 dosis):** …<br><br>**Mayores de 5 años:** … |
```

Se aplicó a Hepatitis B, Antineumocócica, Influenza, dT en gestantes, celdas
combinadas del esquema VIH y a las presentaciones múltiples del Anexo 17
(BCG 20/40/90 %, SPR multidosismonodosis, HVB pediátrica/adulto, SPR/SR).

---

## 4. Herramientas: evaluación y descarte

| Herramienta                            | Prueba                              | Resultado                                                                                          | Decisión                              |
|----------------------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------|
| `pdftotext` / `pdfinfo`                | extracción directa                  | 0 caracteres: sin capa de texto                                                                    | descartado                            |
| Tesseract 5 + `spa.traineddata` (best) | OCR de páginas completas            | legible, pero con errores frecuentes (`N*` por `N°`, `1mi`, `10 5`, tildes, columnas desordenadas) | solo como **apoyo**                   |
| PyMuPDF                                | renderizado 150/300 dpi             | rápido y fiable                                                                                    | **usado**                             |
| dots.ocr (1.7B de LLM) + PyTorch ROCm  | OCR visual con layout y tablas HTML | calidad alta en la página de prueba (acentos, `<sup>`, tablas con rowspan)                         | evaluado y descartado por rendimiento |
| Pandoc                                 | verificación de renderizado         | 81 tablas GFM válidas                                                                              | **usado**                             |

### Por qué se descartó dots.ocr

- GPU AMD RX 6800 XT (16 GB), ROCm 7.2; el modelo cargaba y producía resultados
  excelentes, pero la decodificación era de ~4 tokens/s (~4–5 min/página;
  ~10 h para 140 páginas).
- La cuantización 4-bit degradaba la salida (devolvía una sola caja «Picture»).
- Dos procesos concurrentes provocaron fallos de memoria de la GPU.
- Conclusión: el coste/beneficio no justificaba esperar ~10 h; la lectura visual
  por agentes era más rápida y controlable.

---

## 5. Flujo de trabajo aplicado

### 5.1 Preparación

1. Renderizar cada página a **150 dpi** (vista general) y **300 dpi** (detalle).
2. Ejecutar Tesseract en español para tener un texto de apoyo con coordenadas
   aproximadas.
3. Construir un inventario de páginas y hojas de contacto para detectar páginas
   rotadas o en blanco.

### 5.2 Reparto en rangos

El documento se dividió en **15 rangos de 8–12 páginas**, alineados con secciones:

- Parte 1: 1–10, 11–12, 13–20, 21–27, 28–36, 37–44, 45–53, 54–65, 66–77,
  78–85, 86–97.
- Parte 2: 98–107, 108–117, 118–130, 131–140.

Cada rango lo transcribió un agente con visión, leyendo la imagen como fuente de
verdad y el OCR solo como contraste. Cada agente dejó:

- un fragmento `frag/<tag>_<inicio>-<fin>.md`;
- un archivo `notas-...md` con dudas, celdas ilegibles y continuaciones.

Los rangos con muchas tablas recibieron instrucciones adicionales (fichas
key-value, tablas partidas, subcolumnas, formularios, páginas giradas).

### 5.3 Ensamblado

Un script concatenó los fragmentos en orden, agregó la introducción y normalizó:

- llamadas `(*)`, `(**)` → `(\*)`, `(\*\*)` para que Markdown no las convierta
  en énfasis;
- líneas en blanco repetidas;
- marcadores de página y notas de tablas completadas.

### 5.4 Verificación

Se comprobó automáticamente:

- cobertura de las 140 páginas, sin duplicados;
- número de celdas por fila en las 81 tablas;
- saltos de nivel de encabezado;
- etiquetas `<sup>`/`<sub>` y negritas sin cerrar;
- ítems `TÉRMINO:` sin negrita (pérdida de formato entre páginas).

Y manualmente, por muestreo: portada, una página de texto denso, una ficha
key-value, una tabla con subcolumnas, una tabla partida, una página apaisada y
el final del documento.

---

## 6. Incidencias y correcciones

| Incidencia                                                                         | Causa                                                     | Corrección                                                                      |
|------------------------------------------------------------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------|
| Páginas 15, 16, 18, 19, 26, 30, 38, 41, 46, 51, 52, 71, 91–95 sin contenido propio | su contenido se trasladó a la tabla de la página anterior | marcadores anotados como «continuación de la tabla …» o rango `> _págs. N–M …_` |
| Tablas cortadas entre fragmentos (Hib, SR, índice de anexos, Anexo 5)              | límites de rango                                          | filas movidas a la tabla de la página inicial y nota de completado              |
| Encabezado repetido en una tabla unida                                             | el agente siguiente repitió el encabezado                 | eliminado el encabezado duplicado                                               |
| Pérdida de negrita al pasar de la pág. 10 a la 11                                  | el OCR/transcripción perdió el resaltado de los acrónimos | 15 acrónimos puestos en negrita                                                 |
| `(**)` en prosa ponía texto en negrita                                             | Markdown interpretaba los asteriscos                      | escape sistemático de llamadas                                                  |
| Viñetas `❖` como texto suelto                                                      | símbolo original                                          | convertidas a `-`                                                               |
| Filas duplicadas por presentaciones en el Anexo 17                                 | la celda original tenía varias presentaciones             | unidas en una sola fila con `<br>`                                              |
| Celdas vacías por colspan en el esquema VIH                                        | se había dejado vacía la segunda columna                  | valor repetido en ambas columnas                                                |

---

## 7. Resultado

| Archivo          | Páginas | Palabras | Tablas | Tamaño  |
|------------------|---------|----------|--------|---------|
| `...-parte-1.md` | 1–97    | ~35 900  | 47     | ~229 KB |
| `...-parte-2.md` | 98–140  | ~15 600  | 34     | ~91 KB  |

- **140/140 páginas** con marcador.
- **0 tablas** con columnas inconsistentes.
- **0 saltos** de nivel de encabezado.
- Validación final: `OK` en ambos archivos.

Se conservaron las erratas del original («cuya la implementación», «del del»,
«Naciounal», «DICC50», etc.) y se corrigieron solo errores de OCR.

---

## 8. Formato para base vectorial

Los Markdown se convierten con `pipelines/ocr/md_to_vector.py` (acepta uno o
varios archivos). El resultado es un JSONL con un registro por fragmento:

- campos: `id`, `source`, `doc_title`, `section`, `section_number`, `pages`,
  `page_start`, `page_end`, `type`, `part`, `parts`, `text`, `embed_text`, `chars`;
- `text` es el contenido limpio (sin marcadores de página ni énfasis Markdown);
- `embed_text` agrega la ruta de sección y las páginas para conservar el sujeto
  de la consulta;
- las tablas se mantienen enteras hasta 6000 caracteres; si lo superan, se parten
  por filas repitiendo el encabezado;
- el texto se fragmenta a 1800 caracteres por oración;
- se genera un reporte de auditoría con cobertura de páginas, fragmentos sin
  sección y tablas sobredimensionadas.

Resultado de la conversión de estos dos documentos: **648 fragmentos**
(389 de texto y 49 de tabla en la parte 1; 173 de texto y 37 de tabla en la
parte 2), sin páginas ni secciones faltantes y sin advertencias.

---

## 9. Sistema de consulta

El JSONL de fragmentos alimenta el paquete `src/nts_rag`, que construye un
índice ChromaDB y responde consultas con búsqueda vectorial + BM25 (fusión RRF)
y citas de página y sección. Al ser precisa la ingesta, la consulta no repite
validaciones: no hay léxico corrector, normalización de valores dudosos, fusión
inter-página ni overrides.

```bash
make install     # dependencias de consulta
make index       # data/chunks/*.jsonl → data/index/chroma
make serve       # API FastAPI
make test
```

El detalle de la API, los comandos y la evaluación está en
[`README.md`](../README.md).


---

## 9. Criterios reutilizables

1. La imagen es la fuente de verdad; el OCR es una hipótesis.
2. Dividir por secciones, no por cantidad fija de páginas.
3. Un fragmento por rango + un archivo de notas.
4. Unir tablas partidas y anotar con qué páginas se completaron.
5. Una fila por característica; las subcolumnas van en la misma celda.
6. Repetir celdas combinadas para que cada fila sea autosuficiente.
7. Validar automáticamente cobertura, tablas, encabezados y formato.
8. Revisar por muestreo las zonas de mayor riesgo: tablas, fórmulas y saltos
   de página.

Estos criterios quedaron implementados en la carpeta `pipelines/ocr/`
(`README.md`, `GUIA-DE-ESTILO.md`, `ocr_pipeline.py` y plantillas de prompts).
