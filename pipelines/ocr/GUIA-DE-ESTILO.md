# Guía de estilo para transcripción de documentos escaneados a Markdown

Esta guía define cómo transcribir un documento escaneado (normas, manuales,
informes) a Markdown de forma **fiel, auditable y apta para recuperación
semántica (RAG)**. La fuente de verdad es siempre **la imagen de la página**;
el OCR de Tesseract es solo una ayuda y contiene errores.

---

## 1. Estructura y marcadores de página

1. Cada página original empieza con una línea de marcador:

   ```markdown
   > _pág. N_
   ```

   `N` es el número impreso al pie de la página (no el índice del archivo si el
   PDF empieza numerado en otra página).

2. **Omite** el encabezado y el pie repetidos en todas las páginas (título de la
   norma, código, número de página). Deja constancia de ello en la introducción
   del archivo.

3. Cuando una tabla o un párrafo se prolonga y su contenido se traslada al
   espacio de la página anterior (para no cortar la tabla), deja el marcador de
   la página siguiente anotado:

   ```markdown
   > _pág. 46 — continuación de la tabla 6.2.2 (iniciada en la pág. 45)._
   ```

   Para varias páginas seguidas consumidas por una misma tabla:

   ```markdown
   > _págs. 91–95 — continuación de la tabla del Anexo 6 (iniciada en la pág. 90)._
   ```

4. No dejes marcadores vacíos sin explicación. Si una página no aporta contenido
   propio, indícalo con la anotación anterior.

5. Transcribe **todo** el texto visible: párrafos, listas, notas al pie de tablas,
   llamadas con asteriscos, aclaraciones entre paréntesis, textos de figuras
   relevantes.

---

## 2. Fidelidad

- **No corrijas** la ortografía, gramática ni estilo del autor: si el original
  dice «cuya la implementación», se transcribe así.
- Sí corrige errores evidentes de OCR (`N*` → `N°`, `1mi` → `1 ml`), solo si la
  imagen lo confirma.
- No traduzcas. Conserva mayúsculas/minúsculas, comillas tipográficas (“ ”),
  guiones, símbolos (`N°`, `%`, `°C`, `½`, `¼`, `µ`) y unidades.
- Los párrafos justificados del escaneo se unen en **una sola línea** (sin saltos
  de línea internos). Cada ítem de lista va en su propia línea.
- Conserva las erratas y las duplicaciones del original (p. ej. «del del»).

---

## 3. Títulos y jerarquía

- `##` para capítulos en números romanos: `## I. FINALIDAD`.
- `###` para secciones de primer nivel: `### 6.1. VACUNAS DEL ESQUEMA NACIONAL:`.
- `####` para `6.1.1 ...` y subsecciones tipo `2.1 Objetivo General`.
- `#####` para `6.1.1.1 ...`.
- Respeta el texto exacto del título (mayúsculas incluidas).
- No saltes niveles: `##` → `###` → `####` → `#####`.

---

## 4. Superíndices y subíndices

- Superíndice: `<sup>...</sup>` → `10<sup>5</sup>`, `0,1<sup>e</sup>`, `m<sup>2</sup>`.
- Subíndice: `<sub>...</sub>` → `DICC<sub>50</sub>`.
- No uses caracteres Unicode volados si el original tiene un exponente.

---

## 5. Tablas

### 5.1 Reglas generales

- Usa tablas GFM (`| ... | ... |`).
- **Tablas key-value sin encabezado**: agrega `| Característica | Descripción |`.
- **Celdas combinadas (rowspan)**: repite el valor en cada fila para que cada
  fila sea autosuficiente.
- **Celdas combinadas (colspan)**: repite el valor en cada columna afectada;
  no dejes celdas vacías por combinación.
- Varias líneas dentro de una celda: sepáralas con `<br>`.
- Escapa las barras verticales internas como `\|`.
- Una tabla partida entre páginas se **une** en el lugar donde empieza y se
  anota después: `*Tabla completada con la pág. N.*`.

### 5.2 Subcolumnas dentro de una celda de valor (regla RAG)

**No dividas una característica en varias filas.** Si una celda de valor tiene
dos subcolumnas (por ejemplo, «Vacuna pediátrica» y «Vacuna para adulto»), deja
**una sola fila** con la característica y une ambos contenidos en la misma celda,
etiquetados en negrita:

```markdown
| Característica | Descripción |
| --- | --- |
| COMPOSICIÓN | **Vacuna pediátrica:** Cada dosis de 0.5 ml contiene: …<br><br>**Vacuna para adulto:** Cada dosis de 1 ml contiene: … |
| ESQUEMA DE VACUNACIÓN | **Menores de 5 años (3 dosis):** …<br><br>**Mayores de 5 años:** … |
```

Motivo: al recuperar con una consulta general («esquema de vacunación de la
antineumocócica»), una sola fila contiene **toda** la respuesta. Si se dividiera
en dos filas, el sistema de recuperación podría devolver solo una parte.

La misma regla aplica cuando una celda original contiene varios sub-bloques
(p. ej. «1ra dosis / 2da dosis / 3ra dosis»): únelos en la misma fila con
`<br>` y etiquetas en negrita.

### 5.3 Presentaciones alternativas

Si una fila del original tiene dos presentaciones (Multidosis / Monodosis) con
sus propios valores en otras columnas, **no las separes en dos filas**: únelas
en una sola fila y separa los valores con `<br>`, manteniendo el orden:

```markdown
| … | SPR | 2 | Multidosis (frascos por 5 o 10 dosis)<br>Monodosis | 1.2% (20%)<br>No tiene factor de dosis No utilizada | … |
```

### 5.4 Notas y llamadas

- Conserva las llamadas `(*)`, `(**)`, `(***)` tal cual, y sus notas al pie.
- El script de ensamblado escapa los asteriscos (`(\*)`, `(\*\*)`) para que
  Markdown no los interprete como énfasis. No los escapes a mano.

---

## 6. Formato continuo entre páginas

- Un formato iniciado en una página debe continuar en la siguiente:
  - Listas de acrónimos o términos en negrita: **todos** los ítems llevan negrita.
  - Listas con viñeta: todos los ítems usan `-`, aunque el original cambie el
    símbolo (`•`, `➢`, `❖`) en la página siguiente.
  - Títulos: mismo nivel de encabezado.
-Una definición que empieza en negrita en una página y continúa en la siguiente
  NO lleva negrita en la continuación (es el mismo término).
- Revisa los saltos de página 10→11, 20→21, etc.: son los puntos típicos donde
  se pierde el formato.

---

## 7. Figuras y sellos

- Los sellos del MINSA, membretes laterales, fotos decorativas y grillas vacías
  se omiten.
- Si un diagrama, mapa o esquema aporta contenido, descríbelo en una línea:
  `[Figura: ...]`.
- En formularios con grillas vacías, transcribe los encabezados de columna y
  agrega `*(grilla con N filas en blanco)*`; no generes filas vacías.

---

## 8. Preparación para RAG (recomendaciones)

1. **Unidad de recuperación**: fragmenta por sección (encabezado + párrafos +
   tabla). No fragmentes dentro de una tabla ni entre una tabla y su encabezado.
2. **Contexto**: cada fragmento debe incluir el encabezado de la sección (p. ej.
   «6.1.1.10 VACUNA ANTINEUMOCÓCICA») para que una fila suelta conserve el
   sujeto de la consulta.
3. **Auto-contención**: cada fila de tabla debe poder leerse sola (repite edad,
   población, vacuna y valores combinados).
4. **Metadatos**: conserva `> _pág. N_`; permite citar la fuente y auditar.
5. **Sin duplicados**: al unir tablas partidas, no repitas el encabezado ni las
   notas; usa una sola tabla y una nota de completado.
6. **Texto limpio**: no dejes artefactos de OCR, encabezados repetidos ni
   números de página sueltos.

---

## 9. Entrega

- Un archivo Markdown por PDF (o por parte, si el PDF está dividido).
- Introducción breve con: documento de origen, rango de páginas y convención de
  marcadores.
- Notas de conversión por rango (`notas-*.md`) con dudas, celdas ilegibles,
  tablas que continúan fuera del rango y páginas rotadas.
