# Instrucciones adicionales para rangos con muchas tablas

Añade este bloque al prompt de transcripción cuando el rango contenga fichas
técnicas, tablas de esquemas, formularios o anexos.

---

## Tablas key-value (fichas técnicas)

- Cada tabla key-value sin encabezado lleva `| Característica | Descripción |`.
- La primera columna conserva el rótulo original (preferentemente en mayúsculas:
  `TIPO DE VACUNA`, `COMPOSICIÓN`, `ESQUEMA DE VACUNACIÓN`, ...).
- Si una celda de valor tiene **dos subcolumnas** (p. ej. «Vacuna pediátrica» y
  «Vacuna para adulto», «Menores de 5 años» y «Mayores de 5 años»), deja **una
  sola fila** y une ambos contenidos en la misma celda:

  ```markdown
  | COMPOSICIÓN | **Vacuna pediátrica:** …<br><br>**Vacuna para adulto:** … |
  ```

  No generes dos filas `COMPOSICIÓN — ...`: una consulta general debe recuperar
  el contenido completo.
- Si la celda original tiene varios sub-bloques (1ra/2da/3ra dosis), únelos en la
  misma fila con `<br>` y etiquetas en negrita.
- Si hay dos presentaciones (Multidosis / Monodosis) con valores propios en otras
  columnas, únelas en una fila y separa los valores con `<br>` en el mismo orden.

## Tablas con celdas combinadas

- Rowspan: repite el valor en cada fila.
- Colspan: repite el valor en cada columna afectada; no dejes celdas vacías.
- Encabezados de dos niveles (un grupo sobre dos columnas): usa una fila de
  encabezado con las columnas finales y una línea en cursiva que aclare el grupo:
  `*Columnas bajo el grupo «VACUNA».*`

## Tablas partidas entre páginas

- Únelas en el lugar donde empiezan; no repitas el encabezado.
- Anota después: `*Tabla completada con la pág. N.*` (o `págs. N y M`).
- Si la tabla ocupa varias páginas, deja los marcadores de continuación
  anotados: `> _págs. N–M — continuación de la tabla …_`.

## Formularios y tablas apaisadas

- Si la página está girada, rótala con PIL antes de leerla:
  `Image.open(p).rotate(270, expand=True).save(tmp)`.
- Transcribe el título, los campos de encabezado y los encabezados de columna.
- Para grillas vacías: `*(grilla con N filas en blanco)*`; no generes filas vacías.
- Conserva las opciones y casillas tal cual: `( )`, `☐`, `________`.

## Anexos de programación/presupuesto

- Conserva códigos, montos, porcentajes y rangos de edad.
- Si una misma vacuna aparece en varias filas porque el original tiene varias
  presentaciones, únelas en una sola fila (regla anterior).
