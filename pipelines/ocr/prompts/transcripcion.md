# Plantilla de prompt — transcripción de un rango de páginas

Copia este bloque y rellena los valores entre `{{ }}`.

---

Proyecto: `{{PROYECTO}}`

Lee primero `{{PROYECTO}}/pipelines/ocr/GUIA-DE-ESTILO.md` y síguela al pie
de la letra.

Tarea: transcribir a Markdown las páginas **{{INICIO}} a {{FIN}}** del PDF
`{{PDF}}` (páginas impresas {{INICIO}}–{{FIN}}; imágenes
`{{WORK}}/img150/{{TAG}}/{{INICIO}}…{{FIN}}.png`).

Imágenes:
- Vista general: `{{WORK}}/img150/{{TAG}}/NNN.png` (150 dpi)
- Detalle: `{{WORK}}/img300/{{TAG}}/NNN.png` (300 dpi), úsalo para números,
  acentos, tablas y celdas pequeñas.
- OCR de apoyo (contiene errores): `{{WORK}}/ocr/{{TAG}}/NNN.txt`.
- Si `{{WORK}}/rotations.json` indica un ángulo para una página, rota la imagen
  con PIL (`Image.open(p).rotate(ang, expand=True)`) antes de leerla.
- La fuente de verdad es la imagen.

Contenido esperado: {{DESCRIPCION_DEL_CONTENIDO}}.

Reglas:
- Marcador `> _pág. N_` al inicio de cada página (N = número impreso).
- Omite el encabezado y pie repetidos.
- Tablas GFM; agrega encabezado a las tablas key-value sin encabezado.
- Repite celdas combinadas; no dejes celdas vacías por combinación.
- **Subcolumnas dentro de una celda de valor: una sola fila con ambos contenidos
  etiquetados en negrita** (regla RAG de la guía, sección 5.2). No dividas una
  característica en dos filas.
- Tablas partidas entre páginas: únelas y anota `*Tabla completada con la pág. N.*`.
- Conserva la ortografía del autor; corrige solo errores de OCR.
- Superíndices/subíndices con `<sup>`/`<sub>`.

Entrega:
1. `{{WORK}}/frag/{{TAG}}_{{INICIO}}-{{FIN}}.md` con el contenido transcrito.
2. `{{WORK}}/frag/notas-{{TAG}}_{{INICIO}}-{{FIN}}.md` con dudas, celdas
   ilegibles, tablas que continúan fuera del rango y páginas rotadas.

Responde solo con un resumen de 3–4 líneas (qué páginas hiciste e incidencias).
