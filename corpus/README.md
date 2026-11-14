# Corpus

Fuentes documentales de la NTS N° 246-MINSA/DGIESP-2026.

- `pdf/`: escaneos originales, sin capa de texto. Solo se necesitan si hay que
  repetir el proceso de OCR.
- `markdown/`: transcripción auditada con marcadores de página. Es la **fuente
  de verdad** para la ingesta.

Si se modifica el Markdown, hay que regenerar los chunks (`make chunks`) y el
índice (`make index`).
