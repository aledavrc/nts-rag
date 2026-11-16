# Ruta del intérprete del venv según el sistema operativo.
ifeq ($(OS),Windows_NT)
  VENV_PY := .venv/Scripts/python.exe
else
  VENV_PY := .venv/bin/python
endif

PY ?= $(VENV_PY)
CHUNKS := data/chunks/nts-246.chunks.jsonl
DOCS := corpus/markdown/8265031-nts-n-246-minsa-digiesp-2026-parte-1.md \
        corpus/markdown/8265031-nts-n-246-minsa-digiesp-2026-parte-2.md

.PHONY: install chunks index ingest serve test lint doctor docker-build docker-up docker-down clean-index inspeccionar

install:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]"

# Regenera los chunks desde los Markdown ya auditados.
chunks:
	python3 pipelines/ocr/md_to_vector.py \
		$(DOCS) \
		-o $(CHUNKS) \
		--max-chars 1800 \
		--table-max-chars 6000

index:
	$(PY) -m nts_rag.cli indexar

ingest: chunks index

serve:
	$(PY) -m nts_rag.cli servir --reload

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests pipelines

doctor:
	$(PY) -m nts_rag.cli doctor

# Inspecciona el índice vectorial y busca una pregunta.
# Uso: make inspeccionar PREGUNTA="dosis de la antineumocócica" [K=3]
inspeccionar:
	@test -n "$(PREGUNTA)" || { echo 'Uso: make inspeccionar PREGUNTA="..." [K=3]'; exit 1; }
	$(PY) -m nts_rag.cli inspeccionar "$(PREGUNTA)" $(if $(K),-k $(K),)

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean-index:
	$(PY) -c "import shutil; from pathlib import Path; [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in map(Path, ['data/index', 'data/manifest.json'])]"
