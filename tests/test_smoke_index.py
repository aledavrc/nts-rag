"""Smoke tests del reindexado v2 (docs/CONTEXTO_MEJORAS.md §11).

Usan los chunks y el manifiesto versionados, sin embeddings ni ChromaDB, así que
corren en CI. Las comprobaciones que necesitan recuperar (consulta sin versión,
fila de rescate en el top-5, 20/20 lookups) se cubren con `nts_rag.cli evaluar`.
"""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from nts_rag.catalogo import CANONICAL_FIELDS, FICHAS, NO_APLICA
from nts_rag.config import DOCUMENTS, INDEX_SCHEMA_VERSION, Settings
from nts_rag.index import corpus_lint, freshness_problems, load_chunks, read_manifest

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = Settings(
    chunks_path=ROOT / "data/chunks/nts-246.chunks.jsonl",
    data_dir=ROOT / "data",
    corpus_dir=ROOT / "corpus/markdown",
    _env_file=None,
)


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(SETTINGS.chunks_path)


def test_total_is_close_to_the_estimate(chunks):
    assert 1965 * 0.9 <= len(chunks) <= 1965 * 1.1


def test_every_chunk_has_document_and_vigencia(chunks):
    assert all(chunk["documento_id"] in DOCUMENTS and chunk["vigencia"] for chunk in chunks)
    assert {chunk["documento_id"] for chunk in chunks} == set(DOCUMENTS)


def test_no_label_outside_the_13_canonical(chunks):
    labels = {chunk["campo"] for chunk in chunks if chunk["tipo"] == "campo"}
    assert labels <= set(CANONICAL_FIELDS)


def test_ficha_counts_per_norm(chunks):
    fichas = Counter(
        (chunk["documento_id"], chunk["vacuna_slug"])
        for chunk in chunks
        if chunk["tipo"] == "campo"
    )
    per_doc = Counter(doc_id for doc_id, _ in fichas)
    assert per_doc == {"nts-246-2026": 24, "nts-196-2022": 18}


def test_every_ficha_has_its_13_fields(chunks):
    fields: dict[tuple[str, str], set[str]] = {}
    for chunk in chunks:
        if chunk["tipo"] == "campo":
            fields.setdefault((chunk["documento_id"], chunk["vacuna_slug"]), set()).add(
                chunk["campo"]
            )
    for doc_id, fichas in FICHAS.items():
        for slug in fichas.values():
            expected = {f for f in CANONICAL_FIELDS if slug not in NO_APLICA.get(f, set())}
            assert fields[(doc_id, slug)] == expected, (doc_id, slug)


def test_corpus_lint_is_clean(chunks):
    assert corpus_lint(chunks) == []


def test_no_2022_chunk_is_titled_as_nts_246(chunks):
    old = [chunk for chunk in chunks if chunk["documento_id"] == "nts-196-2022"]
    assert old and all("246" not in chunk["documento"] for chunk in old)
    assert all(chunk["vigencia"] == "derogada" for chunk in old)


def test_ids_are_unique_and_stable(chunks):
    ids = [chunk["id"] for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [chunk["id"] for chunk in load_chunks(SETTINGS.chunks_path)]


def test_manifest_is_v2_and_fresh(monkeypatch):
    manifest = read_manifest(SETTINGS)
    assert manifest is not None
    assert manifest["schema_version"] == INDEX_SCHEMA_VERSION
    assert manifest["fuentes"], "el manifiesto debe registrar los hashes de las fuentes"
    # Las rutas del manifiesto son relativas a la raíz del repositorio.
    monkeypatch.chdir(ROOT)
    assert freshness_problems(Settings(_env_file=None)) == []


def test_chunker_reproduces_the_versioned_jsonl(tmp_path):
    """Regenerar los chunks desde los Markdown da exactamente el JSONL versionado."""

    out = tmp_path / "chunks.jsonl"
    sources = [
        str(ROOT / "corpus/markdown" / name)
        for doc in DOCUMENTS.values()
        for name in doc.fuentes
    ]
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipelines/ocr/md_to_vector.py"),
            *sources,
            "-o",
            str(out),
            "--max-chars",
            "1800",
            "--table-max-chars",
            "6000",
        ],
        check=True,
        capture_output=True,
    )
    normalize = lambda path: path.read_bytes().replace(b"\r\n", b"\n")  # noqa: E731
    assert normalize(out) == normalize(SETTINGS.chunks_path)
