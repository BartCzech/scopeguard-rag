from pathlib import Path

import pytest

from corpus_loader import (
    _parse_md_file,
    filter_by_level,
    get_document_by_id,
    load_corpus,
)

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def test_load_corpus_returns_all_documents() -> None:
    docs = load_corpus(str(CORPUS_DIR))
    assert len(docs) == 46


def test_documents_have_required_fields_and_no_frontmatter() -> None:
    docs = load_corpus(str(CORPUS_DIR))
    for doc in docs:
        assert doc.doc_id
        assert doc.title
        assert doc.access_level in ("public", "internal", "confidential")
        assert isinstance(doc.adversarial, bool)
        assert doc.content
        assert not doc.content.startswith("---")
        assert "doc_id:" not in doc.content.splitlines()[0:5]


def test_filter_by_level_public() -> None:
    docs = load_corpus(str(CORPUS_DIR))
    public_docs = filter_by_level(docs, ["public"])
    assert len(public_docs) == 19
    assert all(doc.access_level == "public" for doc in public_docs)
    assert any(doc.doc_id == "ADV-001" for doc in public_docs)
    assert any(doc.doc_id == "ADV-003" for doc in public_docs)


def test_get_document_by_id_conf_003() -> None:
    docs = load_corpus(str(CORPUS_DIR))
    doc = get_document_by_id(docs, "CONF-003")
    assert doc is not None
    assert doc.title == "Salary Bands & Compensation Framework"
    assert doc.access_level == "confidential"


def test_get_document_by_id_missing_returns_none() -> None:
    docs = load_corpus(str(CORPUS_DIR))
    assert get_document_by_id(docs, "DOES-NOT-EXIST") is None


def test_loads_nested_subdirectory_files() -> None:
    docs = load_corpus(str(CORPUS_DIR))
    doc_ids = {doc.doc_id for doc in docs}
    assert "PUB-001" in doc_ids


def test_missing_frontmatter_raises_value_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "BAD-001.md"
    bad_file.write_text(
        "---\ndoc_id: BAD-001\ntitle: Incomplete\n---\nBody content.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"missing frontmatter field 'access_level'"):
        _parse_md_file(bad_file)
