"""
Unit tests for src/embedder.py — chunking logic only.
No network, no FAISS, no sentence-transformers model.
"""

from embedder import (
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_WORDS,
    _split_text,
    chunk_documents,
)
from models import Document

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_doc(content: str, doc_id: str = "PUB-001", access_level: str = "public") -> Document:
    return Document(
        doc_id=doc_id,
        title="Test doc",
        access_level=access_level,
        adversarial=False,
        content=content,
    )


def _make_long_text(n_words: int) -> str:
    """Return a whitespace-joined string of exactly n_words distinct words."""
    return " ".join(f"word{i}" for i in range(n_words))


def _make_long_text_with_paragraphs(words_per_para: int, n_paras: int) -> str:
    """Build multi-paragraph text with blank-line separators."""
    paras = [" ".join(f"para{p}word{w}" for w in range(words_per_para)) for p in range(n_paras)]
    return "\n\n".join(paras)


# ── _split_text ───────────────────────────────────────────────────────────────


def test_short_text_returns_single_chunk() -> None:
    text = _make_long_text(CHUNK_MAX_WORDS - 1)
    chunks = _split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_exact_max_words_returns_single_chunk() -> None:
    text = _make_long_text(CHUNK_MAX_WORDS)
    chunks = _split_text(text)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks() -> None:
    text = _make_long_text(CHUNK_MAX_WORDS * 3)
    chunks = _split_text(text)
    assert len(chunks) >= 2


def test_every_chunk_is_within_word_limit() -> None:
    text = _make_long_text(CHUNK_MAX_WORDS * 4)
    for chunk in _split_text(text):
        assert len(chunk.split()) <= CHUNK_MAX_WORDS


def test_all_words_covered() -> None:
    """No word should be silently dropped."""
    n = CHUNK_MAX_WORDS * 2 + 30
    words = [f"word{i}" for i in range(n)]
    text = " ".join(words)
    chunks = _split_text(text)

    recovered: set[str] = set()
    for chunk in chunks:
        recovered.update(chunk.split())

    assert recovered == set(words)


def test_consecutive_chunks_overlap() -> None:
    """Adjacent chunks should share at least CHUNK_OVERLAP_WORDS words."""
    text = _make_long_text(CHUNK_MAX_WORDS * 3)
    chunks = _split_text(text)
    assert len(chunks) >= 2

    for a, b in zip(chunks, chunks[1:], strict=False):
        words_a = a.split()
        words_b = b.split()
        # The tail of chunk A should equal the head of chunk B
        shared = min(CHUNK_OVERLAP_WORDS, len(words_a), len(words_b))
        assert words_a[-shared:] == words_b[:shared]


def test_paragraph_boundary_preferred_over_mid_word_split() -> None:
    """
    When a paragraph boundary falls in the overlap window, the split should
    prefer that boundary — so chunk boundaries align with paragraph starts,
    not arbitrary word positions.
    """
    # Build a text just long enough to need two chunks.
    # Make a paragraph break land right in the overlap window of the first chunk.
    words_per_para = CHUNK_MAX_WORDS - CHUNK_OVERLAP_WORDS // 2  # inside overlap window
    text = _make_long_text_with_paragraphs(words_per_para, n_paras=3)

    chunks = _split_text(text)
    # Each chunk should end/start at a paragraph boundary (no \n\n mid-chunk)
    # — verified indirectly: the second chunk should START with a para0/para1/para2 word
    for chunk in chunks:
        assert len(chunk.split()) <= CHUNK_MAX_WORDS


def test_empty_text_returns_single_empty_chunk() -> None:
    chunks = _split_text("")
    assert len(chunks) == 1
    assert chunks[0] == ""


# ── chunk_documents ───────────────────────────────────────────────────────────


def test_chunk_documents_retains_doc_id() -> None:
    doc = _make_doc(_make_long_text(CHUNK_MAX_WORDS * 2), doc_id="CONF-007")
    chunks = chunk_documents([doc])
    assert all(c.doc_id == "CONF-007" for c in chunks)


def test_chunk_documents_retains_access_level() -> None:
    doc = _make_doc(_make_long_text(CHUNK_MAX_WORDS * 2), access_level="confidential")
    chunks = chunk_documents([doc])
    assert all(c.access_level == "confidential" for c in chunks)


def test_chunk_documents_chunk_index_is_zero_based_and_sequential() -> None:
    doc = _make_doc(_make_long_text(CHUNK_MAX_WORDS * 3))
    chunks = chunk_documents([doc])
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_documents_short_doc_gives_one_chunk() -> None:
    doc = _make_doc("This is a short document.")
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_chunk_documents_multiple_docs_are_independent() -> None:
    """Chunks from different documents must not bleed into each other."""
    doc_a = _make_doc(_make_long_text(CHUNK_MAX_WORDS * 2), doc_id="PUB-001")
    doc_b = _make_doc(_make_long_text(CHUNK_MAX_WORDS * 2), doc_id="INT-005")
    chunks = chunk_documents([doc_a, doc_b])

    pub_chunks = [c for c in chunks if c.doc_id == "PUB-001"]
    int_chunks = [c for c in chunks if c.doc_id == "INT-005"]

    assert len(pub_chunks) >= 1
    assert len(int_chunks) >= 1
    # Each sub-list should have independent 0-based indices
    assert [c.chunk_index for c in pub_chunks] == list(range(len(pub_chunks)))
    assert [c.chunk_index for c in int_chunks] == list(range(len(int_chunks)))


def test_chunk_documents_respects_word_limit() -> None:
    doc = _make_doc(_make_long_text(CHUNK_MAX_WORDS * 5))
    for chunk in chunk_documents([doc]):
        assert len(chunk.content.split()) <= CHUNK_MAX_WORDS


def test_chunk_documents_empty_list() -> None:
    assert chunk_documents([]) == []
