import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Flat imports (models, corpus_loader) resolve when src/ is on sys.path — e.g. pytest
# or `python -m src.embedder` from the repo root.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from corpus_loader import load_corpus  # noqa: E402
from models import Chunk, Document  # noqa: E402

# ── Configuration (constants, not tuneable at runtime) ──
CHUNK_MAX_WORDS = 250
CHUNK_OVERLAP_WORDS = 50
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim embeddings
EMBEDDING_DIM = 384
INDEX_DIR = "data/index"


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split each document into word-based chunks of ≤ CHUNK_MAX_WORDS words,
    with CHUNK_OVERLAP_WORDS overlap between consecutive chunks.
    Each Chunk retains the parent doc's doc_id and access_level.
    Chunk index is 0-based within each document.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        text_chunks = _split_text(doc.content)
        for i, text in enumerate(text_chunks):
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_index=i,
                    content=text,
                    access_level=doc.access_level,
                )
            )
    return chunks


def _word_char_ranges(text: str, words: list[str]) -> list[tuple[int, int]]:
    """Map each word to its (start, end) character offsets in the original text."""
    ranges: list[tuple[int, int]] = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        ranges.append((start, start + len(word)))
        pos = start + len(word)
    return ranges


def _paragraph_start_indices(text: str, word_ranges: list[tuple[int, int]]) -> set[int]:
    """Word indices where a new paragraph begins (after a blank line)."""
    para_starts: set[int] = set()
    for i in range(1, len(word_ranges)):
        prev_end = word_ranges[i - 1][1]
        curr_start = word_ranges[i][0]
        gap = text[prev_end:curr_start]
        if "\n\n" in gap:
            para_starts.add(i)
    return para_starts


def _split_text(text: str) -> list[str]:
    """
    Split text into chunks of ≤ CHUNK_MAX_WORDS words with CHUNK_OVERLAP_WORDS overlap.
    Prefers splitting at paragraph boundaries (\\n\\n) when one falls within the overlap
    window. Returns at least one chunk even for short texts.
    """
    words = text.split()
    if len(words) <= CHUNK_MAX_WORDS:
        return [text]

    word_ranges = _word_char_ranges(text, words)
    para_starts = _paragraph_start_indices(text, word_ranges)

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_MAX_WORDS, len(words))
        if end < len(words):
            overlap_start = end - CHUNK_OVERLAP_WORDS
            candidates = [i for i in para_starts if overlap_start < i <= end]
            if candidates:
                end = max(candidates)

        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP_WORDS

    return chunks


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> list[Chunk]:
    """
    Populates the embedding field on each Chunk using model.encode().
    Embeddings are L2-normalized so that inner product == cosine similarity.
    Returns the same list with embeddings filled in.
    """
    texts = [c.content for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk.embedding = embedding.tolist()
    return chunks


def build_index(chunks: list[Chunk], index_dir: str = INDEX_DIR) -> None:
    """
    Creates and saves:
      - {index_dir}/faiss.index  — FAISS IndexFlatIP (inner product; cosine with normalized vecs)
      - {index_dir}/metadata.json — list of dicts, one per vector, same order as index

    The i-th vector in the FAISS index corresponds to the i-th entry in metadata.json.
    """
    model = SentenceTransformer(MODEL_NAME)
    embedded = embed_chunks(chunks, model)
    embeddings = np.array([c.embedding for c in embedded], dtype=np.float32)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path / "faiss.index"))

    metadata = [
        {
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "content": c.content,
            "access_level": c.access_level,
        }
        for c in embedded
    ]
    with open(index_path / "metadata.json", "w") as f:
        json.dump(metadata, f)


def load_index(index_dir: str = INDEX_DIR) -> tuple[faiss.Index, list[dict]]:
    """
    Returns (faiss_index, metadata_list).
    metadata_list[i] corresponds to vector i in the index.
    """
    index_path = Path(index_dir)
    index = faiss.read_index(str(index_path / "faiss.index"))
    with open(index_path / "metadata.json") as f:
        metadata = json.load(f)
    assert index.ntotal == len(metadata), "Index and metadata out of sync"
    return index, metadata


def main() -> None:
    documents = load_corpus()
    chunks = chunk_documents(documents)
    build_index(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents → {INDEX_DIR}/")


if __name__ == "__main__":
    main()
