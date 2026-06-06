import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from embedder import MODEL_NAME, load_index  # noqa: E402
from models import RetrievalResult  # noqa: E402

# Over-fetch multiplier for filtered retrieval.
# We fetch FILTERED_OVERFETCH results from FAISS, filter by access level,
# then return the top top_k that pass. This is a constant, not configurable.
FILTERED_OVERFETCH = 20


class Retriever:
    def __init__(self, index_dir: str = "data/index", model_name: str = MODEL_NAME):
        """Loads FAISS index, metadata, and embedding model at init time."""
        self.model = SentenceTransformer(model_name)
        self.index, self.metadata = load_index(index_dir)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string. Returns a (1, 384) float32 array, L2-normalized."""
        vec = self.model.encode([query], normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Unfiltered retrieval. Searches the full index.
        Returns top_k results sorted by descending score.

        Uses FAISS index.search():
          scores, indices = self.index.search(query_vec, top_k)
        """
        if top_k <= 0:
            return []

        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(
                RetrievalResult(
                    doc_id=meta["doc_id"],
                    chunk_index=meta["chunk_index"],
                    content=meta["content"],
                    access_level=meta["access_level"],
                    score=float(score),
                )
            )
        return results

    def retrieve_filtered(
        self,
        query: str,
        allowed_levels: list[str],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Filtered retrieval. Returns top_k results where
        access_level is in allowed_levels.

        Implementation: fetch FILTERED_OVERFETCH (=20) results from FAISS,
        drop results whose access_level is not in allowed_levels,
        return the first top_k that remain.

        If fewer than top_k results pass the filter, return however many pass.
        This is expected for narrow scopes (e.g., public-only user on a
        confidential-only question).
        """
        if top_k <= 0:
            return []

        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, FILTERED_OVERFETCH)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if meta["access_level"] not in allowed_levels:
                continue
            results.append(
                RetrievalResult(
                    doc_id=meta["doc_id"],
                    chunk_index=meta["chunk_index"],
                    content=meta["content"],
                    access_level=meta["access_level"],
                    score=float(score),
                )
            )
            if len(results) >= top_k:
                break
        return results
