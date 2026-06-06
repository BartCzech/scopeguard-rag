import pytest

from retriever import Retriever

INDEX_DIR = "data/index"


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever(index_dir=INDEX_DIR)


def test_retrieve_returns_top_k_with_scores(retriever: Retriever) -> None:
    results = retriever.retrieve("What is ScopeGuard?", top_k=5)
    assert len(results) == 5
    assert all(r.score > 0 for r in results)


def test_retrieve_results_sorted_by_descending_score(retriever: Retriever) -> None:
    results = retriever.retrieve("What is ScopeGuard?", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_filtered_public_excludes_confidential(retriever: Retriever) -> None:
    results = retriever.retrieve_filtered("salary bands", ["public"], top_k=5)
    assert all(r.access_level == "public" for r in results)
    assert all(r.doc_id != "CONF-003" for r in results)


def test_retrieve_filtered_all_levels_may_include_conf_003(retriever: Retriever) -> None:
    results = retriever.retrieve_filtered(
        "salary bands",
        ["public", "internal", "confidential"],
        top_k=5,
    )
    doc_ids = {r.doc_id for r in results}
    assert "CONF-003" in doc_ids


def test_retrieve_unfiltered_may_include_confidential(retriever: Retriever) -> None:
    """Unfiltered retrieve() is used by strategy 4; may return restricted chunks."""
    results = retriever.retrieve("salary bands", top_k=5)
    access_levels = {r.access_level for r in results}
    assert "confidential" in access_levels
    assert any(r.doc_id == "CONF-003" for r in results)


def test_retrieve_empty_for_zero_top_k(retriever: Retriever) -> None:
    assert retriever.retrieve("What is ScopeGuard?", top_k=0) == []


def test_retrieve_filtered_empty_for_zero_top_k(retriever: Retriever) -> None:
    assert retriever.retrieve_filtered("salary bands", ["public"], top_k=0) == []
