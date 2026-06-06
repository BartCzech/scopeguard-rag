import inspect
import os
from unittest.mock import MagicMock

import pytest

import strategies.strategy_1 as strategy_1_module
from generator import Generator
from models import GenerationResult, RetrievalResult, UserContext
from retriever import Retriever
from strategies.base import StrategyBase, StrategyResult
from strategies.prompts import BASE_RAG_PROMPT, NAIVE_ACCESS_SUFFIX
from strategies.strategy_1 import NaiveRAGStrategy


def _user_context(level: str) -> UserContext:
    scope_map = {
        "public": (["docs:public"], ["public"]),
        "internal": (["docs:public", "docs:internal"], ["public", "internal"]),
        "confidential": (
            ["docs:public", "docs:internal", "docs:confidential"],
            ["public", "internal", "confidential"],
        ),
    }
    scopes, allowed = scope_map[level]
    return UserContext(user_id=f"user-{level}", scopes=scopes, allowed_levels=allowed)


def _chunk(
    doc_id: str,
    access_level: str,
    content: str = "Sample content.",
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        chunk_index=0,
        content=content,
        access_level=access_level,
        score=score,
    )


@pytest.fixture
def mock_chunks() -> list[RetrievalResult]:
    return [
        _chunk("PUB-001", "public", "Product overview."),
        _chunk("CONF-003", "confidential", "Salary bands: $120K–$225K for engineers."),
        _chunk("INT-004", "internal", "Internal roadmap notes."),
    ]


@pytest.fixture
def mock_retriever(mock_chunks: list[RetrievalResult]) -> MagicMock:
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = mock_chunks
    return retriever


@pytest.fixture
def mock_generator() -> MagicMock:
    generator = MagicMock(spec=Generator)
    generator.generate.return_value = GenerationResult(
        answer="I cannot share confidential salary information.",
        cited_doc_ids=[],
        raw_response="I cannot share confidential salary information.",
    )
    return generator


def test_implements_strategy_base(mock_retriever: MagicMock, mock_generator: MagicMock) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    assert isinstance(strategy, StrategyBase)


def test_returns_strategy_result(mock_retriever: MagicMock, mock_generator: MagicMock) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    result = strategy.answer("What are the salary bands?", _user_context("public"))
    assert isinstance(result, StrategyResult)


def test_calls_unfiltered_retrieve(mock_retriever: MagicMock, mock_generator: MagicMock) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    query = "What are the salary bands?"
    strategy.answer(query, _user_context("public"))

    mock_retriever.retrieve.assert_called_once_with(query, top_k=5)
    mock_retriever.retrieve_filtered.assert_not_called()


def test_retrieved_and_context_chunks_are_identical(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_chunks: list[RetrievalResult],
) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    result = strategy.answer("What are the salary bands?", _user_context("public"))

    assert result.retrieved_chunks == mock_chunks
    assert result.context_chunks == mock_chunks


def test_system_prompt_includes_access_level_for_public_user(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    strategy.answer("What is ScopeGuard?", _user_context("public"))

    expected_prompt = BASE_RAG_PROMPT + NAIVE_ACCESS_SUFFIX.format(access_level="public")
    mock_generator.generate.assert_called_once()
    assert mock_generator.generate.call_args.kwargs["system_prompt"] == expected_prompt


def test_system_prompt_uses_highest_allowed_level(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    strategy.answer("What is ScopeGuard?", _user_context("internal"))

    system_prompt = mock_generator.generate.call_args.kwargs["system_prompt"]
    assert "access level: internal" in system_prompt


def test_public_user_context_may_include_confidential_chunks(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_chunks: list[RetrievalResult],
) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    result = strategy.answer("What are the salary bands?", _user_context("public"))

    access_levels = {chunk.access_level for chunk in result.context_chunks}
    assert "confidential" in access_levels
    assert mock_generator.generate.call_args.kwargs["context"] == mock_chunks


def test_does_not_call_generate_no_context(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    strategy.answer("What is ScopeGuard?", _user_context("public"))
    mock_generator.generate_no_context.assert_not_called()


def test_blocked_by_policy_always_false(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
) -> None:
    strategy = NaiveRAGStrategy(mock_retriever, mock_generator)
    result = strategy.answer("What is ScopeGuard?", _user_context("confidential"))
    assert result.blocked_by_policy is False


def test_does_not_use_retrieve_filtered_in_source() -> None:
    source = inspect.getsource(strategy_1_module)
    assert "retrieve_filtered" not in source


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.skipif(
    not os.path.exists("data/index/faiss.index"),
    reason="FAISS index not built — run: uv run python -m src.embedder",
)
def test_smoke_salary_bands_public_user() -> None:
    """
    Live smoke test: public user asking about salary data.
    Either refusal (prompt worked) or leakage (prompt failed) is valid.
    """
    strategy = NaiveRAGStrategy(Retriever(), Generator())
    query = "What are ScopeGuard's salary bands?"
    result = strategy.answer(query, _user_context("public"))

    assert result.generation.answer
    assert len(result.retrieved_chunks) > 0
    assert result.retrieved_chunks == result.context_chunks
    assert result.blocked_by_policy is False

    context_levels = {chunk.access_level for chunk in result.context_chunks}
    # Unfiltered retrieval may surface confidential docs to a public user
    assert len(context_levels) >= 1
