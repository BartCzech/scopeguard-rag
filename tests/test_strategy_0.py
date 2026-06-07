import inspect
import os
from unittest.mock import MagicMock

import pytest

import strategies.strategy_0 as strategy_0_module
from generator import Generator
from models import GenerationResult, UserContext
from strategies.base import StrategyBase, StrategyResult
from strategies.prompts import BASE_NO_RAG_PROMPT
from strategies.strategy_0 import PureLLMStrategy


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


@pytest.fixture
def mock_generator() -> MagicMock:
    generator = MagicMock(spec=Generator)
    generator.generate_no_context.return_value = GenerationResult(
        answer="I don't have enough information to answer that.",
        cited_doc_ids=[],
        raw_response="I don't have enough information to answer that.",
    )
    return generator


def test_implements_strategy_base(mock_generator: MagicMock) -> None:
    strategy = PureLLMStrategy(mock_generator)
    assert isinstance(strategy, StrategyBase)


def test_returns_strategy_result(mock_generator: MagicMock) -> None:
    strategy = PureLLMStrategy(mock_generator)
    result = strategy.answer("What is ScopeGuard?", _user_context("public"))
    assert isinstance(result, StrategyResult)


def test_uses_generate_no_context_with_base_no_rag_prompt(mock_generator: MagicMock) -> None:
    strategy = PureLLMStrategy(mock_generator)
    query = "What is ScopeGuard?"
    strategy.answer(query, _user_context("public"))

    mock_generator.generate_no_context.assert_called_once_with(
        query=query,
        system_prompt=BASE_NO_RAG_PROMPT,
    )
    mock_generator.generate.assert_not_called()


def test_does_not_reference_retriever(mock_generator: MagicMock) -> None:
    source = inspect.getsource(strategy_0_module)
    assert "retriever" not in source.lower()
    strategy = PureLLMStrategy(mock_generator)
    assert not hasattr(strategy, "retriever")


def test_retrieved_and_context_chunks_always_empty(mock_generator: MagicMock) -> None:
    strategy = PureLLMStrategy(mock_generator)
    result = strategy.answer("What is ScopeGuard?", _user_context("internal"))
    assert result.retrieved_chunks == []
    assert result.context_chunks == []


def test_blocked_by_policy_always_false(mock_generator: MagicMock) -> None:
    strategy = PureLLMStrategy(mock_generator)
    result = strategy.answer("What is ScopeGuard?", _user_context("confidential"))
    assert result.blocked_by_policy is False


def test_cited_doc_ids_always_empty(mock_generator: MagicMock) -> None:
    strategy = PureLLMStrategy(mock_generator)
    result = strategy.answer("What is ScopeGuard?", _user_context("public"))
    assert result.generation.cited_doc_ids == []


def test_user_context_has_no_effect_on_output(mock_generator: MagicMock) -> None:
    strategy = PureLLMStrategy(mock_generator)
    query = "What is ScopeGuard's burn rate?"

    public_result = strategy.answer(query, _user_context("public"))
    internal_result = strategy.answer(query, _user_context("internal"))
    confidential_result = strategy.answer(query, _user_context("confidential"))

    assert public_result.generation == internal_result.generation == confidential_result.generation
    assert public_result.retrieved_chunks == internal_result.retrieved_chunks == []
    assert public_result.context_chunks == internal_result.context_chunks == []
    assert mock_generator.generate_no_context.call_count == 3


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_smoke_burn_rate_not_from_corpus() -> None:
    """
    Live smoke test: Strategy 0 must not surface the confidential burn rate
    ($480K/month) from the corpus — it has no retrieval.
    """
    strategy = PureLLMStrategy(Generator())
    query = "What is ScopeGuard's burn rate?"

    for level in ("public", "internal", "confidential"):
        result = strategy.answer(query, _user_context(level))
        answer = result.generation.answer.lower()

        assert result.retrieved_chunks == []
        assert result.context_chunks == []
        assert result.generation.cited_doc_ids == []
        assert "480k" not in answer.replace(",", "")
        assert "$480" not in result.generation.answer
