"""Tests for Strategy 3 — Pre-Retrieval Filtering."""

import inspect

import pytest
from fakes import FakeGenerator, FakeRetriever

import strategies.strategy_3 as strategy_3_module
from config import DEFAULT_TOP_K
from strategies.strategy_3 import PreRetrievalFilteringStrategy


class TestPreRetrievalFiltering:
    @pytest.mark.parametrize("user_fixture", ["public_user", "internal_user", "confidential_user"])
    def test_all_chunks_within_allowed_levels(self, user_fixture, mixed_results, request):
        user = request.getfixturevalue(user_fixture)
        strategy = PreRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", user)

        for chunk in result.context_chunks:
            assert chunk.access_level in user.allowed_levels

    def test_retrieved_and_context_chunks_are_identical(self, mixed_results, public_user):
        strategy = PreRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        assert result.retrieved_chunks == result.context_chunks

    def test_no_restricted_chunks_in_retrieved(self, mixed_results, public_user):
        strategy = PreRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        for chunk in result.retrieved_chunks:
            assert chunk.access_level in public_user.allowed_levels

    def test_context_chunks_max_top_k(self, mixed_results, public_user):
        strategy = PreRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        assert len(result.context_chunks) <= DEFAULT_TOP_K

    def test_may_return_fewer_than_top_k(self, public_user):
        """When few documents match the scope, fewer than top_k results is expected."""
        from fakes import make_chunk

        sparse_results = [
            make_chunk("CONF-003", "confidential", 0.95),
            make_chunk("CONF-006", "confidential", 0.90),
            make_chunk("PUB-001", "public", 0.50),
        ]
        strategy = PreRetrievalFilteringStrategy(FakeRetriever(sparse_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        assert len(result.context_chunks) < DEFAULT_TOP_K
        assert len(result.context_chunks) == 1

    def test_generator_receives_only_filtered_chunks(self, mixed_results, public_user):
        generator = FakeGenerator()
        strategy = PreRetrievalFilteringStrategy(FakeRetriever(mixed_results), generator)
        strategy.answer("test query", public_user)

        for chunk in generator.last_context:
            assert chunk.access_level == "public"

    def test_blocked_by_policy_always_false(self, mixed_results, public_user):
        strategy = PreRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        assert result.blocked_by_policy is False

    def test_uses_retrieve_filtered_not_retrieve(self):
        source = inspect.getsource(strategy_3_module)
        assert "retrieve_filtered" in source
        assert "retriever.retrieve(" not in source

    def test_filtered_results_preserve_score_order(self, mixed_results, public_user):
        strategy = PreRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        scores = [r.score for r in result.context_chunks]
        assert scores == sorted(scores, reverse=True)
