"""Tests for Strategy 2 — Post-Retrieval Filtering."""

from fakes import FakeGenerator, FakeRetriever

from config import DEFAULT_TOP_K, FILTERED_OVERFETCH
from strategies.strategy_2 import PostRetrievalFilteringStrategy


class TestPostRetrievalFiltering:
    def test_context_chunks_contain_only_allowed_levels(self, mixed_results, public_user):
        strategy = PostRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        for chunk in result.context_chunks:
            assert chunk.access_level == "public"

    def test_retrieved_chunks_contain_all_levels(self, mixed_results, public_user):
        strategy = PostRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        levels = {r.access_level for r in result.retrieved_chunks}
        assert "confidential" in levels
        assert "public" in levels

    def test_context_chunks_max_top_k(self, mixed_results, public_user):
        strategy = PostRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        assert len(result.context_chunks) <= DEFAULT_TOP_K

    def test_generator_receives_only_filtered_chunks(self, mixed_results, public_user):
        generator = FakeGenerator()
        strategy = PostRetrievalFilteringStrategy(FakeRetriever(mixed_results), generator)
        strategy.answer("test query", public_user)

        for chunk in generator.last_context:
            assert chunk.access_level == "public"

    def test_confidential_user_sees_all_levels(self, mixed_results, confidential_user):
        strategy = PostRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", confidential_user)

        levels = {r.access_level for r in result.context_chunks}
        assert levels == {"public", "internal", "confidential"}

    def test_blocked_by_policy_always_false(self, mixed_results, public_user):
        strategy = PostRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        assert result.blocked_by_policy is False

    def test_filtered_results_preserve_score_order(self, mixed_results, public_user):
        strategy = PostRetrievalFilteringStrategy(FakeRetriever(mixed_results), FakeGenerator())
        result = strategy.answer("test query", public_user)

        scores = [r.score for r in result.context_chunks]
        assert scores == sorted(scores, reverse=True)

    def test_retriever_called_with_overfetch(self, mixed_results, public_user):
        retriever = FakeRetriever(mixed_results)
        strategy = PostRetrievalFilteringStrategy(retriever, FakeGenerator())
        strategy.answer("test query", public_user)

        assert retriever.last_top_k == FILTERED_OVERFETCH
