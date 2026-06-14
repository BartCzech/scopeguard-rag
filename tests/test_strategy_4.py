"""Tests for Strategy 4 — Taint-Aware Output Guard."""

from fakes import FakeGenerator, FakePolicyEngine, FakeRetriever

from strategies.strategy_4 import REFUSAL_MESSAGE, TaintAwareStrategy


class TestTaintAwareStrategy:
    def test_clean_answer_passes_through(self, mixed_results, public_user):
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(answer="ScopeGuard is a middleware [PUB-001]"),
            FakePolicyEngine(leaked=False),
        )
        result = strategy.answer("What is ScopeGuard?", public_user)

        assert result.generation.answer == "ScopeGuard is a middleware [PUB-001]"
        assert result.blocked_by_policy is False

    def test_leaked_answer_gets_blocked(self, mixed_results, public_user):
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(answer="Salary is $185K [CONF-003]"),
            FakePolicyEngine(leaked=True),
        )
        result = strategy.answer("What are the salaries?", public_user)

        assert result.generation.answer == REFUSAL_MESSAGE
        assert result.blocked_by_policy is True
        assert result.generation.cited_doc_ids == []

    def test_raw_response_preserved_on_block(self, mixed_results, public_user):
        original_answer = "Salary is $185K [CONF-003]"
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(answer=original_answer),
            FakePolicyEngine(leaked=True),
        )
        result = strategy.answer("What are the salaries?", public_user)

        assert result.generation.answer == REFUSAL_MESSAGE
        assert result.generation.raw_response == original_answer

    def test_retrieved_and_context_chunks_identical(self, mixed_results, public_user):
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(),
            FakePolicyEngine(leaked=False),
        )
        result = strategy.answer("test", public_user)

        assert result.retrieved_chunks == result.context_chunks

    def test_context_includes_restricted_chunks(self, mixed_results, public_user):
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(),
            FakePolicyEngine(leaked=False),
        )
        result = strategy.answer("test", public_user)

        levels = {r.access_level for r in result.context_chunks}
        assert "confidential" in levels

    def test_policy_engine_receives_restricted_chunks_only(self, mixed_results, public_user):
        policy = FakePolicyEngine(leaked=False)
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(),
            policy,
        )
        strategy.answer("test", public_user)

        for chunk in policy.last_restricted_chunks:
            assert chunk.access_level not in public_user.allowed_levels

    def test_policy_engine_receives_answer_text(self, mixed_results, public_user):
        policy = FakePolicyEngine(leaked=False)
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(answer="Some specific answer"),
            policy,
        )
        strategy.answer("test", public_user)

        assert policy.last_answer == "Some specific answer"

    def test_generator_receives_all_chunks(self, mixed_results, public_user):
        generator = FakeGenerator()
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            generator,
            FakePolicyEngine(leaked=False),
        )
        strategy.answer("test", public_user)

        context_levels = {r.access_level for r in generator.last_context}
        assert "public" in context_levels
        assert "confidential" in context_levels

    def test_system_prompt_uses_taint_aware_suffix(self, mixed_results, public_user):
        generator = FakeGenerator()
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            generator,
            FakePolicyEngine(leaked=False),
        )
        strategy.answer("test", public_user)

        assert "access level: public" in generator.last_system_prompt

    def test_confidential_user_no_restricted_chunks(self, mixed_results, confidential_user):
        policy = FakePolicyEngine(leaked=False)
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(),
            policy,
        )
        strategy.answer("test", confidential_user)

        assert policy.last_restricted_chunks == []

    def test_confidential_user_never_blocked(self, mixed_results, confidential_user):
        strategy = TaintAwareStrategy(
            FakeRetriever(mixed_results),
            FakeGenerator(),
            FakePolicyEngine(leaked=False),
        )
        result = strategy.answer("test", confidential_user)

        assert result.blocked_by_policy is False
