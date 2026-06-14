from config import DEFAULT_TOP_K
from generator import Generator
from models import GenerationResult, UserContext
from policy_engine import PolicyEngine
from retriever import Retriever
from strategies.base import StrategyBase, StrategyResult
from strategies.prompts import BASE_RAG_PROMPT, TAINT_AWARE_SUFFIX

REFUSAL_MESSAGE = (
    "I'm unable to answer this question fully because it would require "
    "information from documents above your current access level. "
    "Please contact your administrator for elevated access."
)


class TaintAwareStrategy(StrategyBase):
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        policy_engine: PolicyEngine,
    ):
        self.retriever = retriever
        self.generator = generator
        self.policy_engine = policy_engine

    def answer(self, query: str, user_context: UserContext) -> StrategyResult:
        all_results = self.retriever.retrieve(query, top_k=DEFAULT_TOP_K)

        clean = [r for r in all_results if r.access_level in user_context.allowed_levels]
        restricted = [r for r in all_results if r.access_level not in user_context.allowed_levels]

        access_level = max(
            user_context.allowed_levels,
            key=lambda level: ["public", "internal", "confidential"].index(level),
        )
        system_prompt = BASE_RAG_PROMPT + TAINT_AWARE_SUFFIX.format(
            access_level=access_level,
        )

        generation = self.generator.generate(
            query=query,
            context=all_results,
            system_prompt=system_prompt,
        )

        allowed_doc_ids = [r.doc_id for r in clean]
        check_result = self.policy_engine.check(
            answer=generation.answer,
            restricted_chunks=restricted,
            allowed_doc_ids=allowed_doc_ids,
        )

        if check_result.leaked:
            blocked_generation = GenerationResult(
                answer=REFUSAL_MESSAGE,
                cited_doc_ids=[],
                raw_response=generation.raw_response,
            )
            return StrategyResult(
                generation=blocked_generation,
                retrieved_chunks=all_results,
                context_chunks=all_results,
                blocked_by_policy=True,
            )

        return StrategyResult(
            generation=generation,
            retrieved_chunks=all_results,
            context_chunks=all_results,
            blocked_by_policy=False,
        )
