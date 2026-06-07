from generator import Generator
from models import UserContext
from retriever import Retriever
from strategies.base import StrategyBase, StrategyResult
from strategies.prompts import BASE_RAG_PROMPT, NAIVE_ACCESS_SUFFIX


class NaiveRAGStrategy(StrategyBase):
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, user_context: UserContext) -> StrategyResult:
        # Retrieve from FULL index — no scope filtering
        results = self.retriever.retrieve(query, top_k=5)

        # Build system prompt with access level instruction
        access_level = max(
            user_context.allowed_levels,
            key=lambda level: ["public", "internal", "confidential"].index(level),
        )
        system_prompt = BASE_RAG_PROMPT + NAIVE_ACCESS_SUFFIX.format(
            access_level=access_level,
        )

        # Generate — the LLM sees everything, including restricted docs
        generation = self.generator.generate(
            query=query,
            context=results,
            system_prompt=system_prompt,
        )

        return StrategyResult(
            generation=generation,
            retrieved_chunks=results,  # All chunks, unfiltered
            context_chunks=results,  # Same — the LLM saw everything
            blocked_by_policy=False,
        )
