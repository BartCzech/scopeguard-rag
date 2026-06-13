from config import DEFAULT_TOP_K
from generator import Generator
from models import UserContext
from retriever import Retriever
from strategies.base import StrategyBase, StrategyResult
from strategies.prompts import BASE_RAG_PROMPT


class PreRetrievalFilteringStrategy(StrategyBase):
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, user_context: UserContext) -> StrategyResult:
        results = self.retriever.retrieve_filtered(
            query=query,
            allowed_levels=user_context.allowed_levels,
            top_k=DEFAULT_TOP_K,
        )

        generation = self.generator.generate(
            query=query,
            context=results,
            system_prompt=BASE_RAG_PROMPT,
        )

        return StrategyResult(
            generation=generation,
            retrieved_chunks=results,
            context_chunks=results,
            blocked_by_policy=False,
        )
