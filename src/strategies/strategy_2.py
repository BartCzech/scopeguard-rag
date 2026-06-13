from generator import Generator
from models import UserContext
from retriever import Retriever
from strategies.base import StrategyBase, StrategyResult
from strategies.prompts import BASE_RAG_PROMPT


class PostRetrievalFilteringStrategy(StrategyBase):
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, user_context: UserContext) -> StrategyResult:
        # Step 1: Retrieve from FULL index (unfiltered) — same as Strategy 1
        all_results = self.retriever.retrieve(query, top_k=20)

        # Step 2: Filter by access level BEFORE passing to LLM
        filtered_results = [
            r for r in all_results if r.access_level in user_context.allowed_levels
        ][:5]  # Take top 5 that pass the filter

        # Step 3: Generate with clean context only
        generation = self.generator.generate(
            query=query,
            context=filtered_results,
            system_prompt=BASE_RAG_PROMPT,
        )

        return StrategyResult(
            generation=generation,
            retrieved_chunks=all_results,  # Everything that was searched
            context_chunks=filtered_results,  # Only what the LLM saw
            blocked_by_policy=False,
        )
