from generator import Generator
from models import UserContext
from strategies.base import StrategyBase, StrategyResult
from strategies.prompts import BASE_NO_RAG_PROMPT


class PureLLMStrategy(StrategyBase):
    def __init__(self, generator: Generator):
        self.generator = generator

    def answer(self, query: str, user_context: UserContext) -> StrategyResult:
        # No retrieval — LLM answers from parametric knowledge only
        generation = self.generator.generate_no_context(
            query=query,
            system_prompt=BASE_NO_RAG_PROMPT,
        )

        return StrategyResult(
            generation=generation,
            retrieved_chunks=[],  # Nothing retrieved
            context_chunks=[],  # Nothing in context
            blocked_by_policy=False,  # No PolicyEngine involved
        )
