from abc import ABC, abstractmethod

from models import GenerationResult, RetrievalResult, UserContext
from pydantic import BaseModel


class StrategyResult(BaseModel):
    """Returned by every strategy. The evaluation harness consumes this."""

    generation: GenerationResult
    retrieved_chunks: list[RetrievalResult]  # What was retrieved (all chunks, pre-filter)
    context_chunks: list[RetrievalResult]  # What the LLM actually saw (post-filter)
    blocked_by_policy: bool = False  # True if PolicyEngine blocked the answer (S4 only)


class StrategyBase(ABC):
    @abstractmethod
    def answer(self, query: str, user_context: UserContext) -> StrategyResult:
        """Given a query and user context, return a full strategy result."""
