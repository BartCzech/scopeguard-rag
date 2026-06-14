"""Shared fakes for strategy and evaluation tests."""

from models import GenerationResult, RetrievalResult
from policy_engine import LeakageCheckResult


def make_chunk(doc_id: str, access_level: str, score: float) -> RetrievalResult:
    """Create a fake retrieval result."""
    return RetrievalResult(
        doc_id=doc_id,
        chunk_index=0,
        content=f"Content of {doc_id}",
        access_level=access_level,
        score=score,
    )


class FakeRetriever:
    """Predictable retriever that returns pre-set results."""

    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.last_query = query
        self.last_top_k = top_k
        return self.results[:top_k]

    def retrieve_filtered(
        self, query: str, allowed_levels: list[str], top_k: int = 5
    ) -> list[RetrievalResult]:
        self.last_query = query
        self.last_top_k = top_k
        filtered = [r for r in self.results if r.access_level in allowed_levels]
        return filtered[:top_k]


class FakeGenerator:
    """Predictable generator that captures what context it received."""

    def __init__(self, answer: str = "Test answer [PUB-001]", cited: list[str] | None = None):
        self.answer = answer
        self.cited = cited or ["PUB-001"]
        self.last_context: list[RetrievalResult] | None = None
        self.last_system_prompt: str | None = None

    def generate(
        self, query: str, context: list[RetrievalResult], system_prompt: str
    ) -> GenerationResult:
        self.last_context = context
        self.last_system_prompt = system_prompt
        return GenerationResult(
            answer=self.answer,
            cited_doc_ids=self.cited,
            raw_response=self.answer,
        )

    def generate_no_context(self, query: str, system_prompt: str) -> GenerationResult:
        self.last_system_prompt = system_prompt
        self.last_context = []
        return GenerationResult(
            answer=self.answer,
            cited_doc_ids=[],
            raw_response=self.answer,
        )


class FakePolicyEngine:
    """Predictable PolicyEngine that returns pre-set results."""

    def __init__(self, leaked: bool = False, ngram_score: float = 0.0):
        self.leaked = leaked
        self.ngram_score = ngram_score
        self.last_answer: str | None = None
        self.last_restricted_chunks: list | None = None

    def check(self, answer, restricted_chunks, allowed_doc_ids):
        self.last_answer = answer
        self.last_restricted_chunks = restricted_chunks
        result = LeakageCheckResult()
        result.leaked = self.leaked
        result.ngram_score = self.ngram_score
        if self.leaked:
            result.reasons = ["Fake: leakage detected"]
        return result
