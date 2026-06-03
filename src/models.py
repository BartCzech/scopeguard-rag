from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str  # "PUB-001", "CONF-003", "ADV-002"
    title: str
    access_level: str  # "public" | "internal" | "confidential"
    adversarial: bool
    content: str  # Raw markdown body (no frontmatter)


class Chunk(BaseModel):
    doc_id: str
    chunk_index: int  # Position within the document
    content: str
    access_level: str
    embedding: list[float] = Field(default_factory=list, exclude=True)


class RetrievalResult(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    access_level: str
    score: float  # Similarity score, higher = more relevant


class GenerationResult(BaseModel):
    answer: str
    cited_doc_ids: list[str]  # Doc IDs the LLM cited in its answer
    raw_response: str  # Full LLM response before parsing


class UserContext(BaseModel):
    user_id: str
    scopes: list[str]  # ["docs:public"], ["docs:public", "docs:internal"], etc.
    allowed_levels: list[str]  # Derived: ["public"], ["public", "internal"], etc.


class TestQuestion(BaseModel):
    question_id: str
    question: str
    category: str  # "public_only" | "internal_only" | "confidential_only"
    # | "cross_scope" | "unanswerable" | "adversarial"
    gold_answer: str  # Expected correct answer (for correctness scoring)
    gold_doc_ids: list[str]  # Documents that contain the answer
    expected_behavior: dict[str, str]  # {"docs:public": "answer", "docs:internal": "partial", ...}
    # Values: "answer" | "refuse" | "partial"
