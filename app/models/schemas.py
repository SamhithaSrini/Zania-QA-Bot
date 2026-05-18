from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    snippet: str
    page: int | None = None


class RAGScores(BaseModel):
    faithfulness: float
    answer_relevance: float
    completeness: float
    citation_support: float
    confidence: float
    rationale: str = ""


class QAResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    rag_scores: RAGScores | None = None


class QAResponse(BaseModel):
    results: list[QAResult]
    overall_confidence: float | None = None
