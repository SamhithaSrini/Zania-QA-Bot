from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    snippet: str
    page: int | None = None


class QAResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class QAResponse(BaseModel):
    results: list[QAResult]
