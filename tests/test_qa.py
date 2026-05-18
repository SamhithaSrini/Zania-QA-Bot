from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document
import pytest

from app.models.schemas import Citation, QAResult
from app.services.qa import answer_questions


@pytest.mark.asyncio
async def test_answer_questions_returns_qa_results() -> None:
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value={"result": "Blue."})

    results = await answer_questions(["What color is the sky?"], chain)

    assert results == [QAResult(question="What color is the sky?", answer="Blue.")]


@pytest.mark.asyncio
async def test_multiple_questions_resolved_in_parallel() -> None:
    chain = MagicMock()
    chain.ainvoke = AsyncMock(side_effect=[{"result": "Blue."}, {"result": "Clear."}])

    results = await answer_questions(["Color?", "Condition?"], chain)

    assert [result.answer for result in results] == ["Blue.", "Clear."]
    assert chain.ainvoke.await_count == 2
    chain.ainvoke.assert_any_await({"query": "Color?"})
    chain.ainvoke.assert_any_await({"query": "Condition?"})


@pytest.mark.asyncio
async def test_duplicate_questions_reuse_one_llm_call() -> None:
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value={"result": "Blue."})

    results = await answer_questions(["Color?", "Color?"], chain)

    assert [result.answer for result in results] == ["Blue.", "Blue."]
    assert chain.ainvoke.await_count == 1
    chain.ainvoke.assert_awaited_once_with({"query": "Color?"})


@pytest.mark.asyncio
async def test_answer_questions_includes_source_citations() -> None:
    chain = MagicMock()
    chain.ainvoke = AsyncMock(
        return_value={
            "result": "Blue.",
            "source_documents": [
                Document(
                    page_content="The sky is blue because of scattered sunlight.",
                    metadata={"source": "document.pdf", "page": 2},
                )
            ],
        }
    )

    results = await answer_questions(["What color is the sky?"], chain)

    assert results == [
        QAResult(
            question="What color is the sky?",
            answer="Blue.",
            citations=[
                Citation(
                    source="document.pdf",
                    page=3,
                    snippet="The sky is blue because of scattered sunlight.",
                )
            ],
        )
    ]
