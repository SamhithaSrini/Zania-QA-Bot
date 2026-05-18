import asyncio

from fastapi import HTTPException
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.models.schemas import Citation, QAResult

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use only the provided context to answer the question. "
        'If the answer is not supported by the context, say "Not found in the provided document." '
        "Keep the answer concise.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


def build_chain(retriever) -> RetrievalQA:
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )


async def answer_questions(questions: list[str], chain: RetrievalQA) -> list[QAResult]:
    unique_questions = list(dict.fromkeys(questions))
    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS)

    async def invoke(question: str) -> dict:
        async with semaphore:
            return await chain.ainvoke({"query": question})

    try:
        responses = await asyncio.gather(*[invoke(question) for question in unique_questions])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_by_question = dict(zip(unique_questions, responses, strict=True))
    return [
        QAResult(
            question=question,
            answer=str(response_by_question[question]["result"]),
            citations=_citations_from_response(response_by_question[question]),
        )
        for question in questions
    ]


def _citations_from_response(response: dict) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[tuple[str, int | None, str]] = set()

    for index, document in enumerate(response.get("source_documents") or [], start=1):
        metadata = getattr(document, "metadata", {}) or {}
        content = " ".join(getattr(document, "page_content", "").split())
        if not content:
            continue

        snippet = content[:300]
        page = metadata.get("page")
        page = page + 1 if isinstance(page, int) else None
        source = str(metadata.get("source") or f"chunk-{index}")
        key = (source, page, snippet)
        if key in seen:
            continue

        citations.append(Citation(source=source, page=page, snippet=snippet))
        seen.add(key)

    return citations
