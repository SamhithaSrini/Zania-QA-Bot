import asyncio
import json
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.models.schemas import QAResult, RAGScores

JUDGE_SYSTEM = (
    "You are a balanced RAG evaluator for vendor security document QA. "
    "Score only from the question, answer, and returned citation snippets. "
    "Reward conservative 'not found' answers when the snippets do not contain enough evidence. "
    "Return valid JSON only."
)

JUDGE_TEMPLATE = """
Evaluate this answer on four RAG quality dimensions from 0 to 5.

Dimensions:
- faithfulness: answer claims are supported by the citation snippets.
- answer_relevance: answer directly addresses the question.
- completeness: answer covers all parts of the question without unsupported extras.
- citation_support: citations actually support the answer.

Important scoring guidance:
- If the answer says "Not found in the provided document" and the citations do not contain a direct answer, score faithfulness, answer_relevance, and completeness highly. Conservative refusal is good behavior.
- For "Not found" answers, citation_support should be moderate to high when the citations are topically related, even though snippets cannot prove the whole document lacks the answer.
- Do not punish an answer just because it is concise.

Question:
{question}

Answer:
{answer}

Citations:
{citations}

Return JSON exactly in this shape:
{{
  "faithfulness": 0,
  "answer_relevance": 0,
  "completeness": 0,
  "citation_support": 0,
  "rationale": "short explanation"
}}
"""


def citation_text(result: QAResult) -> str:
    if not result.citations:
        return "No citations returned."
    return "\n".join(
        f"- {citation.source}{', page ' + str(citation.page) if citation.page else ''}: "
        f"{citation.snippet}"
        for citation in result.citations
    )


def parse_judge_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def confidence_from_scores(scores: dict[str, float]) -> float:
    weights = {
        "faithfulness": 0.35,
        "answer_relevance": 0.25,
        "completeness": 0.25,
        "citation_support": 0.15,
    }
    weighted_score = sum(scores[key] * weight for key, weight in weights.items())
    return round(weighted_score / 5, 3)


async def score_result(llm: ChatOpenAI, result: QAResult) -> RAGScores:
    prompt = JUDGE_TEMPLATE.format(
        question=result.question,
        answer=result.answer,
        citations=citation_text(result),
    )
    response = await llm.ainvoke(
        [
            ("system", JUDGE_SYSTEM),
            ("human", prompt),
        ]
    )
    parsed = parse_judge_json(str(response.content))
    raw_scores = {
        "faithfulness": clamp_score(parsed.get("faithfulness")),
        "answer_relevance": clamp_score(parsed.get("answer_relevance")),
        "completeness": clamp_score(parsed.get("completeness")),
        "citation_support": clamp_score(parsed.get("citation_support")),
    }
    return RAGScores(
        **raw_scores,
        confidence=confidence_from_scores(raw_scores),
        rationale=str(parsed.get("rationale", "")),
    )


async def add_rag_scores(results: list[QAResult]) -> tuple[list[QAResult], float | None]:
    if not results:
        return results, None

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )
    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JUDGE_CALLS)

    async def guarded(result: QAResult) -> RAGScores:
        async with semaphore:
            return await score_result(llm, result)

    scores = await asyncio.gather(*[guarded(result) for result in results])
    for result, score in zip(results, scores, strict=True):
        result.rag_scores = score

    return results, round(sum(score.confidence for score in scores) / len(scores), 3)
