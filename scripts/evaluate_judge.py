import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings

JUDGE_SYSTEM = (
    "You are a strict evaluator for retrieval-augmented QA over vendor security documents. "
    "Score only from the question, answer, reference answer when present, and returned citations. "
    "Return valid JSON only."
)

JUDGE_TEMPLATE = """
Evaluate this QA result on four dimensions from 0 to 5.

Dimensions:
- faithfulness: answer claims are supported by the citation snippets/reference.
- answer_relevance: answer directly addresses the question.
- completeness: answer covers all parts of the question without adding unsupported extras.
- citation_support: citations actually support the answer.

For unanswerable questions, a faithful and complete answer should say that the answer is not found in the provided document.

Question:
{question}

Expected answerability: {answerable}

Reference answer:
{reference_answer}

Bot answer:
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


def citation_text(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "No citations returned."
    return "\n".join(
        f"- {citation.get('source', 'unknown')}"
        f"{', page ' + str(citation['page']) if citation.get('page') else ''}: "
        f"{citation.get('snippet', '')}"
        for citation in citations
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


async def judge_row(llm: ChatOpenAI, row: dict[str, Any]) -> dict[str, Any]:
    prompt = JUDGE_TEMPLATE.format(
        question=row["question"],
        answerable=row["answerable"],
        reference_answer=row["reference_answer"],
        answer=row["answer"],
        citations=citation_text(row.get("citations", [])),
    )
    response = await llm.ainvoke(
        [
            ("system", JUDGE_SYSTEM),
            ("human", prompt),
        ]
    )
    parsed = parse_judge_json(str(response.content))
    scores = {
        "faithfulness": clamp_score(parsed.get("faithfulness")),
        "answer_relevance": clamp_score(parsed.get("answer_relevance")),
        "completeness": clamp_score(parsed.get("completeness")),
        "citation_support": clamp_score(parsed.get("citation_support")),
    }
    return {
        "id": row["id"],
        "scores": scores,
        "rationale": str(parsed.get("rationale", "")),
    }


async def evaluate_judge(results_path: Path, concurrency: int) -> dict[str, Any]:
    base_report = json.loads(results_path.read_text())
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await judge_row(llm, row)

    judgments = await asyncio.gather(*[guarded(row) for row in base_report["rows"]])
    dimensions = ["faithfulness", "answer_relevance", "completeness", "citation_support"]
    summary = {
        "input": str(results_path),
        "case_count": len(judgments),
        "judge_model": settings.LLM_MODEL,
    }
    for dimension in dimensions:
        summary[f"avg_{dimension}"] = round(
            statistics.mean(judgment["scores"][dimension] for judgment in judgments), 3
        )

    return {"summary": summary, "judgments": judgments}


def print_report(report: dict[str, Any]) -> None:
    print("LLM-as-judge summary")
    print("====================")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")

    print("\nPer-question judge scores")
    print("=========================")
    for judgment in report["judgments"]:
        scores = judgment["scores"]
        print(
            f"{judgment['id']}: "
            f"faithfulness={scores['faithfulness']:.1f} "
            f"relevance={scores['answer_relevance']:.1f} "
            f"completeness={scores['completeness']:.1f} "
            f"citation_support={scores['citation_support']:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge over QA eval results.")
    parser.add_argument(
        "--results",
        default="eval_results/productfruits_results.json",
        type=Path,
        help="Path to JSON output from scripts/evaluate.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write judge results JSON.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent judge calls.",
    )
    args = parser.parse_args()

    report = asyncio.run(evaluate_judge(args.results, args.concurrency))
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
