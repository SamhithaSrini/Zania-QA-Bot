import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ingest import build_retriever
from app.services.parser import parse_document
from app.services.qa import answer_questions, build_chain

NOT_FOUND = "not found in the provided document"


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(predicted: str, reference: str) -> float:
    predicted_tokens = normalize(predicted)
    reference_tokens = normalize(reference)
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0

    common = 0
    remaining = reference_tokens.copy()
    for token in predicted_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)

    if common == 0:
        return 0.0

    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(answer: str, required_keywords: list[str]) -> float:
    if not required_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in required_keywords if keyword.lower() in answer_lower)
    return hits / len(required_keywords)


def is_not_found(answer: str) -> bool:
    return NOT_FOUND in answer.lower()


async def evaluate(eval_path: Path) -> dict[str, Any]:
    dataset = json.loads(eval_path.read_text())
    document_path = Path(dataset["document"])
    document_bytes = document_path.read_bytes()
    document_text = parse_document(document_bytes, document_path.name)
    questions = [case["question"] for case in dataset["cases"]]

    build_start = time.perf_counter()
    retriever = await asyncio.to_thread(build_retriever, document_text)
    chain = build_chain(retriever)
    build_seconds = time.perf_counter() - build_start

    answer_start = time.perf_counter()
    results = await answer_questions(questions, chain)
    answer_seconds = time.perf_counter() - answer_start

    rows = []
    for case, result in zip(dataset["cases"], results, strict=True):
        answer = result.answer
        not_found = is_not_found(answer)
        answerable = bool(case["answerable"])
        rows.append(
            {
                "id": case["id"],
                "answerable": answerable,
                "not_found_correct": (not answerable and not_found)
                or (answerable and not not_found),
                "token_f1": token_f1(answer, case["reference_answer"]),
                "keyword_recall": keyword_recall(answer, case["required_keywords"]),
                "has_citation": bool(result.citations),
                "answer": answer,
            }
        )

    answerable_rows = [row for row in rows if row["answerable"]]
    unanswerable_rows = [row for row in rows if not row["answerable"]]
    summary = {
        "dataset": str(eval_path),
        "document": str(document_path),
        "case_count": len(rows),
        "answerable_count": len(answerable_rows),
        "unanswerable_count": len(unanswerable_rows),
        "build_seconds": round(build_seconds, 2),
        "answer_seconds": round(answer_seconds, 2),
        "avg_seconds_per_question": round(answer_seconds / len(rows), 2),
        "avg_token_f1": round(statistics.mean(row["token_f1"] for row in rows), 3),
        "avg_answerable_token_f1": round(
            statistics.mean(row["token_f1"] for row in answerable_rows), 3
        ),
        "avg_keyword_recall": round(
            statistics.mean(row["keyword_recall"] for row in answerable_rows), 3
        ),
        "unanswerable_accuracy": round(
            statistics.mean(row["not_found_correct"] for row in unanswerable_rows), 3
        ),
        "citation_rate": round(statistics.mean(row["has_citation"] for row in rows), 3),
    }
    return {"summary": summary, "rows": rows}


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Evaluation summary")
    print("==================")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nPer-question results")
    print("====================")
    for row in report["rows"]:
        print(
            f"{row['id']}: f1={row['token_f1']:.3f} "
            f"keyword_recall={row['keyword_recall']:.3f} "
            f"not_found_correct={row['not_found_correct']} "
            f"citation={row['has_citation']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Zania QA Bot on a labeled dataset.")
    parser.add_argument(
        "--dataset",
        default="eval_data/productfruits_qa.json",
        type=Path,
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write full JSON results.",
    )
    args = parser.parse_args()

    report = asyncio.run(evaluate(args.dataset))
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
