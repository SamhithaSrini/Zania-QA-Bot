from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import QAResult, RAGScores

client = TestClient(app)


@patch("app.api.routes.answer_questions", new_callable=AsyncMock)
@patch("app.api.routes.add_rag_scores", new_callable=AsyncMock)
@patch("app.api.routes.build_chain")
@patch("app.api.routes.build_retriever")
def test_post_qa_happy_path(
    mock_build_retriever,
    mock_build_chain,
    mock_add_rag_scores,
    mock_answer_questions,
    sample_pdf_bytes: bytes,
    sample_questions_bytes: bytes,
) -> None:
    mock_build_retriever.return_value = MagicMock()
    mock_build_chain.return_value = MagicMock()
    mock_answer_questions.return_value = [
        QAResult(question="What color is the sky?", answer="Blue.")
    ]
    mock_add_rag_scores.return_value = (
        [
            QAResult(
                question="What color is the sky?",
                answer="Blue.",
                rag_scores=RAGScores(
                    faithfulness=5,
                    answer_relevance=5,
                    completeness=5,
                    citation_support=4,
                    confidence=0.95,
                    rationale="Supported by context.",
                ),
            )
        ],
        0.95,
    )

    response = client.post(
        "/api/v1/qa",
        files={
            "document": ("document.pdf", sample_pdf_bytes, "application/pdf"),
            "questions": ("questions.json", sample_questions_bytes, "application/json"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "overall_confidence": 0.95,
        "results": [
            {
                "question": "What color is the sky?",
                "answer": "Blue.",
                "citations": [],
                "rag_scores": {
                    "faithfulness": 5.0,
                    "answer_relevance": 5.0,
                    "completeness": 5.0,
                    "citation_support": 4.0,
                    "confidence": 0.95,
                    "rationale": "Supported by context.",
                },
            }
        ]
    }


def test_missing_document_field_returns_422(sample_questions_bytes: bytes) -> None:
    response = client.post(
        "/api/v1/qa",
        files={
            "questions": ("questions.json", sample_questions_bytes, "application/json"),
        },
    )

    assert response.status_code == 422


def test_wrong_document_type_returns_400(sample_questions_bytes: bytes) -> None:
    response = client.post(
        "/api/v1/qa",
        files={
            "document": ("document.txt", b"text", "text/plain"),
            "questions": ("questions.json", sample_questions_bytes, "application/json"),
        },
    )

    assert response.status_code == 400


def test_wrong_questions_type_returns_400(sample_pdf_bytes: bytes) -> None:
    response = client.post(
        "/api/v1/qa",
        files={
            "document": ("document.pdf", sample_pdf_bytes, "application/pdf"),
            "questions": ("questions.pdf", b"%PDF", "application/pdf"),
        },
    )

    assert response.status_code == 400


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_upload_ui() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Zania QA Bot" in response.text
    assert "/api/v1/qa" in response.text
