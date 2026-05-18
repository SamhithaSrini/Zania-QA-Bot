from app.services.judge import confidence_from_scores


def test_confidence_weights_citation_support_less_than_core_answer_quality() -> None:
    confidence = confidence_from_scores(
        {
            "faithfulness": 5,
            "answer_relevance": 5,
            "completeness": 5,
            "citation_support": 2,
        }
    )

    assert confidence == 0.91
