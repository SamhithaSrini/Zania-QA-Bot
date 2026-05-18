from scripts.evaluate_judge import citation_text, clamp_score, parse_judge_json


def test_parse_judge_json_accepts_plain_json() -> None:
    parsed = parse_judge_json('{"faithfulness": 5}')

    assert parsed == {"faithfulness": 5}


def test_parse_judge_json_accepts_fenced_json() -> None:
    parsed = parse_judge_json('```json\n{"faithfulness": 4}\n```')

    assert parsed == {"faithfulness": 4}


def test_clamp_score_limits_range() -> None:
    assert clamp_score(7) == 5.0
    assert clamp_score(-2) == 0.0
    assert clamp_score("3.5") == 3.5
    assert clamp_score("nope") == 0.0


def test_citation_text_formats_snippets() -> None:
    text = citation_text(
        [
            {
                "source": "document.pdf",
                "page": 3,
                "snippet": "AWS hosts the service.",
            }
        ]
    )

    assert "document.pdf, page 3" in text
    assert "AWS hosts the service." in text
