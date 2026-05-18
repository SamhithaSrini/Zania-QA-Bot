from scripts.evaluate import is_not_found, keyword_recall, token_f1


def test_token_f1_scores_overlap() -> None:
    assert token_f1("AWS Europe hosting", "AWS hosting in Europe") > 0.7


def test_token_f1_scores_no_overlap() -> None:
    assert token_f1("GitHub", "Microsoft Office") == 0.0


def test_keyword_recall_counts_required_keywords() -> None:
    answer = "Product Fruits uses AWS and GitHub."

    assert keyword_recall(answer, ["AWS", "GitHub", "Office 365"]) == 2 / 3


def test_not_found_detection() -> None:
    assert is_not_found("Not found in the provided document.")
