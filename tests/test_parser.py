import pytest

from app.core.exceptions import (
    EmptyDocumentError,
    MalformedQuestionsFileError,
    UnsupportedFileTypeError,
)
from app.services.parser import parse_document, parse_questions


def test_pdf_parsing_extracts_non_empty_text(sample_pdf_bytes: bytes) -> None:
    assert "sky is blue" in parse_document(sample_pdf_bytes, "document.pdf")


def test_json_doc_parsing_flattens_nested_values() -> None:
    file_bytes = b'{"a": "alpha", "b": {"c": "charlie"}, "d": ["delta", 12]}'

    assert parse_document(file_bytes, "document.json") == "alpha\ncharlie\ndelta"


def test_questions_parsed_from_flat_list_shape() -> None:
    assert parse_questions(b'["q1", "q2"]') == ["q1", "q2"]


def test_questions_parsed_from_object_shape() -> None:
    assert parse_questions(b'{"questions": ["q1", "q2"]}') == ["q1", "q2"]


def test_unsupported_file_type_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        parse_document(b"text", "document.txt")


def test_empty_document_raises() -> None:
    with pytest.raises(EmptyDocumentError):
        parse_document(b'{"content": "  "}', "document.json")


def test_malformed_questions_raises() -> None:
    with pytest.raises(MalformedQuestionsFileError):
        parse_questions(b'{"items": ["q1"]}')


def test_too_many_questions_raises_request_limit() -> None:
    payload = b'[' + b','.join([b'"q"'] * 26) + b']'

    with pytest.raises(Exception) as exc_info:
        parse_questions(payload)

    assert exc_info.value.status_code == 413
