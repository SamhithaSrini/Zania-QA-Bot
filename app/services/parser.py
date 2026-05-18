import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fitz

from app.core.exceptions import (
    EmptyDocumentError,
    MalformedQuestionsFileError,
    RequestLimitError,
    UnsupportedFileTypeError,
)
from app.core.config import settings


def parse_questions(file_bytes: bytes) -> list[str]:
    try:
        payload = json.loads(file_bytes)
    except json.JSONDecodeError as exc:
        raise MalformedQuestionsFileError("Questions file must contain valid JSON") from exc

    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list) or not questions:
        raise MalformedQuestionsFileError(
            'Questions file must be a non-empty list or an object with a "questions" list'
        )

    if not all(isinstance(question, str) and question.strip() for question in questions):
        raise MalformedQuestionsFileError("Questions must be non-empty strings")

    normalized = [question.strip() for question in questions]
    if len(normalized) > settings.MAX_QUESTIONS:
        raise RequestLimitError(f"Questions file may contain at most {settings.MAX_QUESTIONS} questions")

    if any(len(question) > settings.MAX_QUESTION_CHARS for question in normalized):
        raise RequestLimitError(
            f"Each question may contain at most {settings.MAX_QUESTION_CHARS} characters"
        )

    return normalized


def parse_document(file_bytes: bytes, filename: str) -> str:
    if len(file_bytes) > settings.MAX_DOCUMENT_BYTES:
        raise RequestLimitError(
            f"Document may be at most {settings.MAX_DOCUMENT_BYTES} bytes"
        )

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = _parse_pdf(file_bytes)
    elif suffix == ".json":
        text = _parse_json_document(file_bytes)
    else:
        raise UnsupportedFileTypeError("Document file must be a .pdf or .json file")

    if not text.strip():
        raise EmptyDocumentError()
    return text


def _parse_pdf(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def _parse_json_document(file_bytes: bytes) -> str:
    try:
        payload = json.loads(file_bytes)
    except json.JSONDecodeError as exc:
        raise EmptyDocumentError("Document JSON must be valid JSON") from exc

    return "\n".join(_string_values(payload))


def _string_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)
