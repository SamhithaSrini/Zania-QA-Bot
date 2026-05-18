from unittest.mock import MagicMock

import fitz
import pytest


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "The sky is blue.")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


@pytest.fixture
def sample_json_doc_bytes() -> bytes:
    return b'{"content": "The sky is blue."}'


@pytest.fixture
def sample_questions_bytes() -> bytes:
    return b'["What color is the sky?"]'


@pytest.fixture
def mock_retriever() -> MagicMock:
    return MagicMock(name="retriever")


@pytest.fixture
def mock_chain() -> MagicMock:
    chain = MagicMock(name="chain")
    chain.ainvoke.return_value = {"result": "Blue."}
    return chain
