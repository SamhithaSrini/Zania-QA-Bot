import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.config import settings
from app.core.exceptions import UnsupportedFileTypeError
from app.models.schemas import QAResponse
from app.services.ingest import build_retriever
from app.services.parser import parse_document, parse_questions
from app.services.qa import answer_questions, build_chain

router = APIRouter()
logger = logging.getLogger(__name__)


def _extension(filename: str) -> str:
    return Path(filename or "").suffix.lower()


@router.post("/qa", response_model=QAResponse)
async def qa(document: UploadFile, questions: UploadFile) -> QAResponse:
    start = time.perf_counter()
    if _extension(questions.filename) != ".json":
        raise UnsupportedFileTypeError("Questions file must be a .json file")

    if _extension(document.filename) not in {".pdf", ".json"}:
        raise UnsupportedFileTypeError("Document file must be a .pdf or .json file")

    document_bytes, questions_bytes = await asyncio.gather(document.read(), questions.read())
    parsed_questions = parse_questions(questions_bytes)
    document_text = parse_document(document_bytes, document.filename or "")

    try:
        loop = asyncio.get_running_loop()
        retriever = await asyncio.wait_for(
            loop.run_in_executor(None, build_retriever, document_text),
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        chain = build_chain(retriever)
        results = await asyncio.wait_for(
            answer_questions(parsed_questions, chain),
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        response = QAResponse(results=results)
        logger.info(
            "qa_request_completed",
            extra={
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                "question_count": len(parsed_questions),
                "document_filename": document.filename,
            },
        )
        return response
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Request timed out while processing the document or questions",
        ) from exc
