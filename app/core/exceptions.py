from fastapi import HTTPException


class UnsupportedFileTypeError(HTTPException):
    def __init__(self, detail: str = "Unsupported file type") -> None:
        super().__init__(status_code=400, detail=detail)


class MalformedQuestionsFileError(HTTPException):
    def __init__(self, detail: str = "Malformed questions file") -> None:
        super().__init__(status_code=422, detail=detail)


class EmptyDocumentError(HTTPException):
    def __init__(self, detail: str = "Document is empty") -> None:
        super().__init__(status_code=422, detail=detail)


class RequestLimitError(HTTPException):
    def __init__(self, detail: str = "Request exceeds configured limits") -> None:
        super().__init__(status_code=413, detail=detail)
