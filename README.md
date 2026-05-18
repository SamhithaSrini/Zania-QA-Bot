# Zania QA Bot

FastAPI service for answering batches of questions against an uploaded PDF or JSON document. Each request parses the document, builds an in-memory FAISS index, reranks retrieved chunks with a HuggingFace cross-encoder, and asks `gpt-4o-mini` to answer each question.

```text
Upload document + questions
        |
        v
Parse PDF/JSON -> chunk text -> FAISS similarity search
        |                            |
        |                            v
        |                    cross-encoder rerank
        |                            |
        v                            v
     questions --------------> gpt-4o-mini
        |
        v
    JSON answers
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```text
OPENAI_API_KEY=sk-...
```

By default, the app uses FAISS vector retrieval only for fast local demos. Set `USE_RERANKER=true` to enable the HuggingFace cross-encoder reranker; the model downloads on first use and adds latency.

The API also enforces configurable safety limits for document size, question count, question length, and processing timeout. See `.env.example`.

## Run

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000/` for a minimal upload UI, or call the API directly.

## Example Requests

PDF document:

```bash
curl -X POST http://localhost:8000/api/v1/qa \
  -F "document=@sample_data/document.pdf" \
  -F "questions=@sample_data/questions.json"
```

JSON document:

```bash
curl -X POST http://localhost:8000/api/v1/qa \
  -F "document=@sample_data/document.json" \
  -F "questions=@sample_data/questions.json"
```

Example response:

```json
{
  "results": [
    {
      "question": "What is the main topic of the document?",
      "answer": "The document is about automated document question answering.",
      "citations": [
        {
          "source": "chunk-1",
          "snippet": "This sample document is about automated document question answering.",
          "page": null
        }
      ]
    }
  ]
}
```

## Docker

```bash
docker compose up --build
```

The service listens on `http://localhost:8000`.

## Tests

```bash
pytest -v
```

## Design Decisions

The service can use two-stage retrieval. FAISS first performs fast vector similarity search over document chunks. When `USE_RERANKER=true`, the cross-encoder reranks those candidates by jointly scoring each question and chunk. This adds latency, but it can improve answer quality because the LLM receives fewer and more relevant chunks.

The FAISS index is intentionally in-memory and per request. That keeps the API stateless and simple for uploaded documents, while the route runs index construction in an executor so synchronous embedding, FAISS, and cross-encoder setup work does not block the event loop.

For multiple questions, answers are generated concurrently with a configurable cap (`MAX_CONCURRENT_QUESTIONS`) to avoid overwhelming the LLM provider. Duplicate questions in the same request are deduplicated before retrieval and generation, then mapped back into the original response order to avoid unnecessary retriever and LLM calls.

The QA prompt instructs the model to answer only from retrieved context and to return `Not found in the provided document.` when the evidence is missing. Responses include short citations from retrieved source chunks to make answer grounding inspectable.

Requests are logged as JSON and include status code and latency. API responses also include an `X-Process-Time-Ms` header.

## Appendix Sample Files

The bundled `sample_data/questions.json` contains the appendix questions. External evaluation examples provided with the prompt:

- Sample JSON spreadsheet: `https://docs.google.com/spreadsheets/d/1u7z18yNKsL8cMLV6OxYI1-8ageRfFG1j/edit`
- Sample PDF: `https://productfruits.com/docs/soc2-type2.pdf`
