import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if settings.USE_RERANKER:
        logger.info(
            "HuggingFace cross-encoder model will be downloaded on first request if not already cached."
        )
    else:
        logger.info("Cross-encoder reranker disabled; using FAISS vector retrieval only.")
    yield


app = FastAPI(title="Zania QA Bot", version="1.0.0", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")


@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    logger.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.get("/", response_class=HTMLResponse)
async def upload_form() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Zania QA Bot</title>
        <style>
          :root {
            color-scheme: light;
            --bg: #f5f7fb;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #68758b;
            --line: #d9e0ea;
            --accent: #1f7a6d;
            --accent-dark: #155e54;
            --warn: #a45b12;
            --bad: #b42318;
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            min-height: 100vh;
            background: var(--bg);
            color: var(--ink);
            font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }
          main {
            width: min(1120px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 28px 0 44px;
          }
          header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 18px;
            margin-bottom: 18px;
          }
          h1 {
            margin: 0;
            font-size: clamp(28px, 4vw, 42px);
            line-height: 1.05;
            letter-spacing: 0;
          }
          .subhead {
            margin: 8px 0 0;
            color: var(--muted);
            max-width: 680px;
          }
          .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 11px;
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 999px;
            color: var(--muted);
            white-space: nowrap;
          }
          .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--accent);
          }
          .workspace {
            display: grid;
            grid-template-columns: minmax(280px, 360px) 1fr;
            gap: 18px;
            align-items: start;
          }
          .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 30px rgba(23, 32, 51, .06);
          }
          form.panel {
            padding: 18px;
            display: grid;
            gap: 16px;
          }
          .field {
            display: grid;
            gap: 8px;
          }
          label {
            font-weight: 700;
            color: #263449;
          }
          input[type="file"] {
            width: 100%;
            padding: 11px;
            border: 1px dashed #aeb8c7;
            border-radius: 8px;
            background: #fbfcfe;
            color: var(--muted);
          }
          .file-name {
            min-height: 20px;
            color: var(--muted);
            font-size: 13px;
            overflow-wrap: anywhere;
          }
          button {
            border: 0;
            border-radius: 8px;
            padding: 12px 15px;
            background: var(--accent);
            color: white;
            font-weight: 800;
            cursor: pointer;
          }
          button:hover { background: var(--accent-dark); }
          button:disabled { opacity: .65; cursor: wait; }
          .meta {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
          }
          .metric {
            padding: 10px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fbfcfe;
          }
          .metric strong {
            display: block;
            font-size: 18px;
          }
          .metric span {
            color: var(--muted);
            font-size: 12px;
          }
          .results.panel {
            min-height: 480px;
            padding: 0;
            overflow: hidden;
          }
          .results-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 15px 18px;
            border-bottom: 1px solid var(--line);
          }
          .results-head h2 {
            margin: 0;
            font-size: 17px;
          }
          #result-status {
            color: var(--muted);
            font-size: 13px;
            text-align: right;
          }
          #output {
            display: grid;
            gap: 12px;
            padding: 16px;
          }
          .empty {
            min-height: 360px;
            display: grid;
            place-items: center;
            text-align: center;
            color: var(--muted);
          }
          .answer {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            background: #ffffff;
          }
          .answer h3 {
            margin: 0 0 8px;
            font-size: 15px;
          }
          .answer p {
            margin: 0;
          }
          .citations {
            margin-top: 12px;
            display: grid;
            gap: 8px;
          }
          .citation {
            border-left: 3px solid var(--accent);
            background: #f2faf8;
            padding: 9px 10px;
            color: #304258;
            font-size: 13px;
          }
          .error {
            border-color: #f0b8b3;
            background: #fff7f6;
            color: var(--bad);
          }
          @media (max-width: 760px) {
            header { display: block; }
            .status-pill { margin-top: 14px; }
            .workspace { grid-template-columns: 1fr; }
            .meta { grid-template-columns: 1fr; }
          }
        </style>
      </head>
      <body>
        <main>
          <header>
            <div>
              <h1>Zania QA Bot</h1>
              <p class="subhead">Upload a vendor document and a questions JSON file to generate grounded answers with citations.</p>
            </div>
            <div class="status-pill"><span class="status-dot"></span><span>FAISS retrieval mode</span></div>
          </header>

          <section class="workspace">
            <form id="qa-form" class="panel">
              <div class="field">
                <label for="document">Document</label>
                <input id="document" name="document" type="file" accept=".pdf,.json" required />
                <div class="file-name" data-for="document">PDF or JSON</div>
              </div>
              <div class="field">
                <label for="questions">Questions</label>
                <input id="questions" name="questions" type="file" accept=".json" required />
                <div class="file-name" data-for="questions">JSON list of questions</div>
              </div>
              <button id="submit-button" type="submit">Ask Questions</button>
              <div class="meta">
                <div class="metric"><strong>25</strong><span>max questions</span></div>
                <div class="metric"><strong>5</strong><span>concurrent answers</span></div>
              </div>
            </form>

            <section class="results panel">
              <div class="results-head">
                <h2>Answers</h2>
                <div id="result-status">Ready</div>
              </div>
              <div id="output" class="empty">Results will appear here after processing.</div>
            </section>
          </section>
        </main>
        <script>
          const form = document.querySelector("#qa-form");
          const output = document.querySelector("#output");
          const status = document.querySelector("#result-status");
          const button = document.querySelector("#submit-button");
          for (const input of form.querySelectorAll('input[type="file"]')) {
            input.addEventListener("change", () => {
              const target = document.querySelector(`[data-for="${input.id}"]`);
              target.textContent = input.files[0]?.name || target.textContent;
            });
          }
          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const started = performance.now();
            button.disabled = true;
            status.textContent = "Processing...";
            output.className = "empty";
            output.textContent = "Reading document and asking questions...";
            try {
              const response = await fetch("/api/v1/qa", { method: "POST", body: new FormData(form) });
              const contentType = response.headers.get("content-type") || "";
              const payload = contentType.includes("application/json")
                ? await response.json()
                : { detail: await response.text() };
              if (!response.ok) throw new Error(payload.detail || "Request failed");
              output.className = "";
              output.innerHTML = payload.results.map(renderAnswer).join("");
              const seconds = ((performance.now() - started) / 1000).toFixed(1);
              status.textContent = `${payload.results.length} answers in ${seconds}s`;
            } catch (error) {
              output.className = "answer error";
              output.textContent = error.message;
              status.textContent = "Failed";
            } finally {
              button.disabled = false;
            }
          });
          function escapeHtml(value) {
            return String(value).replace(/[&<>"']/g, (char) => ({
              "&": "&amp;",
              "<": "&lt;",
              ">": "&gt;",
              '"': "&quot;",
              "'": "&#039;"
            }[char]));
          }
          function renderAnswer(item) {
            const citations = (item.citations || []).map((citation) => `
              <div class="citation">
                <strong>${escapeHtml(citation.source)}${citation.page ? `, page ${citation.page}` : ""}</strong>
                <div>${escapeHtml(citation.snippet)}</div>
              </div>
            `).join("");
            return `
              <article class="answer">
                <h3>${escapeHtml(item.question)}</h3>
                <p>${escapeHtml(item.answer)}</p>
                ${citations ? `<div class="citations">${citations}</div>` : ""}
              </article>
            `;
          }
        </script>
      </body>
    </html>
    """


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


configure_logging()
