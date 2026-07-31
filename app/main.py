import os
import json
import asyncio
import uuid
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
)

from fastapi.staticfiles import StaticFiles

from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    FileResponse,
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

from .database import db
from .search_engine import search_engine
from .worker import process_document_background


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Enterprise Search & RAG Engine",
    description="Enterprise document ingestion, semantic search and RAG platform",
    version="2.1.0",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# GEMINI
# ============================================================

if GEMINI_API_KEY:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    print("Gemini client configured.")

else:

    client = None

    print(
        "WARNING: GEMINI_API_KEY is not configured."
    )


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATIC FILES
# ============================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(STATIC_DIR)
        ),
        name="static",
    )

else:

    print(
        f"WARNING: Static directory missing: "
        f"{STATIC_DIR}"
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):
    query: str


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def serve_index():

    index_path = (
        STATIC_DIR / "index.html"
    )

    if not index_path.exists():

        return HTMLResponse(
            content=(
                "<h1>index.html missing "
                "from static directory.</h1>"
            ),
            status_code=404,
        )

    return FileResponse(
        index_path
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
async def health_check():

    return {
        "status": "healthy",
        "service": (
            "Enterprise Search & RAG Engine"
        ),
        "gemini_configured": (
            client is not None
        ),
    }


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file was selected.",
        )

    try:

        doc_id = str(
            uuid.uuid4()
        )

        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        db.add_document(
            doc_id,
            file.filename,
        )

        background_tasks.add_task(
            process_document_background,
            file_bytes,
            file.filename,
            doc_id,
        )

        return {
            "message": (
                "Document queued for processing"
            ),
            "document_id": doc_id,
            "filename": file.filename,
            "status": "processing",
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            f"Document upload error: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to upload document."
            ),
        )


# ============================================================
# LIST DOCUMENTS
# ============================================================

@app.get("/api/documents")
async def list_documents():

    try:

        return {
            "documents":
                db.get_all_documents()
        }

    except Exception as error:

        print(
            f"Document listing error: "
            f"{error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve documents."
            ),
        )


# ============================================================
# GEMINI GENERATION
# ============================================================

async def generate_with_gemini(
    prompt: str,
    max_retries: int = 3,
):

    if client is None:

        return None

    for attempt in range(
        max_retries
    ):

        try:

            print(
                "Gemini request attempt "
                f"{attempt + 1}/"
                f"{max_retries}"
            )

            response = (
                await asyncio.to_thread(
                    client.models.generate_content,
                    model=(
                        "gemini-3.5-flash-lite"
                    ),
                    contents=prompt,
                )
            )

            if (
                response
                and response.text
            ):

                print(
                    "Gemini response received."
                )

                return (
                    response.text.strip()
                )

            print(
                "Gemini returned empty response."
            )

        except Exception as error:

            print(
                "Gemini request failed "
                f"({attempt + 1}/"
                f"{max_retries})"
            )

            print(error)

            if (
                attempt
                < max_retries - 1
            ):

                wait_time = (
                    2 ** attempt
                )

                print(
                    "Retrying Gemini in "
                    f"{wait_time} second(s)..."
                )

                await asyncio.sleep(
                    wait_time
                )

    return None


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_rag_prompt(
    query: str,
    search_results: list,
):

    context_parts = []

    for index, result in enumerate(
        search_results,
        start=1,
    ):

        document = result.get(
            "document",
            "Unknown document",
        )

        page = result.get(
            "page"
        )

        text = result.get(
            "text",
            "",
        )

        score = result.get(
            "score",
            0,
        )

        if page is not None:

            source_label = (
                f"{document}, page {page}"
            )

        else:

            source_label = document

        context_parts.append(
            f"""
SOURCE {index}
Document: {source_label}
Similarity Score: {score}

{text}
""".strip()
        )

    context = (
        "\n\n"
        "-----------------------------"
        "\n\n"
    ).join(
        context_parts
    )

    return f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the retrieved
document context below.

RULES:

1. Use only information contained in the retrieved context.

2. Do not invent facts.

3. If the retrieved context does not provide enough
information to answer the question, respond:

"I could not find enough information in the uploaded documents."

4. Give a clear and concise answer.

5. When referring to supporting information, mention the
document and page number when available.

6. Do not use outside knowledge to fill missing information.


RETRIEVED CONTEXT:

{context}


USER QUESTION:

{query}


ANSWER:
""".strip()


# ============================================================
# SSE HELPER
# ============================================================

def create_sse_event(
    data,
    event=None,
):

    if not isinstance(
        data,
        str,
    ):

        data = json.dumps(
            data,
            ensure_ascii=False,
        )

    lines = []

    if event:

        lines.append(
            f"event: {event}"
        )

    # SSE requires each line of multiline
    # data to have its own data: prefix.

    data_lines = (
        data.splitlines()
        or [""]
    )

    for line in data_lines:

        lines.append(
            f"data: {line}"
        )

    return (
        "\n".join(lines)
        + "\n\n"
    )


# ============================================================
# RAG STREAM ENDPOINT
# ============================================================

@app.post("/api/query/stream")
async def query_stream(
    request: QueryRequest
):

    query = (
        request.query.strip()
    )

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    try:

        search_results = (
            await asyncio.to_thread(
                search_engine.search,
                query,
            )
        )

        print(
            f"Retrieved "
            f"{len(search_results)} "
            f"source(s)."
        )

    except Exception as error:

        print(
            f"Search error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Search engine failed.",
        )


    # ========================================================
    # STREAM GENERATOR
    # ========================================================

    async def event_generator():

        try:

            # -----------------------------------------------
            # 1. SEND SOURCES
            # -----------------------------------------------

            yield create_sse_event(
                search_results,
                event="sources",
            )

            await asyncio.sleep(
                0.05
            )


            # -----------------------------------------------
            # 2. NO RELEVANT RESULTS
            # -----------------------------------------------

            if not search_results:

                answer = (
                    "I could not find relevant "
                    "information in the uploaded "
                    "documents."
                )


            # -----------------------------------------------
            # 3. GEMINI AVAILABLE
            # -----------------------------------------------

            elif client is not None:

                prompt = (
                    build_rag_prompt(
                        query,
                        search_results,
                    )
                )

                answer = (
                    await generate_with_gemini(
                        prompt,
                        max_retries=3,
                    )
                )

                if not answer:

                    answer = (
                        "AI generation is temporarily "
                        "unavailable. Relevant document "
                        "sources were retrieved successfully."
                    )


            # -----------------------------------------------
            # 4. GEMINI NOT CONFIGURED
            # -----------------------------------------------

            else:

                answer = (
                    "Gemini API is not configured. "
                    "The retrieval engine successfully "
                    "found relevant document sources."
                )


            # -----------------------------------------------
            # 5. STREAM ANSWER
            # -----------------------------------------------

            print(
                f"Streaming answer "
                f"({len(answer)} characters)."
            )

            # Stream chunks rather than individual words.
            # This preserves spaces and punctuation.

            chunk_size = 40

            for start in range(
                0,
                len(answer),
                chunk_size,
            ):

                chunk = answer[
                    start:
                    start + chunk_size
                ]

                yield create_sse_event(
                    chunk,
                    event="token",
                )

                await asyncio.sleep(
                    0.02
                )


            # -----------------------------------------------
            # 6. END
            # -----------------------------------------------

            yield create_sse_event(
                "[DONE]",
                event="end",
            )

            print(
                "Streaming response completed."
            )


        except asyncio.CancelledError:

            print(
                "Client disconnected "
                "from streaming response."
            )

            raise


        except Exception as error:

            print(
                f"Streaming error: "
                f"{error}"
            )

            yield create_sse_event(
                (
                    "An error occurred while "
                    "generating the answer."
                ),
                event="token",
            )

            yield create_sse_event(
                "[DONE]",
                event="end",
            )


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":
                "no-cache",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no",
        },
    )