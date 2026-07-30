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
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# 2. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Enterprise Search & RAG Engine",
    description="Enterprise document ingestion, search and RAG platform",
    version="1.0.0",
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# 4. GEMINI CLIENT
# ============================================================

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("WARNING: GEMINI_API_KEY is not configured.")


# ============================================================
# 5. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 6. STATIC FILES
# ============================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

else:

    print(
        f"WARNING: Static directory does not exist: {STATIC_DIR}"
    )


# ============================================================
# 7. REQUEST MODELS
# ============================================================

class QueryRequest(BaseModel):
    query: str


# ============================================================
# 8. HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():

    index_path = STATIC_DIR / "index.html"

    if not index_path.exists():

        return HTMLResponse(
            content="<h1>index.html missing in static/ folder!</h1>",
            status_code=404,
        )

    return FileResponse(index_path)


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health_check():

    return {
        "status": "healthy",
        "service": "Enterprise Search & RAG Engine",
        "gemini_configured": client is not None,
    }


# ============================================================
# 10. DOCUMENT UPLOAD
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

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Read uploaded file
        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # Add document metadata to database
        db.add_document(
            doc_id,
            file.filename,
        )

        # Process document in background
        background_tasks.add_task(
            process_document_background,
            file_bytes,
            file.filename,
            doc_id,
        )

        return {
            "message": "Document queued for asynchronous processing",
            "document_id": doc_id,
            "filename": file.filename,
            "status": "processing",
        }

    except HTTPException:
        raise

    except Exception as e:

        print(f"Document upload error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to upload document.",
        )


# ============================================================
# 11. LIST DOCUMENTS
# ============================================================

@app.get("/api/documents")
async def list_documents():

    try:

        documents = db.get_all_documents()

        return {
            "documents": documents
        }

    except Exception as e:

        print(f"Document listing error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve documents.",
        )


# ============================================================
# 12. GEMINI GENERATION FUNCTION
# ============================================================

async def generate_with_gemini(
    prompt: str,
    max_retries: int = 3,
):

    if client is None:

        return None

    for attempt in range(max_retries):

        try:

            print(
                f"Gemini request attempt "
                f"{attempt + 1}/{max_retries}"
            )

            # generate_content() is synchronous.
            # Running it in a thread prevents blocking
            # FastAPI's async event loop.
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3.5-flash-lite",
                contents=prompt,
            )

            if response and response.text:

                return response.text.strip()

            print("Gemini returned an empty response.")

        except Exception as e:

            print(
                f"Gemini request failed "
                f"(attempt {attempt + 1}/{max_retries})"
            )

            print(str(e))

            # Don't sleep after final attempt
            if attempt < max_retries - 1:

                # Exponential backoff:
                #
                # failure 1 -> wait 1 sec
                # failure 2 -> wait 2 sec

                wait_time = 2 ** attempt

                print(
                    f"Retrying in {wait_time} second(s)..."
                )

                await asyncio.sleep(wait_time)

    return None


# ============================================================
# 13. STREAMING RAG QUERY
# ============================================================

@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # --------------------------------------------------------
    # SEARCH DOCUMENTS
    # --------------------------------------------------------

    try:

        search_results = search_engine.search(query)

    except Exception as e:

        print(f"Search error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Search engine failed.",
        )


    # --------------------------------------------------------
    # SSE EVENT GENERATOR
    # --------------------------------------------------------

    async def event_generator():

        # ====================================================
        # SEND SOURCES TO FRONTEND
        # ====================================================

        try:

            sources_json = json.dumps(
                search_results,
                ensure_ascii=False,
            )

            yield (
                f"event: sources\n"
                f"data: {sources_json}\n\n"
            )

        except Exception as e:

            print(f"Source serialization error: {e}")

            yield (
                "event: sources\n"
                "data: []\n\n"
            )


        await asyncio.sleep(0.1)


        # ====================================================
        # NO SEARCH RESULTS
        # ====================================================

        if not search_results:

            response_text = (
                "I could not find relevant information "
                "in the uploaded documents."
            )


        # ====================================================
        # GEMINI AVAILABLE
        # ====================================================

        elif client:

            # Build retrieved context
            context_parts = []

            for result in search_results:

                document = result.get(
                    "document",
                    "Unknown document",
                )

                text = result.get(
                    "text",
                    "",
                )

                context_parts.append(
                    f"Source [{document}]:\n{text}"
                )


            context = "\n\n---\n\n".join(
                context_parts
            )


            # =================================================
            # RAG PROMPT
            # =================================================

            prompt = f"""
You are an enterprise document assistant.

Your job is to answer the user's question using ONLY
the retrieved document context provided below.

Rules:

1. Use only information found in the context.

2. Do not invent facts.

3. If the context does not contain enough information,
say:

"I could not find enough information in the uploaded documents."

4. Give a concise and clear answer.

5. When possible, mention which document supports the answer.

6. Do not claim information that is not present in the context.


RETRIEVED DOCUMENT CONTEXT:

{context}


USER QUESTION:

{query}


ANSWER:
"""


            # =================================================
            # CALL GEMINI
            # =================================================

            response_text = await generate_with_gemini(
                prompt=prompt,
                max_retries=3,
            )


            # =================================================
            # GEMINI FAILED
            # =================================================

            if response_text is None:

                response_text = (
                    "AI generation is temporarily unavailable. "
                    "The relevant document sources were retrieved "
                    "successfully. Please try again shortly."
                )


        # ====================================================
        # GEMINI API KEY MISSING
        # ====================================================

        else:

            context = " ".join(
                result.get("text", "")
                for result in search_results
            )

            response_text = (
                "Gemini API is not configured. "
                "However, the search engine successfully "
                "retrieved this context: "
                + context[:500]
            )


        # ====================================================
        # STREAM RESPONSE TO FRONTEND
        # ====================================================

        words = response_text.split()

        for word in words:

            yield f"data: {word} \n\n"

            await asyncio.sleep(0.03)


        # ====================================================
        # FINISH STREAM
        # ====================================================

        yield (
            "event: end\n"
            "data: [DONE]\n\n"
        )


    # --------------------------------------------------------
    # RETURN SSE STREAM
    # --------------------------------------------------------

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )