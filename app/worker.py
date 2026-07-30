import asyncio
import io
from pypdf import PdfReader
from .database import db
from .search_engine import search_engine

async def process_document_background(file_bytes: bytes, filename: str, doc_id: str):
    await asyncio.sleep(0.5)
    
    extracted_text = ""
    if filename.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + " "
    else:
        extracted_text = file_bytes.decode("utf-8", errors="ignore")

    chunks = search_engine.chunk_text(extracted_text)
    db.update_document_chunks(doc_id, chunks)