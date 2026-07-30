import asyncio
import io

from pypdf import PdfReader

from .database import db
from .search_engine import search_engine


async def process_document_background(
    file_bytes: bytes,
    filename: str,
    doc_id: str,
):

    try:

        print(
            f"Processing document: {filename}"
        )

        await asyncio.sleep(0.5)

        extracted_text = ""

        # ================================================
        # PDF
        # ================================================

        if filename.lower().endswith(".pdf"):

            pdf_reader = PdfReader(
                io.BytesIO(file_bytes)
            )

            for page in pdf_reader.pages:

                text = page.extract_text()

                if text:
                    extracted_text += text + "\n"

        # ================================================
        # TEXT-BASED FILE
        # ================================================

        else:

            extracted_text = file_bytes.decode(
                "utf-8",
                errors="ignore",
            )

        extracted_text = extracted_text.strip()

        # ================================================
        # NO TEXT
        # ================================================

        if not extracted_text:

            print(
                f"No text extracted from {filename}"
            )

            return

        # ================================================
        # CHUNK DOCUMENT
        # ================================================

        chunks = search_engine.chunk_text(
            extracted_text
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        # ================================================
        # STORE CHUNKS IN CURRENT DATABASE
        # ================================================

        db.update_document_chunks(
            doc_id,
            chunks,
        )

        # ================================================
        # CREATE EMBEDDINGS + STORE IN QDRANT
        # ================================================

        await asyncio.to_thread(
            search_engine.index_document,
            doc_id,
            filename,
            chunks,
        )

        print(
            f"Finished processing: {filename}"
        )

    except Exception as e:

        print(
            f"Document processing failed "
            f"for {filename}: {e}"
        )
