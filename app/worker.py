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
        print(f"Processing document: {filename}")

        await asyncio.sleep(0.5)

        chunks_with_metadata = []

        # ==========================================
        # PDF PROCESSING
        # ==========================================

        if filename.lower().endswith(".pdf"):

            pdf_reader = PdfReader(io.BytesIO(file_bytes))

            print(
                f"PDF contains {len(pdf_reader.pages)} pages."
            )

            for page_number, page in enumerate(
                pdf_reader.pages,
                start=1,
            ):

                text = page.extract_text()

                if not text:
                    continue

                text = text.strip()

                if not text:
                    continue

                page_chunks = search_engine.chunk_text(text)

                for page_chunk_index, chunk in enumerate(
                    page_chunks
                ):

                    chunks_with_metadata.append(
                        {
                            "text": chunk,
                            "page": page_number,
                            "page_chunk": page_chunk_index,
                        }
                    )

        # ==========================================
        # TXT / OTHER TEXT FILES
        # ==========================================

        else:

            extracted_text = file_bytes.decode(
                "utf-8",
                errors="ignore",
            ).strip()

            if extracted_text:

                text_chunks = search_engine.chunk_text(
                    extracted_text
                )

                for chunk_index, chunk in enumerate(
                    text_chunks
                ):

                    chunks_with_metadata.append(
                        {
                            "text": chunk,
                            "page": None,
                            "page_chunk": chunk_index,
                        }
                    )

        # ==========================================
        # CHECK IF TEXT WAS EXTRACTED
        # ==========================================

        if not chunks_with_metadata:

            print(
                f"No usable text extracted from {filename}"
            )

            return

        print(
            f"Created {len(chunks_with_metadata)} "
            f"page-aware chunks."
        )

        # ==========================================
        # STORE TEXT IN CURRENT DOCUMENT DATABASE
        # ==========================================

        plain_chunks = [
            item["text"]
            for item in chunks_with_metadata
        ]

        db.update_document_chunks(
            doc_id,
            plain_chunks,
        )

        # ==========================================
        # EMBEDDING + QDRANT INDEXING
        # ==========================================

        await asyncio.to_thread(
            search_engine.index_document,
            doc_id,
            filename,
            chunks_with_metadata,
        )

        print(
            f"Finished processing: {filename}"
        )

    except Exception as e:

        print(
            f"Document processing failed "
            f"for {filename}: {e}"
        )