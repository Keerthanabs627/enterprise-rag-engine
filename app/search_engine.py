from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from .database import db


class VectorSearchEngine:

    def __init__(self):

        # --------------------------------------------------
        # EMBEDDING MODEL
        # --------------------------------------------------

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_size = self.model.get_sentence_embedding_dimension()

        print(
            f"Embedding model loaded. "
            f"Vector size: {self.vector_size}"
        )

        # --------------------------------------------------
        # QDRANT LOCAL DATABASE
        # --------------------------------------------------

        base_dir = Path(__file__).resolve().parent.parent

        qdrant_path = base_dir / "qdrant_storage"

        self.client = QdrantClient(
            path=str(qdrant_path)
        )

        self.collection_name = "enterprise_documents"

        # --------------------------------------------------
        # CREATE COLLECTION
        # --------------------------------------------------

        if not self.client.collection_exists(
            collection_name=self.collection_name
        ):

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

            print(
                f"Created Qdrant collection: "
                f"{self.collection_name}"
            )

        else:

            print(
                f"Using existing Qdrant collection: "
                f"{self.collection_name}"
            )


    # ======================================================
    # TEXT CHUNKING
    # ======================================================

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> List[str]:

        words = text.split()

        if not words:
            return []

        chunks = []

        step = chunk_size - overlap

        for start in range(0, len(words), step):

            chunk_words = words[
                start:start + chunk_size
            ]

            if not chunk_words:
                continue

            chunk = " ".join(chunk_words).strip()

            if chunk:
                chunks.append(chunk)

            # Stop when we've reached the end
            if start + chunk_size >= len(words):
                break

        return chunks


    # ======================================================
    # CREATE EMBEDDING
    # ======================================================

    def create_embedding(
        self,
        text: str,
    ) -> List[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()


    # ======================================================
    # INDEX DOCUMENT
    # ======================================================

    def index_document(
        self,
        doc_id: str,
        title: str,
        chunks: List[str],
    ):

        if not chunks:
            print(
                f"No chunks available for document: {title}"
            )
            return

        print(
            f"Creating embeddings for "
            f"{len(chunks)} chunks..."
        )

        # Generate embeddings in one batch
        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = []

        for chunk_id, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            # Qdrant requires unique point IDs.
            # UUID generated from document ID + chunk number.
            import uuid

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{doc_id}-{chunk_id}",
                )
            )

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "doc_id": doc_id,
                    "document": title,
                    "chunk_id": chunk_id,
                    "text": chunk,
                },
            )

            points.append(point)

        # Store vectors + metadata in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(
            f"Indexed {len(points)} chunks "
            f"for {title}"
        )


    # ======================================================
    # SEMANTIC SEARCH
    # ======================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[dict]:

        query = query.strip()

        if not query:
            return []

        # Convert query into embedding
        query_vector = self.create_embedding(query)

        try:

            # Query Qdrant for semantically similar chunks
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

        except Exception as e:

            print(
                f"Qdrant search error: {e}"
            )

            return []

        results = []

        for point in response.points:

            payload = point.payload or {}

            results.append(
                {
                    "score": round(
                        float(point.score),
                        4,
                    ),
                    "text": payload.get(
                        "text",
                        "",
                    ),
                    "document": payload.get(
                        "document",
                        "Unknown document",
                    ),
                    "chunk_id": payload.get(
                        "chunk_id",
                        0,
                    ),
                }
            )

        return results


search_engine = VectorSearchEngine()