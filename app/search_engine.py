from pathlib import Path
from typing import List
import uuid

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


class VectorSearchEngine:

    def __init__(self):

        # ==========================================
        # EMBEDDING MODEL
        # ==========================================

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # New Sentence Transformers method
        self.vector_size = (
            self.model.get_embedding_dimension()
        )

        print(
            f"Embedding model loaded. "
            f"Vector size: {self.vector_size}"
        )

        # ==========================================
        # QDRANT LOCAL STORAGE
        # ==========================================

        base_dir = Path(__file__).resolve().parent.parent

        qdrant_path = base_dir / "qdrant_storage"

        self.client = QdrantClient(
            path=str(qdrant_path)
        )

        self.collection_name = (
            "enterprise_documents"
        )

        # ==========================================
        # CREATE QDRANT COLLECTION
        # ==========================================

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
                "Created Qdrant collection: "
                f"{self.collection_name}"
            )

        else:

            print(
                "Using existing Qdrant collection: "
                f"{self.collection_name}"
            )

    # ==============================================
    # CHUNK TEXT
    # ==============================================

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> List[str]:

        words = text.split()

        if not words:
            return []

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        chunks = []

        step = chunk_size - overlap

        for start in range(
            0,
            len(words),
            step,
        ):

            chunk_words = words[
                start:start + chunk_size
            ]

            if not chunk_words:
                continue

            chunk = " ".join(
                chunk_words
            ).strip()

            if chunk:
                chunks.append(chunk)

            if start + chunk_size >= len(words):
                break

        return chunks

    # ==============================================
    # CREATE ONE EMBEDDING
    # ==============================================

    def create_embedding(
        self,
        text: str,
    ) -> List[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # ==============================================
    # INDEX DOCUMENT
    # ==============================================

    def index_document(
        self,
        doc_id: str,
        title: str,
        chunks: list,
    ):

        if not chunks:

            print(
                f"No chunks available for document: "
                f"{title}"
            )

            return

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print(
            f"Creating embeddings for "
            f"{len(texts)} chunks..."
        )

        # Generate embeddings in one batch
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = []

        for global_chunk_id, (
            chunk_data,
            embedding,
        ) in enumerate(
            zip(chunks, embeddings)
        ):

            # Create deterministic unique ID
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{doc_id}-{global_chunk_id}",
                )
            )

            payload = {

                "doc_id": doc_id,

                "document": title,

                "chunk_id": global_chunk_id,

                "page": chunk_data.get(
                    "page"
                ),

                "page_chunk": chunk_data.get(
                    "page_chunk"
                ),

                "text": chunk_data["text"],
            }

            point = PointStruct(

                id=point_id,

                vector=embedding.tolist(),

                payload=payload,
            )

            points.append(point)

        # ==========================================
        # STORE IN QDRANT
        # ==========================================

        self.client.upsert(

            collection_name=self.collection_name,

            points=points,
        )

        print(
            f"Indexed {len(points)} "
            f"page-aware chunks for {title}"
        )

    # ==============================================
    # SEMANTIC SEARCH
    # ==============================================

    def search(
        self,
        query: str,
        top_k: int = 3,
        minimum_score: float = 0.20,
    ) -> List[dict]:

        query = query.strip()

        if not query:
            return []

        # ==========================================
        # QUERY → EMBEDDING
        # ==========================================

        query_vector = self.create_embedding(
            query
        )

        try:

            # ======================================
            # VECTOR SEARCH
            # ======================================

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

        # ==========================================
        # FORMAT RESULTS
        # ==========================================

        for point in response.points:

            score = float(
                point.score
            )

            # Ignore very weak matches
            if score < minimum_score:
                continue

            payload = (
                point.payload or {}
            )

            results.append(
                {
                    "score": round(
                        score,
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

                    "page": payload.get(
                        "page"
                    ),

                    "chunk_id": payload.get(
                        "chunk_id",
                        0,
                    ),

                    "page_chunk": payload.get(
                        "page_chunk"
                    ),
                }
            )

        return results

    # ==============================================
    # CLOSE QDRANT
    # ==============================================

    def close(self):

        try:

            self.client.close()

        except Exception as e:

            print(
                f"Qdrant close warning: {e}"
            )


# ==================================================
# GLOBAL SEARCH ENGINE
# ==================================================

search_engine = VectorSearchEngine()