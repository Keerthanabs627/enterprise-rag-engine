import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .database import db

class VectorSearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> list:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def search(self, query: str, top_k: int = 3) -> list:
        all_chunks = []
        metadata = []

        for doc_id, doc in db.documents.items():
            if doc["status"] == "ready":
                for idx, chunk in enumerate(doc["chunks"]):
                    all_chunks.append(chunk)
                    metadata.append({"doc_id": doc_id, "title": doc["title"], "chunk_id": idx})

        if not all_chunks:
            return []

        corpus = all_chunks + [query]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        query_vector = tfidf_matrix[-1]
        document_vectors = tfidf_matrix[:-1]

        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.02:  # Similarity threshold
                results.append({
                    "score": round(score, 4),
                    "text": all_chunks[idx],
                    "document": metadata[idx]["title"],
                    "chunk_id": metadata[idx]["chunk_id"]
                })
        return results

search_engine = VectorSearchEngine()