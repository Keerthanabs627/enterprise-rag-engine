import time

class StorageDB:
    def __init__(self):
        self.documents = {}  # doc_id: {title, status, timestamp, chunks}
        self.query_cache = {} # query: cached_response

    def add_document(self, doc_id: str, title: str):
        self.documents[doc_id] = {
            "title": title,
            "status": "processing",
            "timestamp": time.time(),
            "chunks": []
        }

    def update_document_chunks(self, doc_id: str, chunks: list):
        if doc_id in self.documents:
            self.documents[doc_id]["chunks"] = chunks
            self.documents[doc_id]["status"] = "ready"

    def get_document(self, doc_id: str):
        return self.documents.get(doc_id)

    def get_all_documents(self):
        return [
            {"id": k, "title": v["title"], "status": v["status"], "chunks_count": len(v["chunks"])}
            for k, v in self.documents.items()
        ]

db = StorageDB()