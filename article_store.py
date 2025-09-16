import faiss
import numpy as np
import pickle
import os
import uuid

class ArticleStore:
    def __init__(self, index_path, dimension):
        self.index_path = index_path
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.documents = []
        self.processed_urls = set()
        os.makedirs(self.index_path, exist_ok=True)
        if os.path.exists(f"{self.index_path}.index") and os.path.exists(f"{self.index_path}_metadata.pkl"):
            self.index = faiss.read_index(f"{self.index_path}.index")
            with open(f"{self.index_path}_metadata.pkl", "rb") as f:
                self.metadata, self.documents = pickle.load(f)
                self.processed_urls = {meta["url"] for meta in self.metadata}

    def store(self, article_data: dict):
        if not article_data or article_data["url"] in self.processed_urls:
            return
        embedding = np.array([article_data["embedding"]], dtype=np.float32)
        doc_id = str(uuid.uuid4())
        self.index.add(embedding)
        self.metadata.append({
            "id": doc_id,
            "url": article_data["url"],
            "headline": article_data["headline"],
            "topics": ",".join(article_data["topics"])
        })
        self.documents.append(article_data["summary"])
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(f"{self.index_path}_metadata.pkl", "wb") as f:
            pickle.dump((self.metadata, self.documents), f)
        self.processed_urls.add(article_data["url"])