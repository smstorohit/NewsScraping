import numpy as np

class SemanticSearcher:
    def __init__(self, embedding_model, index, metadata, documents):
        self.embedding_model = embedding_model
        self.index = index
        self.metadata = metadata
        self.documents = documents

    def search(self, query: str, n_results: int = 5, distance_threshold: float = 0.5):
        query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, n_results * 2)
        formatted_results, seen_urls = [], set()
        for i, idx in enumerate(indices[0]):
            if distances[0][i] > distance_threshold or idx >= len(self.metadata):
                continue
            url = self.metadata[idx]["url"]
            if url not in seen_urls:
                formatted_results.append({
                    "id": self.metadata[idx]["id"],
                    "headline": self.metadata[idx]["headline"],
                    "url": url,
                    "summary": self.documents[idx],
                    "topics": self.metadata[idx]["topics"].split(","),
                    "distance": distances[0][i]
                })
                seen_urls.add(url)
            if len(formatted_results) >= n_results:
                break
        return formatted_results