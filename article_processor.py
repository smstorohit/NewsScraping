class ArticleProcessor:
    def __init__(self, openai_api, text_splitter, embedding_model):
        self.openai_api = openai_api
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model

    def process(self, article_data: dict) -> dict:
        full_text = article_data["full_text"]
        chunks = self.text_splitter.split_text(full_text)
        summaries, topics_list = [], []
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                result = self.openai_api.process_chunk(chunk)
                summaries.append(result["summary"])
                topics_list.extend(result["topics"])
        final_summary = " ".join(summaries)
        final_topics = list(set(topics_list))[:5]
        embedding = self.embedding_model.embed_documents([full_text])[0]
        article_data.update({
            "summary": final_summary,
            "topics": final_topics,
            "embedding": embedding
        })
        return article_data