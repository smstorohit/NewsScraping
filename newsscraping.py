import newspaper
from newspaper import Article
import faiss
import numpy as np
import uuid
import re
from typing import List, Dict, Any
import os
import json
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from dotenv import load_dotenv

# Top-level variable for FAISS index storage location
FAISS_INDEX_PATH = "/Users/rohitjain/Documents/pythonprojects/langchainproject/faiss_store_4/newsindex"

# Load environment variables from the .env file
load_dotenv() 

class OpenAIAPI:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def process_chunk(self, chunk: str) -> Dict[str, Any]:
        """
        Generate a summary and identify main topics from a single text chunk
        using a single API call.
        """
        prompt = f"""
        You are a helpful assistant. Analyze the following text and perform two tasks:
        1. Create a concise summary (3-5 sentences) of this chunk.
        2. Identify 3-5 key topics from this chunk.
        
        Return the results in a JSON object with 'summary' and 'topics' keys. The 'topics' value should be a comma-separated string.
        
        Text:
        ---
        {chunk}
        ---
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            topics_str = data.get("topics", "General,News")
            topics = [t.strip() for t in topics_str.split(',') if t.strip()]
            
            return {"summary": data.get("summary", ""), "topics": topics}
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return {"summary": "", "topics": ["General", "News"]}

class NewsAnalyzer:
    def __init__(self):
        self.openai_api = OpenAIAPI()
        
        # Initialize the HuggingFace embeddings model
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.dimension = 384
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.documents = []
        self.processed_urls = set() 

        # create the directory if it doesn't exist
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        # Load FAISS index and metadata if they exist
        if os.path.exists(f"{FAISS_INDEX_PATH}.index") and os.path.exists(f"{FAISS_INDEX_PATH}_metadata.pkl"):
            print(f"Loading FAISS index and metadata from {FAISS_INDEX_PATH}")
            self.index = faiss.read_index(f"{FAISS_INDEX_PATH}.index")
            with open(f"{FAISS_INDEX_PATH}_metadata.pkl", "rb") as f:
                self.metadata, self.documents = pickle.load(f)
                # Populate the set with URLs from the loaded metadata
                self.processed_urls = {meta["url"] for meta in self.metadata}

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text
    
    def extract_article(self, url: str) -> Dict[str, Any]:
        try:
            article = Article(url)
            article.download()
            article.parse()
            headline = self.clean_text(article.title)
            full_text = self.clean_text(article.text)
            
            return {
                "url": url,
                "headline": headline,
                "full_text": full_text
            }
        except Exception as e:
            print(f"Error extracting article from {url}: {str(e)}")
            return None
    
    def process_article(self, url: str) -> Dict[str, Any]:
        """
        Processes a news article from a given URL.
        Includes an early check for duplicate URLs.
        """
        # STEP 1: The most important change. Check if the URL has already been processed.
        if url in self.processed_urls:
            print(f"URL already processed, skipping: {url}")
            return None
        
        article_data = self.extract_article(url)
        if not article_data:
            return None
            
        full_text = article_data["full_text"]
        
        # Use LangChain splitter to chunk the text
        chunks = self.text_splitter.split_text(full_text)
        
        summaries = []
        topics_list = []
        
        # Process each chunk for summary and topics
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                result = self.openai_api.process_chunk(chunk)
                summaries.append(result["summary"])
                topics_list.extend(result["topics"])
        
        # Consolidate topics and summaries
        final_summary = " ".join(summaries)
        final_topics = list(set(topics_list))[:5]
        
        # Generate a single embedding for the whole text using the model's built-in chunking
        embedding = self.embedding_model.embed_documents([full_text])[0]
        
        article_data.update({
            "summary": final_summary,
            "topics": final_topics,
            "embedding": embedding
        })
        
        return article_data
    
    def store_article(self, article_data: Dict[str, Any]) -> None:
        if not article_data:
            return
            
        if any(meta["url"] == article_data["url"] for meta in self.metadata):
            print(f"Skipping duplicate article: {article_data['url']}")
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
        
        faiss.write_index(self.index, f"{FAISS_INDEX_PATH}.index")
        with open(f"{FAISS_INDEX_PATH}_metadata.pkl", "wb") as f:
            pickle.dump((self.metadata, self.documents), f)
    
    def semantic_search(self, query: str, n_results: int = 5,distance_threshold: float = 0.5) -> List[Dict[str, Any]]:
        query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)
        
        distances, indices = self.index.search(query_embedding, n_results * 2)
        
        formatted_results = []
        seen_urls = set()
        for i, idx in enumerate(indices[0]):
            # Check the distance against the threshold
            if distances[0][i] > distance_threshold:
                continue # Skip if distance is too large

            if idx >= len(self.metadata):
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

def main():
    analyzer = NewsAnalyzer()
    
    urls = [
        "https://www.ndtv.com/lifestyle/blind-for-two-decades-canadian-man-sees-again-after-tooth-implant-surgery-9275198?pfrom=home-ndtv_lifestyle",
        "https://edition.cnn.com/2025/09/08/health/diabetes-undiagnosed-half-of-americans-wellness",
        "https://www.bbc.com/news/articles/c930454e77xo",
        "https://www.bbc.com/news/articles/cq8eyzznv0qo"
    ]
    
    for url in urls:
        print(f"Processing {url}...")
        article_data = analyzer.process_article(url)
        if article_data:
            print(f"Generated summary: {article_data['summary'][:100]}...")
            print(f"Identified topics: {article_data['topics']}")
            analyzer.store_article(article_data)
    
    query = "sugar in america"
    results = analyzer.semantic_search(query, distance_threshold=0.7)
    print(f"size of results: {len(results)}")
    print("\nSemantic Search Results:")
    for result in results:
        print(f"Headline: {result['headline']}")
        print(f"URL: {result['url']}")
        print(f"Summary: {result['summary']}")
        print(f"Topics: {', '.join(result['topics'])}")
        print(f"Distance: {result['distance']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()