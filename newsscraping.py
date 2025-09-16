import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from openai_api import OpenAIAPI
from article_extractor import ArticleExtractor
from article_processor import ArticleProcessor
from article_store import ArticleStore
from semantic_searcher import SemanticSearcher

# Load environment variables from the .env file
load_dotenv() 

# Top-level variable for FAISS index storage location (read from env or use default)
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH", "./faiss_index")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Setup
extractor = ArticleExtractor()
openai_api = OpenAIAPI()
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
processor = ArticleProcessor(openai_api, text_splitter, embedding_model)
store = ArticleStore(FAISS_INDEX_PATH, 384)

def main():
    urls = [
        "https://www.ndtv.com/lifestyle/blind-for-two-decades-canadian-man-sees-again-after-tooth-implant-surgery-9275198?pfrom=home-ndtv_lifestyle",
        "https://edition.cnn.com/2025/09/08/health/diabetes-undiagnosed-half-of-americans-wellness",
        "https://www.bbc.com/news/articles/c930454e77xo",
        "https://www.bbc.com/news/articles/cq8eyzznv0qo",
        "https://www.bbc.com/travel/article/20250912-the-japanese-landscapes-that-inspired-studio-ghibli-films"
    ]
    
    for url in urls:
        if url in store.processed_urls:
            logging.info(f"URL already processed, skipping: {url}")
            continue
        logging.info(f"Processing {url}...")
        article_data = extractor.extract(url)
        if article_data:
            processed = processor.process(article_data)
            logging.info(f"Headline: {processed.get('headline', '')}")
            logging.info(f"Summary: {processed.get('summary', '')}")
            logging.info(f"Topics: {', '.join(processed.get('topics', []))}")
            logging.info("-" * 50)
            store.store(processed)
    
    # Search
    searcher = SemanticSearcher(embedding_model, store.index, store.metadata, store.documents)
    results = searcher.search("sugar in america", distance_threshold=0.7)
    logging.info(f"size of results: {len(results)}")
    logging.info("\nSemantic Search Results:")
    for result in results:
        logging.info(f"Headline: {result['headline']}")
        logging.info(f"URL: {result['url']}")
        logging.info(f"Summary: {result['summary']}")
        logging.info(f"Topics: {', '.join(result['topics'])}")
        logging.info(f"Distance: {result['distance']:.4f}")
        logging.info("-" * 50)

if __name__ == "__main__":
    main()