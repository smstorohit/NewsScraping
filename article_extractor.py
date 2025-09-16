from newspaper import Article
import re

class ArticleExtractor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text

    def extract(self, url: str) -> dict:
        try:
            article = Article(url)
            article.download()
            article.parse()
            headline = self.clean_text(article.title)
            full_text = self.clean_text(article.text)
            return {"url": url, "headline": headline, "full_text": full_text}
        except Exception as e:
            return None