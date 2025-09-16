import os
import json
import logging
from openai import OpenAI
from typing import Dict, Any

class OpenAIAPI:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable or add it to your .env file.")
        
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
            logging.error(f"Error processing chunk: {str(e)}")
            return {"summary": "", "topics": ["General", "News"]}