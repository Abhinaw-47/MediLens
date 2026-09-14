from google import genai
from dotenv import load_dotenv
import os
load_dotenv()
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
result=client.models.embed_content(
    model="gemini-embedding-2",
    contents="Hypertension is high blood pressure."
)
embedding=result.embeddings[0].values 
print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])