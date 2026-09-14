import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

gemini_client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client=chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="medical_knowledge"
)


query="what is asthma?"
query_result=gemini_client.models.embed_content(
   model="gemini-embedding-2",
   contents=query
)
query_embedding=query_result.embeddings[0].values

results=collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)
retrieved_documents=results["documents"][0]
print("Retrieved documents:")
for document in retrieved_documents:
    print(document)
context="\n\n".join(retrieved_documents)
prompt=f"""
You are a medical information assistant.

Answer the user's question using only the information provided in the context below.

Context:
{context}

Question:
{query}

If the context does not contain enough information to answer the question, say:
"I don't have enough information in my knowledge base to answer that."


"""

response=gemini_client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)
print("\nAnswer:")
print(response.text)