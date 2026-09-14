import chromadb
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

gemini_client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client=chromadb.PersistentClient(path="./chroma_db")
collection=chroma_client.get_collection(name="medical_knowledge")
query='What are the symptoms of high blood pressure?'
result =gemini_client.models.embed_content(
    model="gemini-embedding-2",
    contents=f"task: question answering | query: {query}",
    config={
        "output_dimensionality":768
    }

)

query_embedding=result.embeddings[0].values
results=collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrieved_documents=results["documents"][0]
retrieved_metadatas=results["metadatas"][0]
context_parts=[]
for document,metadata in zip(
    retrieved_documents,retrieved_metadatas

):
    context_parts.append(
        f"Source: {metadata['source']}"
        f"Page: {metadata['page']}\n"
        f"{document}"
    )
context="\n\n".join(context_parts)

prompt=f"""
You are a medical information assistant.

Answer the user's question using only the information provided in the context below.

Context:
{context}

Question:
{query}

If the context does not contain enough information to answer the question,say:
"I don't have enough information in my knowledge base to answer that."

"""

response=gemini_client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

print("\nRetrieved documents:")

for i, (document, metadata) in enumerate(
    zip(retrieved_documents, retrieved_metadatas)
):

    print(f"\nResult {i + 1}")
    print(f"Source: {metadata['source']}")
    print(f"Page: {metadata['page']}")
    print(document[:500])
    print("-" * 60)

print("\nFinal Answer:")
print(response.text)
