from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="medical_knowledge"
)

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatMessage(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    question: str
    history:list[ChatMessage]=[]

@app.get("/")
def home():
    return{
        "message":"MediRAG API is running",
        "chunks":collection.count()
    }
@app.post("/chat")
def chat(request:ChatRequest):
    query=request.question
    retrieval_query = query
    if request.history:
        previous_conveersation="\n".join(
            f"{message.role}: {message.content}"
            for message in request.history[-4:]
        )
        retrieval_query=f"""
        Previous conversation:
        {previous_conveersation}
        current question:
        {query}


   """
    
    result=gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=f"task: question answering | query: {retrieval_query}",
        config={
            "output_dimensionality":768
        }
    )
    query_embedding=result.embeddings[0].values
    results=collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents","metadatas","distances"]
    )
   

    retrieved_documents = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    # Build context
    context_parts = []

    for document, metadata in zip(
        retrieved_documents,
        retrieved_metadatas
    ):
        context_parts.append(
            f"Source: {metadata['source']}, "
            f"Page: {metadata['page']}\n"
            f"{document}"
        )

    context = "\n\n".join(context_parts)
    history_contents=[]
    for message in request.history:
        role="model" if message.role=="assistant" else "user"
        history_contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(text=message.content)
                ]
            )
        )

    # Ask Gemini to answer using the retrieved context
    prompt = f"""
You are a medical information assistant.

Answer the user's question using only the information
provided in the context below.

Context:
{context}

Question:
{query}

If the context does not contain enough information to answer
the question, say:
"I don't have enough information in my knowledge base to answer that."
"""
    history_contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt)
            ]
        )
    )
    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=history_contents,
         config={
        "temperature": 0.2,
        "max_output_tokens": 500
    }
    )

    # Return answer and sources
    clean_sources = []
    for metadata in retrieved_metadatas:
        clean_sources.append(
            {
                "source": Path(metadata["source"]).stem,
                "page": metadata["page"]
            }
        )
    return {
        "question": query,
        "answer": response.text,
        "sources":clean_sources
    }
    

    