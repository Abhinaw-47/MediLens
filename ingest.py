import pymupdf
from pathlib import Path
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time

load_dotenv()

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="medical_knowledge"
)
data_folder=Path("data")
pdf_files=list(data_folder.glob("*.pdf"))
# for pdf_file in pdf_files:
#     pdf=pymupdf.open(pdf_file)  
#     print(f"\nReading: {pdf_file.name}")
#     for page_number,page in enumerate(pdf):
#         text=page.get_text()
#         print(f"Page {page_number +1}:{len(text)} characters")

#     pdf.close()
def chunk_text(text,chunk_size=1200,overlap=200):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start=end-overlap
    return chunks
pages=[]
for pdf_file in pdf_files:
    pdf=pymupdf.open(pdf_file)
    for page_number,page in enumerate(pdf):
        text=page.get_text().strip()
        if len(text)<100:
            continue
        pages.append({
            "text":text,
            "source":pdf_file.name,
            "page":page_number+1
        })
    pdf.close()
chunks=[]
for page in pages:
    page_chunks=(chunk_text(page["text"]))   
    for chunk_number,chunk_text_value in enumerate(page_chunks):
        chunks.append({
            "text":chunk_text_value,
            "source":page["source"],
            "page":page["page"],
            "chunk":chunk_number+1
        })
print("Pages extracted:",len(pages))
print("Total chunks:",len(chunks))


BATCH_SIZE = 80
WAIT_SECONDS = 60


def make_chunk_id(chunk):
    return (
        f"{Path(chunk['source']).stem}"
        f"_p{chunk['page']}"
        f"_c{chunk['chunk']}"
    )


for start in range(0, len(chunks), BATCH_SIZE):

    batch = chunks[start:start + BATCH_SIZE]

    # Create IDs for this batch
    batch_ids = [
        make_chunk_id(chunk)
        for chunk in batch
    ]

    # Check which chunks are already stored
    existing = collection.get(
        ids=batch_ids
    )

    existing_ids = set(existing["ids"])

    # Only process chunks that are not already in ChromaDB
    missing_chunks = [
        chunk
        for chunk in batch
        if make_chunk_id(chunk) not in existing_ids
    ]

    if not missing_chunks:
        print(
            f"Skipping chunks "
            f"{start + 1}-{min(start + BATCH_SIZE, len(chunks))}"
            f" — already stored"
        )
        continue

    print(
        f"\nEmbedding {len(missing_chunks)} chunks "
        f"from batch "
        f"{start + 1}-{min(start + BATCH_SIZE, len(chunks))}"
    )

    contents = [
        types.Content(
            parts=[
                types.Part(
                    text=(
                        f"title: {chunk['source']} | "
                        f"text: {chunk['text']}"
                    )
                )
            ]
        )
        for chunk in missing_chunks
    ]

    # Keep trying if Gemini rate-limits us
    while True:

        try:

            result = gemini_client.models.embed_content(
                model="gemini-embedding-2",
                contents=contents,
                config={
                    "output_dimensionality": 768
                }
            )

            break

        except Exception as error:

            if "429" not in str(error):
                raise

            print("Rate limit reached.")
            print("Waiting 60 seconds before retrying...")
            time.sleep(WAIT_SECONDS)

    embeddings = [
        embedding.values
        for embedding in result.embeddings
    ]

    documents = [
        chunk["text"]
        for chunk in missing_chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk": chunk["chunk"]
        }
        for chunk in missing_chunks
    ]

    ids = [
        make_chunk_id(chunk)
        for chunk in missing_chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Stored {len(missing_chunks)} new chunks."
    )

print("\nIngestion completed!")
print("Total chunks in ChromaDB:", collection.count())