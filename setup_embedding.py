import pandas as pd
import os
import chromadb
from chromadb.config import Settings

# Load JSON Data
data = pd.read_json("data.json")

df = pd.DataFrame(data)
df.columns = ["id", "title", "text"]

# Manually define chunking strategy with overlap
chunk_size = 1000
chunk_overlap = 200


def manual_chunk_text(text, chunk_size, chunk_overlap):
    """Manually split the text into smaller chunks with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks


# Function for creation of chunks from each document
def create_chunks(row):
    """Split the text into smaller chunks."""
    text = row["text"]
    chunks = manual_chunk_text(text, chunk_size, chunk_overlap)

    # Debug output to confirm chunking works
    print(
        f"Document ID: {row['id']} | Text Length: {len(text)} | Total chunks: {len(chunks)}"
    )
    for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks to verify
        print(
            f"Chunk {i}: {chunk[:200]}..."
        )  # Print a preview of each chunk (first 200 characters)

    return [
        {"id": f"{row['id']}-{i}", "title": row["title"], "chunk": chunk}
        for i, chunk in enumerate(chunks)
    ]


# Flatten Chunks into a DataFrame
chunks = [chunk for _, row in df.iterrows() for chunk in create_chunks(row)]
chunks_df = pd.DataFrame(chunks)

# Verify the total number of chunks
print(f"Total number of chunks: {len(chunks_df)}")

# Initialize Chroma and Save Data
# Configure Chroma
chroma_client = chromadb.PersistentClient(path="db/")

# Create or retrieve a collection
collection = chroma_client.get_or_create_collection(
    name="icici_etf_articles",
    metadata={"description": "Knowledge center articles from ICICI ETF website."},
)

# Add chunks to the collection (Chroma will handle embedding generation)
collection.add(
    ids=chunks_df["id"].tolist(),
    metadatas=[{"title": title} for title in chunks_df["title"]],
    documents=chunks_df["chunk"].tolist(),
)

print("Chunks and metadata stored successfully in Chroma!")
