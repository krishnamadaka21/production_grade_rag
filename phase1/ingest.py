from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# -----------------------------
# Configuration
# -----------------------------

PDF_PATH = "data/Baahubali_Complete_Story.pdf"

COLLECTION_NAME = "phase1_documents"

QDRANT_PATH = "./qdrant_data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# -----------------------------
# 1. Load PDF
# -----------------------------

def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    return pages


# -----------------------------
# 2. Chunk text
# -----------------------------

def create_chunks(pages):

    chunks = []

    chunk_id = 0

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end]

            chunks.append({
                "id": chunk_id,
                "page": page["page"],
                "text": chunk_text
            })

            chunk_id += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# -----------------------------
# 3. Main ingestion
# -----------------------------

def main():

    print("Loading PDF...")

    pages = load_pdf(PDF_PATH)

    print(f"Pages loaded: {len(pages)}")


    print("Creating chunks...")

    chunks = create_chunks(pages)

    print(f"Chunks created: {len(chunks)}")


    # -----------------------------
    # Load embedding model
    # -----------------------------

    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    vector_size = model.get_sentence_embedding_dimension()

    print(f"Embedding dimension: {vector_size}")


    # -----------------------------
    # Create Qdrant
    # -----------------------------

    client = QdrantClient(path=QDRANT_PATH)


    # Delete old collection if it exists

    collections = client.get_collections()

    existing = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME in existing:

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )


    # Create collection

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=vector_size,

            distance=Distance.COSINE
        )
    )


    # -----------------------------
    # Generate embeddings
    # -----------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )


    # -----------------------------
    # Store vectors
    # -----------------------------

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        point = PointStruct(

            id=chunk["id"],

            vector=embedding.tolist(),

            payload={
                "text": chunk["text"],
                "page": chunk["page"]
            }
        )

        points.append(point)


    client.upsert(

        collection_name=COLLECTION_NAME,

        points=points
    )


    print()
    print("================================")
    print("INGESTION COMPLETE")
    print("================================")

    print(f"Pages   : {len(pages)}")
    print(f"Chunks  : {len(chunks)}")
    print(f"Vectors : {len(points)}")

    print("Stored in Qdrant.")


# if __name__ == "__main__":
#     main()