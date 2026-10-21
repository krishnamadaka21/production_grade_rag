
from pathlib import Path
import pickle
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

PHASE1_DIR = Path(__file__).resolve().parent
print(PHASE1_DIR)
PROJECT_ROOT = PHASE1_DIR.parent
print(PROJECT_ROOT)
PDF_PATH = PROJECT_ROOT / "data" / "Baahubali_Complete_Story.pdf"
QDRANT_PATH = PHASE1_DIR / "qdrant_data"
BM25_INDEX_PATH = PHASE1_DIR / "bm25_index.pkl"

COLLECTION_NAME = "phase1_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """Lowercase text and extract words and numbers."""
    return re.findall(r"\b\w+\b", text.lower())


# ============================================================
# 1. LOAD PDF
# ============================================================

def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    return pages


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

def create_chunks(pages):

    chunks = []
    chunk_id = 0

    for page in pages:

        text = page["text"]
        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "id": chunk_id,
                    "page": page["page"],
                    "text": chunk_text
                })

                chunk_id += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ============================================================
# 3. BUILD AND SAVE BM25 INDEX
# ============================================================

def build_and_save_bm25(chunks):

    print("\nBuilding BM25 index...")

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    BM25_INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the index and its matching chunk records together.
    with open(BM25_INDEX_PATH, "wb") as file:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": chunks
            },
            file
        )

    print(f"BM25 index saved to: {BM25_INDEX_PATH}")
    print(f"Indexed chunks: {len(chunks)}")


# ============================================================
# 4. MAIN INGESTION
# ============================================================

def main():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("Loading PDF...")
    pages = load_pdf(PDF_PATH)
    print(f"Pages loaded: {len(pages)}")

    print("\nCreating chunks...")
    chunks = create_chunks(pages)
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No text chunks were extracted.")

    # -----------------------------
    # Load embedding model
    # -----------------------------

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    vector_size = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {vector_size}")

    # -----------------------------
    # Prepare embeddings
    # -----------------------------

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    # -----------------------------
    # Create Qdrant collection
    # -----------------------------

    print("\nConnecting to Qdrant...")
    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        existing = [
            collection.name
            for collection in client.get_collections().collections
        ]

        if COLLECTION_NAME in existing:
            print("Deleting existing Qdrant collection...")
            client.delete_collection(
                collection_name=COLLECTION_NAME
            )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        # -----------------------------
        # Store vectors
        # -----------------------------

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            points.append(
                PointStruct(
                    id=chunk["id"],
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk["text"],
                        "page": chunk["page"]
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    finally:
        client.close()

    # -----------------------------
    # Build BM25 index
    # -----------------------------

    build_and_save_bm25(chunks)

    # -----------------------------
    # Summary
    # -----------------------------

    print("\n================================")
    print("INGESTION COMPLETE")
    print("================================")
    print(f"Pages          : {len(pages)}")
    print(f"Chunks         : {len(chunks)}")
    print(f"Vectors        : {len(chunks)}")
    print(f"BM25 index     : {BM25_INDEX_PATH}")
    print("Qdrant + BM25 indexes are ready.")


if __name__ == "__main__":
    main()
