from pathlib import Path

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_PATH = "../phase1/qdrant_data"
COLLECTION_NAME = "phase1_documents"


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Convert text into tokens for BM25.

    Example:
        "ISO 9001 is a standard"

    becomes:

        ["iso", "9001", "is", "a", "standard"]
    """

    return text.lower().split()


# ============================================================
# LOAD CHUNKS FROM QDRANT
# ============================================================

def load_chunks_from_qdrant():
    """
    Load all document chunks stored in Qdrant.
    """

    # Resolve path relative to this file
    project_root = Path(__file__).resolve().parent.parent
    qdrant_path = project_root / "phase1" / "qdrant_data"

    client = QdrantClient(path=str(qdrant_path))

    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )

    chunks = []

    for record in records:

        payload = record.payload

        if payload is None:
            continue

        text = payload.get("text")

        if not text:
            continue

        chunks.append({
            "id": record.id,
            "text": text,
            "page": payload.get("page")
        })

    client.close()

    return chunks


# ============================================================
# BUILD BM25 INDEX
# ============================================================

def build_bm25_index(chunks):
    """
    Build a BM25 index from document chunks.
    """

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(query, bm25, chunks, top_k=5):
    """
    Search the BM25 index.

    Returns the top-k most relevant chunks.
    """

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    # Sort indices by score in descending order
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "id": chunks[index]["id"],
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": float(scores[index])
        })

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading chunks from Qdrant...")

    chunks = load_chunks_from_qdrant()

    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        print("No chunks found in Qdrant.")
        exit()

    print("Building BM25 index...")

    bm25 = build_bm25_index(chunks)

    print("BM25 index created successfully.")

    # Test query
    # query = "What is ISO 9001?"
    query = "Why did Kattappa kill Baahubali?"
    print("\nQuery:")
    print(query)

    results = bm25_search(
        query=query,
        bm25=bm25,
        chunks=chunks,
        top_k=5
    )

    print("\nBM25 Results:")
    print("=" * 60)

    for i, result in enumerate(results, start=1):

        print(f"\nRank {i}")
        print(f"Chunk ID : {result['id']}")
        print(f"Page     : {result['page']}")
        print(f"Score    : {result['score']:.4f}")
        print(f"Text     : {result['text'][:300]}...")