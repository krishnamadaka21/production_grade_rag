
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "phase1_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


# ============================================================
# DENSE RETRIEVER
# ============================================================

class DenseRetriever:

    def __init__(self, top_k=TOP_K):

        self.top_k = top_k

        # Resolve the existing Phase 1 Qdrant database
        project_root = Path(__file__).resolve().parent.parent
        qdrant_path = project_root / "phase1" / "qdrant_data"

        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("Connecting to Qdrant...")
        self.client = QdrantClient(path=str(qdrant_path))

        print("Dense retriever initialized successfully.")

    def retrieve(self, query):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # 1. Convert query into an embedding vector
        query_vector = self.model.encode(query).tolist()

        # 2. Search the existing Qdrant collection
        points = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=self.top_k,
            with_payload=True
        ).points

        # 3. Normalize results for later fusion
        results = []

        for point in points:

            payload = point.payload or {}

            if not payload.get("text"):
                continue

            results.append({
                "id": point.id,
                "text": payload["text"],
                "page": payload.get("page"),
                "score": float(point.score)
            })

        return results

    def close(self):
        self.client.close()


# ============================================================
# TEST
# ============================================================

# if __name__ == "__main__":
#
#     retriever = DenseRetriever(top_k=5)
#
#     try:
#         query = "Why did Kattappa kill Baahubali?"
#
#         print(f"\nQuery: {query}")
#         print("=" * 60)
#
#         results = retriever.retrieve(query)
#
#         for rank, result in enumerate(results, start=1):
#
#             print(f"\nRank: {rank}")
#             print(f"Chunk ID: {result['id']}")
#             print(f"Page: {result['page']}")
#             print(f"Dense Score: {result['score']:.4f}")
#             print(f"Text: {result['text'][:300]}...")
#
#     finally:
#         retriever.close()
