
from phase2.Keyword_retrieval import KeywordRetriever
from phase2.Dense_retrieval import DenseRetriever
from phase2.Reciprocal_Rank_Fusion import reciprocal_rank_fusion


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search
    and dense vector search using Reciprocal Rank Fusion.
    """

    def __init__(self, top_k=5, rrf_k=60):

        self.top_k = top_k
        self.rrf_k = rrf_k

        print("Initializing Hybrid Retriever...")

        # Initialize both retrieval systems.
        self.keyword_retriever = KeywordRetriever(top_k=top_k)
        self.dense_retriever = DenseRetriever(top_k=top_k)

        print("Hybrid Retriever initialized successfully.")

    def retrieve(self, query):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        print("\nRunning keyword retrieval...")
        keyword_results = self.keyword_retriever.retrieve(query)

        print("Running dense retrieval...")
        dense_results = self.dense_retriever.retrieve(query)

        print("Combining results using RRF...")

        hybrid_results = reciprocal_rank_fusion(
            result_lists=[
                keyword_results,
                dense_results
            ],
            k=self.rrf_k,
            top_k=self.top_k
        )

        return hybrid_results

    def close(self):
        """Release the Qdrant client used by dense retrieval."""
        self.dense_retriever.close()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    retriever = HybridRetriever(top_k=5)

    try:
        query = "Why did Kattappa kill Baahubali?"

        results = retriever.retrieve(query)

        print("\nHYBRID RETRIEVAL RESULTS")
        print("=" * 60)

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"Chunk ID: {result['id']}")
            print(f"Page: {result.get('page')}")
            print(f"RRF Score: {result['rrf_score']:.6f}")
            print(f"Text: {result['text'][:300]}...")

    finally:
        retriever.close()
