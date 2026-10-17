
from BM25 import load_chunks_from_qdrant, build_bm25_index, bm25_search


class KeywordRetriever:
    """
    Keyword-based retriever using BM25.
    """

    def __init__(self, top_k=5):
        self.top_k = top_k

        print("Loading chunks for keyword retrieval...")
        self.chunks = load_chunks_from_qdrant()

        if not self.chunks:
            raise ValueError(
                "No chunks found in Qdrant. "
                "Run Phase 1 ingestion first."
            )

        print(f"Loaded {len(self.chunks)} chunks.")

        print("Building BM25 index...")
        self.bm25 = build_bm25_index(self.chunks)

        print("Keyword retriever initialized successfully.")

    def retrieve(self, query):
        """
        Retrieve the most relevant chunks for a query.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        results = bm25_search(
            query=query,
            bm25=self.bm25,
            chunks=self.chunks,
            top_k=self.top_k
        )

        return results


if __name__ == "__main__":

    retriever = KeywordRetriever(top_k=5)

    query = "Why did Kattappa kill Baahubali?"

    results = retriever.retrieve(query)

    print("\nQuery:", query)
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['id']}")
        print(f"Page: {result['page']}")
        print(f"BM25 Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:300]}...")
