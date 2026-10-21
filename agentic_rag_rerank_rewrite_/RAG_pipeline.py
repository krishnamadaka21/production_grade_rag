from models.generate import generate_answer
from ingest_retrieval.Hybrid_retrieval import HybridRetriever
from ingest_retrieval.Dense_retrieval import DenseRetriever
from ingest_retrieval.Keyword_retrieval import KeywordRetriever
from .Query_Rewriter import QueryRewriter
from .Reranker import ChunkReranker


class AgenticRAG:
    """
    Phase 3 pipeline:
    Original question -> query rewriting -> hybrid retrieval -> reranking -> Groq answer.
    """

    def __init__(
        self,tp,
        retrieval_top_k=10,
        final_top_k=5,
        enable_query_rewriting=True,
        enable_reranking=True,
    ):
        print("Initializing  Agentic RAG pipeline...")
        if tp=="dense":
            self.retriever=DenseRetriever(top_k=retrieval_top_k)
        elif tp=="keyword":
            self.retriever=KeywordRetriever(top_k=retrieval_top_k)
        else:
            self.retriever = HybridRetriever(top_k=retrieval_top_k)
        self.query_rewriter = QueryRewriter() if enable_query_rewriting else None
        self.reranker = (
            ChunkReranker(top_k=final_top_k) if enable_reranking else None
        )
        self.final_top_k = final_top_k

        print("Phase 3 Agentic RAG pipeline initialized.")

    def ask(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        original_question = question.strip()

        # 1. Rewrite the query for retrieval only.
        search_query = (
            self.query_rewriter.rewrite(original_question)
            if self.query_rewriter
            else original_question
        )

        # 2. Existing Phase 2 retrieval: BM25 + dense retrieval + RRF.
        results = self.retriever.retrieve(search_query)

        if not results:
            return {
                "answer": "I don't have enough information in the provided document.",
                "sources": [],
                "original_question": original_question,
                "search_query": search_query,
            }

        # 3. Rerank candidates against the original user question.
        if self.reranker:
            results = self.reranker.rerank(original_question, results)
        else:
            results = results[: self.final_top_k]

        # 4. Build grounded context and deduplicated source list.
        context_parts = []
        sources = []

        for result in results:
            page = result.get("page")
            chunk_id = result.get("id")
            text = result.get("text", "")

            context_parts.append(f"[Page {page}, Chunk {chunk_id}]\n{text}")

            source = {"page": page, "chunk_id": chunk_id}
            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        # Preserve the original wording for answer generation.
        answer = generate_answer(
            question=original_question,
            context=context,
        )

        return {
            "answer": answer,
            "sources": sources,
            "original_question": original_question,
            "search_query": search_query,
            "retrieved_count": len(results),
            "reranked_results": results,
        }

    def close(self):
        self.retriever.close()


# if __name__ == "__main__":
#     question = input("\nAsk a question: ").strip()
#     tp=input("choose\ndense | keyword | both\n").strip().lower()
#     rag = AgenticRAG(tp,retrieval_top_k=10, final_top_k=5)
#
#     try:
#
#         result = rag.ask(question)
#
#         print("\nORIGINAL QUESTION")
#         print(result["original_question"])
#
#         print("\nSEARCH QUERY")
#         print(result["search_query"])
#
#         print("\nANSWER")
#         print(result["answer"])
#
#         print("\nSOURCES")
#         for source in result["sources"]:
#             print(f"Page {source['page']}, Chunk {source['chunk_id']}")
#
#         print("\nFINAL RERANKED CHUNKS")
#         for rank, chunk in enumerate(result.get("reranked_results", []), start=1):
#             print(
#                 f"{rank}. Page={chunk.get('page')} "
#                 f"Rerank score={chunk.get('rerank_score', 'not enabled')}"
#             )
#     finally:
#         if tp!="keyword":
#             rag.close()
