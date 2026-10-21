from ingest_retrieval.Keyword_retrieval import KeywordRetriever
from models.generate import generate_answer


class KeywordRAG:

    def __init__(self, top_k=5):

        print("Initializing Hybrid RAG pipeline...")

        self.retriever = KeywordRetriever(top_k=top_k)

        print("Hybrid RAG pipeline initialized.")

    def ask(self, question):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve chunks using BM25 + Dense + RRF
        results = self.retriever.retrieve(question)

        if not results:
            return {
                "answer": (
                    "I don't have enough information "
                    "in the provided document."
                ),
                "sources": []
            }

        # 2. Build context for the existing Groq function
        context_parts = []
        sources = []

        for result in results:

            page = result.get("page")
            chunk_id = result.get("id")
            text = result.get("text", "")

            context_parts.append(
                f"[Page {page}, Chunk {chunk_id}]\n{text}"
            )

            source = {
                "page": page,
                "chunk_id": chunk_id
            }

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        # 3. Generate answer using your Phase 1 function
        answer = generate_answer(
            question=question,
            context=context
        )

        return {
            "answer": answer,
            "sources": sources
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    rag = KeywordRAG(top_k=5)

    try:

        question = "Why did Kattappa kill Baahubali?"

        result = rag.ask(question)

        print("\nQUESTION")
        print(question)

        print("\nANSWER")
        print(result["answer"])

        print("\nSOURCES")

        for source in result["sources"]:
            print(
                f"Page {source['page']}, "
                f"Chunk {source['chunk_id']}"
            )

    finally:
        pass