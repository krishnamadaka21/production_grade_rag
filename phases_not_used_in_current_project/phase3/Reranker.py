import os
from typing import Any

from sentence_transformers import CrossEncoder


class ChunkReranker:
    """
    Reorders retrieved chunks using a cross-encoder relevance model.

    Expected input:
        [
            {"id": "...", "page": 1, "text": "...", "rrf_score": 0.03},
            ...
        ]

    The original dictionaries are preserved; a `rerank_score` field is added.
    """

    def __init__(self, model_name=None, top_k=5):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.top_k = top_k
        self.model = None

    def _load_model(self):
        # Lazy loading avoids downloading/loading the model until reranking is needed.
        if self.model is None:
            print(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        usable_documents = [
            doc for doc in documents
            if isinstance(doc, dict) and doc.get("text")
        ]
        if not usable_documents:
            return []

        try:
            self._load_model()
            pairs = [(query, doc["text"]) for doc in usable_documents]
            scores = self.model.predict(pairs)

            reranked = []
            for doc, score in zip(usable_documents, scores):
                item = dict(doc)
                item["rerank_score"] = float(score)
                reranked.append(item)

            reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
            return reranked[: self.top_k]

        except Exception as exc:
            # Keep the application usable if the model cannot be loaded or run.
            print(f"Warning: Reranking failed ({exc}). Keeping hybrid retrieval order.")
            return usable_documents[: self.top_k]
