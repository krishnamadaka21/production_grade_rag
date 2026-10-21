import json
import os
import time
from pathlib import Path
from models.judge_score import judge_score
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from agentic_rag_rerank_rewrite_.RAG_Pipeline import AgenticRAG
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    token_f1,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "evaluation_dataset.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
print(ROOT)

def main():
    with open(DATASET_PATH, encoding="utf-8") as file:
        dataset = json.load(file)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    rag = AgenticRAG(
        "hybrid",
        retrieval_top_k=10,
        final_top_k=5,
        enable_query_rewriting=True,
        enable_reranking=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    try:
        for test in dataset:
            question = test["question"]
            reference = test.get("reference_answer", "")
            relevant = set(test.get("relevant_chunk_ids", []))

            print(f"\nEvaluating {test['id']}: {question}")

            start = time.perf_counter()
            result = rag.ask(question)
            latency = time.perf_counter() - start

            chunks = result.get("reranked_results", [])
            retrieved_ids = [
                str(chunk["id"])
                for chunk in chunks
                if chunk.get("id") is not None
            ]

            context = "\n\n".join(
                f"[Page {chunk.get('page')}, "
                f"Chunk {chunk.get('id')}]\n"
                f"{chunk.get('text', '')}"
                for chunk in chunks
            )

            answer = result["answer"]

            record = {
                "id": test["id"],
                "question": question,
                "answer": answer,
                "latency_seconds": round(latency, 3),
                "retrieved_chunk_ids": retrieved_ids,
                "sources": result.get("sources", []),
                "search_query": result.get("search_query"),
                "precision_at_5": precision_at_k(
                    retrieved_ids, relevant, 5
                ),
                "recall_at_5": recall_at_k(
                    retrieved_ids, relevant, 5
                ),
                "mrr": reciprocal_rank(
                    retrieved_ids, relevant
                ),
                "ndcg_at_5": ndcg_at_k(
                    retrieved_ids, relevant, 5
                ),
            }

            if reference:
                record["token_f1"] = token_f1(
                    reference, answer
                )

            for metric in (
                "correctness",
                "faithfulness",
                "relevance",
                "source_support",
            ):
                try:
                    record[metric] = judge_score(
                        client=client,
                        question=question,
                        answer=answer,
                        reference=reference,
                        context=context,
                        metric=metric,
                    )
                except Exception as exc:
                    record[metric] = {
                        "error": str(exc)
                    }

            records.append(record)

            print(f"Answer: {answer}")
            print(f"Latency: {latency:.2f}s")
            print(f"Precision@5: {record['precision_at_5']}")
            print(f"Recall@5: {record['recall_at_5']}")
            print(f"MRR: {record['mrr']}")
            print(f"Token F1: {record.get('token_f1', 'N/A')}")

    finally:
        rag.close()

    output_path = RESULTS_DIR / "evaluation_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
