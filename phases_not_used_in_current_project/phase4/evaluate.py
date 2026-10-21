import json
import os
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from phase3.RAG_Pipeline import AgenticRAG
from phase4.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    token_f1,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "phase4" / "evaluation_dataset.json"
RESULTS_DIR = ROOT / "phase4" / "results"


def judge_score(client, question, answer, reference, context, metric):
    instructions = {
        "correctness": (
            "Compare the answer with the reference answer. "
            "Judge factual agreement and completeness."
        ),
        "faithfulness": (
            "Check whether every factual claim in the answer "
            "is supported by the supplied context. "
            "Penalize unsupported claims."
        ),
        "relevance": (
            "Check whether the answer directly addresses "
            "the user's question."
        ),
        "source_support": (
            "Check whether the supplied source passages support "
            "the factual claims in the answer. Do not assume "
            "that a source is valid merely because it is listed."
        ),
    }

    prompt = f"""
Evaluate this RAG answer.

Metric: {metric}
Task: {instructions[metric]}

Question:
{question}

Reference answer:
{reference or "Not provided"}

Retrieved context:
{context}

Generated answer:
{answer}

Return only valid JSON:
{{
  "score": 0.0,
  "reason": "Brief explanation"
}}

Score must be between 0.0 and 1.0.
Use 0.0 for completely unsuccessful and 1.0 for excellent.
Do not invent evidence.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = json.loads(
        response.choices[0].message.content
    )

    score = float(result["score"])

    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Invalid judge score: {score}")

    return {
        "score": score,
        "reason": result.get("reason", ""),
    }


def main():
    with open(DATASET_PATH, encoding="utf-8") as file:
        dataset = json.load(file)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    rag = AgenticRAG(
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
