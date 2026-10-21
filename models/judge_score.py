import json
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