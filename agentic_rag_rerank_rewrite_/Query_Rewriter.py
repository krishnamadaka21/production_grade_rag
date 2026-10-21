import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq


class QueryRewriter:
    """
    Rewrites a user's question to improve retrieval.
    If Groq is unavailable or rewriting fails, the original query is returned.
    """

    def __init__(self, model=None):
        api_key = os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        )
        self.client = Groq(api_key=api_key) if api_key else None

    def rewrite(self, question: str) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        original_question = question.strip()

        # Safe fallback: Phase 3 should not stop the RAG pipeline.
        if self.client is None:
            print("Warning: GROQ_API_KEY is missing. Using the original query.")
            return original_question

        prompt = f"""
        You are an intelligent query rewriting component for a RAG system.

        Your task is to transform the user's raw query into a clear,
        grammatically correct, retrieval-friendly search query.

        Rules:
        1. Correct spelling mistakes and grammatical errors.
        2. Correct obvious misspellings of names and technical terms
           when the intended term is clear.
        3. Rewrite incomplete or poorly phrased questions into natural,
           complete questions.
        4. Preserve the exact intent of the original query.
        5. Do not answer the question.
        6. Do not add new questions, facts, or assumptions.
        7. Preserve important entities, names, and keywords.
        8. Return only the improved query.
        9. If the query is already clear, return it unchanged.

        Examples:
        Input: whos is the bahubali
        Output: Who is Baahubali?

        Input: explain retrival augmented generation
        Output: Explain Retrieval-Augmented Generation.

        Input: what are the benifits of machine learning
        Output: What are the benefits of machine learning?

        Input: what is the capital of india
        Output: What is the capital of India?

        User query:
        {original_question}
        """.strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You rewrite questions for information retrieval. You do not answer questions."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=160,
            )

            rewritten = (response.choices[0].message.content or "").strip()
            rewritten = rewritten.strip('"').strip("'").strip()

            if not rewritten:
                return original_question

            print(f"Original query : {original_question}")
            print(f"Rewritten query: {rewritten}")
            return rewritten

        except Exception as exc:
            print(f"Warning: Query rewriting failed ({exc}). Using the original query.")
            return original_question
if __name__ == "__main__":
    obj=QueryRewriter()
    print(obj.rewrite(question="who is thebahubali"))
