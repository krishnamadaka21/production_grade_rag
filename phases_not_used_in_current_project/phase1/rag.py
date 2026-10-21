from retrieve import retrieve
from generate import generate_answer


def build_context(results):

    context_parts = []

    for i, result in enumerate(results):

        text = result.payload["text"]

        page = result.payload["page"]

        context_parts.append(

            f"""
SOURCE {i + 1}
Page: {page}

{text}
"""
        )

    return "\n".join(context_parts)


def main():

    question = input("\nAsk a question: ")


    # -----------------------------
    # RETRIEVAL
    # -----------------------------

    results = retrieve(question)


    print("\nRetrieved:")
    print("-" * 60)

    for i, result in enumerate(results):

        print(
            f"{i + 1}. "
            f"Page {result.payload['page']} "
            f"| Score: {result.score:.4f}"
        )


    # -----------------------------
    # BUILD CONTEXT
    # -----------------------------

    context = build_context(results)


    # -----------------------------
    # GENERATION
    # -----------------------------

    answer = generate_answer(

        question,

        context
    )


    print("\n")
    print("=" * 60)

    print("ANSWER")

    print("=" * 60)

    print(answer)


if __name__ == "__main__":
    main()