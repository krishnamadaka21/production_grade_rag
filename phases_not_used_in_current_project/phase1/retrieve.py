from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


COLLECTION_NAME = "phase1_documents"

QDRANT_PATH = "./qdrant_data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5


# -----------------------------
# Load models
# -----------------------------

model = SentenceTransformer(EMBEDDING_MODEL)

client = QdrantClient(path=QDRANT_PATH)


# -----------------------------
# Retrieval function
# -----------------------------

def retrieve(query):

    # Convert question into vector

    query_vector = model.encode(query).tolist()


    # Search Qdrant

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        limit=TOP_K
    ).points


    return results


# -----------------------------
# Test
# -----------------------------
#
# if __name__ == "__main__":
#
#     question = input("Question: ")
#
#     results = retrieve(question)
#
#     print("\nRetrieved chunks:\n")
#
#     for i, result in enumerate(results):
#
#         print("=" * 60)
#
#         print(f"Result: {i + 1}")
#
#         print(f"Score: {result.score}")
#
#         print(f"Page: {result.payload['page']}")
#
#         print()
#
#         print(result.payload["text"])