
def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_k=5
):
    """
    Combine ranked results from multiple retrievers using RRF.

    Args:
        result_lists: List of ranked result lists.
        k: RRF ranking constant.
        top_k: Number of final results to return.

    Returns:
        Fused results sorted by RRF score.
    """

    fused_scores = {}
    documents = {}

    for results in result_lists:

        # Each retriever's results are already ranked.
        for rank, result in enumerate(results, start=1):

            chunk_id = result["id"]

            # Store document information once per unique chunk.
            if chunk_id not in documents:
                documents[chunk_id] = result.copy()

            # RRF uses rank positions, not original scores.
            rrf_score = 1.0 / (k + rank)

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + rrf_score
            )

    # Sort chunks by their combined RRF scores.
    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True
    )

    fused_results = []

    for chunk_id in ranked_ids[:top_k]:

        result = documents[chunk_id].copy()

        result["rrf_score"] = fused_scores[chunk_id]

        fused_results.append(result)

    return fused_results


# if __name__ == "__main__":
#
#     from Keyword_retrieval import KeywordRetriever
#     from Dense_retrieval import DenseRetriever
#
#     query = "Why did Kattappa kill Baahubali?"
#
#     # First retrieve keyword results.
#     keyword_retriever = KeywordRetriever(top_k=5)
#
#     try:
#         keyword_results = keyword_retriever.retrieve(query)
#
#         # Then retrieve dense results.
#         dense_retriever = DenseRetriever(top_k=5)
#
#         try:
#             dense_results = dense_retriever.retrieve(query)
#
#         finally:
#             dense_retriever.close()
#
#     finally:
#         # KeywordRetriever loads its own Qdrant client
#         # inside load_chunks_from_qdrant(), which is closed there.
#         pass
#
#     # Fuse both ranked result lists.
#     hybrid_results = reciprocal_rank_fusion(
#         result_lists=[
#             keyword_results,
#             dense_results
#         ],
#         k=60,
#         top_k=5
#     )
#
#     print("\nQuery:", query)
#     print("\nReciprocal Rank Fusion Results")
#     print("=" * 60)
#
#     for rank, result in enumerate(hybrid_results, start=1):
#
#         print(f"\nRank: {rank}")
#         print(f"Chunk ID: {result['id']}")
#         print(f"Page: {result.get('page')}")
#         print(f"RRF Score: {result['rrf_score']:.6f}")
#         print(f"Text: {result['text'][:300]}...")
