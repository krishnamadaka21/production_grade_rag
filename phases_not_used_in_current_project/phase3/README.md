# Phase 3 — Query Rewriting and Reranking

## Files
- `Query_Rewriter.py`: Uses Groq to rewrite a question for retrieval. If `GROQ_API_KEY`
  is unavailable or the request fails, it falls back to the original question.
- `Reranker.py`: Uses a Sentence Transformers cross-encoder to rerank retrieved chunks.
- `RAG_Pipeline.py`: Integrates Phase 1 generation and Phase 2 hybrid retrieval.

## Expected project structure

```text
project/
├── phase1/
│   └── generate.py
├── phase2/
│   ├── Hybrid_retrieval.py
│   ├── Keyword_retrieval.py
│   ├── Dense_retrieval.py
│   └── Reciprocal_Rank_Fusion.py
└── phase3/
    ├── __init__.py
    ├── Query_Rewriter.py
    ├── Reranker.py
    └── RAG_Pipeline.py
```

The imports assume that the project root is the current working directory and that
Phase 1's `generate.py` exports `generate_answer(question=..., context=...)`.

## Install dependencies

Inside the project's virtual environment:

```bash
pip install groq sentence-transformers
```

Your existing Phase 1/2 dependencies are still required.

Set `GROQ_API_KEY` in the environment or in your existing `.env` loading setup.
Optionally set `GROQ_MODEL` and `RERANKER_MODEL`.

## Run

From the project root:

```bash
python -m phase3.RAG_Pipeline
```

The cross-encoder model is downloaded the first time it is loaded, so the first run
requires internet access. Subsequent runs may use the cached model.

## Important
- `retrieval_top_k=10` fetches candidate chunks from Phase 2 before reranking.
- `final_top_k=5` sends at most five chunks to generation.
- The original user question is used for answer generation; the rewritten query is
  used only for retrieval.
- Reranking is a scoring/reordering step, not a guarantee that the selected context
  contains the answer.
