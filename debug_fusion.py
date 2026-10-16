"""
One-query diagnostic: shows the vector ranking, the BM25 ranking, and the
fused ranking SIDE BY SIDE for a single query, instead of only the final
top-5 the chatbot shows. Doesn't touch production - read-only against the
same gov_docs_norm collection rag_answer_norm.py already built.

Why this exists: the live test of "certificate vs exam-transcript in
English" showed the fused result still ranked the wrong near-duplicate
(exam transcript) ABOVE the correct one (certificate) at position 1, even
though BM25 alone ranks the correct one decisively at #1 (19.298 vs 9.406,
per FINDINGS.md). That only makes sense if vector search ranks the WRONG
one very high (likely #1) AND ranks the CORRECT one surprisingly low (not
just #2 - low enough that RRF's k=60 can't make BM25's #1 win). This script
shows the actual ranks instead of us guessing which.

Run:
    python debug_fusion.py
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_answer_norm import (
    load_all, normalize_arabic, allowed_levels, COLLECTION, EMBED_MODEL,
)
from bm25_search import tokenize, build_stats, bm25_search, reciprocal_rank_fusion
from qdrant_client.models import Filter, FieldCondition, MatchAny

QUERY = "شو الوثائق المطلوبة لاصدار شهادة ثانوية عامة باللغة الانجليزية؟"
ROLE = "employee"
N = 10  # how deep to print each ranking


def main():
    embedder = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path="./qdrant_data")  # reuses the index rag_answer_norm.py already built

    records = load_all()
    bm25_texts = [f"{r['department']} - {r['service']}" for r in records]
    bm25_stats = build_stats([tokenize(t) for t in bm25_texts])

    # --- vector ranking ---
    clean = normalize_arabic(QUERY)
    qvec = embedder.encode(clean).tolist()
    access_filter = Filter(
        must=[FieldCondition(key="access", match=MatchAny(any=allowed_levels(ROLE)))]
    )
    vector_hits = client.query_points(
        collection_name=COLLECTION, query=qvec, limit=N, query_filter=access_filter,
    ).points

    # --- bm25 ranking (unnormalized query, matching retrieve()'s choice) ---
    bm25_hits = bm25_search(QUERY, bm25_stats, records, top_k=N)
    allowed = set(allowed_levels(ROLE))
    bm25_hits = [(idx, score) for idx, score in bm25_hits if records[idx]["access"] in allowed]

    # --- fused ranking ---
    fused = reciprocal_rank_fusion(vector_hits, bm25_hits)

    print(f"Query: {QUERY}\n")

    print(f"{'VECTOR (dense) rank':<40} {'score':<8}")
    for rank, hit in enumerate(vector_hits, start=1):
        print(f"  #{rank:<2} {hit.payload['service_en']:<35} {hit.score:.4f}")

    print(f"\n{'BM25 rank':<40} {'score':<8}")
    for rank, (idx, score) in enumerate(bm25_hits, start=1):
        print(f"  #{rank:<2} {records[idx]['service_en']:<35} {score:.3f}")

    print(f"\n{'FUSED (RRF) rank':<40} {'score':<8}")
    for rank, (idx, score) in enumerate(fused[:N], start=1):
        print(f"  #{rank:<2} {records[idx]['service_en']:<35} {score:.5f}")
    from reranker import rerank
    candidate_ids = [idx for idx, _ in fused[:N]]
    reranked = rerank(QUERY, candidate_ids, records, top_k=N)

    print(f"\n{'RERANKED rank':<40} {'score':<8}")
    for rank, (idx, score) in enumerate(reranked, start=1):
        print(f"  #{rank:<2} {records[idx]['service_en']:<35} {score:.4f}")


if __name__ == "__main__":
    main()
