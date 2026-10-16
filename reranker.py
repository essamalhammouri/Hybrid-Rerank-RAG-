"""
Cross-encoder reranking stage. Takes the fused candidate list from
retrieve() and re-scores each (query, candidate) pair jointly, instead of
comparing independently-computed vectors - see FINDINGS.md for why that
matters for near-duplicate titles.

First run downloads the model from Hugging Face (one-time, needs internet).
"""

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_model = None

def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query, candidate_ids, records, top_k=5):
    """
    candidate_ids: record indices to re-score (e.g. the fused shortlist)
    records: the full records list - same index alignment used everywhere
             else in this project (bm25_records / Qdrant point ids)
    Returns [(idx, cross_encoder_score), ...] sorted best first, length top_k.
    """
    model = get_reranker()
    pairs = [[query, f"{records[idx]['department']} - {records[idx]['service']}"]
             for idx in candidate_ids]
    scores = model.predict(pairs)
    scored = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]