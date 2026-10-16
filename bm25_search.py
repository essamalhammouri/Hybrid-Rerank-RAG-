import re
import math
from collections import Counter


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return tokens


def build_corpus(lang="ar"):
    records = load_all()
    if lang == "en":
        texts = [f"{r['department_en']} - {r['service_en']}" for r in records]
    else:
        texts = [f"{r['department']} - {r['service']}" for r in records]
    tokenized = [tokenize(t) for t in texts]
    return records, tokenized


def build_stats(tokenized):
    N = len(tokenized)
    doc_freqs = [Counter(tokens) for tokens in tokenized]
    doc_lengths = [len(tokens) for tokens in tokenized]
    avgdl = sum(doc_lengths) / N

    df = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    return {
        "N": N,
        "doc_freqs": doc_freqs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "df": df,
    }


def idf(term, stats):
    N = stats["N"]
    n_t = stats["df"][term]
    return math.log((N - n_t + 0.5) / (n_t + 0.5) + 1)


def bm25_score(query_tokens, doc_idx, stats, k1=1.5, b=0.75):
    score = 0.0
    doc_tf = stats["doc_freqs"][doc_idx]
    doc_len = stats["doc_lengths"][doc_idx]
    avgdl = stats["avgdl"]

    for term in query_tokens:
        f = doc_tf.get(term, 0)
        if f == 0:
            continue
        term_idf = idf(term, stats)
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * doc_len / avgdl)
        score += term_idf * (numerator / denominator)

    return score


def bm25_search(query, stats, records, top_k=5):
    query_tokens = tokenize(query)
    scores = [(i, bm25_score(query_tokens, i, stats)) for i in range(stats["N"])]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal rank fusion - combines BM25's ranking with vector search's
# ranking into one, using only each side's RANK (not its raw score), since
# BM25 scores (0 to 20+, open-ended) and cosine scores (0 to 1) aren't on
# comparable scales and can't just be added together.
#
# vector_hits: Qdrant result.points, best first (each has .id and .score -
#              .id is the same array index you upserted it under)
# bm25_results: bm25_search()'s own return value, best first - already
#               (idx, score) tuples using that same array index
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(vector_hits, bm25_results, k=60):
    scores = {}

    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0) + 1 / (k + rank)

    for rank, (doc_idx, _) in enumerate(bm25_results, start=1):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    from rag_answer_norm import load_all

    records, tokenized = build_corpus(lang="ar")
    stats = build_stats(tokenized)

    results = bm25_search("شو الوثائق المطلوبة لاصدار شهادة ثانوية عامة باللغة الانجليزية", stats, records)
    for idx, score in results:
        print(f"{score:.3f}  |  {records[idx]['department']} - {records[idx]['service']}")
