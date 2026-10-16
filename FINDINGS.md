# Retrieval Findings

## Vector search limitation: near-duplicate service titles

Tested whether two genuinely different government services with highly
overlapping titles get confused by pure vector search.

Titles compared:
- A: "اصدار شهادة ثانوية عامة باللغة الانجليزية" (issue English-language secondary certificate)
- B: "اصدار كشف علامات امتحان شهادة الثانوية العامة باللغة الانجليزية" (issue English-language exam results transcript)
- C: "طلب مشروحات / هوية / دفتر عائلة / جواز سفر بدل فاقد" (unrelated control: lost ID/family book/passport replacement)

Pairwise cosine similarity (paraphrase-multilingual-MiniLM-L12-v2):
- A vs B (different services, overlapping wording): 0.793
- A vs C (unrelated control): 0.510
- B vs C (unrelated control): 0.423

Context: typical on-topic query-to-document similarity in this system
ranges 0.63-0.76 (measured earlier in test_retrieval.py evaluation).

**Conclusion:** A and B score *higher* in similarity to each other (0.793)
than a real query typically scores against its own correct document.
Since embeddings average over all tokens, documents that share most of
their wording but differ in a few key distinguishing words collapse
close together in vector space, regardless of whether they describe the
same procedure. This explains an observed live failure: a query for
service A returned service B as the top match.

**Fix in progress:** adding BM25 keyword search, which scores on literal
distinguishing terms rather than blended meaning, combined with vector
search via reciprocal rank fusion.






## BM25 fix: correctly separates the near-duplicate services

Tested whether a from-scratch BM25 keyword search implementation fixes the
near-duplicate confusion documented above.

Query: "issuing a general secondary education certificate in english"
(English-translated version of the real query that originally failed)

Top 5 BM25 results:
1. Ministry of Education - Issuance of high school certificate in English — 13.478
2. Ministry of Education - Issuance of high school exam marks transcript in English — 9.755
3. Ministry of Education - Issuance of high school certificate in Arabic — 9.100
4. General Intelligence Department - Good conduct certificate — 8.768
5. Ministry of Education - Attestation of high school certificate — 6.555

**Conclusion:** BM25 correctly ranks the target service (certificate issuance)
above its near-duplicate sibling (exam transcript issuance) by a clear
margin (13.478 vs 9.755) — directly resolving the specific failure
documented above, where vector search returned the wrong service as its
top match. This confirms the hypothesis: exact keyword matching on
distinguishing terms recovers cases where blended vector embeddings
collapse near-duplicate titles together.

**Caveat:** this is one validated case, not yet a general conclusion.
The Arabic-normalization investigation earlier in this document shows
exactly how a single successful example can mislead — a 5-query test
suggested normalization only helped, but a full 193-query run reversed
that conclusion. Next: test BM25 against the other known near-duplicate
pairs in the dataset (الترخيص بدل فاقد/بدل تالف variants, التأمين الصحي
variants, قاضي القضاة marriage variants) before concluding BM25 reliably
fixes this entire class of problem.



## BM25 confirmed on real Arabic production data, not just the English translation

The previous BM25 test above was run against English-translated titles, not
the real Arabic documents the production system actually searches. Reran the
identical test against the real Arabic corpus, using the real Arabic query
exactly as typed into the chatbot earlier.

Query: "شو الوثائق المطلوبة لاصدار شهادة ثانوية عامة باللغة الانجليزية؟"

Top 5 BM25 results (real Arabic production data):
1. وزارة التربية والتعليم - اصدار شهادة ثانوية عامة باللغة الانجليزية — 19.298
2. وزارة التربية والتعليم - اصدار شهادة ثانوية عامة باللغة العربية — 15.050
3. وزارة التربية والتعليم - اصدار كشف علامات امتحان شهادة الثانوية العامة باللغة الانجليزية — 9.406
4. وزارة التربية والتعليم - اصدار كشف علامات امتحان شهادة الثانوية العامة باللغة العربية — 5.708
5. وزارة الخارجية وشؤون المغتربين - تصديق الوثائق الرسمية — 4.338

**Conclusion:** the gap between the correct service and its near-duplicate
sibling is even larger on the real Arabic data (19.298 vs 9.406, over 2x)
than on the English-translated demo (13.478 vs 9.755). The fix holds on the
actual documents the production system searches, not just a translated
approximation — resolving the caveat raised in the previous entry.





## Hybrid search + reranking: full 191-query validation

**Measurement:** Extended the eval harness to score 6 retrieval configurations (raw dense, normalized dense, query-only-normalized dense, BM25 alone, hybrid RRF fusion of raw+BM25, and hybrid+cross-encoder reranking) across 191 queries — the 25 hand-picked cases plus 168 auto-generated full-coverage queries (one per service). Overall R@5 / P@1:

| Mode | R@5 | P@1 |
|---|---|---|
| raw (dense only) | 89% | 74% |
| normalized (dense only) | 75% | 58% |
| query_only (dense only) | 85% | 71% |
| BM25 only | 95% | 93%* |
| hybrid (RRF, raw+BM25) | 96% | 83% |
| **reranked (hybrid + cross-encoder)** | **96%** | **87%** |

*BM25's overall P@1 is inflated by `full_coverage`: those 168 queries are auto-generated from each service's own text, so they're near-guaranteed keyword overlap — not a hard test for a lexical method. On the categories that actually stress hard cases:

| Category | raw | hybrid | reranked |
|---|---|---|---|
| near_duplicate (P@1) | 67% | 50% | **67%** |
| english (P@1) | 50% | 25% | **75%** |
| paraphrased (P@1) | 80% | 20% | 40% |

**Conclusion:** Reranking is the component that actually fixes the near-duplicate and cross-language failure modes this project set out to fix — RRF fusion alone (hybrid) was not sufficient, and on near_duplicate/english it was *worse* than raw dense search by itself. Adding the cross-encoder rerank stage after fusion recovered or beat raw's precision on every category except paraphrased.

**Caveat:** `paraphrased` is a real, measured regression: raw alone (80% P@1) beats both hybrid (20%) and reranked (40%). BM25 has no signal on paraphrased queries (different wording, no keyword overlap), and fusing its weak/misleading ranking into RRF can push the correct document out of the candidate shortlist entirely before reranking ever sees it (recall drops from 80% to 60%). Reranking can only reorder candidates fusion already kept — it can't recover one fusion dropped. This is a known, accepted trade-off for this project's scope, not something fixed here: BM25 fusion costs some paraphrase robustness in exchange for fixing near-duplicate and cross-language matching, which were the higher-priority failures.

## Arabic normalization retired from the pipeline

**Measurement:** Normalized dense search (corpus + query both run through `normalize_arabic()`) underperforms raw dense search on every category except `typo` (near_duplicate P@1: 33% vs raw's 67%; paraphrased P@1: 0% vs raw's 80%). Full-coverage misses: 47/168 under normalization vs 20/168 under raw.

**Conclusion:** Normalization was only ever earning its keep on typo'd queries, and hybrid+reranking now handles typos without it — the same raw+BM25+rerank config (no normalization anywhere) scores 100%/100% on the `typo` category. Normalization no longer has a job to do, so it was removed from both `parse_department()` (documents indexed unnormalized) and `retrieve()` (query embedded unnormalized).

**Caveat:** The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) was pretrained on raw, unmodified Arabic text. Stripping diacritics and collapsing letter variants (أ/إ/آ→ا, ة→ه, ى→ي) before embedding doesn't "clean" the input from the model's perspective — it shifts it away from the distribution the model actually learned on, which is why it helped the one case it targeted and quietly hurt most others.

## Production pipeline brought in line with validated config

**Measurement:** `retrieve()` in `rag_answer_norm.py` was still fusing the **normalized** Qdrant collection with BM25, with no reranking step — a different, unvalidated configuration from the raw+BM25+rerank pipeline actually measured above.

**Conclusion:** Updated `retrieve()` to embed documents and queries unnormalized (matching the "raw" baseline) and added the reranking stage after RRF fusion, so production now runs the exact configuration validated in the 191-query eval. Verified live: the original near-duplicate bug query ("شو الوثائق المطلوبة لاصدار شهادة ثانوية عامة باللغة الانجليزية؟") now returns the correct certificate service as the top answer, with the exam-transcript near-duplicate correctly demoted to a lower-ranked source instead of winning.

**Caveat:** The refusal/no-answer gate (0.35 cosine-similarity threshold, tested only in the eval harness, not wired into production's code) was calibrated for raw cosine scores. RRF and reranker scores live on different, uncapped scales — so that threshold doesn't transfer and isn't a real calibrated refusal mechanism for the current pipeline. Production currently relies entirely on the LLM's own prompt instruction to refuse when context doesn't answer the question, not on a numeric confidence gate.