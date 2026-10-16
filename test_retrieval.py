"""
Retrieval-only test. No Gemini, no API - so it works even when Google is down.

Why this exists: normalization only affects RETRIEVAL (which chunks get found).
The LLM just phrases the answer afterwards. So to test whether normalization
helped, we only need to look at what comes back from Qdrant.

Eval set has two parts:

1. TESTS (25 hand-picked cases across 6 categories): easy(5) / near-duplicate(6)
   / paraphrased(5) / english(4) / typo(3) / refusal(2). These target the
   SPECIFIC weaknesses Phase 1 measured (specificity penalty, cross-lingual
   gap, orthographic typos). Printed and scored one row per query.

2. FULL COVERAGE (auto-generated, one question per service - all 168): a
   mechanical sweep, not curated. TESTS answers "are the known-hard cases
   handled"; this answers a different question - "is there any service in
   the whole catalog that can't be found AT ALL by its own natural question".
   Generated from docs/ at run time (stays in sync automatically as services
   are added/removed/renamed - nothing to maintain by hand). Only worth
   reading as a summary + the list of outright misses.

THREE modes are compared, not two:

  raw          - documents embedded as-is, query embedded as-is
  normalized   - documents AND query both run through normalize_arabic()
                 before embedding (this is what rag_answer_norm.py does)
  query_only   - documents embedded as-is (SAME vectors as "raw" - reuses
                 the raw collection), only the QUERY is normalized

Why query_only exists: the first real run of this eval set (193 queries)
found that "normalized" was a NET LOSS overall vs "raw" - Recall@5 89%->75%,
Precision@1 74%->58% - even though it still fixed the typo cases it was
designed for. The working hypothesis is that pushing all 168 already-correct
document titles through ة->ه / ى->ي / hamza-stripping produces non-standard
spellings the pretrained multilingual model wasn't trained on, adding noise
to the WHOLE corpus - not just correcting the handful of genuinely-typo'd
queries. query_only isolates that: it fixes spelling only on the query side
(where typos actually happen) and leaves the document corpus untouched. If
query_only recovers close to "raw"'s overall numbers while still fixing the
typo-category cases, that confirms corpus-side normalization was the problem.
If query_only regresses just as badly, something else is going on.

query_only doesn't need its own Qdrant collection - the documents are
identical to "raw", so it just searches the raw collection with a normalized
query vector. Only two collections get built either way.

Run:
    python test_retrieval.py
"""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchAny,
)
from sentence_transformers import SentenceTransformer

DOCS = Path("docs")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Two collections - "query_only" mode reuses COLL_RAW (same document vectors,
# only the query text differs at search time).
COLL_RAW = "test_raw"
COLL_NORM = "test_norm"

# (mode_label, collection_to_search, normalize_the_query) - the three
# conditions every test case gets run under.
MODES = [
    ("raw", COLL_RAW, False),
    ("normalized", COLL_NORM, True),
    ("query_only", COLL_RAW, True),
]
MODE_LABELS = [m[0] for m in MODES] + ["bm25_only", "hybrid", "reranked"]

# Below this cosine score, Phase 1 found real queries basically never sit
# (off-topic queries scored ~0.1, real ones 0.63-0.76). Used only to score
# the REFUSAL category below, where there's no correct service to rank.
REFUSAL_SCORE_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Normalization (same function as in rag_answer_norm.py)
# ---------------------------------------------------------------------------

_DIACRITICS = re.compile(r"[ً-ْٰـ]")

def normalize_arabic(text):
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = re.sub("[أإآٱ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    return text.strip()


# ---------------------------------------------------------------------------
# Load documents
# ---------------------------------------------------------------------------

def parse_department(path):
    text = path.read_text(encoding="utf-8")
    header, *service_blocks = text.split("## ")

    department, access = "", "public"
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("الدائرة:"):
            department = line.split(":", 1)[1].strip()
        elif line.startswith("التصنيف:") and "داخلي" in line:
            access = "internal"

    records = []
    for block in service_blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        service = lines[0].strip()
        records.append({
            "department": department,
            "service": service,
            "access": access,
            "display": f"{department} - {service}",
        })
    return records


def load_all():
    records = []
    for path in sorted(DOCS.glob("*.md")):
        records.extend(parse_department(path))
    return records


def build(collection, records, embedder, client, normalize):
    texts = [normalize_arabic(r["display"]) if normalize else r["display"]
             for r in records]
    vectors = embedder.encode(texts, show_progress_bar=False)

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=collection,
        points=[PointStruct(id=i, vector=v.tolist(), payload=r)
                for i, (v, r) in enumerate(zip(vectors, records))],
    )


def search(collection, query, role, embedder, client, normalize, top_k=5):
    q = normalize_arabic(query) if normalize else query
    qvec = embedder.encode(q).tolist()
    allowed = ["public"] if role == "citizen" else ["public", "internal"]
    result = client.query_points(
        collection_name=collection,
        query=qvec,
        limit=top_k,
        query_filter=Filter(
            must=[FieldCondition(key="access", match=MatchAny(any=allowed))]
        ),
    )
    return result.points


def search_all_modes(query, role, embedder, client, bm25_stats, records):
    hits = {
        label: search(collection, query, role, embedder, client, normalize=norm_q)
        for label, collection, norm_q in MODES
    }
    bm25_only, hybrid, reranked = search_hybrid(query, role, embedder, client, bm25_stats, records)
    hits["bm25_only"], hits["hybrid"], hits["reranked"] = bm25_only, hybrid, reranked
    return hits


# ---------------------------------------------------------------------------
# The eval set. Each row: (category, query, expected_service_or_None, note)
#
# expected_service is the exact "## " heading text from the source .md, since
# that's what rank_of() below compares against - it's what search() puts in
# hit.payload["service"]. For REFUSAL rows there is no correct service, so
# expected_service is None: those rows are scored on top-1 SCORE instead of
# rank (see the refusal block in main()).
#
# All 25 queried as role="employee" (like the original 5) so internal docs
# are reachable - this file tests RETRIEVAL only, not the RBAC filter itself.
# ---------------------------------------------------------------------------

EASY = "easy"
NEAR_DUP = "near_duplicate"
PARAPHRASE = "paraphrased"
ENGLISH = "english"
TYPO = "typo"
REFUSAL = "refusal"

TESTS = [
    # --- easy / direct (5): near-exact title wording, single unambiguous match
    (EASY, "تصديق شهادة الخبرة", "تصديق شهادة الخبرة",
     "labour.md - straight title match"),
    (EASY, "عدم تقاضي راتب", "عدم تقاضي راتب",
     "finance.md - straight title match"),
    (EASY, "تجديد جواز السفر", "تجديد جواز السفر",
     "civil_status.md - straight title match"),
    (EASY, "عدم المحكومية", "عدم المحكومية",
     "justice_criminal_record.md - straight title match"),
    (EASY, "حسن سيرة وسلوك", "حسن سيرة وسلوك",
     "intelligence.md - straight title match"),

    # --- near-duplicate (6): query must be specific enough to pick ONE
    # member of a near-identical cluster - this is the known weak spot,
    # since embeddings average over words and dilute the distinguishing part
    (NEAR_DUP, "بدي اصدار شهادة الثانوية العامة بس بالانجليزي",
     "اصدار شهادة ثانوية عامة باللغة الانجليزية",
     "education.md - 4-way cluster (شهادة/كشف علامات x عربي/انجليزي)"),
    (NEAR_DUP, "ضيعت رخصة السواقة تبعي شو بدي اعمل؟",
     "اصدار رخصة قيادة بدل فاقد / تالف (فئة ثالثة)",
     "vehicle_licensing.md - رخصة قيادة (person) vs رخصة مركبة (vehicle)"),
    (NEAR_DUP, "بطاقة التأمين الصحي تبعي تخربت شو الوثائق المطلوبة؟",
     "اصدار بطاقة تأمين صحي بدل تالف",
     "health_insurance.md - بدل تالف vs بدل فاقد"),
    (NEAR_DUP, "بدي اتزوج بس هاي مو اول مرة، كنت مطلق قبل",
     "تنظيم عقد الزواج - بناء على طلاق سابق",
     "sharia_judge.md - 3-way marriage cluster"),
    (NEAR_DUP, "شو بدي لتنظيم معرض تجاري؟",
     "طلب إقامة وتنظيم معرض",
     "jedco.md - بازار vs معرض (byte-identical requirements)"),
    (NEAR_DUP, "ضاع مني دفتر العائلة، كيف بقدر استخرج بدل عنه؟",
     "اصدار دفتر عائلة بدل فاقد",
     "civil_status.md - دفتر عائلة: لأول مرة/تجديد/بدل فاقد/بدل تالف"),

    # --- paraphrased (5): real phrasing, not quoting the title
    (PARAPHRASE, "ابني خلص توه، وين بسجله؟",
     "تسجيل واقعة ولادة",
     "civil_status.md"),
    (PARAPHRASE, "بدي اعرف اذا في مخالفات سير على سيارتي والي بحاسبها",
     "الاستعلام / دفع مخالفات السير",
     "amman_municipality.md"),
    (PARAPHRASE, "بدي افتح شركة جديدة، شو الاجراء؟",
     "تسجيل شركة",
     "companies_control.md"),
    (PARAPHRASE, "زوجي توفي، شو لازم اسوي رسمياً؟",
     "تسجيل واقعة وفاة",
     "civil_status.md"),
    (PARAPHRASE, "بدي اعرف كيف اقدم طلب للحصول على مساعدة مالية شهرية من الدولة",
     "تقديم طلب لبرنامج الدعم النقدي الموحد",
     "aid_fund.md"),

    # --- english (4): measured weakest in Phase 1 (cross-lingual gap +
    # vocab collapse, e.g. "permit" maps to both تصريح and رخصة)
    (ENGLISH, "What documents do I need to get a work permit?",
     "طباعة تصاريح العمل",
     "labour.md - the exact case that scored #3 in the Phase 1 baseline"),
    (ENGLISH, "I lost my national ID, what do I need to request a replacement paper?",
     "طلب مشروحات / هوية / دفتر عائلة / جواز سفر بدل فاقد",
     "civil_status.md - bundled service, tests 'ID' -> هوية mapping"),
    (ENGLISH, "What do I need to certify my marriage contract?",
     "تصديق عقد زواج",
     "sharia_judge.md - distinct from the near-dup 'organize a marriage' cluster"),
    (ENGLISH, "What are the requirements to get health insurance for the first time?",
     "اصدار بطاقة تأمين صحي لأول مرة",
     "health_insurance.md"),

    # --- typo / spelling variants (3): tests normalize_arabic() specifically
    (TYPO, "كيف اكشف تزوير الوثايق؟",
     "إجراءات الكشف الداخلي عن تزوير الوثائق",
     "internal_fraud_check.md - hamza variant (ء dropped), ORTHOGRAPHIC: "
     "normalization should fix this (this is the original calibration case)"),
    (TYPO, "اصدار شهاده ثانويه عامه باللغه العربيه",
     "اصدار شهادة ثانوية عامة باللغة العربية",
     "education.md - every ة written as ه, ORTHOGRAPHIC: normalization should fix this"),
    (TYPO, "تجديد دفر العائله",
     "تجديد دفتر العائلة",
     "civil_status.md - GENUINE typo: dropped ت in دفتر (+ ة->ه). "
     "normalize_arabic() only fixes spelling VARIANTS, not missing letters - "
     "so this one is expected to still be hard/miss. Kept in the set on "
     "purpose, as a diagnostic, not a pass requirement."),

    # --- should return nothing (2): refusal behaviour. Scored on score, not rank.
    (REFUSAL, "كيف أطبخ المنسف؟", None,
     "completely off-topic control (same query used in the Phase 1 baseline)"),
    (REFUSAL, "كيف بقدر اجدد رخصتي كسائق باص عمومي (كوستر)؟", None,
     "in-domain sounding but NOT covered - vehicle_licensing.md only has "
     "فئة ثالثة / ركوب صغير / دراجة آلية, no public-transport license"),
]


# ---------------------------------------------------------------------------
# Full coverage: one auto-generated question per service, over ALL records.
#
# Deliberately NOT hand-written (unlike TESTS above) - with 168 services,
# hand-writing one query each would just be typing the title back with a
# question wrapped around it, which teaches nothing a human reviewer couldn't
# already tell by reading the title. What full coverage is actually FOR is
# catching accidents: a service whose title got mangled during parsing, an
# empty/duplicate block, an embedding that comes out degenerate - the kind of
# bug that TESTS (25 hand-picked, human-approved wordings) would never expose
# because a human wouldn't pick a broken title to test with in the first place.
#
# The template keeps the service name (so it's still a fair "can retrieval
# find its own document" check) but wraps it in a real question rather than
# feeding back the literal "department - service" string that's exactly what
# got embedded - that would make every single case a trivial #1 and prove
# nothing.
# ---------------------------------------------------------------------------

FULL_COVERAGE = "full_coverage"
FULL_COVERAGE_TEMPLATE = "شو الوثائق المطلوبة لـ {service}؟"


def generate_full_coverage_tests(records):
    """One (query, expected_service) pair per record, in the FULL_COVERAGE category."""
    return [
        (FULL_COVERAGE, FULL_COVERAGE_TEMPLATE.format(service=r["service"]), r["service"], r["department"])
        for r in records
    ]


def rank_of(expected, hits):
    """Position (1-based) of the expected service in the results, or None."""
    if expected is None:
        return None
    for i, h in enumerate(hits, start=1):
        if h.payload["service"] == expected:
            return i
    return None
from bm25_search import tokenize, build_stats, bm25_search, reciprocal_rank_fusion
from reranker import rerank

CANDIDATE_N = 20

class Hit:
    __slots__ = ("payload", "score")
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score

def search_hybrid(query, role, embedder, client, bm25_stats, records, top_k=5):
    """raw dense + BM25 -> RRF -> rerank. This is the production pipeline."""
    vector_hits = search(COLL_RAW, query, role, embedder, client, normalize=False, top_k=CANDIDATE_N)
    allowed = ["public"] if role == "citizen" else ["public", "internal"]
    bm25_hits = bm25_search(query, bm25_stats, records, top_k=CANDIDATE_N)
    bm25_hits = [(i, s) for i, s in bm25_hits if records[i]["access"] in allowed]

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits)
    candidate_ids = [i for i, _ in fused[:CANDIDATE_N]]
    reranked = rerank(query, candidate_ids, records, top_k=top_k)

    return (
        [Hit(records[i], s) for i, s in bm25_hits[:top_k]],
        [Hit(records[i], s) for i, s in fused[:top_k]],
        [Hit(records[i], s) for i, s in reranked],
    )


def main():
    embedder = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path="./qdrant_test")

    records = load_all()
    bm25_stats = build_stats([tokenize(r["display"]) for r in records])
    print(f"Loaded {len(records)} services.\n")

    print("Building indexes (raw + normalized - query_only reuses raw)...")
    build(COLL_RAW, records, embedder, client, normalize=False)
    build(COLL_NORM, records, embedder, client, normalize=True)
    print("Done.\n")

    scored_rows = []   # (category, query, {mode: rank})
    refusal_rows = []  # (query, {mode: top_score})

    print(f"{'CATEGORY':<14} {'QUERY':<38} {'RAW':<8} {'NORM':<8} {'QUERY-ONLY':<10}")
    print("-" * 80)

    for category, query, expected, note in TESTS:
        hits = search_all_modes(query, "employee", embedder, client, bm25_stats, records)
        short = query[:36] + "..." if len(query) > 36 else query

        if category == REFUSAL:
            scores = {label: (h[0].score if h else 0.0) for label, h in hits.items()}
            refusal_rows.append((query, scores))
            fmt_score = lambda s: f"{s:.3f}"
            print(f"{category:<14} {short:<38} "
                  f"{fmt_score(scores['raw']):<8} {fmt_score(scores['normalized']):<8} "
                  f"{fmt_score(scores['query_only']):<10}")
            continue

        ranks = {label: rank_of(expected, h) for label, h in hits.items()}
        scored_rows.append((category, query, ranks))

        fmt_rank = lambda r: f"#{r}" if r else "MISS"
        print(f"{category:<14} {short:<38} "
              f"{fmt_rank(ranks['raw']):<8} {fmt_rank(ranks['normalized']):<8} "
              f"{fmt_rank(ranks['query_only']):<10}")

    # ---- full coverage: run silently (168 rows would just be noise), keep
    # only the numbers and the outright misses ----
    full_coverage_tests = generate_full_coverage_tests(records)
    full_coverage_rows = []  # (query, department, expected, {mode: rank})
    for i, (category, query, expected, department) in enumerate(full_coverage_tests):
        print(f"  ...{i+1}/{len(full_coverage_tests)}", end="\r")
        hits = search_all_modes(query, "employee", embedder, client, bm25_stats, records)      
        ranks = {label: rank_of(expected, h) for label, h in hits.items()}
        full_coverage_rows.append((query, department, expected, ranks))
        scored_rows.append((FULL_COVERAGE, query, ranks))

    # ---- per-category metrics (Recall@5, Precision@1), scored rows only ----
    print("\n" + "=" * 96)
    print(f"{'CATEGORY':<16} {'N':<4} " + " ".join(f"{m}(R@5/P@1)" for m in MODE_LABELS))
    print("-" * 96)

    by_cat = defaultdict(list)
    for category, query, ranks in scored_rows:
        by_cat[category].append(ranks)

    def recall_at_5(ranks_list):
        return sum(1 for r in ranks_list if r is not None) / len(ranks_list)

    def precision_at_1(ranks_list):
        return sum(1 for r in ranks_list if r == 1) / len(ranks_list)

    def print_metrics_row(label, rows):
       r5 = {m: recall_at_5([row[m] for row in rows]) for m in MODE_LABELS}
       p1 = {m: precision_at_1([row[m] for row in rows]) for m in MODE_LABELS}
       print(f"{label:<16} {len(rows):<4} " + " ".join(f"{m}:{r5[m]:.0%}/{p1[m]:.0%}" for m in MODE_LABELS))

    for category in [EASY, NEAR_DUP, PARAPHRASE, ENGLISH, TYPO, FULL_COVERAGE]:
        rows = by_cat[category]
        if not rows:
            continue
        label = category if category != FULL_COVERAGE else f"{category} (168)"
        print_metrics_row(label, rows)

    print("-" * 96)
    print_metrics_row("OVERALL", [ranks for _, _, ranks in scored_rows])

    # ---- full coverage misses: the actionable output of that sweep ----
    def miss_set(mode):
        return {(q, d, e) for q, d, e, ranks in full_coverage_rows if ranks[mode] is None}

    raw_miss, norm_miss, qonly_miss = miss_set("raw"), miss_set("normalized"), miss_set("query_only")
    all_three_miss = raw_miss & norm_miss & qonly_miss

    print(f"\nFULL COVERAGE misses (service not found in top 5 by its own "
          f"auto-generated question):")
    print(f"  raw={len(raw_miss)}/168, normalized={len(norm_miss)}/168, "
          f"query_only={len(qonly_miss)}/168, missed by ALL THREE={len(all_three_miss)}/168")

    if all_three_miss:
        print("  Missed under every mode (not a normalization question at all - "
              "look at these first, likely a genuine retrieval/parsing issue):")
        for q, d, e in sorted(all_three_miss):
            print(f"    [{d}] {e}")

    # the diagnostic this whole experiment is for: cases normalization "fixed"
    # (missed raw, found normalized) - does query_only ALSO find them, without
    # needing to touch the document corpus?
    norm_fixed = raw_miss - norm_miss
    if norm_fixed:
        print("  Missed raw but found once fully normalized - does query_only "
              "also find these (without touching the docs)?")
        for q, d, e in sorted(norm_fixed):
            got_it = (q, d, e) not in qonly_miss
            print(f"    [{d}] {e}  -> query_only: {'FOUND' if got_it else 'still MISS'}")

    # cases query_only breaks that raw did NOT - if this list is non-empty,
    # even query-side normalization alone has a cost, not just corpus-side
    qonly_broke = qonly_miss - raw_miss
    if qonly_broke:
        print("  Found in raw but MISSED once only the query was normalized "
              "(query-side normalization has a cost too, not just corpus-side):")
        for q, d, e in sorted(qonly_broke):
            print(f"    [{d}] {e}")

    # ---- refusal cases: judged on score vs threshold, not rank ----
    print("\nREFUSAL cases (judged on top-1 score vs "
          f"{REFUSAL_SCORE_THRESHOLD} threshold from the Phase 1 baseline, "
          "not on matching a service):")
    for query, scores in refusal_rows:
        short = query[:50] + "..." if len(query) > 50 else query
        verdict = lambda s: "OK" if s < REFUSAL_SCORE_THRESHOLD else "RISK (above threshold)"
        print(f"  {short}")
        for label in MODE_LABELS:
            print(f"    {label:<10}: {scores[label]:.3f}  {verdict(scores[label])}")

    print("\n(#1 = expected service ranked first. MISS = not in top 5.)")
    print("Compare the three columns: raw is the baseline, normalized touches "
          "both docs and query, query_only touches only the query. If "
          "query_only tracks raw on non-typo categories while still fixing "
          "the typo category, that confirms corpus-side normalization was "
          "the source of the earlier regression.")


if __name__ == "__main__":
    main()
