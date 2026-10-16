"""
Phase 1 - RAG core (v2, structure-aware chunking).

What changed from v1 and why:

v1 split text every 500 characters. That is wrong for this data. The
government documents have natural boundaries - one service, one list of
required documents - and a character-based split would cut a service in
half or separate its title from its requirements.

So: one chunk = one service. And because dozens of services across
different departments have IDENTICAL bodies ("الهوية الشخصية." and
nothing else), the department name and service title are prepended to
the chunk text before embedding. Otherwise those chunks are
indistinguishable and retrieval cannot possibly pick the right one.

Setup:
    pip install sentence-transformers qdrant-client

    docs/            one .md file per department (see finance.md)

Run:
    python rag_core_v2.py
"""

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

DOCS = Path("docs")
COLLECTION = "gov_docs"

# Multilingual: same meaning in Arabic and English lands in the same place,
# so you can ask in English and match an Arabic document.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# 1. Parsing + chunking - one chunk per service.
# ---------------------------------------------------------------------------

def parse_department(path):
    """
    Read one department file and return a list of service records.

    Expected shape:

        الدائرة: <department name>
        ساعات تقديم الخدمات: <hours>

        ## <service name>
        - requirement
        - requirement

        ## <next service>
        ...
    """
    text = path.read_text(encoding="utf-8")

    header, *service_blocks = text.split("## ")

    department, hours = "", ""
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("الدائرة:"):
            department = line.split(":", 1)[1].strip()
        elif line.startswith("ساعات تقديم الخدمات:"):
            hours = line.split(":", 1)[1].strip()

    records = []
    for block in service_blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        service = lines[0].strip()
        body = "\n".join(l.strip() for l in lines[1:] if l.strip())

        # The text we EMBED includes the department and service name.
        # This is the whole trick: without it, every "الهوية الشخصية."
        # service across 12 departments produces the same vector.
        embed_text = f"{department} - {service}"
        records.append({
            "department": department,
            "service": service,
            "requirements": body,
            "hours": hours,
            "source": path.name,
            "text": embed_text,
        })

    return records


def load_all():
    records = []
    for path in sorted(DOCS.glob("*.md")):
        records.extend(parse_department(path))
    return records


# ---------------------------------------------------------------------------
# 2. Embed + store
# ---------------------------------------------------------------------------

def build_index(records, model, client):
    vectors = model.encode([r["text"] for r in records], show_progress_bar=True)

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=v.tolist(), payload=r)
            for i, (v, r) in enumerate(zip(vectors, records))
        ],
    )
    print(f"Indexed {len(records)} services from {len(set(r['department'] for r in records))} departments.")


# ---------------------------------------------------------------------------
# 3. Retrieve
# ---------------------------------------------------------------------------

def search(question, model, client, top_k=5):
    qvec = model.encode(question).tolist()
    result = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=top_k,
    )
    return result.points


def main():
    if not DOCS.exists() or not any(DOCS.glob("*.md")):
        raise SystemExit("Put department .md files in docs/ first.")

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path="./qdrant_data")   # local, no Docker

    records = load_all()
    build_index(records, model, client)

    print("\nاسأل سؤالك / ask a question (blank line to quit).\n")
    while True:
        question = input("> ").strip()
        if not question:
            break
        for hit in search(question, model, client):
            p = hit.payload
            print(f"\n[{hit.score:.3f}] {p['department']} — {p['service']}")
            print(p["requirements"])
        print()


if __name__ == "__main__":
    main()
