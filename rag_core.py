"""
Phase 1 - RAG core.

Documents -> chunks -> embeddings -> Qdrant -> retrieval.

No web app, no security, no roles. Just prove retrieval works.

Setup:
    pip install sentence-transformers qdrant-client

    mkdir docs
    # put a few .txt files in docs/  (bank services, fees, account opening...)

Run:
    python rag_core.py
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
# 1. Chunking - you write this yourself, it's the point.
# ---------------------------------------------------------------------------

def chunk_text(text, size=500, overlap=50):
    """
    Split text into overlapping pieces.

    Overlap matters: without it, a sentence split across two chunks
    loses its meaning in both.
    """
    text = " ".join(text.split())          # normalise whitespace
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap              # step back to create the overlap
    return chunks


def load_chunks():
    """Read every .txt in docs/ and chunk it."""
    records = []
    for path in sorted(DOCS.glob("*.txt")):
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            records.append({
                "text": chunk,
                "source": path.name,
                "chunk_index": i,
            })
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
    print(f"Indexed {len(records)} chunks.")


# ---------------------------------------------------------------------------
# 3. Retrieve
# ---------------------------------------------------------------------------

def search(question, model, client, top_k=3):
    qvec = model.encode(question).tolist()
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=top_k,
    )
    return hits


def main():
    if not DOCS.exists() or not any(DOCS.glob("*.txt")):
        raise SystemExit("Put some .txt files in docs/ first.")

    model = SentenceTransformer(MODEL_NAME)
    # Local storage - no Docker needed for Phase 1.
    client = QdrantClient(path="./qdrant_data")

    records = load_chunks()
    build_index(records, model, client)

    print("\nAsk questions (blank line to quit).\n")
    while True:
        question = input("> ").strip()
        if not question:
            break
        for hit in search(question, model, client):
            src = hit.payload["source"]
            print(f"\n[{hit.score:.3f}] {src}")
            print(hit.payload["text"][:300], "...")
        print()


if __name__ == "__main__":
    main()
