import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "poslovniModeli")
EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

def get_qdrant_context(query: str, top_k: int = 3) -> str:
    url = os.getenv("QDRANT_URL"); key = os.getenv("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY missing.")

    client   = QdrantClient(url=url, api_key=key, timeout=60)
    embedder = SentenceTransformer(EMB_MODEL)
    qvec     = embedder.encode(query).tolist()

    hits = client.search(collection_name=COLLECTION, query_vector=qvec, limit=top_k)
    if not hits:
        return ""

    # Sort by similarity (higher is better with cosine in Qdrant)
    hits = sorted(hits, key=lambda h: h.score, reverse=True)
    return "\n".join((h.payload or {}).get("text", "") for h in hits if (h.payload or {}).get("text"))

if __name__ == "__main__":
    ctx = get_qdrant_context("This is a query document about Alutech info", top_k=3)
    print("Context for ML model:\n", ctx)
