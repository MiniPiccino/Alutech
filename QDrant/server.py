import os, glob
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
import PyPDF2
from dotenv import load_dotenv
load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "poslovniModeli")
DATA_DIR   = os.getenv("DATA_DIR", "Dokumenti")  # your folder with books

EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast

def read_txt(path: str) -> str:
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def read_pdf(path: str) -> str:
    r = PyPDF2.PdfReader(path)
    return "".join(page.extract_text() or "" for page in r.pages)

def chunk(text: str, size=500, overlap=150) -> List[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += size - overlap
    return out

def main():
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY missing. Put them in env or .streamlit/secrets.toml")

    client   = QdrantClient(url=url, api_key=key, timeout=60)
    embedder = SentenceTransformer(EMB_MODEL)
    emb_dim  = len(embedder.encode("dim-probe"))

    # Create collection if needed
    try:
        client.get_collection(COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=emb_dim,
                distance=qmodels.Distance.COSINE
            ),
        )

    # Gather documents
    docs, ids, payloads = [], [], []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        text = read_txt(p)
        for j, ch in enumerate(chunk(text)):
            docs.append(ch)
            ids.append(f"{os.path.basename(p)}:chunk:{j}")
            payloads.append({"source": os.path.basename(p)})

    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf"))):
        text = read_pdf(p)
        for j, ch in enumerate(chunk(text)):
            docs.append(ch)
            ids.append(f"{os.path.basename(p)}:chunk:{j}")
            payloads.append({"source": os.path.basename(p)})

    if not docs:
        print(f"No files found under {DATA_DIR}/. Add .txt or .pdf files and rerun.")
        return

    # Upsert in batches
    B = 512
    for i in range(0, len(docs), B):
        batch = docs[i:i+B]
        vecs  = embedder.encode(batch, show_progress_bar=False).tolist()
        points = [
            qmodels.PointStruct(
                id=ids[i+k],
                vector=vecs[k],
                payload={"text": batch[k], **payloads[i+k]},
            )
            for k in range(len(batch))
        ]
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        print(f"Upserted {min(i+B, len(docs))}/{len(docs)}")

    count = client.count(collection_name=COLLECTION, exact=True).count
    print(f"Done. {count} chunks in collection '{COLLECTION}'.")

if __name__ == "__main__":
    main()
