# -*- coding: utf-8 -*-
"""
Streamlit app: users upload documents (PDF/TXT) → chunks → embeddings → Qdrant.
Then chat retrieves top-k chunks from the same Qdrant collection for RAG.

Now includes: **automatic Qdrant Cloud connectivity self-test + REST ping fallback + clearer diagnostics**.
"""

import io
import os
import uuid
import logging
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import PyPDF2
import requests, urllib.parse, socket

# Load .env from the working directory or nearest parent
_dotenv_path = find_dotenv(usecwd=True)
load_dotenv(_dotenv_path, override=False)
logging.basicConfig(level=logging.INFO)

# --- Proxy bypass ---
def _apply_proxy_bypass():
    additions = ["*.qdrant.tech", "localhost", "127.0.0.1", "::1"]
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
    if no_proxy:
        merged = set([s.strip() for s in no_proxy.split(",") if s.strip()]) | set(additions)
        os.environ["NO_PROXY"] = ",".join(sorted(merged))
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
    else:
        os.environ["NO_PROXY"] = ",".join(additions)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        val = os.environ.get(var)
        if val and ("localhost" in val or "127.0.0.1" in val):
            os.environ.pop(var, None)

_apply_proxy_bypass()

# -----------------------------
# Cached resources
# -----------------------------

@st.cache_resource(show_spinner=False)
def _embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# --- REST ping like curl ---
def _ping_qdrant_rest() -> tuple[bool, str]:
    #url = os.getenv("QDRANT_URL", "")
    url = st.secrets["QDRANT_URL"]
    #key = os.getenv("QDRANT_API_KEY", "")
    key = st.secrets["QDRANT_API_KEY"]
    if not url:
        return False, "QDRANT_URL not set."
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        ipv4s = sorted({ai[4][0] for ai in socket.getaddrinfo(host, 443, family=socket.AF_INET)})
        headers = {"api-key": key} if key else {}
        r = requests.get(url.rstrip("/") + "/healthz", headers=headers, timeout=5)
        return (r.status_code == 200, f"REST healthz {r.status_code}; IPv4 {ipv4s}")
    except Exception as e:
        return False, f"REST healthz failed: {e}"


def _mk_qdrant_client_from_url():
    #raw_url = os.getenv("QDRANT_URL")
    raw_url = st.secrets["QDRANT_URL"]
    #api_key = os.getenv("QDRANT_API_KEY")
    api_key = st.secrets["QDRANT_API_KEY"]
    if not raw_url:
        raise RuntimeError("QDRANT_URL is not set.")
    u = urllib.parse.urlparse(raw_url)
    if u.scheme != "https":
        raise RuntimeError("Qdrant Cloud requires HTTPS URL.")
    host = u.hostname
    port = u.port or 443
    try:
        ipv4s = {ai[4][0] for ai in socket.getaddrinfo(host, port, family=socket.AF_INET)}
    except Exception:
        ipv4s = set()
    client = QdrantClient(host=host, port=port, https=True, api_key=api_key, prefer_grpc=False, timeout=10)
    return client, host, port, sorted(ipv4s)


@st.cache_resource(show_spinner=False)
def _qdrant() -> Tuple[QdrantClient, str, int]:
    #coll = os.getenv("QDRANT_COLLECTION", "poslovniModeli")
    coll = st.secrets["QDRANT_COLLECTION"]
    client, host, port, ipv4s = _mk_qdrant_client_from_url()
    try:
        client.get_collections()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Qdrant via qdrant-client. Host={host} Port={port} IPv4={ipv4s}\nError: {e}"
        )
    emb_dim = len(_embedder().encode("dim-probe"))
    try:
        client.get_collection(coll)
    except Exception:
        client.create_collection(
            collection_name=coll,
            vectors_config=qmodels.VectorParams(size=emb_dim, distance=qmodels.Distance.COSINE),
            timeout=30,
        )
    return client, coll, emb_dim


# -----------------------------
# UI / App
# -----------------------------

def main():
    st.set_page_config(page_title="Document Chatbot", page_icon="💬", layout="wide")
    st.title("Document Chatbot (Qdrant)")
    st.caption("Upload PDF/TXT. We embed, store in Qdrant Cloud, and chat over your docs.")

    MODEL_OPTIONS = ["HF Pro Models", "HF Standard Models", "DeepSeek R1 (cloud)"]
    with st.sidebar:
        selected_model = st.selectbox("Select model", MODEL_OPTIONS, index=0)
        st.divider()
        st.subheader("Vector DB")
        #q_url = os.getenv("QDRANT_URL")
        q_url = st.secrets["QDRANT_URL"]
        #q_coll = os.getenv("QDRANT_COLLECTION", "poslovniModeli")
        q_coll = st.secrets["QDRANT_COLLECTION"]

        ok_rest, msg_rest = _ping_qdrant_rest()
        if ok_rest:
            st.success(f"REST ✅ {msg_rest}")
        else:
            st.error(msg_rest)

        try:
            client, coll, emb_dim = _qdrant()
            st.success(f"Qdrant client ✅ collection: {q_coll}")
            ok = True
        except Exception as e:
            st.error(str(e))
            ok = False

        if st.button("Retry connection", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        with st.expander("Diagnostics", expanded=False):
            st.write({
                "QDRANT_URL": q_url,
                "QDRANT_COLLECTION": q_coll,
                "Has Qdrant API key": bool(os.getenv("QDRANT_API_KEY")),
                "Has HF token": bool(os.getenv("HUGGINGFACE_TOKEN")),
                "Has OpenRouter key": bool(os.getenv("OPENROUTER_API_KEY")),
                "HTTP_PROXY": os.getenv("HTTP_PROXY"),
                "HTTPS_PROXY": os.getenv("HTTPS_PROXY"),
                "NO_PROXY": os.getenv("NO_PROXY"),
            })

    if not ok_rest or not ok:
        st.stop()

    # Upload & ingest
    st.subheader("Upload your documents")
    uploaded_files = st.file_uploader("Drop PDFs or TXTs here", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Embedding and uploading to Qdrant..."):
            total_chunks = 0
            for f in uploaded_files:
                text = extract_text_from_upload(f)
                if not text.strip():
                    st.warning(f"No text extracted from {f.name} – skipped.")
                    continue
                chunks = chunk_text(text, size=1000, overlap=200)
                n = upsert_chunks_to_qdrant(chunks, source_name=f.name)
                total_chunks += n
            st.success(f"Finished ingestion. Total chunks upserted: {total_chunks}")

    # Chat
    st.subheader("2) Chat")
    if "history" not in st.session_state:
        st.session_state.history = []

    user_msg = st.chat_input("Pitaj chat što Vas zanima o Alutech-u i njihovim uslugama...")
    if user_msg:
        context = get_qdrant_context(user_msg, top_k=4)
        prompt = build_prompt(user_msg, context)
        reply = get_model_response(prompt, selected_model)
        st.session_state.history.append({"role": "user", "content": user_msg})
        st.session_state.history.append({"role": "assistant", "content": reply})
        with st.sidebar.expander("Last retrieved context", expanded=False):
            st.write(context or "<no context>")

    for m in st.session_state.history[-30:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])


# -----------------------------
# RAG helpers
# -----------------------------

def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            return "".join(page.extract_text() or "" for page in reader.pages)
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"<extract error: {e}>"


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    chunks, i = [], 0
    L = len(text)
    while i < L:
        chunks.append(text[i : i + size])
        i += max(1, size - overlap)
    return [c for c in chunks if c.strip()]


def upsert_chunks_to_qdrant(chunks: List[str], source_name: str) -> int:
    if not chunks:
        return 0
    client, coll, _ = _qdrant()
    embedder = _embedder()
    BATCH = 256
    uploaded = 0
    now_iso = datetime.utcnow().isoformat() + "Z"
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vecs = embedder.encode(batch, show_progress_bar=False).tolist()
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vecs[k],
                payload={"text": batch[k], "source": source_name, "uploaded_at": now_iso},
            )
            for k in range(len(batch))
        ]
        client.upsert(collection_name=coll, points=points, wait=True)
        uploaded += len(points)
    return uploaded


def get_qdrant_context(query: str, top_k: int = 4) -> str:
    client, coll, _ = _qdrant()
    qvec = _embedder().encode(query).tolist()
    hits = client.search(collection_name=coll, query_vector=qvec, limit=top_k)
    if not hits:
        return ""
    hits = sorted(hits, key=lambda h: h.score, reverse=True)
    return "\n\n".join(
        f"[score={h.score:.3f}] { (h.payload or {}).get('text','') }" for h in hits if (h.payload or {}).get("text")
    )


def build_prompt(user_question: str, context: str) -> str:
    return (
        f"Use the following context to answer the user's question:\n\n"
        f"Please use only information from the documents give concise answer that will asnwer most following questions immediatelly\n\n"
        f"Please answer without showing your internal thought process or meta-comments like <think>.\n\n"
        f"Please answer in croatian all the time.\n\n"
        f"Context:\n{context or '<none>'}\n\n"
        f"User question: {user_question}\n"
        "Answer:"
    )


# -----------------------------
# LLM backends
# -----------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from huggingface_hub import InferenceClient

def get_model_response(prompt: str, model_choice: str) -> str:
    # --- Hugging Face Pro Models (using latest Inference Providers) ---
    if model_choice == "HF Pro Models":
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            return "Please set HUGGINGFACE_TOKEN environment variable for HF Pro access."
        
        try:
            # Use the latest Inference Providers system - routed through HF
            client = InferenceClient(token=hf_token)
            
            # Try chat completion with popular models that support it
            chat_models = [
                "HuggingFaceTB/SmolLM3-3B"
                ]
            
            for model_id in chat_models:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat_completion(
                        messages=messages,
                        model=model_id,
                        max_tokens=800,
                        temperature=0.7,
                        stream=False
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    logging.info(f"Chat model {model_id} failed: {e}")
                    continue
            
            # Fallback to text generation models
            text_models = [
                "gpt2-large",
                "EleutherAI/gpt-neo-2.7B",
                "bigscience/bloom-1b7",
            ]
            
            for model_id in text_models:
                try:
                    response = client.text_generation(
                        prompt,
                        model=model_id,
                        max_new_tokens=800,
                        temperature=0.7,
                        do_sample=True,
                        stream=False,
                        return_full_text=False
                    )
                    result = response.strip() if response else ""
                    if result and len(result) > 10:
                        return result
                except Exception as e:
                    logging.info(f"Text model {model_id} failed: {e}")
                    continue
            
            # Final fallback - let HF choose the model
            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=800,
                    temperature=0.7,
                    do_sample=True,
                    stream=False,
                    return_full_text=False
                )
                result = response.strip() if response else ""
                if result:
                    return result
            except Exception as e:
                logging.info(f"Default model failed: {e}")
            
        except Exception as e:
            # Check if it's an old version issue
            if "unexpected keyword argument 'provider'" in str(e):
                return "Please upgrade huggingface_hub: pip install --upgrade huggingface_hub"
            return f"HF Pro error: {e}"
        
        return "All HF models failed. This might be a temporary API issue."

    # --- Hugging Face with Third-Party Providers (Latest feature) ---
    if model_choice == "HF Standard Models":
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

        if not hf_token:
            return "HUGGINGFACE_TOKEN environment variable is not set."

        try:
            # List of third-party providers and respective models
            providers_to_try = [
                ("together", "meta-llama/Llama-3.2-3B-Instruct"),
                ("fireworks", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
                ("replicate", "meta/llama-2-7b-chat"),
            ]

            # Try each provider sequentially
            for provider, model_id in providers_to_try:
                try:
                    client = InferenceClient(provider=provider, token=hf_token)
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat_completion(
                        messages=messages,
                        model=model_id,
                        max_tokens=800,
                        temperature=0.7,
                        stream=False
                    )
                    return response.choices[0].message.content.strip()

                except Exception as e:
                    logging.info(f"Provider {provider} failed: {e}")
                    continue

            # Fallback to basic Hugging Face models without provider param
            client = InferenceClient(token=hf_token)
            basic_models = ["gpt2", "distilgpt2", "gpt2-medium"]

            for model_id in basic_models:
                try:
                    response = client.text_generation(
                        prompt,
                        model=model_id,
                        max_new_tokens=500,
                        temperature=0.7,
                        do_sample=True,
                        stream=False,
                        return_full_text=False
                    )
                    if response:
                        return response.strip()
                except Exception:
                    continue

        except Exception as e:
            if "unexpected keyword argument 'provider'" in str(e):
                return "Please upgrade huggingface_hub: pip install --upgrade huggingface_hub"
            return f"HF Standard error: {e}"

        return "All HF standard models failed."

    # else:
    #     return "Model choice is not HF Standard Models."

    # --- DeepSeek via OpenRouter (backup option) ---
    if model_choice == "DeepSeek R1 (cloud)":
        if not OpenAI:
            return "DeepSeek R1 backend unavailable. Please install openai package."
        
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            return "Please set OPENROUTER_API_KEY environment variable."
            
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"DeepSeek R1 error: {e}"

    # # --- Local/Offline (basic response) ---
    # if model_choice == "Local/Offline":
    #     # Simple pattern-based responses for when no API is available
    #     context_lines = prompt.split('\n')
    #     context_text = ""
    #     for line in context_lines:
    #         if "Context:" in line:
    #             idx = context_lines.index(line)
    #             context_text = '\n'.join(context_lines[idx+1:])
    #             break
        
    #     if "no context" in context_text.lower() or len(context_text.strip()) < 10:
    #         return "I don't have enough context from your uploaded documents to answer this question. Could you please upload some relevant documents first?"
        
    #     # Extract key information from context
    #     lines = [line.strip() for line in context_text.split('\n') if line.strip() and not line.startswith('[score=')]
    #     if lines:
    #         return f"Based on your uploaded documents: {' '.join(lines[:3])}... Please note this is a simplified response. For better answers, configure HF Pro or other AI model APIs."
        
    #     return "No relevant information found in uploaded documents for your question."

    # return "Model not implemented."

if __name__ == "__main__":
    main()