# -*- coding: utf-8 -*-
"""
Streamlit app: users upload documents (PDF/TXT) → chunks → embeddings → Qdrant.
Then chat retrieves top-k chunks from the same Qdrant collection for RAG.

Now includes: **automatic Qdrant Cloud connectivity self-test + REST ping fallback + clearer diagnostics**.
Also includes: **token budgeting** for prompts so the model always has room to "read" and answer.
"""

import io
import os
import uuid
import logging
from datetime import datetime
from typing import List, Tuple
import re

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
    url = st.secrets["QDRANT_URL"]
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

def clean_llm_output(text: str) -> str:
    """
    Uklanja sve <think> tagove i druge poznate meta-oznake iz LLM odgovora.
    """
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

def _mk_qdrant_client_from_url():
    raw_url = st.secrets["QDRANT_URL"]
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
        q_url = st.secrets["QDRANT_URL"]
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
                "Has Qdrant API key": bool(st.secrets.get("QDRANT_API_KEY")),
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
        contexts = get_qdrant_context(user_msg, top_k=6)  # možeš povisiti K; prompt builder reže višak
        prompt = build_prompt_with_budget(user_msg, contexts, selected_model, reply_tokens=800)
        reply = get_model_response(prompt, selected_model)
        st.session_state.history.append({"role": "user", "content": user_msg})
        st.session_state.history.append({"role": "assistant", "content": reply})
        with st.sidebar.expander("Last retrieved context", expanded=False):
            if contexts:
                st.write([
                    {"score": f"{s:.3f}", "snippet": t[:240] + ("..." if len(t) > 240 else "")}
                    for s, t in contexts
                ])
            else:
                st.write("<no context>")

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

def get_qdrant_context(query: str, top_k: int = 4) -> List[Tuple[float, str]]:
    client, coll, _ = _qdrant()
    qvec = _embedder().encode(query).tolist()
    hits = client.search(collection_name=coll, query_vector=qvec, limit=top_k)
    if not hits:
        return []
    hits = sorted(hits, key=lambda h: h.score, reverse=True)
    out: List[Tuple[float, str]] = []
    for h in hits:
        txt = (h.payload or {}).get("text", "")
        if txt and txt.strip():
            out.append((h.score, txt))
    return out

# -----------------------------
# Prompt building with token budgeting
# -----------------------------

def estimate_tokens(text: str) -> int:
    """Gruba procjena tokena, 4 znaka ~ 1 token (radi za većinu LLM-ova)."""
    if not text:
        return 0
    return max(1, len(text) // 4)

def build_prompt_with_budget(user_question: str,
                             contexts: List[Tuple[float, str]],
                             model_choice: str,
                             reply_tokens: int = 800) -> str:
    """
    Složi prompt i kontekst tako da stane u context window modela + ostavi mjesta za odgovor.
    """
    # Procjena context window-a po backendu (konzervativno)
    ctx_windows = {
        "HF Pro Models": 8192,
        "HF Standard Models": 8192,
        "DeepSeek R1 (cloud)": 16384,
    }
    max_ctx = ctx_windows.get(model_choice, 8192)

    header = (
        "Tvoj zadatak je odgovoriti na korisničko pitanje koristeći ISKLJUČIVO informacije iz sljedećeg konteksta.\n\n"
        "🎯 CILJ:\n"
        "Daj jasan, sažet i točan odgovor koji pokriva bit pitanja tako da korisnik odmah dobije najveću moguću vrijednost.\n\n"
        "📜 PRAVILA:\n"
        "- Koristi isključivo informacije iz danog konteksta (nema izmišljanja).\n"
        "- Ako kontekst ne sadrži odgovor, reci to izravno (\"Nema dovoljno informacija u dostupnim dokumentima.\").\n"
        "- Odgovaraj isključivo na hrvatskom jeziku, gramatički i stilski prirodno.\n"
        "- Izbjegavaj meta-komentare, razmišljanja, oznake poput <think> i slično.\n"
        "- Odgovor formuliraj kao da si stručnjak koji zna objasniti jasno i profesionalno, ali bez suvišne formalnosti.\n"
        "- Poželjno je da prvi redak odmah sadrži sažeti odgovor, a ako je potrebno, ispod možeš dodati kratko pojašnjenje.\n\n"
        "📚 KONTEKST:\n"
    )
    footer = f"\n\n❓ PITANJE KORISNIKA:\n{user_question}\n\n💬 ODGOVOR:\n"

    # Rezerviraj tokene za header, footer i odgovor
    header_tokens = estimate_tokens(header)
    footer_tokens = estimate_tokens(footer)
    budget = max_ctx - reply_tokens - header_tokens - footer_tokens
    if budget < 256:
        budget = 256  # safety net

    # Spakiraj kontekst dok ima budžeta
    context_blob = []
    used = 0
    for score, text in contexts:
        if not text:
            continue
        line = f"[score={score:.3f}] {text}\n\n"
        t = estimate_tokens(line)
        if used + t > budget:
            remaining_tokens = max(0, budget - used)
            remaining_chars = remaining_tokens * 4
            if remaining_chars > 64:  # izbjegni trivijalne repove
                line = line[:remaining_chars]
                context_blob.append(line)
                used += estimate_tokens(line)
            break
        context_blob.append(line)
        used += t

    context_text = "".join(context_blob) if context_blob else "<nema dostupnog konteksta>"
    prompt = f"{header}{context_text}{footer}"
    return prompt


# -----------------------------
# LLM backends
# -----------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from huggingface_hub import InferenceClient

def get_model_response(prompt: str, model_choice: str) -> str:
    # --- Hugging Face Pro Models ---
    if model_choice == "HF Pro Models":
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            return "Please set HUGGINGFACE_TOKEN environment variable for HF Pro access."
        try:
            client = InferenceClient(token=hf_token)
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
                    raw_text = response.choices[0].message.content
                    return clean_llm_output(raw_text)
                except Exception as e:
                    logging.info(f"Chat model {model_id} failed: {e}")
                    continue

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
                        return clean_llm_output(result)
                except Exception as e:
                    logging.info(f"Text model {model_id} failed: {e}")
                    continue

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
                    return clean_llm_output(result)
            except Exception as e:
                logging.info(f"Default model failed: {e}")

        except Exception as e:
            if "unexpected keyword argument 'provider'" in str(e):
                return "Please upgrade huggingface_hub: pip install --upgrade huggingface_hub"
            return f"HF Pro error: {e}"

        return "All HF models failed. This might be a temporary API issue."

    # --- Hugging Face with Third-Party Providers ---
    if model_choice == "HF Standard Models":
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            return "HUGGINGFACE_TOKEN environment variable is not set."
        try:
            providers_to_try = [
                ("together", "meta-llama/Llama-3.2-3B-Instruct"),
                ("fireworks", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
                ("replicate", "meta/llama-2-7b-chat"),
            ]
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
                    raw_text = response.choices[0].message.content
                    return clean_llm_output(raw_text)

                except Exception as e:
                    logging.info(f"Provider {provider} failed: {e}")
                    continue

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

    # --- DeepSeek via OpenRouter ---
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
            raw_text = completion.choices[0].message.content
            clean_text = clean_llm_output(raw_text)
            return clean_text
        except Exception as e:
            return f"DeepSeek R1 error: {e}"

if __name__ == "__main__":
    main()
