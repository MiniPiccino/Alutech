# -*- coding: utf-8 -*-
"""
Streamlit app: users upload documents (PDF/TXT) → chunks → embeddings → Qdrant.
RAG preko Qdranta (semantic + full-text + keyword fallback) s token budgetingom.

Uključuje:
- Qdrant Cloud self-test + REST ping
- Automatsko kreiranje kolekcije i FULL-TEXT indeksa nad payload.text
- Dohvat konteksta: semantic → full-text → keyword scroll (reranking po sličnosti)
- Token budgeting (prompt uvijek ostavlja mjesta za odgovor)
- Dijagnostika: broj točaka u kolekciji
"""

import io
import os
import uuid
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import re

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import PyPDF2
import requests, urllib.parse, socket

# Load .env
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
    # create collection if missing
    try:
        client.get_collection(coll)
    except Exception:
        client.create_collection(
            collection_name=coll,
            vectors_config=qmodels.VectorParams(size=emb_dim, distance=qmodels.Distance.COSINE),
            timeout=30,
        )
    # ensure full-text index
    ensure_fulltext_index(client, coll, field_name="text")
    return client, coll, emb_dim

def ensure_fulltext_index(client: QdrantClient, coll: str, field_name: str = "text"):
    """
    Pokuša kreirati/namjestiti FULL-TEXT indeks nad payload poljem 'text'.
    Radi i ako je već postojao (ignorira grešku).
    """
    try:
        # Qdrant full-text (word tokenizer, lowercase)
        client.create_payload_index(
            collection_name=coll,
            field_name=field_name,
            field_schema=qmodels.TextIndexParams(
                type="text",                   # ekvivalent PayloadSchemaType.TEXT
                tokenizer=qmodels.TokenizerType.WORD,
                min_token_len=2,
                lowercase=True,
                stopwords=[],
                use_diacritics=False,
            ),
        )
    except Exception as e:
        # Ako već postoji ili verzija ne podržava detaljne parametre, probaj minimalnu definiciju
        try:
            client.create_payload_index(
                collection_name=coll,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.TEXT
            )
        except Exception:
            # Najvjerojatnije već postoji — to je ok.
            pass

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

        # count
        try:
            c = qdrant_count()
            st.info(f"Broj vektora u kolekciji **{q_coll}**: {c}")
        except Exception as e:
            st.warning(f"Ne mogu dohvatiti count: {e}")

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
        contexts = get_context_smart(user_msg, top_k=10, min_vec_score=0.25)  # semantic → full-text → keyword
        if not contexts:
            reply = compose_noinfo_reply(user_msg)
        else:
            prompt = build_prompt_with_budget(user_msg, contexts, selected_model, reply_tokens=800)
            reply = get_model_response(prompt, selected_model)
            reply = clean_llm_output(reply)

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
    # ensure index (in slučaju prvog ingest-a)
    ensure_fulltext_index(client, coll, field_name="text")
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

def qdrant_count() -> int:
    client, coll, _ = _qdrant()
    try:
        res = client.count(coll, count_filter=None, exact=True)
        return int(res.count) if hasattr(res, "count") else int(res)
    except Exception:
        return -1

# -------- Context retrieval (semantic + full-text + keyword fallback) --------
def get_qdrant_context_vector(query: str, top_k: int = 10) -> List[Tuple[float, str]]:
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

def _split_keywords(q: str) -> List[str]:
    q = q.lower()
    q = re.sub(r"[^a-ž0-9\s]", " ", q, flags=re.IGNORECASE)
    toks = [t for t in q.split() if len(t) >= 3]
    stop = {"što","sto","kako","koliko","koji","koja","kada","gdje","jeli","je","su","sam","smo","ste","u","na","za","od","do","i","ili","te"}
    return [t for t in toks if t not in stop]

def search_keyword_in_qdrant(keywords: List[str], max_points: int = 200) -> List[str]:
    if not keywords:
        return []
    client, coll, _ = _qdrant()
    out: List[str] = []
    next_offset = None
    fetched = 0
    while fetched < max_points:
        batch_limit = min(64, max_points - fetched)
        res, next_offset = client.scroll(
            collection_name=coll,
            limit=batch_limit,
            with_payload=True,
            with_vectors=False,
            offset=next_offset
        )
        if not res:
            break
        for pt in res:
            txt = ((pt.payload or {}).get("text") or "").strip()
            if not txt:
                continue
            low = txt.lower()
            if any(k in low for k in keywords):
                out.append(txt)
        fetched += len(res)
        if next_offset is None:
            break
    uniq = []
    seen = set()
    for t in out:
        key = t[:200]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
        if len(uniq) >= 12:
            break
    return uniq

def get_qdrant_context_fulltext(query: str, limit: int = 24) -> List[str]:
    """
    Full-text pretraga preko Qdrantovog 'MatchText' nad payload.text.
    Vraća listu payload tekstova (bez score-a), kasnije ćemo ih re-rankati semantički.
    """
    client, coll, _ = _qdrant()
    try:
        filt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="text",
                    match=qmodels.MatchText(text=query)
                )
            ]
        )
        # koristimo scroll da pokupimo više pogodaka; može i .search s filterom + vector
        out: List[str] = []
        next_offset = None
        fetched = 0
        while fetched < limit:
            batch = min(64, limit - fetched)
            res, next_offset = client.scroll(
                collection_name=coll,
                limit=batch,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
                scroll_filter=filt
            )
            if not res:
                break
            for pt in res:
                txt = ((pt.payload or {}).get("text") or "").strip()
                if txt:
                    out.append(txt)
            fetched += len(res or [])
            if next_offset is None:
                break
        return out[:limit]
    except Exception as e:
        logging.info(f"Full-text search failed: {e}")
        return []

def rerank_by_similarity(query: str, texts: List[str], top_k: int) -> List[Tuple[float, str]]:
    if not texts:
        return []
    em = _embedder()
    qv = em.encode(query).tolist()
    scored: List[Tuple[float, str]] = []
    import math
    for txt in texts:
        sv = em.encode(txt).tolist()
        dot = sum(a*b for a,b in zip(qv, sv))
        nq = math.sqrt(sum(a*a for a in qv)) or 1.0
        ns = math.sqrt(sum(a*a for a in sv)) or 1.0
        sim = dot/(nq*ns)
        scored.append((float(sim), txt))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def get_context_smart(query: str, top_k: int = 10, min_vec_score: float = 0.25) -> List[Tuple[float, str]]:
    """
    1) Semantic vector search
    2) If weak → FULL-TEXT MatchText nad payload.text + semantički reranking
    3) Ako i dalje prazno → keyword scroll fallback + reranking
    """
    vec = get_qdrant_context_vector(query, top_k=top_k)
    if vec and (vec[0][0] >= min_vec_score or len(vec) >= max(4, top_k//2)):
        return vec

    # Full-text
    ft_hits = get_qdrant_context_fulltext(query, limit=32)
    if ft_hits:
        return rerank_by_similarity(query, ft_hits, top_k=top_k)

    # Keyword scroll fallback
    kws = _split_keywords(query)
    kw_hits = search_keyword_in_qdrant(kws, max_points=400)
    if kw_hits:
        return rerank_by_similarity(query, kw_hits, top_k=top_k)

    return []

# -----------------------------
# Prompt building with token budgeting
# -----------------------------
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

def build_prompt_with_budget(user_question: str,
                             contexts: List[Tuple[float, str]],
                             model_choice: str,
                             reply_tokens: int = 800) -> str:
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

    header_tokens = estimate_tokens(header)
    footer_tokens = estimate_tokens(footer)
    budget = max_ctx - reply_tokens - header_tokens - footer_tokens
    if budget < 256:
        budget = 256

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
            if remaining_chars > 64:
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
# "No info" reply composer
# -----------------------------
def compose_noinfo_reply(user_question: str) -> str:
    kws = _split_keywords(user_question)
    tips = [
        "Učitajte dokumente koji sadrže traženi pojam, definiciju ili brojke.",
        "Provjerite pravopis i moguće varijante pojma (množina/jednina, padeži, sinonimi).",
        "Ako tražite broj (npr. 'koliko poduzeća...'), osigurajte izvješće/tablicu s tim podatkom.",
        "Ako PDF nema teksta (scan/slika), optimizirajte OCR ili učitajte TXT/PDF s izvornim tekstom.",
    ]
    msg = "Nema dovoljno informacija u dostupnim dokumentima za traženo pitanje.\n\n"
    if kws:
        msg += f"• Pretraživani ključni pojmovi: {', '.join(kws)}\n"
    msg += "• Razlozi: pojam se ne pojavljuje u indeksu ili je sadržaj nedostupan kao tekst.\n"
    msg += "• Što možete napraviti:\n  - " + "\n  - ".join(tips)
    return msg

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
            chat_models = ["HuggingFaceTB/SmolLM3-3B"]
            for model_id in chat_models:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat_completion(
                        messages=messages, model=model_id,
                        max_tokens=800, temperature=0.7, stream=False
                    )
                    raw_text = response.choices[0].message.content
                    return clean_llm_output(raw_text)
                except Exception as e:
                    logging.info(f"Chat model {model_id} failed: {e}")
                    continue
            text_models = ["gpt2-large","EleutherAI/gpt-neo-2.7B","bigscience/bloom-1b7"]
            for model_id in text_models:
                try:
                    response = client.text_generation(
                        prompt, model=model_id, max_new_tokens=800,
                        temperature=0.7, do_sample=True, stream=False, return_full_text=False
                    )
                    result = response.strip() if response else ""
                    if result and len(result) > 10:
                        return clean_llm_output(result)
                except Exception as e:
                    logging.info(f"Text model {model_id} failed: {e}")
                    continue
            try:
                response = client.text_generation(
                    prompt, max_new_tokens=800, temperature=0.7,
                    do_sample=True, stream=False, return_full_text=False
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
                        messages=messages, model=model_id,
                        max_tokens=800, temperature=0.7, stream=False
                    )
                    raw_text = response.choices[0].message.content
                    return clean_llm_output(raw_text)
                except Exception as e:
                    logging.info(f"Provider {provider} failed: {e}")
                    continue
            client = InferenceClient(token=hf_token)
            basic_models = ["gpt2","distilgpt2","gpt2-medium"]
            for model_id in basic_models:
                try:
                    response = client.text_generation(
                        prompt, model=model_id, max_new_tokens=500,
                        temperature=0.7, do_sample=True, stream=False, return_full_text=False
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
