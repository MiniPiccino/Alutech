# -*- coding: utf-8 -*-
"""
Streamlit app: users upload documents (PDF/TXT) → chunks → embeddings → Qdrant.
RAG preko Qdranta (semantic + full-text + keyword fallback) s token budgetingom.

Major updates (HR-first & robust retrieval):
- Multilingual E5 embeddings: intfloat/multilingual-e5-base (bolji HR↔EN semantički match).
- E5 prefixes: 'query:' i 'passage:' za konzistentne vektore.
- Dvojezična i sinonimska ekspanzija upita (direktor/ravnatelj/predsjednik + EN prijevod).
- Ne prevodimo PROMPT (kontekst ostaje točan); eventualno prevodimo SAMO ODGOVOR natrag u HR.
- Heuristike za kvalitetu HR odgovora.
- Ispravke: total_chunks akumulacija, dijagnostika, i sitni cleanup.

Dependencies:
pip install streamlit python-dotenv sentence-transformers qdrant-client PyPDF2 huggingface_hub transformers langdetect requests

Environment/Secrets (Streamlit):
- st.secrets["QDRANT_URL"] (https://<cluster>.<region>.qdrant.tech:443)
- st.secrets["QDRANT_API_KEY"]
- st.secrets["QDRANT_COLLECTION"]
- st.secrets["OPENAI_API_KEY"]
- env HUGGINGFACE_TOKEN (for HF Inference)
- env OPENROUTER_API_KEY (optional, for DeepSeek route)
"""

import io
import os
import uuid
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import re
import hashlib
import math
import time
import socket
import urllib.parse
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv, find_dotenv
import requests
import PyPDF2

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Language detection & translation
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
from transformers import pipeline

# Optional OpenRouter
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from huggingface_hub import InferenceClient

# -----------------------------
# Init
# -----------------------------
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

BASE_DIR = Path(__file__).resolve().parent
COMPANY_MEDIA_DIR = BASE_DIR / "company_media"
COMPANY_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
COMPANY_LOGO_PATH = COMPANY_MEDIA_DIR / "image.png"  # prilagodite naziv datoteke po potrebi
OPENAI_PRIMARY_CHAT_MODEL = "gpt-4o-mini"
OPENAI_PRIMARY_LABEL = "OpenAI GPT-4o-mini (primary)"
PRODUCTION_MODEL_CHOICE = OPENAI_PRIMARY_LABEL  # uredite po potrebi
PRODUCTION_PAGE_TITLE = "Alutech Chatbot (Production)"


def _get_config_value(key: str, required: bool = False) -> str | None:
    value = None
    try:
        value = st.secrets.get(key)
    except Exception:
        # st.secrets might not be initialised yet in bare execution
        pass
    if value is None:
        value = os.getenv(key)
    if required and not value:
        raise RuntimeError(f"Configuration value '{key}' is not set. Define it in Streamlit secrets or environment.")
    return value

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def _embedder() -> SentenceTransformer:
    # Multilingual, strong cross-lingual retrieval
    return SentenceTransformer("intfloat/multilingual-e5-base")

def _add_e5_prefix(s: str, kind: str) -> str:
    return f"{kind}: {s.strip()}" if s else f"{kind}:"

@st.cache_resource(show_spinner=False)
def _translator_to_en():
    try:
        return pipeline("translation", model="Helsinki-NLP/opus-mt-hr-en", device_map="auto")
    except Exception as exc:
        logging.warning("Translator HR->EN unavailable: %s", exc)
        return None

@st.cache_resource(show_spinner=False)
def _translator_to_hr():
    try:
        return pipeline("translation", model="Helsinki-NLP/opus-mt-en-hr", device_map="auto")
    except Exception as exc:
        logging.warning("Translator EN->HR unavailable: %s", exc)
        return None

def _encode_query(text: str):
    return _embedder().encode(_add_e5_prefix(text, "query"), show_progress_bar=False)

def _encode_passages(texts: List[str]):
    return _embedder().encode([_add_e5_prefix(t, "passage") for t in texts], show_progress_bar=False)

def _encode_passage(text: str):
    return _embedder().encode(_add_e5_prefix(text, "passage"), show_progress_bar=False)

# --- REST ping like curl ---
def _ping_qdrant_rest() -> tuple[bool, str]:
    url = _get_config_value("QDRANT_URL")
    key = _get_config_value("QDRANT_API_KEY")
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
    s = text
    s = re.sub(r"(?is)<\s*(think|thinking|analysis|reasoning)[^>]*>.*?<\s*/\s*\1\s*>", "", s)
    s = re.sub(r"(?is)```(?:thinking|think|analysis|reasoning|xml).*?```", "", s)
    s = re.sub(r"(?im)^\s*(Thought|Thinking|Analysis|Reasoning)\s*:\s.*?$", "", s)
    s = re.sub(r"(?is)<\s*reflection[^>]*>.*?<\s*/\s*reflection\s*>", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return _normalize_to_hr(s)

# -----------------------------
# Qdrant client & collection
# -----------------------------
def _mk_qdrant_client_from_url():
    raw_url = _get_config_value("QDRANT_URL", required=True)
    api_key = _get_config_value("QDRANT_API_KEY")
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
    coll = _get_config_value("QDRANT_COLLECTION", required=True)
    client, host, port, ipv4s = _mk_qdrant_client_from_url()
    try:
        client.get_collections()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Qdrant via qdrant-client. Host={host} Port={port} IPv4={ipv4s}\nError: {e}"
        )
    emb_dim = len(_encode_query("dim-probe"))
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
    try:
        client.create_payload_index(
            collection_name=coll,
            field_name=field_name,
            field_schema=qmodels.TextIndexParams(
                type="text",
                tokenizer=qmodels.TokenizerType.WORD,
                min_token_len=2,
                lowercase=True,
                stopwords=[],
                use_diacritics=False,
            ),
        )
    except Exception:
        try:
            client.create_payload_index(
                collection_name=coll,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.TEXT
            )
        except Exception:
            pass

# -----------------------------
# UI / App
# -----------------------------
def main():
    st.set_page_config(page_title=PRODUCTION_PAGE_TITLE, page_icon='📄', layout='wide')
    st.title('Alutech Chatbot')

    ok_rest = False
    ok_client = False

    with st.sidebar:
        if COMPANY_LOGO_PATH.exists():
            st.image(str(COMPANY_LOGO_PATH), use_column_width=True)
        else:
            st.markdown('**Alutech**')

        st.markdown('---')

        ok_rest, msg_rest = _ping_qdrant_rest()
        if ok_rest:
            st.success('Veza s Qdrantom je aktivna.')
        else:
            st.error(msg_rest)

        try:
            _qdrant()
            ok_client = True
        except Exception as e:
            st.error(f'Povezivanje na Qdrant klijent nije uspjelo: {e}')
            logging.error('Qdrant client init failed: %s', e)

    if not ok_rest or not ok_client:
        st.stop()

    model_choice = PRODUCTION_MODEL_CHOICE

    st.subheader('Chat')
    if 'history' not in st.session_state:
        st.session_state.history = []

    user_msg = st.chat_input('Postavite pitanje o učitanim dokumentima…')
    if user_msg:
        # --- NORMALIZACIJA + HISTORY ---
        if 'last_company' not in st.session_state:
            st.session_state.last_company = None
        normalized_q, current_company = normalize_question(user_msg, st.session_state.last_company)
        st.session_state.last_company = current_company

        user_lang_guess = safe_detect(user_msg)

        # --- RETRIEVAL NA NORMALIZIRANO PITANJE ---
        contexts = get_context_smart(normalized_q, top_k=10, min_vec_score=0.15)

        if not contexts:
            reply = compose_noinfo_reply(normalized_q)
        else:
            # --- PROMPT NA NORMALIZIRANO PITANJE ---
            prompt = build_prompt_with_budget(normalized_q, contexts, model_choice, reply_tokens=800)
            primary_raw = get_model_response(prompt, model_choice, user_lang_guess=user_lang_guess)
            reply = clean_llm_output(primary_raw)


            if is_backend_failure_message(reply):
                logging.warning("Primary model %s failed: %s", model_choice, reply)
                failure_notes = [f"{model_choice}: {reply}"]
                fallback_applied = False
                fallback_candidates: List[str] = []

                if model_choice == OPENAI_PRIMARY_LABEL:
                    fallback_candidates.extend([
                        "HF Pro Models (HR-first)",
                        "HF Standard Models (router)",
                        "DeepSeek R1 (cloud)",
                    ])
                elif model_choice == "HF Pro Models (HR-first)":
                    fallback_candidates.extend([
                        "HF Standard Models (router)",
                        "DeepSeek R1 (cloud)",
                    ])
                elif model_choice == "HF Standard Models (router)":
                    fallback_candidates.append("DeepSeek R1 (cloud)")

                for alt_choice in fallback_candidates:
                    alt_raw = get_model_response(prompt, alt_choice, user_lang_guess=user_lang_guess)
                    alt_reply = clean_llm_output(alt_raw)
                    if not is_backend_failure_message(alt_reply):
                        reply = alt_reply
                        fallback_applied = True
                        logging.info("Fallback to %s succeeded", alt_choice)
                        break
                    failure_notes.append(f"{alt_choice}: {alt_reply}")
                    logging.warning("Fallback model %s failed: %s", alt_choice, alt_reply)

                if not fallback_applied:
                    failure_summary = summarize_backend_failure("; ".join(failure_notes))
                    reply = compose_noinfo_reply(user_msg)
                    reply += (
                        "\n\nUpozorenje: LLM backend trenutno nije dostupan "
                        f"({failure_summary}). Pokusajte ponovno kasnije ili provjerite API postavke."
                    )

        st.session_state.history.append({'role': 'user', 'content': user_msg})
        st.session_state.history.append({'role': 'assistant', 'content': reply})

    for m in st.session_state.history[-30:]:
        with st.chat_message(m['role']):
            st.write(m['content'])

# -----------------------------
# RAG helpers
# -----------------------------
def extract_text_from_upload(uploaded_file) -> List[dict]:
    """
    Vrati listu: [{"page": 1, "text": "..."} , ...]
    Ako je TXT, tretiraj sve kao jednu stranicu.
    """
    name = uploaded_file.name.lower()
    records = []
    try:
        if name.endswith(".pdf"):
            data = uploaded_file.read()
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                records.append({"page": i+1, "text": txt})
        else:
            txt = uploaded_file.read().decode("utf-8", errors="ignore")
            records.append({"page": 1, "text": txt})
    except Exception as e:
        records.append({"page": 1, "text": f"<extract error: {e}>"})
    return records

def _split_sentences(text: str) -> List[str]:
    # jednostavan: razdvajanje po . ! ? ; + dvostruki novi red
    text = re.sub(r"[ \t]+", " ", text)
    parts = re.split(r"(?<=[\.\!\?\;])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]

def chunk_text_smart(text: str, size: int = 1200, overlap: int = 300) -> List[str]:
    """
    Pametno dijeljenje teksta u veće chunkove.
    Povećani size i overlap da se zadrži cijeli opis tvrtke u istom vektoru.
    """
    if not text or not text.strip():
        return []
    sents = _split_sentences(text)
    chunks, cur = [], ""
    target = max(400, size)
    for s in sents:
        if len(cur) + len(s) + 1 <= target:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                chunks.append(cur)
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                cur = (tail + " " + s).strip()
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def make_chunks_from_pages(pages: List[dict], size: int = 1200, overlap: int = 300) -> List[dict]:
    out = []
    for rec in pages:
        page = rec["page"]
        txt = rec.get("text", "") or ""
        # --- FIX: eksplicitni kontekst firme ---
        txt = re.sub(r"(?m)^(\d+\.\s*)([A-ZČĆŽŠĐ].+)$", r"Tvrtka: \2\nOpis:", txt)
        txt = re.sub(r"(?m)^###\s*([A-ZČĆŽŠĐ].+)$", r"Tvrtka: \1\nOpis:", txt)
        page_chunks = chunk_text_smart(txt, size=size, overlap=overlap)
        for i, c in enumerate(page_chunks):
            out.append({"text": c, "page": page, "chunk_idx_on_page": i})
    return out

def reset_collection() -> str:
    client, coll, _ = _qdrant()
    try:
        client.delete_collection(coll)
    except Exception as e:
        logging.info(f"Delete collection warning (maybe missing): {e}")
    # recreate (with backoff)
    for _ in range(20):
        try:
            client.create_collection(
                collection_name=coll,
                vectors_config=qmodels.VectorParams(
                    size=len(_encode_query("dimension-probe")),
                    distance=qmodels.Distance.COSINE,
                ),
                timeout=30,
            )
            break
        except Exception:
            time.sleep(0.25)
    else:
        raise RuntimeError("Could not recreate collection (still deleting?).")
    ensure_fulltext_index(client, coll, field_name="text")
    try:
        for field in ("hash", "doc_id", "source"):
            client.create_payload_index(
                collection_name=coll,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
    except Exception:
        pass
    return coll

def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def upsert_chunks_to_qdrant(chunks: List[dict], source_name: str) -> int:
    if not chunks:
        return 0
    client, coll, _ = _qdrant()
    BATCH = 128
    uploaded = 0
    now_iso = datetime.utcnow().isoformat() + "Z"
    ensure_fulltext_index(client, coll, field_name="text")
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        texts = [b["text"] for b in batch]
        vecs = _encode_passages(texts).tolist()
        points = []
        for k, ch in enumerate(batch):
            txt = ch["text"]
            pid_uuid = uuid.uuid5(uuid.NAMESPACE_URL, txt)
            points.append(
                qmodels.PointStruct(
                    id=str(pid_uuid),
                    vector=vecs[k],
                    payload={
                        "text": txt,
                        "source": source_name,
                        "page": ch.get("page"),
                        "chunk_idx_on_page": ch.get("chunk_idx_on_page"),
                        "hash": _hash_text(txt),
                        "uploaded_at": now_iso,
                    },
                )
            )
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

# -------- Context retrieval (semantic + full-text + keyword + bilingual) --------
ROLE_ALIASES = {
    "direktor": ["ravnatelj", "predsjednik", "voditelj"],
    "predsjednik": ["ravnatelj", "direktor", "voditelj"],
    "ravnatelj": ["direktor", "predsjednik", "voditelj"],
}

def expand_query_hr(q: str) -> List[str]:
    ql = q.lower()
    out = [q]
    for k, alts in ROLE_ALIASES.items():
        if k in ql:
            for a in alts:
                out.append(re.sub(k, a, q, flags=re.IGNORECASE))
    # dedup preserving order
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

def get_qdrant_context_vector(query: str, top_k: int = 10) -> List[Tuple[float, str]]:
    client, coll, _ = _qdrant()
    qvec = _encode_query(query).tolist()
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
    qv = _encode_query(query).tolist()
    scored: List[Tuple[float, str]] = []
    for txt in texts:
        sv = _encode_passage(txt).tolist()
        dot = sum(a*b for a,b in zip(qv, sv))
        nq = math.sqrt(sum(a*a for a in qv)) or 1.0
        ns = math.sqrt(sum(a*a for a in sv)) or 1.0
        sim = dot/(nq*ns)
        scored.append((float(sim), txt))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def _prioritize_company_contexts(query: str, contexts: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    if not contexts:
        return contexts
    q_low = (query or "").lower()
    if any(term in q_low for term in ALUTECH_TERMS):
        hits = []
        for item in contexts:
            text_low = item[1].lower()
            if any(term in text_low for term in ALUTECH_TERMS):
                hits.append(item)
        if hits:
            return hits
    return contexts

def _default_to_company_contexts(query: str, contexts: List[Tuple[float, str]], top_k: int) -> List[Tuple[float, str]]:
    # Ako već imamo kontekste gdje se pojavljuje "alutech", zadrži
    if contexts and any(term in item[1].lower() for item in contexts for term in ALUTECH_TERMS):
        return contexts

    q_low = (query or "").lower()
    # Ako u upitu eksplicitno stoji alutech, zadrži postojeće
    if any(term in q_low for term in ALUTECH_TERMS):
        return contexts

    # Inače – preferiraj direktan fallback na Ricat (jer je to default tvog sustava)
    ricat_fallback = get_qdrant_context_vector("Ricat", top_k=top_k)
    if ricat_fallback:
        return ricat_fallback

    # Kao sekundarni fallback, može i "Alutech" (rijetko će trebati)
    alutech_fallback = get_qdrant_context_vector("Alutech", top_k=top_k)
    return alutech_fallback if alutech_fallback else contexts


def get_context_smart(query: str, top_k: int = 10, min_vec_score: float = 0.25) -> List[Tuple[float, str]]:
    """
    1) Semantic vector search (original + HR synonym expansion + EN translation)
    2) If weak → FULL-TEXT (HR, EN) + semantički reranking
    3) Ako i dalje prazno → keyword scroll fallback + reranking
    """
    lang = safe_detect(query)

    # Build candidate queries
    queries = [query]
    if lang in ("hr", "sh", "bs", "sr"):
        queries = expand_query_hr(query)
        # add English translation of the primary query
        try:
            to_en = _translator_to_en()
            if to_en:
                en_q = to_en(query)[0]["translation_text"]
                queries.append(en_q)
        except Exception:
            pass

    # 1) Semantic over all candidate queries, then merge by text with max score
    merged_scores = {}
    for q in queries:
        vec = get_qdrant_context_vector(q, top_k=top_k)
        for s, t in vec:
            merged_scores[t] = max(merged_scores.get(t, 0.0), s)
    vec_merged = sorted([(s, t) for t, s in merged_scores.items()], key=lambda x: x[0], reverse=True)[:top_k]
    vec_merged = _prioritize_company_contexts(query, vec_merged)
    if vec_merged and (vec_merged[0][0] >= min_vec_score or len(vec_merged) >= max(4, top_k//2)):
        return _default_to_company_contexts(query, vec_merged, top_k)

    # 2) Full-text on all candidate queries
    ft_hits_all: List[str] = []
    for q in queries:
        ft_hits_all += get_qdrant_context_fulltext(q, limit=32)
    if ft_hits_all:
        ft_ranked = rerank_by_similarity(query, ft_hits_all, top_k=top_k)
        return _default_to_company_contexts(query, _prioritize_company_contexts(query, ft_ranked), top_k)

    # 3) Keyword fallback on HR tokens from the original query
    kws = _split_keywords(query)
    kw_hits = search_keyword_in_qdrant(kws, max_points=400)
    if kw_hits:
        kw_ranked = rerank_by_similarity(query, kw_hits, top_k=top_k)
        return _default_to_company_contexts(query, _prioritize_company_contexts(query, kw_ranked), top_k)

    fallback = _default_to_company_contexts(query, [], top_k)
    if not vec_merged:
        alt_query = f"tvrtka {query}"
        alt_vec = get_qdrant_context_vector(alt_query, top_k=top_k)
        if alt_vec:
            return alt_vec
    return fallback if fallback else []

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
        OPENAI_PRIMARY_LABEL: 128000,
        "HF Pro Models (HR-first)": 8192,
        "HF Standard Models (router)": 8192,
        "DeepSeek R1 (cloud)": 16384,
    }
    max_ctx = ctx_windows.get(model_choice, 8192)

    # --- NORMALIZACIJA UPITA ---
    normalized_question = user_question.strip()
    if not normalized_question:
        normalized_question = "Koje usluge nudi Ricat?"
    else:
        # zamijeni sve oblike "alutech" s "ricat"
        normalized_question = re.sub(r'\b[Aa][Ll][Uu][ -]?[Tt][Ee][Cc][Hh]\b', 'Ricat', normalized_question)
        # ako se ne spominje nijedna firma, pretpostavi Ricat
        if not re.search(r'\b[Rr][Ii][Cc][Aa][Tt]\b', normalized_question):
            normalized_question = f"{normalized_question} (pitanje se odnosi na Ricat)"

    # --- HEADER + FOOTER ---
    header = (
        "Tvoj zadatak je odgovoriti na korisničko pitanje koristeći ISKLJUČIVO informacije iz sljedećeg konteksta.\n\n"
        "🎯 CILJ:\n"
        "Daj jasan, sažet i točan odgovor koji pokriva bit pitanja tako da korisnik odmah dobije najveću moguću vrijednost.\n\n"
        "📜 PRAVILA:\n"
        "- Koristi isključivo informacije iz danog konteksta (nema izmišljanja).\n"
        "- Poduzeća RICAT i ALUTECH (uključujući sve varijante zapisa poput 'Alutech', 'alutech', 'ALutech' itd.) smatraju se istim poduzećem.\n"
        "- Ako korisnik spomene Alutech u bilo kojem obliku, tretiraj to kao da se radi o RICAT-u i koristi sve informacije koje se odnose na Ricat.\n"
        "- Ako korisnik ne navede poduzeće o kojem se radi, odgovori kao da se radi o RICAT poduzeću.\n"
        "- Ako kontekst ne sadrži odgovor, reci ovo što je u zagradi (\"Nema dovoljno informacija u dostupnim dokumentima. Provjerite na https://alutech.hr/\").\n"
        "- Odgovaraj isključivo na hrvatskom jeziku, gramatički i stilski prirodno.\n"
        "- Ne spominji dokumente.\n"
        "- Ako pitanje ne spominje drugu organizaciju, najprije provjeri ima li kontekst informacija o poduzeću Ricat; ako ih nema, reci da nema dovoljno informacija (nemoj koristiti podatke o drugim kompanijama).\n"
        "- Ako korisnik izričito pita za drugu organizaciju, odgovori samo ako kontekst sadrži provjerljive informacije za tu organizaciju; u suprotnom objasni da informacija nema.\n"
        "- Izbjegavaj meta-komentare i oznake poput <think>.\n\n"
        "📚 KONTEKST:\n"
    )
    footer = f"\n\n❓ PITANJE KORISNIKA:\n{normalized_question}\n\n💬 ODGOVOR:\n"

    # --- TOKEN BUDŽET ---
    header_tokens = estimate_tokens(header)
    footer_tokens = estimate_tokens(footer)
    budget = max_ctx - reply_tokens - header_tokens - footer_tokens
    if budget < 256:
        budget = 256

    # --- KONTEKST ---
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
# LLM backends (HR-first router)
# -----------------------------
HR_KEYWORDS = set("""

da li može možete hrvatski usluga cijena tvrtka poduzeće odgovori sažetak zaključak primjer politika izjava rokovi uvjeti direktor ravnatelj predsjednik
""".split())

ALUTECH_TERMS = ('alutech', 'alu tech', 'alu-tech')

SR_LATIN_MARKERS = {
    "tačnije", "tačno", "tačan", "tačna", "tačne", "tačnosti",
    "preduzeće", "preduzeća", "preduzeću", "preduzećima",
    "preduzetnik", "preduzetnici", "preduzetništvo",
    "opština", "opštine", "opštinski",
    "svetski", "svetska", "svetsko",
}

_SR_TO_HR_MAP = {
    "tačnije": "točnije",
    "tačno": "točno",
    "tačan": "točan",
    "tačna": "točna",
    "tačne": "točne",
    "tačnosti": "točnosti",
    "preduzeće": "poduzeće",
    "preduzeća": "poduzeća",
    "preduzeću": "poduzeću",
    "preduzećima": "poduzećima",
    "preduzetnik": "poduzetnik",
    "preduzetnici": "poduzetnici",
    "preduzetništvo": "poduzetništvo",
    "opština": "općina",
    "opštine": "općine",
    "opštinski": "općinski",
    "svetski": "svjetski",
    "svetska": "svjetska",
    "svetsko": "svjetsko",
}

_SR_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in _SR_TO_HR_MAP.keys()) + r")\b", flags=re.IGNORECASE)


def _normalize_to_hr(text: str) -> str:
    if not text:
        return ""

    def repl(match: re.Match) -> str:
        original = match.group(0)
        target = _SR_TO_HR_MAP.get(original.lower())
        if not target:
            return original
        if original.isupper():
            return target.upper()
        if original[0].isupper():
            return target.capitalize()
        return target

    return _SR_PATTERN.sub(repl, text)


def safe_detect(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        if re.search(r"[čćšđžČĆŠĐŽ]", text or ""):
            return "hr"
        return "en"

def normalize_question(user_question: str, last_company: str | None = None):
    """
    - Alutech (bilo koji oblik) -> Ricat
    - Ako je eksplicitno Ricat -> ostavi
    - Ako nema firme -> koristi last_company ili Ricat (default)
    Vraća: (normalized_question, current_company)
    """
    q = (user_question or "").strip()

    if not q:
        return "Koje usluge nudi Ricat?", "Ricat"

    # Alutech -> Ricat (s varijantama)
    if re.search(r'\b[Aa][Ll][Uu][ -]?[Tt][Ee][Cc][Hh]\b', q):
        q = re.sub(r'\b[Aa][Ll][Uu][ -]?[Tt][Ee][Cc][Hh]\b', 'Ricat', q)
        return q, "Ricat"

    # Ricat eksplicitno
    if re.search(r'\b[Rr][Ii][Cc][Aa][Tt]\b', q):
        return q, "Ricat"

    # Bez firme → koristi last_company ili default Ricat
    company = last_company or "Ricat"
    q = f"{q} (pitanje se odnosi na {company})"
    return q, company


def hr_quality_score(s: str) -> float:
    """Lightweight heuristic for HR-ness and fluency."""
    if not s:
        return 0.0
    s_low = s.lower()
    diacritics = len(re.findall(r"[čćšđž]", s_low))
    letters = max(1, len(re.findall(r"[a-zA-Zčćšđž]", s_low)))
    dia_ratio = diacritics / letters
    kw_hits = sum(1 for k in HR_KEYWORDS if k in s_low)
    en_hits = len(re.findall(r"\b(the|and|is|are|you|we|our|with|for|of)\b", s_low))
    score = 0.5*min(1.0, dia_ratio*4) + 0.4*min(1.0, kw_hits/5) + 0.2*(1.0/(1+en_hits))
    sr_hits = sum(1 for marker in SR_LATIN_MARKERS if marker in s_low)
    score -= 0.4*min(1.0, sr_hits/2)
    return max(0.0, min(1.0, score))

def is_low_quality_hr(s: str) -> bool:
    lang = safe_detect(s)
    if lang != "hr":
        if lang in ("sh", "bs", "sr"):
            return hr_quality_score(s) < 0.35
        return True
    return hr_quality_score(s) < 0.30


def is_backend_failure_message(msg: Optional[str]) -> bool:
    if not msg:
        return True
    lowered = msg.strip().lower()
    if not lowered:
        return True
    failure_markers = (
        "all hf models failed",
        "hf pro error",
        "hf standard error",
        "deepseek r1 error",
        "backend unavailable",
        "unexpected keyword argument 'provider'",
        "please set huggingface_token",
        "please set openrouter_api_key",
        "model route not implemented",
    )
    if any(marker in lowered for marker in failure_markers):
        return True
    return False


def summarize_backend_failure(msg: Optional[str]) -> str:
    if not msg:
        return "bez detalja"
    first_line = msg.strip().splitlines()[0]
    if len(first_line) > 160:
        first_line = first_line[:157] + "..."
    return first_line or "bez detalja"



def get_model_response(prompt: str, model_choice: str, user_lang_guess: str = "hr") -> str:
    """
    HR-first router with retry logic:
    1) Try multilingual instruct model → HR answer.
    2) If English-ish/low HR → try a stronger model on the SAME (HR) prompt.
    3) If still not HR → translate final ANSWER to HR (do NOT translate prompt/context).
    """

    def try_with_retries(call_fn, retries=3, delay=2):
        """Helper to retry flaky HF calls."""
        for attempt in range(1, retries + 1):
            try:
                return call_fn()
            except Exception as e:
                logging.warning(f"Attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    raise

    # --- OpenAI primary route ---
    if model_choice == OPENAI_PRIMARY_LABEL:
        if not OpenAI:
            return "OpenAI backend unavailable. Please install the openai package."

        openai_key = _get_config_value("OPENAI_API_KEY")
        if not openai_key:
            return "OPENAI_API_KEY is not configured. Add it to Streamlit secrets or the environment."

        try:
            client = OpenAI(api_key=openai_key)

            def call():
                return client.chat.completions.create(
                    model=OPENAI_PRIMARY_CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.6,
                )

            completion = try_with_retries(call)
            txt = clean_llm_output(completion.choices[0].message.content)

            if user_lang_guess in ("hr", "sh", "bs", "sr") and is_low_quality_hr(txt):
                to_hr = _translator_to_hr()
                if to_hr:
                    return clean_llm_output(to_hr(txt)[0]["translation_text"])
            return txt

        except Exception as e:
            return f"{model_choice} error: {e}"

    # --- HF Pro Models (HR-first) ---
    if model_choice == "HF Pro Models (HR-first)":
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            return "Please set HUGGINGFACE_TOKEN environment variable for HF Pro access."

        try:
            client = InferenceClient(token=hf_token)
            candidates_primary = [
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
            ]
            fallback_stronger = [
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
            ]

            # Step 1: direct HR attempt(s)
            for model_id in candidates_primary:
                try:
                    def call():
                        resp = client.chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            model=model_id,
                            max_tokens=800,
                            temperature=0.5,
                            stream=False,
                        )
                        return resp
                    response = try_with_retries(call)
                    txt = clean_llm_output(response.choices[0].message.content)
                    if not is_low_quality_hr(txt):
                        return txt
                except Exception as e:
                    logging.info(f"Chat model {model_id} failed: {e}")

            # Step 2: stronger model
            for model_id in fallback_stronger:
                try:
                    def call():
                        resp = client.chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            model=model_id,
                            max_tokens=800,
                            temperature=0.1,
                            stream=False,
                        )
                        return resp
                    response = try_with_retries(call)
                    txt2 = clean_llm_output(response.choices[0].message.content)
                    if not is_low_quality_hr(txt2):
                        return txt2
                    to_hr = _translator_to_hr()
                    if to_hr:
                        return clean_llm_output(to_hr(txt2)[0]["translation_text"])
                    return txt2
                except Exception as e:
                    logging.info(f"Fallback model {model_id} failed: {e}")

            return "All HF models failed after retries. This might be a temporary API issue."

        except Exception as e:
            if "unexpected keyword argument 'provider'" in str(e):
                return "Please upgrade huggingface_hub: pip install --upgrade huggingface_hub"
            return f"HF Pro error: {e}"

    # --- HF Standard Models (router) ---
    if model_choice == "HF Standard Models (router)":
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

                    def call():
                        return client.chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            model=model_id,
                            max_tokens=800,
                            temperature=0.6,
                            stream=False,
                        )

                    response = try_with_retries(call)
                    txt = clean_llm_output(response.choices[0].message.content)
                    if not is_low_quality_hr(txt):
                        return txt
                    break
                except Exception as e:
                    logging.info(f"Provider {provider} failed: {e}")
                    continue

            # Direct fallback
            client = InferenceClient(token=hf_token)
            for model_id in [
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
            ]:
                try:
                    def call():
                        return client.chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            model=model_id,
                            max_tokens=800,
                            temperature=0.6,
                            stream=False,
                        )

                    response = try_with_retries(call)
                    txt2 = clean_llm_output(response.choices[0].message.content)
                    if not is_low_quality_hr(txt2):
                        return txt2
                    to_hr = _translator_to_hr()
                    if to_hr:
                        return clean_llm_output(to_hr(txt2)[0]["translation_text"])
                    return txt2
                except Exception as e:
                    logging.info(f"HF direct fallback {model_id} failed: {e}")

            return "All HF standard models failed after retries."

        except Exception as e:
            if "unexpected keyword argument 'provider'" in str(e):
                return "Please upgrade huggingface_hub: pip install --upgrade huggingface_hub"
            return f"HF Standard error: {e}"

    # --- DeepSeek via OpenRouter ---
    if model_choice == "DeepSeek R1 (cloud)":
        if not OpenAI:
            return "DeepSeek R1 backend unavailable. Please install openai package."

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            return "Please set OPENROUTER_API_KEY environment variable."

        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

            def call():
                return client.chat.completions.create(
                    model="deepseek/deepseek-chat-v3.1:free",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1200,
                    temperature=0.7,
                )

            completion = try_with_retries(call)
            txt = clean_llm_output(completion.choices[0].message.content)

            if user_lang_guess in ("hr", "sh", "bs", "sr") and is_low_quality_hr(txt):
                to_hr = _translator_to_hr()
                if to_hr:
                    return clean_llm_output(to_hr(txt)[0]["translation_text"])
            return txt

        except Exception as e:
            return f"DeepSeek R1 error: {e}"

    # --- Fallback route ---
    return "Model route not implemented."



# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
