"""
Streamlit RAG chatbot for the Oracle IDAM support-doc collection.

Sequence implemented in this file:
  1. Load retrieval models/clients (cached, loaded once per app instance)
  2. Retrieval functions (hybrid_search, fetch_units, search_with_context, rerank)
     -- copied straight from your notebook, unchanged
  3. contextualize_query()  -> rewrites follow-up questions into standalone queries
  4. generate_answer()      -> calls Groq with retrieved context + recent history
  5. st.session_state       -> per-session chat memory (list of {role, content})
  6. Streamlit chat UI loop -> ties 3 -> 4 -> 5 together
"""

import os
import re
import streamlit as st
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient, models
from groq import Groq

st.set_page_config(page_title="Oracle IDAM Support Assistant", page_icon="🛡️")


COLLECTION_NAME = "oracle_docs_321"

_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"}


def clean_text(text: str) -> str:
    """
    Strip leftover HTML markup from ingested source text (e.g. <br> from a
    Confluence/HTML export that wasn't stripped during chunking). Run this on
    every chunk BEFORE it reaches the LLM, so the model never treats markup as
    real content worth preserving verbatim.
    """
    if not text:
        return text
    text = _BR_TAG_RE.sub("\n", text)   # real line breaks, not literal "<br>"
    text = _ANY_TAG_RE.sub("", text)    # drop any other stray tags (<b>, <table>, etc.)
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    return text

# ---------------------------------------------------------------------------
# 1. SECRETS / CONFIG
# On Streamlit Cloud: put these in the app's "Secrets" panel (Settings -> Secrets)
# as a TOML block:
#
#   QDRANT_URL = "..."
#   QDRANT_API_KEY = "..."
#   GROQ_API_KEY = "..."
#
# st.secrets reads that automatically. Locally, you can instead set them as
# environment variables and the fallback below will pick them up.
# ---------------------------------------------------------------------------
def _secret(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value

    try:
        return st.secrets[name]
    except Exception:
        raise RuntimeError(f"Missing secret: {name}")


# ---------------------------------------------------------------------------
# 2. CACHED RESOURCES -- loaded once, reused across all users/reruns
# st.cache_resource is the right cache type for models/clients (not st.cache_data)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading retrieval models (first run only)...")
def load_resources():
    qdrant_url = _secret("QDRANT_URL").strip()
    qdrant_key = _secret("QDRANT_API_KEY").strip()

    # Sanity-check the URL before handing it to the network stack. A malformed
    # value here (stray quotes, a pasted-in secrets block, a newline, the API
    # key concatenated in by mistake) doesn't fail cleanly -- it surfaces as a
    # confusing 'idna codec / label too long' DNS error several layers deep in
    # httpx. Catch it here instead, with a message that says what's wrong.
    if not qdrant_url.startswith(("http://", "https://")):
        raise RuntimeError(
            f"QDRANT_URL doesn't look like a URL (starts with: {qdrant_url[:40]!r}). "
            "Check for stray quotes, whitespace, or an accidentally pasted secrets block."
        )
    if len(qdrant_url) > 200 or "\n" in qdrant_url:
        raise RuntimeError(
            "QDRANT_URL is unusually long or contains a newline -- it likely has "
            "extra text pasted into it. It should be just the cluster URL, e.g. "
            "'https://xxxx.cloud.qdrant.io:6333'."
        )

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=torch.cuda.is_available())
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=torch.cuda.is_available())
    groq_client = Groq(api_key=_secret("GROQ_API_KEY").strip())
    return client, embed_model, reranker, groq_client


client, embed_model, reranker, groq_client = load_resources()


# ---------------------------------------------------------------------------
# 2b. RETRIEVAL FUNCTIONS -- unchanged from your notebook
# ---------------------------------------------------------------------------
def sparse_to_qdrant(w: dict) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(k) for k in w.keys()],
        values=[float(v) for v in w.values()],
    )


def _filter(doc=None, types=None):
    must = []
    if doc:
        must.append(models.FieldCondition(key="doc", match=models.MatchValue(value=doc)))
    if types:
        must.append(models.FieldCondition(key="type", match=models.MatchAny(any=list(types))))
    return models.Filter(must=must) if must else None


def hybrid_search(query: str, top_k: int = 8, mode: str = "hybrid",
                   doc: str | None = None, types: list[str] | None = None,
                   prefetch_k: int = 50):
    out = embed_model.encode([query], max_length=512, return_dense=True,
                              return_sparse=True, return_colbert_vecs=False)
    dense = out["dense_vecs"][0].tolist()
    sparse = sparse_to_qdrant(out["lexical_weights"][0])
    flt = _filter(doc, types)

    if mode == "dense":
        res = client.query_points(COLLECTION_NAME, query=dense, using="dense",
                                   query_filter=flt, limit=top_k, with_payload=True)
    elif mode == "sparse":
        res = client.query_points(COLLECTION_NAME, query=sparse, using="sparse",
                                   query_filter=flt, limit=top_k, with_payload=True)
    else:
        res = client.query_points(
            COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense, using="dense", filter=flt, limit=prefetch_k),
                models.Prefetch(query=sparse, using="sparse", filter=flt, limit=prefetch_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
    return res.points


def fetch_units(doc: str, indices):
    indices = sorted(set(indices))
    pts, _ = client.scroll(
        COLLECTION_NAME,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="doc", match=models.MatchValue(value=doc)),
            models.FieldCondition(key="unit_index", match=models.MatchAny(any=indices)),
        ]),
        limit=len(indices),
        with_payload=True,
        with_vectors=False,
    )
    return sorted(pts, key=lambda p: p.payload["unit_index"])


def search_with_context(query: str, top_k: int = 5, radius: int = 5, **kw):
    hits = hybrid_search(query, top_k=top_k, **kw)
    seen, results = set(), []
    for h in hits:
        d, i = h.payload["doc"], h.payload["unit_index"]
        if (d, i) in seen:
            continue
        window = fetch_units(d, range(max(0, i - radius), i + radius + 1))
        seen.update((d, p.payload["unit_index"]) for p in window)
        results.append({
            "doc": d,
            "pages": sorted({p.payload.get("page") for p in window if p.payload.get("page") is not None}),
            "score": h.score,
            "hit_type": h.payload["type"],
            "text": clean_text("\n\n".join(p.payload["content"] for p in window)),
        })
    return results


def rerank(query, hits, keep=5):
    scores = reranker.compute_score(
        [[query, h.payload["content"]] for h in hits], normalize=True
    )
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return ranked[:keep]


# Below this rerank score, we don't trust anything found -- treat as "not in docs"
# rather than handing the LLM a weak/irrelevant chunk and letting it guess.
MIN_CONFIDENT_SCORE = 0.15


def robust_search(raw_question: str, standalone_query: str, top_k: int = 8, keep: int = 5):
    """
    Retrieve using BOTH the raw question and the (possibly rewritten)
    standalone query, merge the candidate pool, then score every candidate
    against BOTH queries and keep whichever score is higher per chunk.

    Why: a single bad rewrite used to be able to hijack the entire answer
    (search + rerank + generation all inherited its mistake). Here, a bad
    rewrite can only ever ADD irrelevant candidates to the pool -- it can
    never suppress the correct chunks, because those chunks are still
    fetched via the raw question and still score well against it.
    """
    queries = [raw_question]
    if standalone_query and standalone_query.strip() != raw_question.strip():
        queries.append(standalone_query)

    seen_ids, pool = set(), []
    for q in queries:
        for h in hybrid_search(q, top_k=top_k):
            if h.id in seen_ids:
                continue
            seen_ids.add(h.id)
            pool.append(h)

    if not pool:
        return []

    raw_scores = reranker.compute_score(
        [[raw_question, h.payload["content"]] for h in pool], normalize=True
    )
    if len(queries) > 1:
        std_scores = reranker.compute_score(
            [[standalone_query, h.payload["content"]] for h in pool], normalize=True
        )
    else:
        std_scores = raw_scores

    combined = [(h, max(r, s)) for h, r, s in zip(pool, raw_scores, std_scores)]
    ranked = sorted(combined, key=lambda x: x[1], reverse=True)
    return ranked[:keep]


# ---------------------------------------------------------------------------
# 3. QUERY CONTEXTUALIZATION -- rewrites follow-ups into standalone queries
# ---------------------------------------------------------------------------
def format_history(history):
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


# Words that suggest a question is actually referring back to the previous turn
# (pronouns, continuations) rather than introducing a new, unrelated topic.
_FOLLOWUP_SIGNALS = (
    "it", "that", "this", "those", "these", "also", "again", "what about",
    "and if", "same", "above", "previous", "instead", "either", "further",
    "more on", "else", "other one", "other option",
)


def _looks_like_followup(question: str) -> bool:
    q = question.lower()
    return any(sig in q for sig in _FOLLOWUP_SIGNALS)


def contextualize_query(history: list[dict], new_question: str) -> str:
    if not history:
        return new_question

    # Skip the LLM rewrite entirely when the question doesn't look like a
    # follow-up -- this is the main fix: a weak/fast rewrite model can
    # otherwise drag an unrelated new question back toward the previous
    # topic just because the history is topically dominant.
    if not _looks_like_followup(new_question):
        return new_question

    recent = history[-2:]  # just the last exchange -- less topic to latch onto
    prompt = f"""You rewrite follow-up questions into standalone questions for a
search engine. You will be shown the last exchange of a conversation and a new
question.

Rules:
- If the new question is a follow-up that depends on the previous topic (uses
  words like "it", "that", "also", "what about"), rewrite it to include the
  missing context explicitly.
- If the new question introduces a DIFFERENT, unrelated topic, return it
  completely UNCHANGED. Do not merge it with the previous topic.
- Never answer the question. Only output the rewritten (or unchanged) question.

Example 1:
Previous topic: generating B2B OAuth tokens in IDAM
New question: "what about the client secret?"
Output: "How do I get the client secret when generating a B2B OAuth token in IDAM?"

Example 2:
Previous topic: generating B2B OAuth tokens in IDAM
New question: "how to upload banners in mobile app"
Output: "how to upload banners in mobile app"
(unchanged -- this is a new, unrelated topic, not a follow-up)

Now do the same for this conversation:

Last exchange:
{format_history(recent)}

New question: {new_question}
Output:"""

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 4. GENERATION -- answers using retrieved context + recent conversation
# ---------------------------------------------------------------------------
def generate_answer(query: str, contexts: list[dict], history: list[dict]):
    context_parts = []
    total_chars = 0
    max_context_chars = 16000  # bumped up since radius=5 windows are larger than radius=3

    for c in contexts:
        text = c.get("text", "")
        remaining = max_context_chars - total_chars

        if remaining <= 0:
            break

        text = text[:remaining]

        context_parts.append(
            f"[{c['doc']} p.{c['pages']}]\n{text}"
        )

        total_chars += len(text)

    context_block = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are an enterprise technical support assistant specializing in "
        "Oracle Identity and Access Management (Oracle IDAM) and "
        "Oracle WebCenter Content (WCC). "
        "Answer using ONLY the retrieved documentation provided in the context. "
        "Clearly distinguish between Oracle IDAM and WebCenter Content. "
        "Never invent configuration steps, commands, error codes, or solutions. "
        "If the retrieved context does not contain the answer, say so explicitly. "
        "Provide clear, numbered troubleshooting steps when supported by the "
        "documentation. Preserve exact product names, error messages, and "
        "configuration values from the source documents."
    )

    recent_history = history[-4:]

    messages = [{"role": "system", "content": system_prompt}]

    for m in recent_history:
        messages.append({
            "role": m["role"],
            "content": m["content"][:1500]
        })

    messages.append({
        "role": "user",
        "content": (
            f"Context:\n{context_block}"
            f"\n\nQuestion: {query}"
        )
    })

    stream = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2,
        max_tokens=700,
        stream=True,
    )

    # Yield text pieces as they arrive. Some chunks (role/finish/reasoning
    # deltas) carry no content, so skip those.
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece


# 5. SESSION MEMORY -- one chat_history per browser session, reset on refresh
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# 6. STREAMLIT UI
# ---------------------------------------------------------------------------

st.title("🛡️ Oracle IDAM Support Assistant")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask a question about Oracle IDAM...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            standalone_query = contextualize_query(st.session_state.chat_history, user_question)
            reranked = robust_search(user_question, standalone_query, top_k=8, keep=5)

            # Confidence floor: nothing found that actually looks relevant --
            # answer directly instead of feeding the LLM weak/irrelevant context
            # and hoping it admits it doesn't know.
            if not reranked or reranked[0][1] < MIN_CONFIDENT_SCORE:
                answer = (
                    "I couldn't find anything in the documentation that looks "
                    "relevant to this question. It may not be covered in the "
                    "indexed docs, or it may need to be phrased differently."
                )
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.stop()

            RADIUS = 5
            contexts = []
            for h, score in reranked:
                d, i = h.payload["doc"], h.payload["unit_index"]
                window = fetch_units(d, range(max(0, i - RADIUS), i + RADIUS + 1))
                contexts.append({
                    "doc": d,
                    "pages": sorted({p.payload.get("page") for p in window if p.payload.get("page") is not None}),
                    "text": clean_text("\n\n".join(p.payload["content"] for p in window)),
                })

        # st.write_stream renders each piece as it arrives (typing effect) and
        # returns the complete text once finished, which we store in memory.
        answer = st.write_stream(
            generate_answer(standalone_query, contexts, st.session_state.chat_history)
        )

        with st.expander("🔍 Debug: retrieval details"):
            st.write("**Raw question:**")
            st.code(user_question)
            st.write("**Standalone query (rewrite, if any):**")
            st.code(standalone_query)
            st.write(f"**Retrieved {len(contexts)} chunks (radius={RADIUS}):**")
            for idx, (c, (h, score)) in enumerate(zip(contexts, reranked)):
                st.write(f"— Chunk {idx+1}: `{c['doc']}` p.{c['pages']} | combined_score={score:.3f} | {len(c['text'])} chars")
                st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
