"""Streamlit app: side-by-side comparison of traditional RAG vs PageIndex on Tesla's 2024 10-K."""
import json
import time
from pathlib import Path

import streamlit as st

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config
from app.rag_pipeline import answer as rag_answer, get_embedder, load_index
from app.pageindex_agent import answer as pageindex_answer, get_client as get_pageindex_client, get_tree_structure


SAMPLE_QUESTIONS = [
    "What were Tesla's total automotive revenues in 2024, and how did they change from 2023?",
    "What does Tesla disclose about cybersecurity risk management and incidents?",
    "How is Tesla's Energy Generation and Storage segment performing, and what are the key risks specific to it?",
    "Summarize the material legal proceedings Tesla is involved in as of the 2024 10-K.",
    "What is Tesla's strategy and exposure for operations in China, and what risks does the filing flag?",
]


# ---------- caching ----------
@st.cache_resource(show_spinner=False)
def _warm_caches():
    """Preload embedder, FAISS index, PageIndex client. Returns a dict of status."""
    status = {}
    t = time.time()
    embedder = get_embedder()
    status["embedder"] = f"{time.time()-t:.1f}s"
    t = time.time()
    index, chunks = load_index()
    status["faiss"] = f"{index.ntotal} vecs, {time.time()-t:.1f}s"
    t = time.time()
    try:
        get_pageindex_client()
        tree = get_tree_structure()
        status["pageindex"] = f"loaded, {time.time()-t:.1f}s"
        status["tree"] = tree
    except FileNotFoundError:
        status["pageindex"] = "not built yet (run scripts/build_index.py pageindex)"
        status["tree"] = None
    return status


@st.cache_data(show_spinner=False)
def _cached_rag(query: str) -> dict:
    return rag_answer(query)


@st.cache_data(show_spinner=False)
def _cached_pageindex(query: str) -> dict:
    return pageindex_answer(query)


# ---------- UI ----------
st.set_page_config(page_title="PageIndex vs Traditional RAG", layout="wide")

st.title("PageIndex vs Traditional RAG")
st.markdown(
    "Same document, same question, two retrieval architectures. "
    f"The document is **{config.DOC_NAME}** (227 pages). "
    "All LLM calls go through whichever provider is configured in `.env`."
)

status = _warm_caches()
with st.expander("System status", expanded=False):
    st.write(f"- Embedder: `{config.EMBED_MODEL}` ({status['embedder']})")
    st.write(f"- FAISS index: {status['faiss']}")
    st.write(f"- PageIndex tree: {status['pageindex']}")
    st.write(f"- LLM endpoint: `{config.LLM_API_BASE}`")
    st.write(f"- Indexing model: `{config.INDEX_MODEL}`")
    st.write(f"- Answer model: `{config.ANSWER_MODEL}`")
    st.write(f"- Retrieval reasoning model: `{config.RETRIEVE_MODEL}`")

st.markdown("---")
st.markdown("**Try a sample question, or write your own:**")
cols = st.columns(len(SAMPLE_QUESTIONS))
chosen = None
for i, q in enumerate(SAMPLE_QUESTIONS):
    if cols[i].button(f"#{i+1}", help=q, use_container_width=True):
        chosen = q

query = st.text_area("Question", value=chosen or st.session_state.get("last_query", ""), height=80)
go = st.button("Run both pipelines", type="primary", disabled=not query.strip())

if go and query.strip():
    st.session_state["last_query"] = query

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Traditional RAG")
        st.caption("Chunk → embed → FAISS top-k similarity → LLM generates answer from chunks.")
        with st.spinner("Retrieving + generating..."):
            t = time.time()
            rag = _cached_rag(query)
            rag_time = time.time() - t

        st.markdown(f"**Answer** *(generated in {rag_time:.1f}s)*")
        st.write(rag["answer"])
        with st.expander(f"Retrieved chunks (top {len(rag['hits'])} by cosine similarity)", expanded=True):
            for h in rag["hits"]:
                page_label = f"p. {h['page_start']}" if h["page_start"] == h["page_end"] else f"pp. {h['page_start']}–{h['page_end']}"
                st.markdown(f"**{page_label}** &nbsp;·&nbsp; similarity `{h['score']:.3f}` &nbsp;·&nbsp; {h['tokens']} tokens")
                st.text(h["text"][:600] + ("…" if len(h["text"]) > 600 else ""))
                st.markdown("---")
        if rag.get("thinking"):
            with st.expander("Model reasoning trace"):
                st.text(rag["thinking"])

    with right:
        st.subheader("PageIndex")
        st.caption("Read the document's table-of-contents tree → reason about which sections are relevant → read only those pages → answer.")
        if status["tree"] is None:
            st.warning("PageIndex tree not built yet. Run `python scripts/build_index.py pageindex`.")
        else:
            with st.spinner("Walking the tree..."):
                t = time.time()
                pi = _cached_pageindex(query)
                pi_time = time.time() - t

            st.markdown(f"**Answer** *(generated in {pi_time:.1f}s)*")
            st.write(pi["answer"])

            steps = pi["steps"]
            for step in steps:
                title = "Step 1 — Plan: pick relevant sections from the tree" if step["name"] == "plan" else "Step 2 — Answer using only those pages"
                with st.expander(title, expanded=(step["name"] == "plan")):
                    if step.get("thinking"):
                        st.markdown("*Model reasoning:*")
                        st.text(step["thinking"])
                    if step["name"] == "plan":
                        st.markdown(f"**Selected pages:** `{step.get('parsed_pages','')}`")
                        if step.get("reasoning"):
                            st.markdown("*Stated reason:*")
                            st.info(step["reasoning"])
            if pi.get("page_text"):
                with st.expander(f"Content of pages read ({pi['pages_read']})"):
                    st.text(pi["page_text"][:5000] + ("…" if len(pi["page_text"]) > 5000 else ""))

    st.markdown("---")
    with st.expander("How they differ"):
        st.markdown("""
**Traditional RAG (left)** splits the document into fixed-size chunks, computes a vector for each chunk, and ranks chunks by cosine similarity to the *query vector*. The model never sees the document's structure — only a flat list of high-similarity passages.
*Failure modes:* questions that require reasoning across distant sections, questions whose answer doesn't lexically match the query, questions about *absence* (what isn't in the document).

**PageIndex (right)** first builds a hierarchical table-of-contents tree (one-time cost, at indexing). At query time, the model reads the tree, decides which sections are relevant *by reasoning about the section headings*, then reads only those pages — much like a human flipping through the table of contents of a textbook. There are no vector embeddings at query time.
*Failure modes:* documents without clear hierarchical structure; questions where the section title isn't suggestive of the content.
""")

else:
    if status.get("tree"):
        with st.expander(f"PageIndex tree (top-level sections of {config.DOC_NAME})", expanded=False):
            def _render(node, depth=0):
                if isinstance(node, list):
                    for n in node:
                        _render(n, depth)
                    return
                title = node.get("title", "?")
                page = node.get("physical_index") or node.get("page_number")
                page_str = f" — p. {page}" if page else ""
                st.markdown(f"{'&nbsp;' * (depth * 4)}• **{title}**{page_str}")
                for k, v in node.items():
                    if k.startswith("nodes"):
                        _render(v, depth + 1)
            _render(status["tree"])
