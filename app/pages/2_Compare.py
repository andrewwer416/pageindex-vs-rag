"""Compare page: pick a doc from your library and run side-by-side comparison of
Traditional RAG vs PageIndex on it. Also surfaces the PageIndex hierarchy tree
that was extracted at indexing time.
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st

# Make the project root importable when streamlit launches this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import config, library  # noqa: E402
from app.rag_pipeline import answer as rag_answer  # noqa: E402
from app.pageindex_agent import answer as pi_answer, get_tree_structure  # noqa: E402


st.set_page_config(page_title="Compare — PageIndex vs RAG", layout="wide")
st.title("Compare on your document")

# ──────────────── doc picker ───────────────────────────────────────────────────

docs = library.list_documents()
ready_docs = [d for d in docs if d.get("status") in ("ready", "partial")]

if not ready_docs:
    st.info(
        "Your library is empty (or no documents are indexed yet). Open the **app** "
        "page in the sidebar to upload a document, wait for indexing to finish, "
        "then come back here."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Document")
    picked_id = st.selectbox(
        "Choose a document",
        options=[d["doc_id"] for d in ready_docs],
        format_func=lambda did: next(d["name"] for d in ready_docs if d["doc_id"] == did),
    )

doc = next(d for d in ready_docs if d["doc_id"] == picked_id)
kw = library.doc_kwargs(picked_id)

st.markdown(
    f"Querying **{doc['name']}** &nbsp;·&nbsp; "
    f"{doc.get('page_count') or '?'} pages &nbsp;·&nbsp; status: `{doc.get('status')}`"
)
if doc.get("status") == "partial":
    st.warning("PageIndex tree wasn't built for this doc — only the Traditional-RAG side will produce answers.")


# ──────────────── document hierarchy ──────────────────────────────────────────

if doc.get("status") == "ready":
    with st.expander("📚 Document hierarchy (the tree PageIndex extracted)", expanded=False):
        try:
            tree = get_tree_structure(**{k: v for k, v in kw["pi"].items() if k != "doc_name"})
        except Exception as e:
            st.error(f"Could not load tree: {e!r}")
            tree = None

        if tree:
            def _render(node, depth=0):
                if isinstance(node, list):
                    for n in node:
                        _render(n, depth)
                    return
                title = node.get("title", "?")
                page = node.get("physical_index") or node.get("start_index") or node.get("page_number")
                page_str = f" — p. {page}" if page else ""
                summary = node.get("summary") or ""
                indent = "&nbsp;" * (depth * 4)
                st.markdown(f"{indent}• **{title}**{page_str}")
                if summary:
                    st.markdown(f"{indent}&nbsp;&nbsp;&nbsp;&nbsp;_{summary[:300]}_")
                for k, v in node.items():
                    if k.startswith("nodes") and isinstance(v, list) and v:
                        _render(v, depth + 1)
            _render(tree)

            with st.expander("Raw tree JSON"):
                import json as _json
                st.code(_json.dumps(tree, indent=2)[:20000], language="json")


# Generic sample questions that work on most documents.
SAMPLES = [
    "Summarize this document in one paragraph.",
    "What are the most important takeaways from this document?",
    "What sections discuss risk, uncertainty, or limitations?",
    "List the main entities, organizations, or people mentioned.",
    "What conclusions or recommendations does the document make?",
]

st.markdown("**Try a sample question, or write your own:**")
if "custom_query" not in st.session_state:
    st.session_state["custom_query"] = ""

cols = st.columns(len(SAMPLES))
for i, q in enumerate(SAMPLES):
    if cols[i].button(f"#{i+1}", help=q, use_container_width=True):
        st.session_state["custom_query"] = q

query = st.text_area("Question", key="custom_query", height=80)
go = st.button("Run both pipelines", type="primary", disabled=not query.strip(),
               key=f"go_{picked_id}")


# ──────────────── run + render ────────────────────────────────────────────────

def _run(query: str) -> dict:
    rag = pi = None
    rag_err = pi_err = None
    rag_time = pi_time = 0.0
    try:
        t = time.time()
        rag = rag_answer(query, **kw["rag"])
        rag_time = time.time() - t
    except Exception as e:
        rag_err = repr(e)
    if doc.get("status") == "ready":
        try:
            t = time.time()
            pi = pi_answer(query, **kw["pi"])
            pi_time = time.time() - t
        except Exception as e:
            pi_err = repr(e)
    return {"query": query, "rag": rag, "rag_err": rag_err, "rag_time": rag_time,
            "pi": pi, "pi_err": pi_err, "pi_time": pi_time}


if go and query.strip():
    with st.spinner("Running both pipelines..."):
        st.session_state[f"results_{picked_id}"] = _run(query)

results = st.session_state.get(f"results_{picked_id}")
if results:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Traditional RAG")
        if results["rag_err"]:
            st.error(f"```\n{results['rag_err']}\n```")
        elif results["rag"] is not None:
            rag = results["rag"]
            st.markdown(f"**Answer** *(generated in {results['rag_time']:.1f}s)*")
            if not rag.get("answer", "").strip():
                st.warning("LLM returned an empty answer.")
            st.write(rag["answer"])
            with st.expander(f"Retrieved chunks (top {len(rag['hits'])})", expanded=True):
                for h in rag["hits"]:
                    page_label = f"p. {h['page_start']}" if h["page_start"] == h["page_end"] else f"pp. {h['page_start']}–{h['page_end']}"
                    st.markdown(f"**{page_label}** &nbsp;·&nbsp; sim `{h['score']:.3f}` &nbsp;·&nbsp; {h['tokens']} tok")
                    st.text(h["text"][:600] + ("…" if len(h["text"]) > 600 else ""))
                    st.markdown("---")
            if rag.get("thinking"):
                with st.expander("Model reasoning trace"):
                    st.text(rag["thinking"])

    with right:
        st.subheader("PageIndex")
        if doc.get("status") != "ready":
            st.info("PageIndex tree not available for this doc.")
        elif results["pi_err"]:
            st.error(f"```\n{results['pi_err']}\n```")
        elif results["pi"] is not None:
            pi = results["pi"]
            st.markdown(f"**Answer** *(generated in {results['pi_time']:.1f}s)*")
            if not pi.get("answer", "").strip():
                st.warning("LLM returned an empty answer.")
            st.write(pi["answer"])
            for step in pi["steps"]:
                title = "Step 1 — Plan: pick relevant sections" if step["name"] == "plan" else "Step 2 — Answer from selected pages"
                with st.expander(title, expanded=(step["name"] == "plan")):
                    thinking = step.get("thinking", "").strip()
                    if thinking:
                        st.markdown("*Model reasoning:*")
                        st.text(thinking)
                    else:
                        st.caption("_(This model didn't surface an explicit reasoning trace.)_")
                    if step["name"] == "plan":
                        st.markdown(f"**Selected pages:** `{step.get('parsed_pages','')}`")
                        if step.get("reasoning"):
                            st.info(step["reasoning"])
                    else:
                        mo = step.get("model_output", "").strip()
                        if mo:
                            st.markdown("*Raw step output:*")
                            st.text(mo[:2000] + ("…" if len(mo) > 2000 else ""))
            if pi.get("page_text"):
                with st.expander(f"Content of pages read ({pi['pages_read']})"):
                    st.text(pi["page_text"][:5000] + ("…" if len(pi["page_text"]) > 5000 else ""))
