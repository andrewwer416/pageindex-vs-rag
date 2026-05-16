"""Compare page: pick an indexed doc and run questions through all three
retrieval architectures side-by-side. Each pipeline gets its own tab so the
UI works on laptop-width screens while keeping a parallel structure.
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
from app.graphrag_pipeline import answer as graphrag_answer  # noqa: E402


st.set_page_config(page_title="Compare — PageIndex vs RAG vs GraphRAG", layout="wide")
st.title("Compare retrieval architectures")
st.caption(
    "Same document, same question, three pipelines. Each tab shows that "
    "pipeline's answer plus its own trace (retrieved chunks / tree walk / "
    "community partials)."
)

# ──────────────── doc picker ───────────────────────────────────────────────────

docs = library.list_documents()
ready_docs = [d for d in docs if d.get("status") in ("ready", "partial", "graphrag-indexing")]

if not ready_docs:
    st.info(
        "Your library is empty (or no documents have finished indexing). Upload "
        "a document on the **app** page in the sidebar, wait for indexing to "
        "finish, then come back here."
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
graphrag_built = library.has_graphrag(picked_id)

st.markdown(
    f"Querying **{doc['name']}** &nbsp;·&nbsp; "
    f"{doc.get('page_count') or '?'} pages &nbsp;·&nbsp; status: `{doc.get('status')}`"
)
if doc.get("status") == "partial":
    st.warning("PageIndex tree build had issues — that tab may show partial results.")


# ──────────────── document hierarchy ──────────────────────────────────────────

if doc.get("status") in ("ready", "partial"):
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


# ──────────────── question input ─────────────────────────────────────────────

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
go = st.button("Run all pipelines", type="primary", disabled=not query.strip(),
               key=f"go_{picked_id}")


# ──────────────── run pipelines ──────────────────────────────────────────────

def _run(query: str) -> dict:
    out = {"query": query}
    # RAG
    try:
        t = time.time()
        out["rag"] = rag_answer(query, **kw["rag"])
        out["rag_time"] = time.time() - t
        out["rag_err"] = None
    except Exception as e:
        out["rag"] = None
        out["rag_err"] = repr(e)
        out["rag_time"] = 0.0
    # PageIndex
    if doc.get("status") in ("ready", "partial"):
        try:
            t = time.time()
            out["pi"] = pi_answer(query, **kw["pi"])
            out["pi_time"] = time.time() - t
            out["pi_err"] = None
        except Exception as e:
            out["pi"] = None
            out["pi_err"] = repr(e)
            out["pi_time"] = 0.0
    else:
        out["pi"] = None
        out["pi_err"] = "PageIndex tree not available"
        out["pi_time"] = 0.0
    # GraphRAG
    if graphrag_built:
        try:
            t = time.time()
            out["gr"] = graphrag_answer(query, **kw["graphrag"])
            out["gr_time"] = time.time() - t
            out["gr_err"] = None
        except Exception as e:
            out["gr"] = None
            out["gr_err"] = repr(e)
            out["gr_time"] = 0.0
    else:
        out["gr"] = None
        out["gr_err"] = "GraphRAG index not built for this document"
        out["gr_time"] = 0.0
    return out


if go and query.strip():
    with st.spinner("Running all pipelines..."):
        st.session_state[f"results_{picked_id}"] = _run(query)

results = st.session_state.get(f"results_{picked_id}")


# ──────────────── render tabs ────────────────────────────────────────────────

def _render_empty_warning(label: str, ans: str):
    if not ans.strip():
        st.warning(f"{label} LLM returned an empty answer.")


def render_rag(results: dict):
    if results.get("rag_err"):
        st.error(f"```\n{results['rag_err']}\n```")
        return
    rag = results.get("rag")
    if rag is None:
        st.info("No result.")
        return
    st.markdown(f"**Answer** *(generated in {results['rag_time']:.1f}s)*")
    _render_empty_warning("RAG", rag.get("answer", ""))
    st.write(rag.get("answer", ""))
    with st.expander(f"Retrieved chunks (top {len(rag['hits'])})", expanded=True):
        for h in rag["hits"]:
            page_label = (
                f"p. {h['page_start']}" if h["page_start"] == h["page_end"]
                else f"pp. {h['page_start']}–{h['page_end']}"
            )
            st.markdown(f"**{page_label}** &nbsp;·&nbsp; sim `{h['score']:.3f}` &nbsp;·&nbsp; {h['tokens']} tok")
            st.text(h["text"][:600] + ("…" if len(h["text"]) > 600 else ""))
            st.markdown("---")
    if rag.get("thinking"):
        with st.expander("Model reasoning trace"):
            st.text(rag["thinking"])


def render_pageindex(results: dict):
    if results.get("pi_err"):
        if doc.get("status") not in ("ready", "partial"):
            st.info(results["pi_err"])
        else:
            st.error(f"```\n{results['pi_err']}\n```")
        return
    pi = results.get("pi")
    if pi is None:
        st.info("No result.")
        return
    st.markdown(f"**Answer** *(generated in {results['pi_time']:.1f}s)*")
    _render_empty_warning("PageIndex", pi.get("answer", ""))
    st.write(pi.get("answer", ""))
    for step in pi["steps"]:
        title = "Step 1 — Plan: pick relevant sections" if step["name"] == "plan" else "Step 2 — Answer from selected pages"
        with st.expander(title, expanded=(step["name"] == "plan")):
            thinking = step.get("thinking", "").strip()
            if thinking:
                st.markdown("*Model reasoning trace:*")
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


def render_graphrag(results: dict):
    if results.get("gr_err") and not graphrag_built:
        st.info(results["gr_err"])
        return
    if results.get("gr_err"):
        st.error(f"```\n{results['gr_err']}\n```")
        return
    gr = results.get("gr")
    if gr is None:
        st.info("No result.")
        return

    n_comm = gr.get("n_communities", 0)
    if n_comm == 0:
        # Diagnostic path: the answer field contains a markdown-formatted
        # explanation of why we have no communities — render it as such
        # rather than as a normal answer.
        st.warning(
            "GraphRAG produced 0 communities for this document — "
            "nothing to query. See the diagnostic below."
        )
        st.markdown(gr.get("answer", ""))
        return

    st.markdown(f"**Answer** *(aggregated across {n_comm} communities, "
                f"generated in {results['gr_time']:.1f}s)*")
    _render_empty_warning("GraphRAG", gr.get("answer", ""))
    st.write(gr.get("answer", ""))

    partials = gr.get("partials", [])
    if partials:
        with st.expander(f"Per-community partial answers ({len(partials)})", expanded=False):
            for p in partials:
                st.markdown(f"**Community {p['community_id']}**")
                st.caption(f"_Summary:_ {p['summary']}")
                st.write(p["answer"])
                st.markdown("---")


if results:
    tab_rag, tab_pi, tab_gr = st.tabs(
        ["Traditional RAG", "PageIndex", "GraphRAG (LlamaIndex)"]
    )
    with tab_rag:
        render_rag(results)
    with tab_pi:
        render_pageindex(results)
    with tab_gr:
        render_graphrag(results)
