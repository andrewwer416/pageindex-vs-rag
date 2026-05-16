"""Compare retrieval architectures on a *collection* — multi-doc queries.

UX mirrors the single-doc Compare page (tabs per pipeline, expandable
reasoning traces) but each tab adds per-doc attribution: which docs
contributed chunks (RAG), which docs the router picked (PageIndex),
which entities span which docs (GraphRAG).
"""
import sys
import time
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import collections, collection_query, library  # noqa: E402


st.set_page_config(page_title="Collection Compare", layout="wide")
st.title("Compare across a collection")
st.caption(
    "Pick a built collection and query it through all three pipelines. Each "
    "tab shows the answer plus its own trace; the trace also tells you "
    "which documents in the collection contributed."
)

# ──────────────── picker ──────────────────────────────────────────────────────

cols = collections.list_collections()
queryable = [c for c in cols if c.get("status") in ("ready", "partial")]
stale = [c for c in cols if c.get("status") == "stale"]

if not queryable and not stale:
    st.info(
        "No built collections. Go to the **Collections** page in the sidebar "
        "to create and build one."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Collection")
    options = queryable + stale
    picked = st.selectbox(
        "Choose a collection",
        options=[c["col_id"] for c in options],
        format_func=lambda cid: next(
            f"{c['name']} ({c.get('status', '?')})" for c in options if c["col_id"] == cid
        ),
    )

col = next(c for c in options if c["col_id"] == picked)

if col.get("status") == "stale":
    st.warning(
        f"**{col['name']}** is stale — one or more member docs were re-indexed "
        "since this collection was built. Queries are blocked until you "
        "rebuild on the **Collections** page."
    )
    st.stop()

if col.get("status") == "partial":
    st.warning(
        f"**{col['name']}** was built with errors — some pipeline tabs may "
        "show 'not available' messages. Rebuild on the Collections page to "
        "retry the failed pipelines."
    )

# Header info
member_names = []
for did in col.get("doc_ids", []):
    try:
        member_names.append(library.get_document(did).get("name", did[:8]))
    except Exception:
        member_names.append(f"{did[:8]} (missing)")

st.markdown(
    f"Querying **{col['name']}** &nbsp;·&nbsp; {len(member_names)} doc(s) &nbsp;·&nbsp; "
    f"status: `{col.get('status')}`"
)
with st.expander("Member documents", expanded=False):
    for n in member_names:
        st.markdown(f"- {n}")

kw = collections.collection_kwargs(picked)
stats = col.get("stats", {})

# ──────────────── question input ─────────────────────────────────────────────

SAMPLES = [
    "What are the main topics covered across these documents?",
    "Which documents mention <topic>?  (Edit topic before running.)",
    "Compare what these documents say about the same subject.",
    "Are there contradictions between the documents?",
    "Summarize the collection in one paragraph.",
]

st.markdown("**Try a sample question, or write your own:**")
if "col_query" not in st.session_state:
    st.session_state["col_query"] = ""
sample_cols = st.columns(len(SAMPLES))
for i, q in enumerate(SAMPLES):
    if sample_cols[i].button(f"#{i+1}", help=q, use_container_width=True):
        st.session_state["col_query"] = q

query = st.text_area("Question", key="col_query", height=80)
go = st.button("Run all pipelines", type="primary", disabled=not query.strip(),
               key=f"col_go_{picked}")


# ──────────────── run pipelines ──────────────────────────────────────────────

def _has_rag(col) -> bool:
    return collections.rag_faiss_path(picked).exists()


def _has_pi(col) -> bool:
    return collections.doc_summaries_path(picked).exists()


def _has_gr(col) -> bool:
    return (collections.graphrag_dir(picked) / "community_summaries.json").exists()


def _run(query: str) -> dict:
    out = {"query": query}
    # RAG
    if _has_rag(col):
        try:
            t = time.time()
            out["rag"] = collection_query.rag_answer(query, **kw["rag"])
            out["rag_time"] = time.time() - t
            out["rag_err"] = None
        except Exception as e:
            out["rag"] = None
            out["rag_err"] = f"{type(e).__name__}: {e}"
            out["rag_time"] = 0.0
    else:
        out["rag"] = None
        out["rag_err"] = "RAG index not built for this collection."
        out["rag_time"] = 0.0
    # PageIndex
    if _has_pi(col):
        try:
            t = time.time()
            out["pi"] = collection_query.pageindex_answer_collection(query, **kw["pi"])
            out["pi_time"] = time.time() - t
            out["pi_err"] = None
        except Exception as e:
            out["pi"] = None
            out["pi_err"] = f"{type(e).__name__}: {e}"
            out["pi_time"] = 0.0
    else:
        out["pi"] = None
        out["pi_err"] = "PageIndex doc-router not built for this collection."
        out["pi_time"] = 0.0
    # GraphRAG
    if _has_gr(col):
        try:
            t = time.time()
            out["gr"] = collection_query.graphrag_answer_collection(query, **kw["graphrag"])
            out["gr_time"] = time.time() - t
            out["gr_err"] = None
        except Exception as e:
            out["gr"] = None
            out["gr_err"] = f"{type(e).__name__}: {e}"
            out["gr_time"] = 0.0
    else:
        out["gr"] = None
        out["gr_err"] = "GraphRAG not built for this collection."
        out["gr_time"] = 0.0
    return out


if go and query.strip():
    with st.spinner("Running all pipelines..."):
        st.session_state[f"col_results_{picked}"] = _run(query)

results = st.session_state.get(f"col_results_{picked}")


# ──────────────── render tabs ────────────────────────────────────────────────

def render_rag(r):
    if r.get("rag_err"):
        (st.info if r["rag"] is None and "not built" in (r["rag_err"] or "")
         else st.error)(r["rag_err"])
        if r["rag"] is None:
            return
    rag = r.get("rag")
    if rag is None:
        return
    st.markdown(f"**Answer** *(generated in {r['rag_time']:.1f}s)*")
    st.write(rag.get("answer", ""))

    contrib = rag.get("doc_contributions", {})
    if contrib:
        st.markdown("**Per-doc contributions** (chunks pulled):")
        st.markdown(" · ".join(f"`{k}`: {v}" for k, v in sorted(contrib.items(), key=lambda kv: -kv[1])))

    with st.expander(f"Retrieved chunks (top {len(rag['hits'])})", expanded=False):
        for h in rag["hits"]:
            pp = (f"p. {h.get('page_start')}" if h.get("page_start") == h.get("page_end")
                  else f"pp. {h.get('page_start')}-{h.get('page_end')}")
            st.markdown(f"**{h.get('doc_name', '?')}** &nbsp;·&nbsp; {pp} &nbsp;·&nbsp; sim `{h['score']:.3f}` &nbsp;·&nbsp; {h.get('tokens', '?')} tok")
            t = h.get("text", "")
            st.text(t[:600] + ("…" if len(t) > 600 else ""))
            st.markdown("---")


def render_pi(r):
    if r.get("pi_err"):
        (st.info if r["pi"] is None and "not built" in (r["pi_err"] or "")
         else st.error)(r["pi_err"])
        if r["pi"] is None:
            return
    pi = r.get("pi")
    if pi is None:
        return
    st.markdown(f"**Answer** *(generated in {r['pi_time']:.1f}s)*")
    st.write(pi.get("answer", ""))

    router = pi.get("router", {})
    if router:
        st.markdown("**Router decision:**")
        st.caption(router.get("reasoning", ""))
        confs = router.get("confidences", {})
        if confs:
            id_to_name = {did: library.get_document(did).get("name", did[:8])
                          for did in confs if did}
            rows = sorted(confs.items(), key=lambda kv: -kv[1])
            for did, conf in rows:
                bar = "█" * int(round(conf * 20))
                in_picked = did in pi.get("selected_doc_ids", [])
                mark = "▶" if in_picked else "  "
                st.markdown(f"`{mark}` **{id_to_name.get(did, did[:8])}** — conf {conf:.2f} {bar}")
        if router.get("fallback_used"):
            st.info(
                "No document cleared the router threshold "
                f"({router.get('threshold')}); walked all docs as fallback."
            )

    per_doc = pi.get("per_doc", [])
    if per_doc:
        with st.expander(f"Per-doc tree walks ({len(per_doc)})", expanded=False):
            for d in per_doc:
                st.markdown(f"**{d['doc_name']}** (confidence {d['confidence']:.2f})")
                if d.get("error"):
                    st.error(d["error"])
                    st.markdown("---")
                    continue
                res = d.get("result") or {}
                st.caption(f"_Pages read:_ {res.get('pages_read', '')}")
                st.write((res.get("answer") or "(empty)")[:1500])
                st.markdown("---")


def render_gr(r):
    if r.get("gr_err"):
        (st.info if r["gr"] is None and "not built" in (r["gr_err"] or "")
         else st.error)(r["gr_err"])
        if r["gr"] is None:
            return
    gr = r.get("gr")
    if gr is None:
        return
    n_comm = gr.get("n_communities", 0)
    if n_comm == 0:
        st.warning(
            "GraphRAG: the merged collection graph produced 0 communities "
            "to answer from — see the diagnostic below."
        )
        st.markdown(gr.get("answer", ""))
        return
    st.markdown(f"**Answer** *(aggregated across {n_comm} communities, "
                f"generated in {r['gr_time']:.1f}s)*")
    st.write(gr.get("answer", ""))

    partials = gr.get("partials", [])
    if partials:
        with st.expander(f"Per-community partial answers ({len(partials)})"):
            for p in partials:
                st.markdown(f"**Community {p['community_id']}**")
                st.caption(f"_Summary:_ {p['summary']}")
                st.write(p["answer"])
                st.markdown("---")


if results:
    tab_rag, tab_pi, tab_gr = st.tabs(
        ["Traditional RAG (merged FAISS)", "PageIndex (router + walks)", "GraphRAG (merged graph)"]
    )
    with tab_rag:
        render_rag(results)
    with tab_pi:
        render_pi(results)
    with tab_gr:
        render_gr(results)
