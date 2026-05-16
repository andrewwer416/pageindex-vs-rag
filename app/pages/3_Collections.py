"""Collections page: group library docs into named collections that get
queried as one cross-document corpus on the Collection Compare page.

UX:
  - Top: list of existing collections, each with a status badge and
    member-doc count. 'stale' collections show a 'Rebuild required' notice.
  - Middle: 'Create new collection' form — pick a name, multi-select docs
    from the library, click Create. After creation the user is prompted to
    Build.
  - Build runs all three pipelines; each phase reports progress via the
    same st.status pattern as the Library page.
"""
import sys
import time
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import collections, library  # noqa: E402


st.set_page_config(page_title="Collections — PageIndex vs RAG", layout="wide")
st.title("Collections")
st.caption(
    "Group documents into a collection to query the whole corpus at once. "
    "Each pipeline gets a collection-level index built on top of the per-doc "
    "indices: a merged FAISS for RAG, a doc-router for PageIndex, and a "
    "cross-doc entity-resolved knowledge graph for GraphRAG."
)


# ─── existing collections ─────────────────────────────────────────────────────

st.markdown("---")
cols = collections.list_collections()
st.markdown(f"### Your collections &nbsp;·&nbsp; {len(cols)} collection(s)")


def _badge(status: str) -> str:
    return {
        "ready":                     "✅ ready",
        "stale":                     "⚠️ rebuild required",
        "partial":                   "⚠️ partial (some pipelines failed)",
        "created":                   "🆕 not yet built",
        "rag-building":              "⏳ building (RAG)",
        "pageindex-router-building": "⏳ building (PageIndex router)",
        "graphrag-resolving":        "⏳ building (GraphRAG)",
        "failed":                    "❌ failed",
    }.get(status, status)


if not cols:
    st.info("No collections yet. Create one below.")
else:
    for col in cols:
        col_id = col["col_id"]
        member_names = []
        for did in col.get("doc_ids", []):
            try:
                member_names.append(library.get_document(did).get("name", did[:8]))
            except Exception:
                member_names.append(f"{did[:8]} (missing)")

        c1, c2, c3, c4 = st.columns([4, 2, 3, 1])
        c1.markdown(f"**{col.get('name', '?')}**  \n`{col_id}`")
        c2.markdown(f"_{_badge(col.get('status', '?'))}_")
        c3.markdown(f"{len(member_names)} doc(s) · " + ", ".join(member_names[:3])
                    + (f" +{len(member_names) - 3} more" if len(member_names) > 3 else ""))
        if c4.button("🗑️", key=f"del_col_{col_id}", help="Delete this collection"):
            ok, msg = collections.delete_collection(col_id)
            if msg:
                (st.warning if ok else st.error)(msg)
            if ok:
                st.rerun()

        if col.get("status") == "stale":
            st.warning(
                "One or more member documents were re-indexed (or the doc set "
                "changed) since this collection was built. Queries are blocked "
                "until you rebuild — the existing indices may not reflect the "
                "current state of the library."
            )
        if col.get("error"):
            st.error(col["error"])

        # Build / rebuild button
        needs_build = col.get("status") in ("created", "stale", "partial", "failed")
        label = "Build collection" if col.get("status") == "created" else "Rebuild collection"
        if needs_build and st.button(label, key=f"build_{col_id}", type="primary"):
            with st.status(f"Building **{col['name']}** …", expanded=True) as status:
                t0 = time.time()

                def progress(phase: str, info: dict):
                    elapsed = time.time() - t0
                    if phase == "rag":
                        status.write(f"[{elapsed:6.1f}s] 🔍 merging RAG indices across {info.get('doc_count', '?')} docs")
                    elif phase == "rag-done":
                        status.write(f"[{elapsed:6.1f}s] ✅ RAG: {info.get('n_chunks', '?')} chunks, "
                                     f"{info.get('vec_dim', '?')}-dim")
                    elif phase == "rag-failed":
                        status.write(f"[{elapsed:6.1f}s] ❌ RAG: {info.get('error', '')}")
                    elif phase == "pageindex-router":
                        status.write(f"[{elapsed:6.1f}s] 🧭 building PageIndex doc-router "
                                     f"({info.get('doc_count', '?')} summaries)")
                    elif phase == "pageindex-router-done":
                        status.write(f"[{elapsed:6.1f}s] ✅ PageIndex router: "
                                     f"{info.get('n_docs_summarized', '?')} doc summaries")
                    elif phase == "pageindex-router-failed":
                        status.write(f"[{elapsed:6.1f}s] ❌ PageIndex router: {info.get('error', '')}")
                    elif phase == "graphrag-resolution":
                        status.write(f"[{elapsed:6.1f}s] 🔗 GraphRAG: LLM entity resolution across "
                                     f"{info.get('doc_count', '?')} graphs")
                    elif phase == "graphrag-merge":
                        status.write(f"[{elapsed:6.1f}s] 🕸️ GraphRAG: merging graphs")
                    elif phase == "graphrag-communities":
                        status.write(f"[{elapsed:6.1f}s] 🗂️ GraphRAG: community detection + summaries")
                    elif phase == "graphrag-done":
                        status.write(f"[{elapsed:6.1f}s] ✅ GraphRAG: "
                                     f"{info.get('n_entities', '?')} merged entities, "
                                     f"{info.get('n_communities', '?')} communities")
                    elif phase == "graphrag-failed":
                        err = info.get("error", "")
                        first = err.split("\n")[0]
                        status.write(f"[{elapsed:6.1f}s] ❌ GraphRAG: {first}")
                        if "\n" in err:
                            status.code(err, language="python")
                    elif phase == "done":
                        status.write(f"[{elapsed:6.1f}s] 🎉 collection build complete (status={info.get('status', '?')})")
                    elif phase == "failed":
                        status.write(f"[{elapsed:6.1f}s] ❌ {info.get('error', 'failed')}")

                result = collections.build_collection(col_id, progress=progress)

            if result.get("status") == "ready":
                st.success(f"**{col['name']}** is ready.")
                st.balloons()
            elif result.get("status") == "partial":
                st.warning(f"**{col['name']}** built with errors in some pipelines — see status panel above.")
            else:
                st.error(f"Build failed: {result.get('error', 'unknown')}")
            st.rerun()
        st.markdown("---")


# ─── create new collection ────────────────────────────────────────────────────

st.markdown("### Create a new collection")

docs = library.list_documents()
eligible = [d for d in docs if d.get("status") in ("ready", "partial")]

if not eligible:
    st.info(
        "No ready documents in your library yet. Upload and index documents "
        "on the **app** page first."
    )
else:
    with st.form("new_collection_form", clear_on_submit=True):
        name = st.text_input("Collection name", placeholder="e.g. 'AI research papers'")
        picked = st.multiselect(
            "Pick documents to include",
            options=[d["doc_id"] for d in eligible],
            format_func=lambda did: next(
                f"{d['name']} ({d.get('page_count') or '?'} pages, {d.get('status')})"
                for d in eligible if d["doc_id"] == did
            ),
            help="Only ready (or partial) docs can be added.",
        )
        submitted = st.form_submit_button("Create collection")

    if submitted:
        if not name.strip():
            st.error("Give the collection a name.")
        elif not picked:
            st.error("Pick at least one document.")
        else:
            try:
                col_id = collections.create_collection(name, picked)
                st.success(f"Created `{col_id}`. Hit **Build collection** above to index it.")
                time.sleep(0.4)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
