"""Library + Upload — the entry point.

Two columns: the existing library on the left, an upload form on the right.
Once a document finishes indexing, the user picks it on the **Compare** page
in the sidebar to start querying.
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st

# Make the project root importable when streamlit launches this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import config, library  # noqa: E402
from app.vision import is_enabled as vision_enabled  # noqa: E402


st.set_page_config(page_title="PageIndex vs Traditional RAG — Library", layout="wide")
st.title("PageIndex vs Traditional RAG")
st.markdown(
    "Upload a document, index it once, then query it on the **Compare** page "
    "with both retrieval pipelines side by side. Pages live in the sidebar."
)


# ─── upload form ──────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Upload a new document")

if vision_enabled():
    st.success(
        "🖼️ **Vision processing is ON.** Each image in the document will be "
        "sent to your vision model and the description injected into the page "
        "text — scanned pages, charts, and diagrams become queryable. Adds one "
        "VLM call per image to indexing time."
    )
else:
    st.info(
        "🖼️ Vision processing is **off**. Tables are still parsed structurally, "
        "but images, charts, and scanned-only pages are skipped. To enable, set "
        "`VISION_ENABLED=true` (and a `VISION_MODEL`) in `.env` and restart."
    )

uploaded = st.file_uploader(
    "Choose a PDF, Markdown, or plain-text file",
    type=[ext.lstrip(".") for ext in library.SUPPORTED_EXTS],
    accept_multiple_files=False,
    help="PDF works best (preserves page numbers). Markdown uses heading structure. "
    "Plain text is wrapped into a single-section Markdown.",
)

if uploaded is not None:
    if st.button(f"Index '{uploaded.name}'", type="primary"):
        doc_id = library.add_document(uploaded.getvalue(), uploaded.name)
        st.success(f"Saved as `{doc_id}`. Indexing will block this page until done — keep it open.")

        with st.status(f"Indexing **{uploaded.name}** …", expanded=True) as status:
            t0 = time.time()

            def progress(phase: str, info: dict):
                elapsed = time.time() - t0
                if phase == "chunking":
                    status.write(f"[{elapsed:6.1f}s] 📄 reading + chunking source file")
                elif phase == "embedding":
                    status.write(f"[{elapsed:6.1f}s] 🧮 embedding {info.get('chunks', '?')} chunks for FAISS")
                elif phase == "pageindex":
                    status.write(
                        f"[{elapsed:6.1f}s] 🌳 building PageIndex tree "
                        "(many LLM calls — slow phase)"
                    )
                elif phase == "graphrag":
                    status.write(
                        f"[{elapsed:6.1f}s] 🕸️ building GraphRAG: entity extraction + community summaries"
                    )
                elif phase == "graphrag-done":
                    status.write(
                        f"[{elapsed:6.1f}s] ✅ GraphRAG: {info.get('n_chunks','?')} chunks, "
                        f"{info.get('n_entities','?')} entities, {info.get('n_communities','?')} communities"
                    )
                elif phase == "graphrag-failed":
                    status.write(f"[{elapsed:6.1f}s] ⚠️ GraphRAG: {info.get('error', '')}")
                elif phase == "graphrag-skipped":
                    status.write(f"[{elapsed:6.1f}s] ⏭️ GraphRAG skipped (GRAPHRAG_ENABLED=false)")
                elif phase == "done":
                    status.write(
                        f"[{elapsed:6.1f}s] ✅ PageIndex done — {info.get('page_count') or '?'} pages, "
                        f"doc_id={info.get('pi_doc_id')}"
                    )
                elif phase == "failed":
                    status.write(f"[{elapsed:6.1f}s] ❌ {info.get('error')}")

            result = library.build_indices(doc_id, progress=progress)

        if result.get("status") == "ready":
            st.success(
                f"**{uploaded.name}** is ready. Open the **Compare** page in the "
                "sidebar to start asking questions."
            )
            st.balloons()
        elif result.get("status") == "partial":
            st.warning(
                "RAG side succeeded but the PageIndex tree build failed. The doc is "
                "queryable on the Traditional-RAG side only. Error below:"
            )
            st.code(result.get("error", ""))
        else:
            st.error(f"Indexing failed: {result.get('error', 'unknown')}")


# ─── existing library ─────────────────────────────────────────────────────────

st.markdown("---")
docs = library.list_documents()
st.markdown(f"### Your library &nbsp;·&nbsp; {len(docs)} document(s)")

if not docs:
    st.info("Nothing in your library yet. Upload above.")
else:
    for doc in docs:
        status = doc.get("status", "?")
        badge = {
            "ready": "✅ ready",
            "partial": "⚠️ partial (RAG only)",
            "rag-indexing": "⏳ indexing (RAG)",
            "pageindex-indexing": "⏳ indexing (PageIndex tree)",
            "graphrag-indexing": "⏳ indexing (GraphRAG)",
            "uploaded": "🆕 uploaded",
            "failed": "❌ failed",
        }.get(status, status)
        # Mark which pipelines this doc has built indices for.
        flags = []
        if (library.faiss_path(doc["doc_id"])).exists(): flags.append("RAG")
        if (library.pageindex_doc_id_file(doc["doc_id"])).exists(): flags.append("PageIndex")
        if library.has_graphrag(doc["doc_id"]): flags.append("GraphRAG")
        pipelines_str = " · ".join(flags) if flags else "—"

        col1, col2, col3, col4 = st.columns([4, 2, 3, 1])
        col1.markdown(f"**{doc.get('name', '?')}**  \n`{doc['doc_id']}`")
        col2.markdown(f"_{badge}_")
        col3.markdown(
            (f"{doc.get('page_count') or '—'} pages · "
             if doc.get('ext') == '.pdf' else f"{doc.get('ext', '?')} file · ")
            + pipelines_str
        )
        if col4.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete this document and its indices"):
            ok, msg = library.delete_document(doc["doc_id"])
            if msg:
                (st.warning if ok else st.error)(msg)
            if ok:
                st.rerun()
        if doc.get("error"):
            st.error(doc["error"])
        st.markdown("---")


# ─── system status ────────────────────────────────────────────────────────────

if os.environ.get("SHOW_SYSTEM_STATUS", "true").lower() != "false":
    with st.expander("System status", expanded=False):
        st.markdown("**LLM transport** (redacted):")
        from app.llm import describe_client_config
        st.json(describe_client_config())
        st.markdown("**Embedding transport**:")
        from app.embed import describe_embed_config
        st.json(describe_embed_config())
        st.markdown("**Vision (VLM) transport**:")
        from app.vision import describe_vision_config
        st.json(describe_vision_config())
        st.markdown("**GraphRAG transport**:")
        from app.graphrag_pipeline import describe_graphrag_config
        st.json(describe_graphrag_config())
