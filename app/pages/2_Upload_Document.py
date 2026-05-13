"""Upload page: add a PDF/Markdown/TXT to the library and build both indices.

Indexing is blocking — the page shows a live status panel while RAG embeddings
and the PageIndex tree are built. Total time scales with doc size + LLM speed;
expect minutes-to-tens-of-minutes per doc.
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
from app.vision import is_enabled as vision_enabled  # noqa: E402


st.set_page_config(page_title="Upload — PageIndex vs RAG", layout="wide")
st.title("Upload a document")
st.caption(
    "Add a PDF, Markdown, or plain-text document. Both indices (FAISS for RAG + "
    "PageIndex tree) are built once at upload time. After that the document shows "
    "up under **Compare Custom** for side-by-side querying."
)

if vision_enabled():
    st.success(
        "🖼️ **Vision processing is ON.** During indexing each image will be sent "
        "to your vision model and its description will be injected into the "
        "page text — scanned pages, charts, and diagrams become queryable. "
        "Adds one VLM call per image to indexing time."
    )
else:
    st.info(
        "🖼️ Vision processing is **off**. Tables are still parsed structurally, "
        "but images, charts, and scanned-only pages are skipped. To enable, set "
        "`VISION_ENABLED=true` (and a `VISION_MODEL`) in `.env` and restart."
    )


# ──────────────── existing library ─────────────────────────────────────────────

docs = library.list_documents()
st.markdown(f"### Your library &nbsp;·&nbsp; {len(docs)} document(s)")

if not docs:
    st.info("Nothing in your library yet. Upload below.")
else:
    for doc in docs:
        status = doc.get("status", "?")
        badge = {
            "ready": "✅ ready",
            "partial": "⚠️ partial (RAG only)",
            "rag-indexing": "⏳ indexing (RAG)",
            "pageindex-indexing": "⏳ indexing (PageIndex tree)",
            "uploaded": "🆕 uploaded",
            "failed": "❌ failed",
        }.get(status, status)

        col1, col2, col3, col4 = st.columns([5, 2, 2, 1])
        col1.markdown(f"**{doc.get('name', '?')}**  \n`{doc['doc_id']}`")
        col2.markdown(f"_{badge}_")
        col3.markdown(
            f"{doc.get('page_count') or '—'} pages"
            if doc.get("ext") == ".pdf"
            else f"{doc.get('ext', '?')} file"
        )
        if col4.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete this document and its indices"):
            library.delete_document(doc["doc_id"])
            st.rerun()
        if doc.get("error"):
            st.error(doc["error"])
        st.markdown("---")


# ──────────────── upload form ──────────────────────────────────────────────────

st.markdown("### Upload new document")
uploaded = st.file_uploader(
    "Choose a file",
    type=[ext.lstrip(".") for ext in library.SUPPORTED_EXTS],
    accept_multiple_files=False,
    help="PDF works best (preserves page numbers). Markdown uses heading structure. "
    "Plain text is wrapped into a single-section Markdown.",
)

if uploaded is not None:
    if st.button(f"Index '{uploaded.name}'", type="primary"):
        # Save first so we have a stable doc_id throughout indexing.
        doc_id = library.add_document(uploaded.getvalue(), uploaded.name)
        st.success(f"Saved as `{doc_id}`. Indexing will block this page until done — keep it open.")

        with st.status(f"Indexing **{uploaded.name}** …", expanded=True) as status:
            phases_seen = []
            t0 = time.time()

            def progress(phase: str, info: dict):
                phases_seen.append(phase)
                elapsed = time.time() - t0
                if phase == "chunking":
                    status.write(f"[{elapsed:6.1f}s] 📄 reading + chunking source file")
                elif phase == "embedding":
                    status.write(f"[{elapsed:6.1f}s] 🧮 embedding {info.get('chunks', '?')} chunks for FAISS")
                elif phase == "pageindex":
                    status.write(
                        f"[{elapsed:6.1f}s] 🌳 building PageIndex tree "
                        f"(many LLM calls — this is the slow phase)"
                    )
                elif phase == "done":
                    status.write(
                        f"[{elapsed:6.1f}s] ✅ done — {info.get('page_count') or '?'} pages, "
                        f"PageIndex doc_id={info.get('pi_doc_id')}"
                    )
                elif phase == "failed":
                    status.write(f"[{elapsed:6.1f}s] ❌ {info.get('error')}")

            result = library.build_indices(doc_id, progress=progress)

        if result.get("status") == "ready":
            st.success(
                f"**{uploaded.name}** is ready. Switch to the **Compare Custom** page "
                "in the sidebar to start asking questions."
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
